#!/usr/bin/env python3
"""
Stage 2b: Cross-check waveform inventory against EHR data.

Filters the waveform patient list to only those with EHR events.
This runs AFTER stage 1 (waveform scan) and stage 2 (EHR extraction),
and BEFORE stage 3 (waveform extraction).

Purpose: don't waste hours extracting waveforms for patients with no labs/vitals.

Run:  python workzone/mimic3/stage2b_cross_check.py
Output: workzone/outputs/mimic3/record_inventory_final.parquet
"""
import os
import json
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO_ROOT / "workzone" / "outputs" / "mimic3"

# Minimum number of EHR events to keep a patient
MIN_LAB_EVENTS = 1
MIN_TOTAL_EHR_EVENTS = 3  # at least 3 measurements (any variable)


def main():
    log.info("Stage 2b: Cross-check waveform patients against EHR availability")
    t0 = time.time()

    # Load waveform inventory (from stage 1)
    wav_inv = pd.read_parquet(OUT_DIR / "record_inventory_filtered.parquet")
    log.info(f"Waveform inventory: {len(wav_inv)} patients with PLETH+II")

    # Load EHR data (from stage 2)
    labs = pl.read_parquet(OUT_DIR / "labs_filtered.parquet")
    vitals = pl.read_parquet(OUT_DIR / "vitals_filtered.parquet")

    # Count EHR events per SUBJECT_ID
    lab_counts = labs.group_by("SUBJECT_ID").agg(
        pl.count().alias("n_lab_events"),
        pl.col("var_id").n_unique().alias("n_lab_variables"),
    ).to_pandas()

    vital_counts = vitals.group_by("SUBJECT_ID").agg(
        pl.count().alias("n_vital_events"),
        pl.col("var_id").n_unique().alias("n_vital_variables"),
    ).to_pandas()

    # Merge with waveform inventory
    merged = wav_inv.merge(lab_counts, left_on="subject_id", right_on="SUBJECT_ID", how="left")
    merged = merged.merge(vital_counts, left_on="subject_id", right_on="SUBJECT_ID", how="left")

    # Fill NaN (patients with no EHR) with 0
    for col in ["n_lab_events", "n_lab_variables", "n_vital_events", "n_vital_variables"]:
        merged[col] = merged[col].fillna(0).astype(int)

    merged["n_total_ehr"] = merged["n_lab_events"] + merged["n_vital_events"]

    # Stats before filtering
    n_with_labs = (merged["n_lab_events"] > 0).sum()
    n_with_vitals = (merged["n_vital_events"] > 0).sum()
    n_with_both = ((merged["n_lab_events"] > 0) & (merged["n_vital_events"] > 0)).sum()
    n_with_none = (merged["n_total_ehr"] == 0).sum()

    log.info(f"\n=== EHR Overlap ===")
    log.info(f"  Waveform patients:         {len(merged)}")
    log.info(f"  With labs:                 {n_with_labs}")
    log.info(f"  With vitals:               {n_with_vitals}")
    log.info(f"  With both labs + vitals:    {n_with_both}")
    log.info(f"  With NO EHR at all:        {n_with_none}")

    # Filter: must have minimum EHR events
    final = merged[
        (merged["n_lab_events"] >= MIN_LAB_EVENTS) &
        (merged["n_total_ehr"] >= MIN_TOTAL_EHR_EVENTS)
    ].copy()

    log.info(f"\n=== After Filter (>={MIN_LAB_EVENTS} lab events, >={MIN_TOTAL_EHR_EVENTS} total EHR) ===")
    log.info(f"  Patients remaining: {len(final)} (dropped {len(merged) - len(final)})")

    # Distribution of EHR events
    if len(final) > 0:
        log.info(f"\n=== EHR Distribution (filtered patients) ===")
        log.info(f"  Lab events:   median={final['n_lab_events'].median():.0f}, "
                 f"min={final['n_lab_events'].min()}, max={final['n_lab_events'].max()}")
        log.info(f"  Vital events: median={final['n_vital_events'].median():.0f}, "
                 f"min={final['n_vital_events'].min()}, max={final['n_vital_events'].max()}")
        log.info(f"  Total EHR:    median={final['n_total_ehr'].median():.0f}")
        log.info(f"  Lab variables covered:   median={final['n_lab_variables'].median():.0f}")
        log.info(f"  Vital variables covered: median={final['n_vital_variables'].median():.0f}")

    # Save final inventory (this is what stage 3 reads)
    out_path = OUT_DIR / "record_inventory_final.parquet"
    final.to_parquet(out_path, index=False)
    log.info(f"\nSaved: {out_path} ({len(final)} patients)")

    # Summary
    summary = {
        "waveform_patients": len(wav_inv),
        "with_labs": int(n_with_labs),
        "with_vitals": int(n_with_vitals),
        "with_both": int(n_with_both),
        "with_no_ehr": int(n_with_none),
        "after_filter": len(final),
        "dropped": len(merged) - len(final),
        "min_lab_events": MIN_LAB_EVENTS,
        "min_total_ehr": MIN_TOTAL_EHR_EVENTS,
    }
    with open(OUT_DIR / "stage2b_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    elapsed = time.time() - t0

    # Verification
    log.info(f"\n=== Verification ===")
    assert len(final) > 0, "ABORT: No patients with waveform + EHR overlap!"
    assert final["n_lab_events"].min() >= MIN_LAB_EVENTS
    assert final["n_total_ehr"].min() >= MIN_TOTAL_EHR_EVENTS
    log.info(f"  [PASS] {len(final)} patients have waveform + EHR")
    log.info(f"  [PASS] All patients have >= {MIN_LAB_EVENTS} lab events")
    log.info(f"  Time: {elapsed:.1f}s")

    log.info(f"\nNext: python workzone/mimic3/stage3_extract_waveforms.py")


if __name__ == "__main__":
    main()
