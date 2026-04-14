#!/usr/bin/env python3
"""
Stage 2b: Cross-check waveform inventory against EHR data.

Filters the waveform patient list to only those with ANY EHR events.
Loose filter: any patient with at least 1 EHR event is kept.
Rationale: EHR events are ~14 KB per patient vs ~14 MB waveforms = free to store.
           Downstream tasks filter variables at runtime.

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

# Loose filter: just need at least 1 EHR event
MIN_TOTAL_EHR_EVENTS = 1


def main():
    log.info("Stage 2b: Cross-check waveform patients against EHR availability")
    log.info(f"  Filter: any patient with >= {MIN_TOTAL_EHR_EVENTS} total EHR events")
    t0 = time.time()

    # Load waveform inventory (from stage 1)
    wav_inv = pd.read_parquet(OUT_DIR / "record_inventory_filtered.parquet")
    log.info(f"Waveform inventory: {len(wav_inv)} patients with PLETH+II")

    # Load EHR data (from stage 2)
    labs = pl.read_parquet(OUT_DIR / "labs_filtered.parquet")
    vitals = pl.read_parquet(OUT_DIR / "vitals_filtered.parquet")
    log.info(f"Labs: {len(labs):,} events, Vitals: {len(vitals):,} events")

    # Compute per-patient event counts and variable coverage (for reporting)
    lab_counts = labs.group_by("SUBJECT_ID").agg(
        pl.count().alias("n_lab_events"),
        pl.col("var_id").n_unique().alias("n_lab_variables"),
    ).to_pandas()

    vital_counts = vitals.group_by("SUBJECT_ID").agg(
        pl.count().alias("n_vital_events"),
        pl.col("var_id").n_unique().alias("n_vital_variables"),
    ).to_pandas()

    # Merge all with waveform inventory
    merged = wav_inv.merge(lab_counts, left_on="subject_id", right_on="SUBJECT_ID", how="left")
    merged = merged.merge(vital_counts, left_on="subject_id", right_on="SUBJECT_ID", how="left")

    # Fill NaN with 0
    for col in ["n_lab_events", "n_lab_variables", "n_vital_events", "n_vital_variables"]:
        merged[col] = merged[col].fillna(0).astype(int)

    merged["n_total_ehr"] = merged["n_lab_events"] + merged["n_vital_events"]
    merged["n_total_variables"] = merged["n_lab_variables"] + merged["n_vital_variables"]

    # Stats before filtering
    n_with_labs = (merged["n_lab_events"] > 0).sum()
    n_with_vitals = (merged["n_vital_events"] > 0).sum()
    n_with_both = ((merged["n_lab_events"] > 0) & (merged["n_vital_events"] > 0)).sum()
    n_with_any = (merged["n_total_ehr"] > 0).sum()
    n_with_none = (merged["n_total_ehr"] == 0).sum()

    log.info(f"\n=== EHR Overlap (before filter) ===")
    log.info(f"  Waveform patients:         {len(merged)}")
    log.info(f"  With any labs:             {n_with_labs}")
    log.info(f"  With any vitals:           {n_with_vitals}")
    log.info(f"  With both labs + vitals:   {n_with_both}")
    log.info(f"  With any EHR:              {n_with_any}")
    log.info(f"  With NO EHR at all:        {n_with_none}")

    # Variable coverage distribution (for reporting only, not filtering)
    combined = pl.concat([
        labs.select(["SUBJECT_ID", "var_id"]),
        vitals.select(["SUBJECT_ID", "var_id"]),
    ])
    per_patient_vars = combined.group_by("SUBJECT_ID").agg(
        pl.col("var_id").n_unique().alias("n_vars")
    ).to_pandas()

    merged = merged.merge(per_patient_vars, left_on="subject_id", right_on="SUBJECT_ID", how="left")
    merged["n_vars"] = merged["n_vars"].fillna(0).astype(int)

    log.info(f"\n=== Variable Coverage Distribution ===")
    for n_vars in range(0, merged["n_vars"].max() + 1):
        count = (merged["n_vars"] == n_vars).sum()
        if count > 0:
            log.info(f"  {n_vars:2d} vars: {count:5d} patients")

    # Apply loose filter: any EHR overlap
    final = merged[merged["n_total_ehr"] >= MIN_TOTAL_EHR_EVENTS].copy()

    log.info(f"\n=== After Filter (>= {MIN_TOTAL_EHR_EVENTS} EHR events) ===")
    log.info(f"  Patients remaining: {len(final)} (dropped {len(merged) - len(final)})")

    # Distribution of EHR events
    if len(final) > 0:
        log.info(f"\n=== EHR Distribution (filtered patients) ===")
        log.info(f"  Lab events:   median={final['n_lab_events'].median():.0f}, "
                 f"min={final['n_lab_events'].min()}, max={final['n_lab_events'].max()}")
        log.info(f"  Vital events: median={final['n_vital_events'].median():.0f}, "
                 f"min={final['n_vital_events'].min()}, max={final['n_vital_events'].max()}")
        log.info(f"  Total EHR:    median={final['n_total_ehr'].median():.0f}")
        log.info(f"  Variables:    median={final['n_vars'].median():.0f}")

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
        "with_any_ehr": int(n_with_any),
        "with_no_ehr": int(n_with_none),
        "after_filter": len(final),
        "dropped": len(merged) - len(final),
        "filter_rules": {
            "min_total_ehr_events": MIN_TOTAL_EHR_EVENTS,
            "note": "Loose filter: store all EHR, filter at runtime",
        },
    }
    with open(OUT_DIR / "stage2b_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    elapsed = time.time() - t0

    # Verification
    log.info(f"\n=== Verification ===")
    assert len(final) > 0, "ABORT: No patients with waveform + EHR overlap!"
    assert final["n_total_ehr"].min() >= MIN_TOTAL_EHR_EVENTS
    log.info(f"  [PASS] {len(final)} patients have waveform + EHR")
    log.info(f"  Time: {elapsed:.1f}s")

    log.info(f"\nNext: python workzone/mimic3/stage3_extract_waveforms.py")


if __name__ == "__main__":
    main()
