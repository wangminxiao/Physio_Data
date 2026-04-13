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

# ---- Filtering rules ----
# Total target variables: 6 labs + 3 vitals = 9
TOTAL_TARGET_VARS = 9
# At least 70% of target variables must have data
MIN_VAR_COVERAGE = 0.70  # -> need at least 7 out of 9 variables present
MIN_VARS_PRESENT = int(np.ceil(TOTAL_TARGET_VARS * MIN_VAR_COVERAGE))
# Each present variable must have at least 2 data points
MIN_POINTS_PER_VAR = 2


def compute_per_patient_var_coverage(labs, vitals):
    """For each SUBJECT_ID, count how many distinct var_ids have >= MIN_POINTS_PER_VAR."""
    combined = pl.concat([
        labs.select(["SUBJECT_ID", "var_id"]),
        vitals.select(["SUBJECT_ID", "var_id"]),
    ])

    # Count events per (SUBJECT_ID, var_id)
    per_var = combined.group_by(["SUBJECT_ID", "var_id"]).agg(
        pl.count().alias("n_events")
    )

    # Keep only variables with >= MIN_POINTS_PER_VAR
    qualified = per_var.filter(pl.col("n_events") >= MIN_POINTS_PER_VAR)

    # Count how many qualified variables each patient has
    coverage = qualified.group_by("SUBJECT_ID").agg(
        pl.col("var_id").n_unique().alias("n_vars_qualified"),
        pl.col("var_id").alias("qualified_var_ids"),
    ).to_pandas()

    return coverage


def main():
    log.info("Stage 2b: Cross-check waveform patients against EHR availability")
    log.info(f"  Rules: >={MIN_VAR_COVERAGE*100:.0f}% of {TOTAL_TARGET_VARS} target vars present "
             f"(>= {MIN_VARS_PRESENT} vars), each with >= {MIN_POINTS_PER_VAR} data points")
    t0 = time.time()

    # Load waveform inventory (from stage 1)
    wav_inv = pd.read_parquet(OUT_DIR / "record_inventory_filtered.parquet")
    log.info(f"Waveform inventory: {len(wav_inv)} patients with PLETH+II")

    # Load EHR data (from stage 2)
    labs = pl.read_parquet(OUT_DIR / "labs_filtered.parquet")
    vitals = pl.read_parquet(OUT_DIR / "vitals_filtered.parquet")

    # Compute per-patient variable coverage
    coverage = compute_per_patient_var_coverage(labs, vitals)

    # Also get basic event counts for reporting
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
    merged = merged.merge(coverage, left_on="subject_id", right_on="SUBJECT_ID", how="left")

    # Fill NaN with 0
    for col in ["n_lab_events", "n_lab_variables", "n_vital_events", "n_vital_variables", "n_vars_qualified"]:
        merged[col] = merged[col].fillna(0).astype(int)

    merged["n_total_ehr"] = merged["n_lab_events"] + merged["n_vital_events"]

    # Stats before filtering
    n_with_labs = (merged["n_lab_events"] > 0).sum()
    n_with_vitals = (merged["n_vital_events"] > 0).sum()
    n_with_both = ((merged["n_lab_events"] > 0) & (merged["n_vital_events"] > 0)).sum()
    n_with_none = (merged["n_total_ehr"] == 0).sum()

    log.info(f"\n=== EHR Overlap (before filter) ===")
    log.info(f"  Waveform patients:         {len(merged)}")
    log.info(f"  With any labs:             {n_with_labs}")
    log.info(f"  With any vitals:           {n_with_vitals}")
    log.info(f"  With both labs + vitals:    {n_with_both}")
    log.info(f"  With NO EHR at all:        {n_with_none}")

    # Variable coverage distribution
    log.info(f"\n=== Variable Coverage (vars with >= {MIN_POINTS_PER_VAR} points) ===")
    for n_vars in range(TOTAL_TARGET_VARS + 1):
        count = (merged["n_vars_qualified"] == n_vars).sum()
        marker = " <-- threshold" if n_vars == MIN_VARS_PRESENT else ""
        log.info(f"  {n_vars}/{TOTAL_TARGET_VARS} vars: {count} patients{marker}")

    # Apply filter: >= MIN_VARS_PRESENT variables, each with >= MIN_POINTS_PER_VAR points
    final = merged[merged["n_vars_qualified"] >= MIN_VARS_PRESENT].copy()

    log.info(f"\n=== After Filter (>= {MIN_VARS_PRESENT}/{TOTAL_TARGET_VARS} vars, each >= {MIN_POINTS_PER_VAR} points) ===")
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
        "filter_rules": {
            "min_var_coverage": MIN_VAR_COVERAGE,
            "min_vars_present": MIN_VARS_PRESENT,
            "total_target_vars": TOTAL_TARGET_VARS,
            "min_points_per_var": MIN_POINTS_PER_VAR,
        },
    }
    with open(OUT_DIR / "stage2b_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    elapsed = time.time() - t0

    # Verification
    log.info(f"\n=== Verification ===")
    assert len(final) > 0, "ABORT: No patients with waveform + EHR overlap!"
    assert final["n_vars_qualified"].min() >= MIN_VARS_PRESENT
    log.info(f"  [PASS] {len(final)} patients have waveform + EHR")
    log.info(f"  [PASS] All patients have >= {MIN_VARS_PRESENT}/{TOTAL_TARGET_VARS} vars, each >= {MIN_POINTS_PER_VAR} points")
    log.info(f"  Time: {elapsed:.1f}s")

    log.info(f"\nNext: python workzone/mimic3/stage3_extract_waveforms.py")


if __name__ == "__main__":
    main()
