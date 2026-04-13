#!/usr/bin/env python3
"""
Stage 2: Extract and filter EHR data (labs + vitals) from raw MIMIC-III CSVs.

Reads LABEVENTS.csv and CHARTEVENTS.csv.gz, filters to target variables,
joins with ADMISSIONS for HADM_ID linkage, computes normalization stats.

Run:  python workzone/mimic3/stage2_extract_ehr.py
Output:
  workzone/outputs/mimic3/labs_filtered.parquet
  workzone/outputs/mimic3/vitals_filtered.parquet
  workzone/outputs/mimic3/normalization_stats.json
  workzone/outputs/mimic3/stage2_summary.json
"""
import os
import json
import time
import logging
from pathlib import Path

import numpy as np
import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO_ROOT / "workzone" / "outputs" / "mimic3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

import yaml
with open(REPO_ROOT / "workzone" / "configs" / "server_paths.yaml") as f:
    cfg = yaml.safe_load(f)

EHR_ROOT = cfg["mimic3"]["raw_ehr_dir"]

# Variable definitions from API.md / var_registry.json
LAB_ITEMS = {
    # var_id: (name, [ITEMIDs], unit, physiological_min, physiological_max)
    0: ("Potassium",  [50971, 50822], "mEq/L", 1.0, 10.0),
    1: ("Calcium",    [50893],        "mg/dL", 4.0, 15.0),
    2: ("Sodium",     [50983, 50824], "mEq/L", 100, 180),
    3: ("Glucose",    [50931, 50809], "mg/dL", 10, 1000),
    4: ("Lactate",    [50813],        "mmol/L", 0.1, 30.0),
    5: ("Creatinine", [50912],        "mg/dL", 0.1, 25.0),
}

VITAL_ITEMS = {
    # var_id: (name, [ITEMIDs], unit, physiological_min, physiological_max)
    6: ("NBPs", [220179], "mmHg", 30, 300),
    7: ("NBPd", [220180], "mmHg", 10, 200),
    8: ("NBPm", [220181], "mmHg", 20, 250),
}

# Build ITEMID -> var_id lookup
ITEMID_TO_VARID = {}
for var_id, (name, itemids, unit, vmin, vmax) in {**LAB_ITEMS, **VITAL_ITEMS}.items():
    for iid in itemids:
        ITEMID_TO_VARID[iid] = var_id


def extract_labs():
    """Extract lab events from LABEVENTS.csv."""
    log.info("=== Extracting Labs ===")
    lab_path = os.path.join(EHR_ROOT, "LABEVENTS.csv")
    if not os.path.exists(lab_path):
        lab_path = os.path.join(EHR_ROOT, "LABEVENTS.csv.gz")
    log.info(f"  Reading: {lab_path}")

    target_itemids = []
    for var_id, (name, itemids, *_) in LAB_ITEMS.items():
        target_itemids.extend(itemids)

    t0 = time.time()
    df = pl.scan_csv(lab_path, infer_schema_length=1000).filter(
        pl.col("ITEMID").is_in(target_itemids) &
        pl.col("VALUENUM").is_not_null() &
        pl.col("VALUENUM").is_not_nan()
    ).select([
        "SUBJECT_ID", "HADM_ID", "ITEMID", "CHARTTIME", "VALUENUM"
    ]).collect()

    elapsed = time.time() - t0
    log.info(f"  Loaded {len(df)} lab events in {elapsed:.1f}s")

    # Map ITEMID -> var_id
    itemid_map = pl.DataFrame({
        "ITEMID": list(ITEMID_TO_VARID.keys()),
        "var_id": [ITEMID_TO_VARID[k] for k in ITEMID_TO_VARID.keys()],
    }).filter(pl.col("var_id") < 6)  # labs only

    df = df.join(itemid_map, on="ITEMID", how="inner")

    # Apply physiological range filters
    rows_before = len(df)
    range_filters = []
    for var_id, (name, _, _, vmin, vmax) in LAB_ITEMS.items():
        range_filters.append(
            (pl.col("var_id") == var_id) &
            (pl.col("VALUENUM") >= vmin) &
            (pl.col("VALUENUM") <= vmax)
        )

    combined_filter = range_filters[0]
    for rf in range_filters[1:]:
        combined_filter = combined_filter | rf

    df = df.filter(combined_filter)
    rows_after = len(df)
    log.info(f"  Range filter: {rows_before} -> {rows_after} ({rows_before - rows_after} removed)")

    # Deduplicate: keep first per (SUBJECT_ID, CHARTTIME, var_id)
    df = df.sort("CHARTTIME").unique(
        subset=["SUBJECT_ID", "CHARTTIME", "var_id"], keep="first"
    )
    log.info(f"  After dedup: {len(df)} events")

    # Parse CHARTTIME
    df = df.with_columns(
        pl.col("CHARTTIME").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S").alias("charttime_dt")
    )

    return df


def extract_vitals():
    """Extract vital events from CHARTEVENTS.csv.gz."""
    log.info("\n=== Extracting Vitals ===")
    # Try .csv.gz first, then .csv
    vital_path = os.path.join(EHR_ROOT, "CHARTEVENTS.csv.gz")
    if not os.path.exists(vital_path):
        vital_path = os.path.join(EHR_ROOT, "CHARTEVENTS.csv")
    log.info(f"  Reading: {vital_path}")

    target_itemids = []
    for var_id, (name, itemids, *_) in VITAL_ITEMS.items():
        target_itemids.extend(itemids)

    t0 = time.time()
    df = pl.scan_csv(vital_path, infer_schema_length=1000).filter(
        pl.col("ITEMID").is_in(target_itemids) &
        pl.col("VALUENUM").is_not_null() &
        pl.col("VALUENUM").is_not_nan()
    ).select([
        "SUBJECT_ID", "HADM_ID", "ITEMID", "CHARTTIME", "VALUENUM"
    ]).collect()

    elapsed = time.time() - t0
    log.info(f"  Loaded {len(df)} vital events in {elapsed:.1f}s")

    # Map ITEMID -> var_id
    itemid_map = pl.DataFrame({
        "ITEMID": list(ITEMID_TO_VARID.keys()),
        "var_id": [ITEMID_TO_VARID[k] for k in ITEMID_TO_VARID.keys()],
    }).filter(pl.col("var_id") >= 6)  # vitals only

    df = df.join(itemid_map, on="ITEMID", how="inner")

    # Apply physiological range filters
    rows_before = len(df)
    range_filters = []
    for var_id, (name, _, _, vmin, vmax) in VITAL_ITEMS.items():
        range_filters.append(
            (pl.col("var_id") == var_id) &
            (pl.col("VALUENUM") >= vmin) &
            (pl.col("VALUENUM") <= vmax)
        )

    combined_filter = range_filters[0]
    for rf in range_filters[1:]:
        combined_filter = combined_filter | rf

    df = df.filter(combined_filter)
    rows_after = len(df)
    log.info(f"  Range filter: {rows_before} -> {rows_after} ({rows_before - rows_after} removed)")

    # Deduplicate
    df = df.sort("CHARTTIME").unique(
        subset=["SUBJECT_ID", "CHARTTIME", "var_id"], keep="first"
    )
    log.info(f"  After dedup: {len(df)} events")

    # Parse CHARTTIME
    df = df.with_columns(
        pl.col("CHARTTIME").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S").alias("charttime_dt")
    )

    return df


def compute_normalization(labs_df, vitals_df):
    """Compute per-variable normalization stats."""
    log.info("\n=== Normalization Stats ===")
    all_items = {**LAB_ITEMS, **VITAL_ITEMS}
    stats = {}

    combined = pl.concat([
        labs_df.select(["var_id", "VALUENUM"]),
        vitals_df.select(["var_id", "VALUENUM"]),
    ])

    for var_id, (name, _, unit, vmin, vmax) in all_items.items():
        subset = combined.filter(pl.col("var_id") == var_id)["VALUENUM"]
        if len(subset) == 0:
            log.warning(f"  {name} (var_id={var_id}): no data!")
            continue

        vals = subset.to_numpy()
        stats[str(var_id)] = {
            "name": name,
            "unit": unit,
            "count": int(len(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "p01": float(np.percentile(vals, 1)),
            "p99": float(np.percentile(vals, 99)),
            "median": float(np.median(vals)),
        }
        log.info(f"  {name}: n={len(vals)}, range=[{stats[str(var_id)]['p01']:.2f}, {stats[str(var_id)]['p99']:.2f}]")

    return stats


def main():
    log.info("Stage 2: Extract EHR from MIMIC-III")
    t0 = time.time()

    labs_df = extract_labs()
    vitals_df = extract_vitals()

    # Compute normalization
    norm_stats = compute_normalization(labs_df, vitals_df)

    # Save
    labs_path = OUT_DIR / "labs_filtered.parquet"
    labs_df.write_parquet(labs_path)
    log.info(f"\nSaved labs: {labs_path} ({len(labs_df)} rows)")

    vitals_path = OUT_DIR / "vitals_filtered.parquet"
    vitals_df.write_parquet(vitals_path)
    log.info(f"Saved vitals: {vitals_path} ({len(vitals_df)} rows)")

    norm_path = OUT_DIR / "normalization_stats.json"
    with open(norm_path, "w") as f:
        json.dump(norm_stats, f, indent=2)
    log.info(f"Saved normalization: {norm_path}")

    # Summary
    n_lab_subjects = labs_df["SUBJECT_ID"].n_unique()
    n_vital_subjects = vitals_df["SUBJECT_ID"].n_unique()

    summary = {
        "total_lab_events": len(labs_df),
        "total_vital_events": len(vitals_df),
        "n_lab_subjects": n_lab_subjects,
        "n_vital_subjects": n_vital_subjects,
        "per_variable": {
            str(var_id): {
                "name": name,
                "count": int(norm_stats.get(str(var_id), {}).get("count", 0)),
            }
            for var_id, (name, *_) in {**LAB_ITEMS, **VITAL_ITEMS}.items()
        },
    }
    with open(OUT_DIR / "stage2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    elapsed = time.time() - t0

    # Verification
    log.info(f"\n=== Verification ===")
    assert len(labs_df) > 0, "No lab events extracted"
    assert len(vitals_df) > 0, "No vital events extracted"
    assert labs_df["SUBJECT_ID"].null_count() == 0, "Null SUBJECT_IDs in labs"
    assert vitals_df["SUBJECT_ID"].null_count() == 0, "Null SUBJECT_IDs in vitals"
    log.info(f"  [PASS] {len(labs_df)} lab events from {n_lab_subjects} subjects")
    log.info(f"  [PASS] {len(vitals_df)} vital events from {n_vital_subjects} subjects")
    log.info(f"  [PASS] No null SUBJECT_IDs")
    log.info(f"  Total time: {elapsed:.1f}s")

    log.info(f"\nNext: python workzone/mimic3/stage3_extract_waveforms.py")


if __name__ == "__main__":
    main()
