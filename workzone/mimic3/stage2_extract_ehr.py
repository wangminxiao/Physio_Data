#!/usr/bin/env python3
"""
Stage 2: Extract and filter EHR data (labs + vitals) from raw MIMIC-III CSVs.

Reads var_registry.json for variable definitions (ITEMIDs, physio ranges).
Extracts ALL labs from LABEVENTS.csv and ALL vitals from CHARTEVENTS.csv.gz.
Handles unit conversions (e.g. Fahrenheit -> Celsius for Temperature).

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

# --------------------------------------------------------------------------
# Load variable definitions from var_registry.json
# --------------------------------------------------------------------------
with open(REPO_ROOT / "indices" / "var_registry.json") as f:
    VAR_REGISTRY = json.load(f)

# Build per-category variable dicts: {var_id: {name, itemids, unit, physio_min, physio_max, convert}}
LAB_VARS = {}    # from LABEVENTS (category == "lab", has mimic_itemids)
VITAL_VARS = {}  # from CHARTEVENTS (category == "vital", has mimic_itemids)

for v in VAR_REGISTRY["variables"]:
    if "mimic_itemids" not in v:
        continue  # skip actions/scores without direct ITEMID extraction
    entry = {
        "name": v["name"],
        "itemids": v["mimic_itemids"],
        "unit": v["unit"],
        "physio_min": v.get("physio_min"),
        "physio_max": v.get("physio_max"),
        "convert": v.get("mimic_convert", {}),
    }
    if v["category"] == "lab":
        LAB_VARS[v["id"]] = entry
    elif v["category"] == "vital":
        VITAL_VARS[v["id"]] = entry

# Build ITEMID -> var_id lookup (separate for labs and vitals since they come from different tables)
LAB_ITEMID_TO_VARID = {}
for var_id, info in LAB_VARS.items():
    for iid in info["itemids"]:
        LAB_ITEMID_TO_VARID[iid] = var_id

VITAL_ITEMID_TO_VARID = {}
# Track which ITEMIDs need unit conversion
VITAL_ITEMID_CONVERT = {}  # {itemid: conversion_type}
for var_id, info in VITAL_VARS.items():
    for iid in info["itemids"]:
        VITAL_ITEMID_TO_VARID[iid] = var_id
        str_iid = str(iid)
        if str_iid in info["convert"]:
            VITAL_ITEMID_CONVERT[iid] = info["convert"][str_iid]

log.info(f"Loaded var_registry: {len(LAB_VARS)} labs (var_ids {sorted(LAB_VARS.keys())}), "
         f"{len(VITAL_VARS)} vitals (var_ids {sorted(VITAL_VARS.keys())})")


# --------------------------------------------------------------------------
# Unit conversions
# --------------------------------------------------------------------------
def fahrenheit_to_celsius(f):
    """Convert Fahrenheit to Celsius."""
    return (f - 32) * 5 / 9


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------

def extract_labs():
    """Extract lab events from LABEVENTS.csv."""
    log.info("=== Extracting Labs ===")
    lab_path = os.path.join(EHR_ROOT, "LABEVENTS.csv")
    if not os.path.exists(lab_path):
        lab_path = os.path.join(EHR_ROOT, "LABEVENTS.csv.gz")
    log.info(f"  Reading: {lab_path}")
    log.info(f"  Target: {len(LAB_VARS)} variables, {len(LAB_ITEMID_TO_VARID)} ITEMIDs")

    target_itemids = list(LAB_ITEMID_TO_VARID.keys())

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
        "ITEMID": list(LAB_ITEMID_TO_VARID.keys()),
        "var_id": list(LAB_ITEMID_TO_VARID.values()),
    })
    df = df.join(itemid_map, on="ITEMID", how="inner")

    # Apply physiological range filters
    rows_before = len(df)
    range_filters = []
    for var_id, info in LAB_VARS.items():
        vmin, vmax = info["physio_min"], info["physio_max"]
        if vmin is not None and vmax is not None:
            range_filters.append(
                (pl.col("var_id") == var_id) &
                (pl.col("VALUENUM") >= vmin) &
                (pl.col("VALUENUM") <= vmax)
            )
        else:
            range_filters.append(pl.col("var_id") == var_id)

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

    # Per-variable counts
    for var_id, info in sorted(LAB_VARS.items()):
        n = df.filter(pl.col("var_id") == var_id).height
        log.info(f"    var_id={var_id:3d} {info['name']:15s}: {n:>10,} events")

    return df


def extract_vitals():
    """Extract vital events from CHARTEVENTS.csv.gz."""
    log.info("\n=== Extracting Vitals ===")
    vital_path = os.path.join(EHR_ROOT, "CHARTEVENTS.csv.gz")
    if not os.path.exists(vital_path):
        vital_path = os.path.join(EHR_ROOT, "CHARTEVENTS.csv")
    log.info(f"  Reading: {vital_path}")
    log.info(f"  Target: {len(VITAL_VARS)} variables, {len(VITAL_ITEMID_TO_VARID)} ITEMIDs")

    target_itemids = list(VITAL_ITEMID_TO_VARID.keys())

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

    # Apply unit conversions BEFORE mapping to var_id
    # (conversion is per-ITEMID, not per-var_id)
    if VITAL_ITEMID_CONVERT:
        for itemid, conv_type in VITAL_ITEMID_CONVERT.items():
            if conv_type == "F_to_C":
                n_before = df.filter(pl.col("ITEMID") == itemid).height
                df = df.with_columns(
                    pl.when(pl.col("ITEMID") == itemid)
                    .then((pl.col("VALUENUM") - 32) * 5 / 9)
                    .otherwise(pl.col("VALUENUM"))
                    .alias("VALUENUM")
                )
                log.info(f"  Converted {n_before} ITEMID={itemid} values from F to C")

    # Map ITEMID -> var_id
    itemid_map = pl.DataFrame({
        "ITEMID": list(VITAL_ITEMID_TO_VARID.keys()),
        "var_id": list(VITAL_ITEMID_TO_VARID.values()),
    })
    df = df.join(itemid_map, on="ITEMID", how="inner")

    # Apply physiological range filters (after conversion)
    rows_before = len(df)
    range_filters = []
    for var_id, info in VITAL_VARS.items():
        vmin, vmax = info["physio_min"], info["physio_max"]
        if vmin is not None and vmax is not None:
            range_filters.append(
                (pl.col("var_id") == var_id) &
                (pl.col("VALUENUM") >= vmin) &
                (pl.col("VALUENUM") <= vmax)
            )
        else:
            range_filters.append(pl.col("var_id") == var_id)

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

    # Per-variable counts
    for var_id, info in sorted(VITAL_VARS.items()):
        n = df.filter(pl.col("var_id") == var_id).height
        log.info(f"    var_id={var_id:3d} {info['name']:15s}: {n:>10,} events")

    return df


def compute_normalization(labs_df, vitals_df):
    """Compute per-variable normalization stats."""
    log.info("\n=== Normalization Stats ===")
    all_vars = {**LAB_VARS, **VITAL_VARS}
    stats = {}

    combined = pl.concat([
        labs_df.select(["var_id", "VALUENUM"]),
        vitals_df.select(["var_id", "VALUENUM"]),
    ])

    for var_id, info in sorted(all_vars.items()):
        subset = combined.filter(pl.col("var_id") == var_id)["VALUENUM"]
        if len(subset) == 0:
            log.warning(f"  {info['name']} (var_id={var_id}): no data!")
            continue

        vals = subset.to_numpy()
        stats[str(var_id)] = {
            "name": info["name"],
            "unit": info["unit"],
            "count": int(len(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "p01": float(np.percentile(vals, 1)),
            "p99": float(np.percentile(vals, 99)),
            "median": float(np.median(vals)),
        }
        log.info(f"  {info['name']:15s}: n={len(vals):>10,}, "
                 f"range=[{stats[str(var_id)]['p01']:.2f}, {stats[str(var_id)]['p99']:.2f}]")

    return stats


def main():
    log.info("Stage 2: Extract EHR from MIMIC-III (full var_registry)")
    log.info(f"  Labs: {len(LAB_VARS)} variables from LABEVENTS")
    log.info(f"  Vitals: {len(VITAL_VARS)} variables from CHARTEVENTS")
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
        "n_lab_variables": len(LAB_VARS),
        "n_vital_variables": len(VITAL_VARS),
        "per_variable": {},
    }
    for var_id in sorted({**LAB_VARS, **VITAL_VARS}.keys()):
        info = LAB_VARS.get(var_id) or VITAL_VARS.get(var_id)
        summary["per_variable"][str(var_id)] = {
            "name": info["name"],
            "category": "lab" if var_id in LAB_VARS else "vital",
            "count": int(norm_stats.get(str(var_id), {}).get("count", 0)),
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
    log.info(f"  [PASS] {len(labs_df):,} lab events from {n_lab_subjects} subjects ({len(LAB_VARS)} vars)")
    log.info(f"  [PASS] {len(vitals_df):,} vital events from {n_vital_subjects} subjects ({len(VITAL_VARS)} vars)")
    log.info(f"  [PASS] No null SUBJECT_IDs")
    log.info(f"  Total time: {elapsed:.1f}s")

    log.info(f"\nNext: python workzone/mimic3/stage2b_cross_check.py")


if __name__ == "__main__":
    main()
