#!/usr/bin/env python3
"""
Stage 2: Extract and normalize EHR data from MC_MED CSVs (labs + vitals).

Maps MC_MED lab Component_name and vital Measure names to var_registry.json IDs.
Filters to physiological ranges. Computes normalization stats.

Run:  python workzone/mcmed/stage2_extract_ehr.py
Output:
  workzone/outputs/mcmed/labs_filtered.parquet
  workzone/outputs/mcmed/vitals_filtered.parquet
  workzone/outputs/mcmed/normalization_stats.json
  workzone/outputs/mcmed/stage2_summary.json
"""
import json
import time
import logging
from pathlib import Path

import numpy as np
import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO_ROOT / "workzone" / "outputs" / "mcmed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

import yaml, os
with open(REPO_ROOT / "workzone" / "configs" / "server_paths.yaml") as f:
    cfg = yaml.safe_load(f)

CSV_DIR = cfg["mcmed"]["raw_csv_dir"]
if not os.path.exists(CSV_DIR):
    CSV_DIR = os.path.expanduser("~/workspace/mc_med_csv")

# Load var_registry
with open(REPO_ROOT / "indices" / "var_registry.json") as f:
    VAR_REGISTRY = json.load(f)

# ------------------------------------------------------------------
# MC_MED Component_name -> var_id mapping
# MC_MED labs use Component_name (e.g. "SODIUM", "POTASSIUM")
# ------------------------------------------------------------------
MCMED_LAB_MAP = {
    # Component_name -> var_id (from var_registry.json)
    "POTASSIUM":     0,
    "CALCIUM":       1,
    "SODIUM":        2,
    "GLUCOSE":       3,
    "LACTIC ACID":   4,   # Lactate
    "LACTATE":       4,
    "CREATININE":    5,
    "BILIRUBIN TOTAL": 6,
    "BILIRUBIN, TOTAL": 6,
    "PLATELET COUNT": 7,
    "PLATELETS":     7,
    "WBC":           8,
    "WHITE BLOOD CELL COUNT": 8,
    "HEMOGLOBIN":    9,
    "INR":           10,
    "BUN":           11,
    "UREA NITROGEN": 11,
    "ALBUMIN":       12,
    "PH ARTERIAL":   13,   # Arterial_pH
    "PH":            13,
    "PO2 ARTERIAL":  14,   # paO2
    "PO2":           14,
    "PCO2 ARTERIAL": 15,   # paCO2
    "PCO2":          15,
    "BICARBONATE":   16,   # HCO3
    "HCO3":          16,
    "CO2":           16,
}

# MC_MED vitals Measure -> var_id
MCMED_VITAL_MAP = {
    "HR":    100,
    "SpO2":  101,
    "RR":    102,
    "Temp":  103,
    "SBP":   104,   # maps to NBPs
    "DBP":   105,   # maps to NBPd
    "MAP":   106,   # maps to NBPm
}

# Build var_id -> physio range lookup
VAR_RANGES = {}
for v in VAR_REGISTRY["variables"]:
    VAR_RANGES[v["id"]] = (v.get("physio_min", -1e9), v.get("physio_max", 1e9))


def extract_labs():
    """Extract labs from labs.csv, map to var_ids, filter by physio range."""
    labs_path = Path(CSV_DIR) / "labs.csv"
    log.info(f"Reading labs from {labs_path}")

    df = pl.read_csv(
        labs_path,
        columns=["CSN", "Result_time", "Component_name", "Component_value"],
        dtypes={"CSN": pl.Utf8, "Component_value": pl.Float64},
        null_values=["", "NA", "None"],
        ignore_errors=True,
    )

    log.info(f"  Raw lab rows: {len(df)}")

    # Drop nulls
    df = df.drop_nulls(subset=["Component_value", "Result_time", "Component_name"])

    # Uppercase component names for matching
    df = df.with_columns(pl.col("Component_name").str.to_uppercase().alias("comp_upper"))

    # Map to var_id
    map_expr = pl.col("comp_upper").replace(MCMED_LAB_MAP, default=None).cast(pl.UInt16)
    df = df.with_columns(map_expr.alias("var_id"))
    df = df.drop_nulls(subset=["var_id"])

    log.info(f"  After var_id mapping: {len(df)}")

    # Parse timestamp
    df = df.with_columns(
        pl.col("Result_time").str.to_datetime(strict=False).alias("time")
    )
    df = df.drop_nulls(subset=["time"])

    # Convert time to milliseconds since epoch
    df = df.with_columns(
        (pl.col("time").dt.epoch("ms")).alias("time_ms")
    )

    # Filter by physiological range
    results = []
    for var_id, (lo, hi) in VAR_RANGES.items():
        subset = df.filter(pl.col("var_id") == var_id)
        if len(subset) == 0:
            continue
        subset = subset.filter(
            (pl.col("Component_value") >= lo) & (pl.col("Component_value") <= hi)
        )
        results.append(subset)

    if results:
        df = pl.concat(results)
    else:
        df = df.head(0)

    # Select final columns
    df = df.select([
        pl.col("CSN").alias("csn"),
        pl.col("time_ms"),
        pl.col("var_id"),
        pl.col("Component_value").cast(pl.Float32).alias("value"),
    ]).sort(["csn", "time_ms"])

    log.info(f"  Final lab events: {len(df)}, {df['var_id'].n_unique()} variables, {df['csn'].n_unique()} CSNs")
    return df


def extract_vitals():
    """Extract vitals from numerics.csv, map to var_ids."""
    vitals_path = Path(CSV_DIR) / "numerics.csv"
    log.info(f"Reading vitals from {vitals_path}")

    df = pl.read_csv(
        vitals_path,
        columns=["CSN", "Measure", "Value", "Time"],
        dtypes={"CSN": pl.Utf8, "Value": pl.Float64},
        null_values=["", "NA", "None"],
        ignore_errors=True,
    )

    log.info(f"  Raw vital rows: {len(df)}")
    df = df.drop_nulls(subset=["Value", "Time", "Measure"])

    # Map Measure to var_id
    map_expr = pl.col("Measure").replace(MCMED_VITAL_MAP, default=None).cast(pl.UInt16)
    df = df.with_columns(map_expr.alias("var_id"))
    df = df.drop_nulls(subset=["var_id"])

    log.info(f"  After var_id mapping: {len(df)}")

    # Parse timestamp
    df = df.with_columns(
        pl.col("Time").str.to_datetime(strict=False).alias("time")
    )
    df = df.drop_nulls(subset=["time"])

    df = df.with_columns(
        (pl.col("time").dt.epoch("ms")).alias("time_ms")
    )

    # Temperature: check if Fahrenheit (MC_MED Temp should be Fahrenheit based on data)
    # Convert F -> C for var_id 103
    temp_mask = pl.col("var_id") == 103
    df = df.with_columns(
        pl.when(temp_mask & (pl.col("Value") > 50))
        .then((pl.col("Value") - 32) * 5 / 9)
        .otherwise(pl.col("Value"))
        .alias("Value")
    )

    # Filter by physiological range
    results = []
    for var_id, (lo, hi) in VAR_RANGES.items():
        subset = df.filter(pl.col("var_id") == var_id)
        if len(subset) == 0:
            continue
        subset = subset.filter(
            (pl.col("Value") >= lo) & (pl.col("Value") <= hi)
        )
        results.append(subset)

    if results:
        df = pl.concat(results)
    else:
        df = df.head(0)

    df = df.select([
        pl.col("CSN").alias("csn"),
        pl.col("time_ms"),
        pl.col("var_id"),
        pl.col("Value").cast(pl.Float32).alias("value"),
    ]).sort(["csn", "time_ms"])

    log.info(f"  Final vital events: {len(df)}, {df['var_id'].n_unique()} variables, {df['csn'].n_unique()} CSNs")
    return df


def compute_normalization_stats(labs_df, vitals_df):
    """Compute per-variable normalization statistics."""
    combined = pl.concat([labs_df, vitals_df])
    stats = {}
    for var_id in sorted(combined["var_id"].unique().to_list()):
        subset = combined.filter(pl.col("var_id") == var_id)["value"]
        arr = subset.to_numpy()
        name = "unknown"
        for v in VAR_REGISTRY["variables"]:
            if v["id"] == var_id:
                name = v["name"]
                break
        stats[str(var_id)] = {
            "name": name,
            "var_id": int(var_id),
            "count": int(len(arr)),
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
            "mean": float(np.nanmean(arr)),
            "std": float(np.nanstd(arr)),
            "median": float(np.nanmedian(arr)),
            "p01": float(np.nanpercentile(arr, 1)),
            "p99": float(np.nanpercentile(arr, 99)),
        }
    return stats


def main():
    t0 = time.time()
    log.info("Stage 2: Extracting MC_MED EHR data")

    labs_df = extract_labs()
    vitals_df = extract_vitals()

    # Normalization stats
    norm_stats = compute_normalization_stats(labs_df, vitals_df)

    elapsed = time.time() - t0

    # Summary
    summary = {
        "total_lab_events": len(labs_df),
        "total_vital_events": len(vitals_df),
        "lab_csns": int(labs_df["csn"].n_unique()),
        "vital_csns": int(vitals_df["csn"].n_unique()),
        "lab_variables": int(labs_df["var_id"].n_unique()),
        "vital_variables": int(vitals_df["var_id"].n_unique()),
        "extraction_time_sec": round(elapsed, 1),
    }

    log.info("\n=== Summary ===")
    for k, v in summary.items():
        log.info(f"  {k}: {v}")

    # Save
    labs_df.to_pandas().to_parquet(OUT_DIR / "labs_filtered.parquet", index=False)
    vitals_df.to_pandas().to_parquet(OUT_DIR / "vitals_filtered.parquet", index=False)

    with open(OUT_DIR / "normalization_stats.json", "w") as f:
        json.dump(norm_stats, f, indent=2)
    with open(OUT_DIR / "stage2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"\nSaved to {OUT_DIR}")
    log.info(f"Next: python workzone/mcmed/stage3_convert_to_canonical.py")


if __name__ == "__main__":
    main()
