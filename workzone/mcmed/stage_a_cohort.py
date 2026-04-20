#!/usr/bin/env python3
"""
Stage A — MC_MED cohort from waveform_summary.csv + visits.csv.

Output: workzone/outputs/mcmed/valid_cohort.parquet
  entity_id (str=CSN), CSN, MRN, Arrival_ms, Departure_ms, ED_LOS_hours,
  Hosp_LOS_days, n_pleth_seg, n_ii_seg, n_resp_seg, duration_pleth_s,
  age, gender, race, ethnicity, triage_acuity, CC, dispo, dx_icd10, dx_name,
  visit_no, visits_total.

Filter: keep CSNs with Pleth segments >= 1 AND arrival_ms <= departure_ms.
EHR overlap check deferred to Stage F manifest.
"""
import argparse
import json
import os
import time
from pathlib import Path

import polars as pl

RAW_ROOT = "/opt/localdata100tb/UNIPHY_Plus/raw_datasets/physionet.org/files/mc-med/1.0.1/data"
WAVE_SUMMARY = f"{RAW_ROOT}/waveform_summary.csv"
VISITS_CSV = f"{RAW_ROOT}/visits.csv"

OUT_DIR = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/mcmed"
OUT_PARQUET = f"{OUT_DIR}/valid_cohort.parquet"
OUT_SUMMARY = f"{OUT_DIR}/stage_a_summary.json"


def iso_z_to_ms(col: str) -> pl.Expr:
    # MC_MED strings end with "Z" (UTC). Handle both with and without fractional seconds.
    return (
        pl.col(col).cast(pl.Utf8)
        .str.strip_suffix("Z")
        .str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S%.f", strict=False)
        .dt.replace_time_zone("UTC")
        .dt.timestamp("ms")
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    print(f"[A] reading {WAVE_SUMMARY}")
    wave = pl.read_csv(WAVE_SUMMARY)  # CSN, Type, Segments, Duration
    print(f"    rows={wave.height}  unique CSNs={wave['CSN'].n_unique()}")

    # Wide format: CSN x Type (Pleth/II/Resp) -> (segments, duration)
    wave_wide = (
        wave.pivot(on="Type", index="CSN", values=["Segments", "Duration"],
                   aggregate_function="first")
    )
    # Normalize column names (polars pivots to "Segments_Pleth", "Duration_Pleth", ...)
    rename_map = {}
    for t in ("Pleth", "II", "Resp"):
        for base in ("Segments", "Duration"):
            src = f"{base}_{t}"
            if src in wave_wide.columns:
                rename_map[src] = f"n_{t.lower()}_seg" if base == "Segments" else f"duration_{t.lower()}_s"
    wave_wide = wave_wide.rename(rename_map)
    for c in ["n_pleth_seg", "n_ii_seg", "n_resp_seg",
              "duration_pleth_s", "duration_ii_s", "duration_resp_s"]:
        if c not in wave_wide.columns:
            wave_wide = wave_wide.with_columns(pl.lit(0).alias(c))
    wave_wide = wave_wide.with_columns([
        pl.col("n_pleth_seg").cast(pl.Int32, strict=False).fill_null(0),
        pl.col("n_ii_seg").cast(pl.Int32, strict=False).fill_null(0),
        pl.col("n_resp_seg").cast(pl.Int32, strict=False).fill_null(0),
        pl.col("duration_pleth_s").cast(pl.Float64, strict=False).fill_null(0),
        pl.col("duration_ii_s").cast(pl.Float64, strict=False).fill_null(0),
        pl.col("duration_resp_s").cast(pl.Float64, strict=False).fill_null(0),
    ])

    print(f"[A] reading {VISITS_CSV}")
    visits = pl.read_csv(VISITS_CSV, infer_schema_length=10000, ignore_errors=True)
    print(f"    rows={visits.height}  unique CSNs={visits['CSN'].n_unique()}")

    visits = visits.with_columns([
        iso_z_to_ms("Arrival_time").alias("arrival_ms"),
        iso_z_to_ms("Departure_time").alias("departure_ms"),
        iso_z_to_ms("Roomed_time").alias("roomed_ms"),
        iso_z_to_ms("Dispo_time").alias("dispo_ms"),
        iso_z_to_ms("Admit_time").alias("admit_ms"),
    ]).select([
        "MRN", "CSN", "Visit_no", "Visits",
        pl.col("Age").cast(pl.Int32, strict=False).alias("age"),
        "Gender", "Race", "Ethnicity", "Means_of_arrival",
        "Triage_acuity", "CC", "ED_dispo", "Admit_service",
        "Dx_ICD10", "Dx_name",
        pl.col("ED_LOS").cast(pl.Float64, strict=False).alias("ed_los_hours"),
        pl.col("Hosp_LOS").cast(pl.Float64, strict=False).alias("hosp_los_days"),
        "Payor_class",
        "arrival_ms", "roomed_ms", "dispo_ms", "admit_ms", "departure_ms",
    ])

    cohort = wave_wide.join(visits, on="CSN", how="inner")
    before = cohort.height
    cohort = cohort.filter(
        (pl.col("n_pleth_seg") >= 1)
        & pl.col("arrival_ms").is_not_null()
        & pl.col("departure_ms").is_not_null()
        & (pl.col("departure_ms") >= pl.col("arrival_ms"))
    )
    print(f"[A] after filter (Pleth>=1, valid episode): {cohort.height} / {before}")

    cohort = cohort.with_columns([
        pl.col("CSN").cast(pl.Utf8).alias("entity_id"),
        pl.col("CSN").cast(pl.Int64, strict=False).alias("csn"),
        pl.col("MRN").cast(pl.Int64, strict=False).alias("mrn"),
    ]).drop(["CSN", "MRN"]).rename({
        "Gender": "gender", "Race": "race", "Ethnicity": "ethnicity",
        "Means_of_arrival": "means_of_arrival", "Triage_acuity": "triage_acuity",
        "CC": "chief_complaint", "ED_dispo": "ed_dispo",
        "Admit_service": "admit_service", "Dx_ICD10": "dx_icd10",
        "Dx_name": "dx_name", "Payor_class": "payor_class",
        "Visit_no": "visit_no", "Visits": "visits_total",
    })

    col_order = [
        "entity_id", "csn", "mrn", "visit_no", "visits_total",
        "arrival_ms", "roomed_ms", "dispo_ms", "admit_ms", "departure_ms",
        "ed_los_hours", "hosp_los_days",
        "age", "gender", "race", "ethnicity", "means_of_arrival",
        "triage_acuity", "chief_complaint", "ed_dispo", "admit_service",
        "dx_icd10", "dx_name", "payor_class",
        "n_pleth_seg", "n_ii_seg", "n_resp_seg",
        "duration_pleth_s", "duration_ii_s", "duration_resp_s",
    ]
    cohort = cohort.select(col_order).unique("entity_id", keep="first").sort("csn")

    cohort.write_parquet(OUT_PARQUET)

    summary = {
        "n_entities": cohort.height,
        "n_unique_mrn": int(cohort["mrn"].n_unique()),
        "n_with_ii": int((cohort["n_ii_seg"] >= 1).sum()),
        "n_with_resp": int((cohort["n_resp_seg"] >= 1).sum()),
        "total_pleth_seg": int(cohort["n_pleth_seg"].sum()),
        "total_ii_seg": int(cohort["n_ii_seg"].sum()),
        "total_duration_pleth_h": round(cohort["duration_pleth_s"].sum() / 3600, 1),
        "median_ed_los_hours": round(float(cohort["ed_los_hours"].median() or 0), 2),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(OUT_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== Stage A summary ===")
    print(json.dumps(summary, indent=2))
    print(cohort.head(3))


if __name__ == "__main__":
    main()
