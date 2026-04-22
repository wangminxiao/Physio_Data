#!/usr/bin/env python3
"""
Stage A - VitalDB cohort from clinical_data.csv + .vital file presence.

Output: workzone/outputs/vitaldb/valid_cohort.parquet
  entity_id (= f'{caseid:04d}'), caseid, subjectid,
  casestart_s, caseend_s, anestart_s, aneend_s, opstart_s, opend_s,
  vital_file_path,
  age, sex, height_cm, weight_kg, bmi, asa, emop,
  department, optype, dx, opname, approach, position, ane_type,
  adm_s, dis_s, icu_days, death_inhosp.

Filter: keep caseids with a .vital file AND valid anestart < aneend.
"""
import argparse
import json
import os
import time
from pathlib import Path

import polars as pl

RAW_ROOT = "/opt/localdata100tb/UNIPHY_Plus/raw_datasets/vitalDB"
CLINICAL_CSV = f"{RAW_ROOT}/clinical_data.csv"
VITAL_DIR = f"{RAW_ROOT}/vital_files"

OUT_DIR = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/vitaldb"
OUT_PARQUET = f"{OUT_DIR}/valid_cohort.parquet"
OUT_SUMMARY = f"{OUT_DIR}/stage_a_summary.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    print(f"[A] reading {CLINICAL_CSV}")
    df = pl.read_csv(CLINICAL_CSV, infer_schema_length=10000, ignore_errors=True,
                     encoding="utf8-lossy")
    print(f"    rows={df.height}  unique caseid={df['caseid'].n_unique()}  "
          f"unique subjectid={df['subjectid'].n_unique()}")

    df = df.rename({
        "caseid": "caseid", "subjectid": "subjectid",
        "casestart": "casestart_s", "caseend": "caseend_s",
        "anestart": "anestart_s", "aneend": "aneend_s",
        "opstart": "opstart_s",   "opend": "opend_s",
        "adm": "adm_s", "dis": "dis_s",
        "height": "height_cm", "weight": "weight_kg",
    })

    # Attach .vital file path
    vital_dir = Path(VITAL_DIR)
    df = df.with_columns([
        pl.col("caseid").cast(pl.Utf8).str.zfill(4).alias("entity_id"),
        (pl.lit(f"{vital_dir}/") + pl.col("caseid").cast(pl.Utf8).str.zfill(4) + pl.lit(".vital"))
          .alias("vital_file_path"),
    ])

    # Verify .vital file exists per row
    exists = [Path(p).exists() for p in df["vital_file_path"].to_list()]
    df = df.with_columns(pl.Series("vital_exists", exists))
    print(f"    .vital files present: {sum(exists)}/{df.height}")

    before = df.height
    df = df.filter(
        pl.col("vital_exists")
        & pl.col("anestart_s").is_not_null()
        & pl.col("aneend_s").is_not_null()
        & (pl.col("aneend_s") > pl.col("anestart_s"))
    ).drop("vital_exists")
    print(f"[A] after filter: {df.height} / {before}")

    keep_cols = [
        "entity_id", "caseid", "subjectid",
        "casestart_s", "caseend_s", "anestart_s", "aneend_s",
        "opstart_s", "opend_s", "adm_s", "dis_s",
        "age", "sex", "height_cm", "weight_kg", "bmi", "asa", "emop",
        "department", "optype", "dx", "opname", "approach", "position",
        "ane_type", "icu_days", "death_inhosp",
        "vital_file_path",
    ]
    df = df.select([c for c in keep_cols if c in df.columns]).unique("entity_id",
                                                                     keep="first").sort("caseid")
    df.write_parquet(OUT_PARQUET)

    summary = {
        "n_entities": df.height,
        "n_unique_subject": int(df["subjectid"].n_unique()),
        "median_an_duration_min": round(float(
            ((df["aneend_s"] - df["anestart_s"]) / 60.0).median() or 0), 1),
        "median_case_duration_min": round(float(
            ((df["caseend_s"] - df["casestart_s"]) / 60.0).median() or 0), 1),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(OUT_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== Stage A summary ===")
    print(json.dumps(summary, indent=2))
    print(df.head(3))


if __name__ == "__main__":
    main()
