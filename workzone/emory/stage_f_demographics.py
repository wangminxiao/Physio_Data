#!/usr/bin/env python3
"""
Stage F-2 — Emory demographics.csv.

One row per entity_id. Most fields come directly from the cohort parquet
(already joined to JGSEPSIS_ENCOUNTER in Stage A). Only gender/race/
ethnicity/DOB/DEATH come from the separate JGSEPSIS_DEMOGRAPHICS.csv,
joined via PAT_ID.

Output columns (per datasets/emory/API.md §Demographics):
  entity_id, empi_nbr, encounter_nbr, pat_id,
  gender, dob_iso, age_years,
  race, ethnicity, insurance,
  admit_utc_ms, discharge_utc_ms, los_days,
  admit_dx_icd10, admit_dx_desc,
  entity_healthcare_nm,
  death_date_utc_ms,
  wave_start_ms, wave_end_ms,
  type   (case/control from cohort, kept for convenience)

Time convention: EHR `DEATH_DATE` is date-only; treat as NY-midnight→UTC ms.
"""
import os
import sys
import csv
import json
import time
import argparse
import logging
from pathlib import Path

import polars as pl

EHR_ROOT = "/labs/hulab/Emory_EHR/CDW_Pull_ICU_Data_Siva_version"
DEMOGRAPHICS_CSV = f"{EHR_ROOT}/JGSEPSIS_DEMOGRAPHICS.csv"

OUT_ROOT = "/opt/localdata100tb/physio_data/emory"
COHORT_PARQUET = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/emory/valid_wave_window.parquet"
OUTPUTS_DIR = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/emory"
SUMMARY_JSON = f"{OUTPUTS_DIR}/stage_f_demographics_summary.json"

NY_TZ = "America/New_York"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=str, default=OUT_ROOT)
    args = ap.parse_args()

    t0 = time.time()
    out_root = Path(args.out_root)

    cohort = (
        pl.read_parquet(COHORT_PARQUET)
          .unique("entity_id", keep="first")
    )
    # Restrict to entities that have Stage B output on disk
    on_disk = {p.name for p in out_root.iterdir()
               if p.is_dir() and (p / "meta.json").exists()}
    cohort = cohort.filter(pl.col("entity_id").is_in(list(on_disk)))
    log.info(f"cohort entities with output: {cohort.height}")

    # Pull demographics (DOB / GENDER / RACE / ETHNICITY / DEATH_DATE) by pat_id
    pat_ids = cohort["pat_id"].unique().to_list()
    log.info(f"unique pat_ids to look up: {len(pat_ids)}")

    log.info(f"scanning {DEMOGRAPHICS_CSV}")
    demo_lf = pl.scan_csv(DEMOGRAPHICS_CSV, low_memory=True,
                          infer_schema_length=10000, ignore_errors=True)
    demo = (
        demo_lf.filter(pl.col("PAT_ID").cast(pl.Utf8).is_in(pat_ids))
               .select([
                   pl.col("PAT_ID").cast(pl.Utf8).alias("pat_id"),
                   pl.col("GENDER").cast(pl.Utf8, strict=False).alias("gender"),
                   pl.col("DOB").cast(pl.Utf8, strict=False).alias("dob_raw"),
                   pl.col("RACE").cast(pl.Utf8, strict=False).alias("race"),
                   pl.col("ETHNICITY").cast(pl.Utf8, strict=False).alias("ethnicity"),
                   pl.col("DEATH_DATE").cast(pl.Utf8, strict=False).alias("death_raw"),
               ])
               .unique("pat_id")
               .collect(engine="streaming")
    )
    log.info(f"demographics rows matched: {demo.height}")

    # Parse DOB (YYYY-MM-DD) and DEATH_DATE (MM/DD/YYYY or YYYY-MM-DD)
    demo = demo.with_columns([
        pl.col("dob_raw")
          .str.strptime(pl.Date, format="%Y-%m-%d", strict=False)
          .alias("dob"),
        pl.coalesce([
            pl.col("death_raw").str.strptime(pl.Datetime,
                                             format="%m/%d/%Y %H:%M:%S",
                                             strict=False),
            pl.col("death_raw").str.strptime(pl.Datetime,
                                             format="%m/%d/%Y",
                                             strict=False),
            pl.col("death_raw").str.strptime(pl.Datetime,
                                             format="%Y-%m-%d %H:%M:%S",
                                             strict=False),
            pl.col("death_raw").str.strptime(pl.Datetime,
                                             format="%Y-%m-%d",
                                             strict=False),
        ]).alias("death_dt"),
    ]).with_columns([
        pl.col("dob").dt.strftime("%Y-%m-%d").alias("dob_iso"),
        pl.col("death_dt")
          .dt.replace_time_zone(NY_TZ, ambiguous="earliest", non_existent="null")
          .dt.convert_time_zone("UTC")
          .dt.timestamp("ms")
          .alias("death_date_utc_ms"),
    ])

    # Left join cohort ← demo
    joined = cohort.join(demo.select(
        ["pat_id", "gender", "dob_iso", "race", "ethnicity", "death_date_utc_ms"]
    ), on="pat_id", how="left")
    log.info(f"joined rows: {joined.height}")

    # Wave bounds already in cohort parquet (wave_start_ms, wave_end_ms)
    out_csv = out_root / "demographics.csv"
    n_written = 0
    n_missing_demo = 0
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "entity_id", "empi_nbr", "encounter_nbr", "pat_id",
            "gender", "dob_iso", "age_years",
            "race", "ethnicity", "insurance",
            "admit_utc_ms", "discharge_utc_ms", "los_days",
            "admit_dx_icd10", "admit_dx_desc",
            "entity_healthcare_nm",
            "death_date_utc_ms",
            "wave_start_ms", "wave_end_ms",
            "type",
        ])
        for row in joined.iter_rows(named=True):
            if row.get("gender") is None and row.get("dob_iso") is None:
                n_missing_demo += 1
            w.writerow([
                row["entity_id"],
                int(row["empi_nbr"]) if row.get("empi_nbr") is not None else "",
                int(row["encounter_nbr"]) if row.get("encounter_nbr") is not None else "",
                row.get("pat_id") or "",
                row.get("gender") or "",
                row.get("dob_iso") or "",
                int(row["age"]) if row.get("age") is not None else "",
                row.get("race") or "",
                row.get("ethnicity") or "",
                row.get("insurance_status") or "",
                int(row["admit_ms"]) if row.get("admit_ms") is not None else "",
                int(row["discharge_ms"]) if row.get("discharge_ms") is not None else "",
                round(float(row["los_days"]), 3) if row.get("los_days") is not None else "",
                row.get("admit_dx_icd10") or "",
                row.get("admit_dx_desc") or "",
                row.get("entity_healthcare_nm") or "",
                int(row["death_date_utc_ms"]) if row.get("death_date_utc_ms") is not None else "",
                int(row["wave_start_ms"]) if row.get("wave_start_ms") is not None else "",
                int(row["wave_end_ms"]) if row.get("wave_end_ms") is not None else "",
                row.get("type") or "",
            ])
            n_written += 1

    elapsed = time.time() - t0
    summary = {
        "stage": "f_demographics",
        "ran_at_unix": int(time.time()),
        "elapsed_sec": round(elapsed, 1),
        "n_rows": n_written,
        "n_missing_demographics_join": n_missing_demo,
        "output_csv": str(out_csv),
    }
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"wrote {out_csv}  rows={n_written}  missing_demo={n_missing_demo}  "
             f"elapsed={elapsed:.1f}s")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
