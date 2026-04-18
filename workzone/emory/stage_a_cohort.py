#!/usr/bin/env python3
"""
Stage A — build valid_wave_window.parquet for the Emory sepsis cohort.

Aggregates the sepsis task list (uniq_combine) to one row per encounter with:
  - entity_id, empi_nbr, encounter_nbr (ID columns)
  - wfdb_records (list of wfdb_record IDs belonging to the encounter)
  - wave_start_ms / wave_end_ms (full waveform span from whole list, UTC)
  - valid_start_ms / valid_end_ms (quality-gated span from task list, UTC)
  - valid_duration_hour, valid_ratio (task list quality stats)
  - sepsis_time_zero_ms, type (case/control) (task labels)
  - pat_id (from JGSEPSIS_ENCOUNTER, the EHR patient key)
  - admit_ms, discharge_ms, los_days (from JGSEPSIS_ENCOUNTER, NY->UTC)
  - admit_dx_icd10, admit_dx_desc (primary diagnosis)
  - entity_healthcare_nm, insurance_status

Output:  /labs/hulab/mxwang/Physio_Data/workzone/outputs/emory/valid_wave_window.parquet
         /labs/hulab/mxwang/Physio_Data/workzone/outputs/emory/stage_a_summary.json

No heavy I/O — should finish in under a minute.
"""
import os
import json
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import polars as pl

t0 = time.time()

UTC = timezone.utc
NY = ZoneInfo("America/New_York")

CSV_EHR_DIR = "/labs/hulab/Emory_EHR/CDW_Pull_ICU_Data_Siva_version"
WHOLE_CSV = "/labs/hulab/mxwang/data/sepsis/Wav/sepsis_cc_2025_06_13_all_collab.csv"
TASK_CSV = "/labs/hulab/mxwang/data/sepsis/Wav/sepsis_cc_2025_06_13_all_collab_uniq_combine.csv"

OUT_DIR = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/emory"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PARQUET = os.path.join(OUT_DIR, "valid_wave_window.parquet")
OUT_SUMMARY = os.path.join(OUT_DIR, "stage_a_summary.json")


def ny_local_str_to_utc_ms(col: str) -> pl.Expr:
    return (
        pl.col(col)
        .str.strptime(pl.Datetime, "%m/%d/%Y %H:%M:%S", strict=False)
        .dt.replace_time_zone(
            "America/New_York",
            ambiguous="earliest",      # fall-back DST
            non_existent="null",       # spring-forward: hour doesn't exist → null
        )
        .dt.convert_time_zone("UTC")
        .dt.epoch("ms")
    )


def naive_utc_str_to_ms(col: str) -> pl.Expr:
    return (
        pl.col(col)
        .str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%.f", strict=False)
        .dt.replace_time_zone("UTC")
        .dt.epoch("ms")
    )


def iso_z_str_to_ms(col: str) -> pl.Expr:
    return (
        pl.col(col)
        .str.strip_suffix("Z")
        .str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S", strict=False)
        .dt.replace_time_zone("UTC")
        .dt.epoch("ms")
    )


# ------------------------------------------------------------
# 1. Aggregate task CSV per encounter (valid_start/end + sepsis + type)
# ------------------------------------------------------------
print("Reading task list (uniq_combine) ...")
task = pl.read_csv(TASK_CSV)
print(f"  rows={task.height}  encounters={task['encounter_nbr'].n_unique()}")

task_agg = (
    task.with_columns([
        naive_utc_str_to_ms("valid_start").alias("valid_start_ms"),
        naive_utc_str_to_ms("valid_end").alias("valid_end_ms"),
        naive_utc_str_to_ms("sepsis_time_zero_dttm").alias("sepsis_time_zero_ms"),
    ])
    .group_by("encounter_nbr")
    .agg([
        pl.col("valid_start_ms").min().alias("valid_start_ms"),
        pl.col("valid_end_ms").max().alias("valid_end_ms"),
        pl.col("valid_duration_hour").sum().alias("valid_duration_hour_sum"),
        pl.col("valid_ratio").mean().alias("valid_ratio_mean"),
        pl.col("sepsis_time_zero_ms").first().alias("sepsis_time_zero_ms"),
        pl.col("type").first().alias("type"),
        pl.col("wfdb_record").unique().sort().alias("wfdb_records"),
    ])
)
print(f"  per-encounter rows: {task_agg.height}")

# ------------------------------------------------------------
# 2. Attach empi_nbr + full wave span from whole list
# ------------------------------------------------------------
print("Reading whole list for empi_nbr + full wave span ...")
whole = (
    pl.scan_csv(WHOLE_CSV)
    .filter(pl.col("encounter_nbr").is_in(task_agg["encounter_nbr"]))
    .with_columns([
        iso_z_str_to_ms("wfdb_start").alias("wfdb_start_ms"),
        iso_z_str_to_ms("wfdb_end").alias("wfdb_end_ms"),
    ])
    .group_by("encounter_nbr")
    .agg([
        pl.col("empi_nbr").first().alias("empi_nbr"),
        pl.col("wfdb_start_ms").min().alias("wave_start_ms"),
        pl.col("wfdb_end_ms").max().alias("wave_end_ms"),
        pl.col("wfdb_record").unique().sort().alias("wfdb_records_all"),
    ])
    .collect()
)
print(f"  whole-list encounters matched: {whole.height}")

cohort = task_agg.join(whole, on="encounter_nbr", how="inner")

# ------------------------------------------------------------
# 3. Attach encounter-level fields from JGSEPSIS_ENCOUNTER
# ------------------------------------------------------------
print("Reading JGSEPSIS_ENCOUNTER ...")
enc_cols = [
    "ENCOUNTER_NBR", "PAT_ID", "AGE",
    "HOSPITAL_ADMISSION_DATE_TIME", "HOSPITAL_DISCHARGE_DATE_TIME",
    "DIAGNOSIS_ICD10_CD", "DIAGNOSIS_ICD10_DESC",
    "ENCOUNTER_TYPE", "INSURANCE_STATUS", "ENTITY_HEALTHCARE_NM",
]
enc = (
    pl.scan_csv(
        f"{CSV_EHR_DIR}/JGSEPSIS_ENCOUNTER.csv",
        infer_schema_length=10000,
        ignore_errors=True,
        schema_overrides={"PAT_ID": pl.Utf8},
    )
    .select([pl.col(c) for c in enc_cols])
    .with_columns(pl.col("ENCOUNTER_NBR").cast(pl.Int64, strict=False))
    .filter(pl.col("ENCOUNTER_NBR").is_in(cohort["encounter_nbr"]))
    .with_columns([
        ny_local_str_to_utc_ms("HOSPITAL_ADMISSION_DATE_TIME").alias("admit_ms"),
        ny_local_str_to_utc_ms("HOSPITAL_DISCHARGE_DATE_TIME").alias("discharge_ms"),
    ])
    .rename({
        "ENCOUNTER_NBR": "encounter_nbr",
        "PAT_ID": "pat_id",
        "AGE": "age",
        "DIAGNOSIS_ICD10_CD": "admit_dx_icd10",
        "DIAGNOSIS_ICD10_DESC": "admit_dx_desc",
        "ENCOUNTER_TYPE": "encounter_type",
        "INSURANCE_STATUS": "insurance_status",
        "ENTITY_HEALTHCARE_NM": "entity_healthcare_nm",
    })
    .drop(["HOSPITAL_ADMISSION_DATE_TIME", "HOSPITAL_DISCHARGE_DATE_TIME"])
    .unique(subset=["encounter_nbr"])
    .collect()
)
print(f"  JGSEPSIS_ENCOUNTER rows for cohort: {enc.height}")

cohort = cohort.join(enc, on="encounter_nbr", how="left")

# ------------------------------------------------------------
# 4. Derive final columns
# ------------------------------------------------------------
cohort = cohort.with_columns([
    # Cast empi_nbr as Int64 explicitly
    pl.col("empi_nbr").cast(pl.Int64, strict=False).alias("empi_nbr"),
    (pl.col("empi_nbr").cast(pl.Utf8) + "_" + pl.col("encounter_nbr").cast(pl.Utf8)).alias("entity_id"),
    ((pl.col("discharge_ms") - pl.col("admit_ms")) / 86_400_000).alias("los_days"),
    ((pl.col("wave_end_ms") - pl.col("wave_start_ms")) / 86_400_000).alias("wave_duration_days"),
    ((pl.col("valid_end_ms") - pl.col("valid_start_ms")) / 3_600_000).alias("valid_duration_hour_max"),
    (pl.col("wfdb_records_all").list.len()).alias("n_wfdb_records"),
])

# Order columns nicely
col_order = [
    "entity_id", "empi_nbr", "encounter_nbr", "pat_id", "type",
    "wave_start_ms", "wave_end_ms", "wave_duration_days",
    "valid_start_ms", "valid_end_ms", "valid_duration_hour_max",
    "valid_duration_hour_sum", "valid_ratio_mean",
    "sepsis_time_zero_ms",
    "admit_ms", "discharge_ms", "los_days",
    "age", "admit_dx_icd10", "admit_dx_desc",
    "encounter_type", "insurance_status", "entity_healthcare_nm",
    "n_wfdb_records", "wfdb_records", "wfdb_records_all",
]
cohort = cohort.select(col_order).sort("encounter_nbr")

# ------------------------------------------------------------
# 5. Save + summary
# ------------------------------------------------------------
cohort.write_parquet(OUT_PARQUET)
print(f"\nSaved {OUT_PARQUET}  ({cohort.height} rows)")

summary = {
    "n_entities": cohort.height,
    "n_cases": int((cohort["type"] == "case").sum()),
    "n_controls": int((cohort["type"] == "control").sum()),
    "n_unique_empi": int(cohort["empi_nbr"].n_unique()),
    "n_wfdb_records_total": int(cohort["n_wfdb_records"].sum()),
    "n_no_empi": int(cohort["empi_nbr"].is_null().sum()),
    "n_no_admit": int(cohort["admit_ms"].is_null().sum()),
    "los_days_median": float(cohort["los_days"].median()),
    "wave_duration_days_median": float(cohort["wave_duration_days"].median()),
    "valid_duration_hour_max_median": float(cohort["valid_duration_hour_max"].median()),
    "wave_start_utc_min": datetime.fromtimestamp(int(cohort["wave_start_ms"].min()) / 1000, UTC).isoformat(),
    "wave_end_utc_max": datetime.fromtimestamp(int(cohort["wave_end_ms"].max()) / 1000, UTC).isoformat(),
    "n_sepsis_time_zero_non_null": int(cohort["sepsis_time_zero_ms"].is_not_null().sum()),
    "elapsed_sec": round(time.time() - t0, 1),
}
with open(OUT_SUMMARY, "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== Stage A summary ===")
print(json.dumps(summary, indent=2))
print(f"\nHead:\n{cohort.head(3)}")
