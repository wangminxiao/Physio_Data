#!/usr/bin/env python3
"""
Stage A - MOVER/SIS cohort from patient_information.csv + waveform dir scan.

Output: workzone/outputs/mover/valid_cohort.parquet
  entity_id (= str(PID)), PID, age, gender, height_cm, weight_kg,
  or_start_ms, or_end_ms, surgery_start_ms, surgery_end_ms,
  n_xml_files, procedure.

Filter: keep PIDs with >=1 XML file AND valid OR window.

Times: patient_information.csv is naive Pacific local -> UTC (LA tz,
ambiguous=earliest, non_existent=null).
"""
import argparse
import json
import os
import time
from pathlib import Path

import polars as pl

RAW_ROOT = "/opt/localdata100tb/UNIPHY_Plus/raw_datasets/MOVER"
PAT_INFO_CSV = f"{RAW_ROOT}/EMR/patient_information.csv"
SIS_WAVE_ROOT = f"{RAW_ROOT}/sis_wave_v2/UCI_deidentified_part3_SIS_11_07/Waveforms"

OUT_DIR = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/mover"
OUT_PARQUET = f"{OUT_DIR}/valid_cohort.parquet"
OUT_SUMMARY = f"{OUT_DIR}/stage_a_summary.json"

LA_TZ = "America/Los_Angeles"


def pt_local_to_utc_ms(col: str) -> pl.Expr:
    return (
        pl.col(col).cast(pl.Utf8)
        .str.strptime(pl.Datetime, format="%m/%d/%y %H:%M", strict=False)
        .dt.replace_time_zone(LA_TZ, ambiguous="earliest", non_existent="null")
        .dt.convert_time_zone("UTC")
        .dt.timestamp("ms")
    )


def scan_wave_dirs() -> pl.DataFrame:
    """Scan the SIS Waveforms tree -> DataFrame(PID, n_xml_files)."""
    root = Path(SIS_WAVE_ROOT)
    rows = []
    for pid_dir in root.glob("*/*"):
        if not pid_dir.is_dir():
            continue
        pid = pid_dir.name
        n = sum(1 for _ in pid_dir.glob("*.xml"))
        if n > 0:
            rows.append({"pid": pid, "n_xml_files": n})
    return pl.DataFrame(rows, schema={"pid": pl.Utf8, "n_xml_files": pl.Int32})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    print(f"[A] reading {PAT_INFO_CSV}")
    info = pl.read_csv(PAT_INFO_CSV, infer_schema_length=10000, ignore_errors=True)
    print(f"    rows={info.height}  unique PIDs={info['PID'].n_unique()}")

    info = info.with_columns([
        pt_local_to_utc_ms("OR_start").alias("or_start_ms"),
        pt_local_to_utc_ms("OR_end").alias("or_end_ms"),
        pt_local_to_utc_ms("Surgery_start").alias("surgery_start_ms"),
        pt_local_to_utc_ms("Surgery_end").alias("surgery_end_ms"),
    ]).rename({"PID": "pid", "Age": "age", "Ht": "height_cm",
               "Wt": "weight_kg", "Gender": "gender",
               "Procedure": "procedure"}).with_columns([
        pl.col("age").cast(pl.Int32, strict=False),
        pl.col("height_cm").cast(pl.Float32, strict=False),
        pl.col("weight_kg").cast(pl.Float32, strict=False),
    ])

    print(f"[A] scanning waveform dirs under {SIS_WAVE_ROOT}")
    waves = scan_wave_dirs()
    print(f"    PIDs with waveforms: {waves.height}")

    cohort = info.join(waves, on="pid", how="inner")
    before = cohort.height
    cohort = cohort.filter(
        pl.col("or_start_ms").is_not_null()
        & pl.col("or_end_ms").is_not_null()
        & (pl.col("or_end_ms") >= pl.col("or_start_ms"))
    )
    print(f"[A] after filter (have waveform + valid OR window): {cohort.height} / {before}")

    cohort = cohort.with_columns(pl.col("pid").alias("entity_id"))
    col_order = [
        "entity_id", "pid",
        "or_start_ms", "or_end_ms", "surgery_start_ms", "surgery_end_ms",
        "n_xml_files", "age", "gender", "height_cm", "weight_kg", "procedure",
    ]
    cohort = cohort.select(col_order).unique("entity_id", keep="first").sort("pid")
    cohort.write_parquet(OUT_PARQUET)

    summary = {
        "n_entities": cohort.height,
        "n_unique_pid": int(cohort["pid"].n_unique()),
        "total_xml_files": int(cohort["n_xml_files"].sum()),
        "median_xml_per_pid": int(cohort["n_xml_files"].median()),
        "median_or_duration_h": round(float(
            ((cohort["or_end_ms"] - cohort["or_start_ms"]) / 3_600_000).median() or 0), 2),
        "n_info_only": int(info.height - cohort.height),
        "n_wave_only": int(waves.height - cohort.height),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(OUT_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== Stage A summary ===")
    print(json.dumps(summary, indent=2))
    print(cohort.head(3))


if __name__ == "__main__":
    main()
