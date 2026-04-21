#!/usr/bin/env python3
"""
Stage A - MOVER/EPIC cohort.

Enumerates all XMLs under `epic_wave_{1,2,3}_v2/UCI_deidentified_part*/Waveforms/`,
parses filenames to extract (MRN, class=CB|IP, file_datetime_ms), joins against
the EPIC_MRN_PAT_ID crosswalk + patient_information.csv to attribute each XML
to a LOG_ID whose anesthesia window (AN_START..AN_STOP +/- 1h) contains the
file timestamp.

Output: workzone/outputs/mover_epic/valid_cohort.parquet
  entity_id (= str(LOG_ID)), log_id, mrn, pat_id,
  n_xml_files, xml_paths (list[str]),
  an_start_ms, an_stop_ms, in_or_ms, out_or_ms,
  hosp_admsn_ms, hosp_disch_ms, los_days,
  birth_date, height_cm, weight_kg, sex, asa_rating, procedure, patient_class_nm,
  icu_admin_flag.

Filter: keep LOG_IDs with >=1 XML attributed AND valid AN window.
"""
import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

UTC = timezone.utc
RAW_ROOT = "/opt/localdata100tb/UNIPHY_Plus/raw_datasets/MOVER"
CROSSWALK_CSV = f"{RAW_ROOT}/EPIC_MRN_PAT_ID.csv"
PAT_INFO_CSV = f"{RAW_ROOT}/EPIC_EMR/EMR/patient_information.csv"
WAVE_DIRS = [
    f"{RAW_ROOT}/epic_wave_1_v2/UCI_deidentified_part1_EPIC_07_22/Waveforms",
    f"{RAW_ROOT}/epic_wave_2_v2/UCI_deidentified_part2_EPIC_08_10/Waveforms",
    f"{RAW_ROOT}/epic_wave_3_v2/UCI_deidentified_part4_EPIC_11_28/Waveforms",
]

OUT_DIR = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/mover_epic"
OUT_PARQUET = f"{OUT_DIR}/valid_cohort.parquet"
OUT_SUMMARY = f"{OUT_DIR}/stage_a_summary.json"
WINDOW_BUFFER_MS = 3600 * 1000  # +/- 1 h around anesthesia window

LA_TZ = "America/Los_Angeles"

# Filename pattern: {16-hex-MRN}{CB|IP}-{YYYY-MM-DD-HH-MM-SS-mmm}Z.xml
FNAME_RE = re.compile(
    r"^([0-9a-f]{16})(CB|IP)-(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{3})Z\.xml$"
)


def filename_to_ms(name: str) -> tuple[str, str, int] | None:
    """Return (pat_id_16hex, class, file_ms). XML filename prefix is PAT_ID
    (not MRN as naming in the raw files might suggest) - verified 30/30
    membership in the crosswalk PAT_ID column."""
    m = FNAME_RE.match(name)
    if not m:
        return None
    pat_id, cls, y, mo, d, h, mi, s, ms = m.groups()
    dt = datetime(int(y), int(mo), int(d), int(h), int(mi), int(s),
                  int(ms) * 1000, tzinfo=UTC)
    return pat_id, cls, int(dt.timestamp() * 1000)


def pt_local_to_utc_ms(col: str) -> pl.Expr:
    return (
        pl.col(col).cast(pl.Utf8)
        .str.strptime(pl.Datetime, format="%m/%d/%y %H:%M", strict=False)
        .dt.replace_time_zone(LA_TZ, ambiguous="earliest", non_existent="null")
        .dt.convert_time_zone("UTC")
        .dt.timestamp("ms")
    )


def enumerate_xmls() -> pl.DataFrame:
    """Walk all 3 EPIC wave roots, yield (pat_id, class, file_ms, path)."""
    rows = []
    for root_s in WAVE_DIRS:
        root = Path(root_s)
        if not root.exists():
            print(f"[A] WARN: wave dir missing: {root_s}")
            continue
        for p in root.glob("*/*.xml"):
            parsed = filename_to_ms(p.name)
            if parsed is None:
                continue
            pat_id, cls, file_ms = parsed
            rows.append({"pat_id": pat_id, "class": cls, "file_ms": file_ms,
                         "path": str(p)})
    return pl.DataFrame(rows, schema={
        "pat_id": pl.Utf8, "class": pl.Utf8, "file_ms": pl.Int64, "path": pl.Utf8,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    print(f"[A] reading {CROSSWALK_CSV}")
    xwalk = pl.read_csv(CROSSWALK_CSV, infer_schema_length=10000, ignore_errors=True)
    print(f"    crosswalk rows={xwalk.height}  unique MRN={xwalk['MRN'].n_unique()}  "
          f"unique LOG_ID={xwalk['LOG_ID'].n_unique()}")

    print(f"[A] reading {PAT_INFO_CSV}")
    info = pl.read_csv(PAT_INFO_CSV, infer_schema_length=10000, ignore_errors=True)
    print(f"    patient_info rows={info.height}")
    info = info.with_columns([
        pt_local_to_utc_ms("HOSP_ADMSN_TIME").alias("hosp_admsn_ms"),
        pt_local_to_utc_ms("HOSP_DISCH_TIME").alias("hosp_disch_ms"),
        pt_local_to_utc_ms("IN_OR_DTTM").alias("in_or_ms"),
        pt_local_to_utc_ms("OUT_OR_DTTM").alias("out_or_ms"),
        pt_local_to_utc_ms("AN_START_DATETIME").alias("an_start_ms"),
        pt_local_to_utc_ms("AN_STOP_DATETIME").alias("an_stop_ms"),
    ]).rename({
        "LOG_ID": "log_id", "MRN": "mrn", "SEX": "sex",
        "HEIGHT": "height_cm", "WEIGHT": "weight_kg",
        "ASA_RATING": "asa_rating", "PRIMARY_PROCEDURE_NM": "procedure",
        "PATIENT_CLASS_NM": "patient_class_nm",
        "ICU_ADMIN_FLAG": "icu_admin_flag", "LOS": "los_days",
        "BIRTH_DATE": "birth_date", "PRIMARY_ANES_TYPE_NM": "anes_type",
        "DISCH_DISP": "disch_disp",
    }).select([
        "log_id", "mrn",
        "hosp_admsn_ms", "hosp_disch_ms",
        "in_or_ms", "out_or_ms",
        "an_start_ms", "an_stop_ms",
        "los_days", "icu_admin_flag",
        "birth_date", "height_cm", "weight_kg", "sex",
        "anes_type", "asa_rating", "patient_class_nm", "procedure",
        "disch_disp",
    ]).unique("log_id", keep="first")

    print(f"[A] enumerating EPIC XMLs across 3 wave dirs ...")
    t1 = time.time()
    xmls = enumerate_xmls()
    print(f"    total XMLs indexed: {xmls.height:,}  elapsed={time.time()-t1:.1f}s")

    # Join XML.pat_id -> crosswalk.PAT_ID (one PAT_ID -> many LOG_IDs; one
    # patient can have multiple surgical encounters). Crosswalk: 65k rows /
    # 39k unique PAT_IDs -> ~1.7 encounters per patient on average.
    xml_joined = xmls.join(xwalk.rename({"MRN": "mrn", "LOG_ID": "log_id",
                                         "PAT_ID": "pat_id"}),
                           on="pat_id", how="inner")
    print(f"[A] XMLs matched to at least one LOG_ID: {xml_joined.height:,}")

    # For each candidate (XML, LOG_ID), keep only when file_ms in
    # [an_start - buffer, an_stop + buffer].
    xml_with_win = xml_joined.join(info.select(["log_id", "an_start_ms",
                                                "an_stop_ms"]),
                                   on="log_id", how="inner")
    xml_with_win = xml_with_win.filter(
        pl.col("an_start_ms").is_not_null()
        & pl.col("an_stop_ms").is_not_null()
        & (pl.col("file_ms") >= (pl.col("an_start_ms") - WINDOW_BUFFER_MS))
        & (pl.col("file_ms") <= (pl.col("an_stop_ms")  + WINDOW_BUFFER_MS))
    )
    print(f"[A] XMLs in anesthesia window: {xml_with_win.height:,}")

    # Group by log_id -> list of xml paths
    per_log = (xml_with_win.group_by("log_id").agg([
        pl.col("path").unique().sort().alias("xml_paths"),
        pl.col("path").n_unique().alias("n_xml_files"),
        pl.col("file_ms").min().alias("xml_ms_min"),
        pl.col("file_ms").max().alias("xml_ms_max"),
    ]))
    print(f"[A] LOG_IDs with at least 1 attributed XML: {per_log.height}")

    cohort = info.join(per_log, on="log_id", how="inner")
    cohort = cohort.filter(
        pl.col("an_start_ms").is_not_null()
        & pl.col("an_stop_ms").is_not_null()
        & (pl.col("an_stop_ms") >= pl.col("an_start_ms"))
    )
    # entity_id = log_id
    cohort = cohort.with_columns(pl.col("log_id").alias("entity_id")).sort("log_id")
    cohort.write_parquet(OUT_PARQUET)

    summary = {
        "n_entities": cohort.height,
        "n_xmls_total_indexed": int(xmls.height),
        "n_xmls_attributed": int(xml_with_win.height),
        "n_log_ids_in_info": int(info.height),
        "n_log_ids_in_crosswalk": int(xwalk["LOG_ID"].n_unique()),
        "median_xml_per_log": int(cohort["n_xml_files"].median() or 0),
        "median_an_duration_h": round(float(
            ((cohort["an_stop_ms"] - cohort["an_start_ms"]) / 3_600_000).median() or 0), 2),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(OUT_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== Stage A summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
