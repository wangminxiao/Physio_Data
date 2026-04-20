#!/usr/bin/env python3
"""
Stage F-2 — MC_MED demographics.csv.

One row per entity. All fields come from the cohort parquet (visits.csv join
in Stage A) — no separate demographics table in MC_MED.

Columns: entity_id, csn, mrn, visit_no, visits_total,
  gender, age, race, ethnicity, means_of_arrival, triage_acuity,
  chief_complaint, ed_dispo, admit_service, payor_class,
  dx_icd10, dx_name,
  arrival_utc_ms, departure_utc_ms, admit_utc_ms, dispo_utc_ms,
  ed_los_hours, hosp_los_days,
  wave_start_ms, wave_end_ms.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import polars as pl

OUT_ROOT = "/opt/localdata100tb/physio_data/mcmed"
COHORT_PARQUET = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/mcmed/valid_cohort.parquet"
SUMMARY_JSON = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/mcmed/stage_f_demographics_summary.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=OUT_ROOT)
    args = ap.parse_args()

    t0 = time.time()
    out_root = Path(args.out_root)
    cohort = pl.read_parquet(COHORT_PARQUET).unique("entity_id", keep="first")
    on_disk = {p.name for p in out_root.iterdir()
               if p.is_dir() and (p / "meta.json").exists()}
    cohort = cohort.filter(pl.col("entity_id").is_in(list(on_disk)))
    print(f"cohort entities with output: {cohort.height}")

    out_csv = out_root / "demographics.csv"
    n = 0
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "entity_id", "csn", "mrn", "visit_no", "visits_total",
            "gender", "age", "race", "ethnicity",
            "means_of_arrival", "triage_acuity", "chief_complaint",
            "ed_dispo", "admit_service", "payor_class",
            "dx_icd10", "dx_name",
            "arrival_utc_ms", "departure_utc_ms",
            "admit_utc_ms", "dispo_utc_ms",
            "ed_los_hours", "hosp_los_days",
            "wave_start_ms", "wave_end_ms",
        ])
        for row in cohort.iter_rows(named=True):
            # Read wave bounds from meta.json (Stage B writes them)
            meta_path = out_root / row["entity_id"] / "meta.json"
            wave_start = wave_end = ""
            try:
                m = json.loads(meta_path.read_text())
                wave_start = int(m.get("wave_start_ms") or 0)
                wave_end   = int(m.get("wave_end_ms") or 0)
            except Exception:
                pass
            w.writerow([
                row["entity_id"],
                int(row["csn"]) if row.get("csn") is not None else "",
                int(row["mrn"]) if row.get("mrn") is not None else "",
                int(row["visit_no"]) if row.get("visit_no") is not None else "",
                int(row["visits_total"]) if row.get("visits_total") is not None else "",
                row.get("gender") or "",
                int(row["age"]) if row.get("age") is not None else "",
                row.get("race") or "",
                row.get("ethnicity") or "",
                row.get("means_of_arrival") or "",
                row.get("triage_acuity") or "",
                row.get("chief_complaint") or "",
                row.get("ed_dispo") or "",
                row.get("admit_service") or "",
                row.get("payor_class") or "",
                row.get("dx_icd10") or "",
                row.get("dx_name") or "",
                int(row["arrival_ms"]) if row.get("arrival_ms") is not None else "",
                int(row["departure_ms"]) if row.get("departure_ms") is not None else "",
                int(row["admit_ms"]) if row.get("admit_ms") is not None else "",
                int(row["dispo_ms"]) if row.get("dispo_ms") is not None else "",
                round(float(row["ed_los_hours"]), 3) if row.get("ed_los_hours") is not None else "",
                round(float(row["hosp_los_days"]), 3) if row.get("hosp_los_days") is not None else "",
                wave_start, wave_end,
            ])
            n += 1

    elapsed = time.time() - t0
    summary = {
        "stage": "f_demographics",
        "ran_at_unix": int(time.time()),
        "elapsed_sec": round(elapsed, 1),
        "n_rows": n,
        "output_csv": str(out_csv),
    }
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
