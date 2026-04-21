#!/usr/bin/env python3
"""
Stage F-2 - MOVER/EPIC demographics.csv.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import polars as pl

OUT_ROOT = "/opt/localdata100tb/physio_data/mover_epic"
COHORT_PARQUET = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/mover_epic/valid_cohort.parquet"
SUMMARY_JSON = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/mover_epic/stage_f_demographics_summary.json"


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
        w.writerow(["entity_id", "log_id", "mrn",
                    "sex", "birth_date", "height_cm", "weight_kg",
                    "asa_rating", "anes_type", "patient_class_nm",
                    "procedure", "icu_admin_flag",
                    "hosp_admsn_ms", "hosp_disch_ms",
                    "in_or_ms", "out_or_ms",
                    "an_start_ms", "an_stop_ms",
                    "los_days",
                    "wave_start_ms", "wave_end_ms",
                    "n_xml_files"])
        for row in cohort.iter_rows(named=True):
            meta_path = out_root / row["entity_id"] / "meta.json"
            wave_start = wave_end = ""
            try:
                m = json.loads(meta_path.read_text())
                wave_start = int(m.get("wave_start_ms") or 0)
                wave_end   = int(m.get("wave_end_ms") or 0)
            except Exception:
                pass
            w.writerow([
                row["entity_id"], row["log_id"], row.get("mrn") or "",
                row.get("sex") or "", row.get("birth_date") or "",
                round(float(row["height_cm"]), 1) if row.get("height_cm") is not None else "",
                round(float(row["weight_kg"]), 2) if row.get("weight_kg") is not None else "",
                row.get("asa_rating") or "",
                row.get("anes_type") or "",
                row.get("patient_class_nm") or "",
                row.get("procedure") or "",
                row.get("icu_admin_flag") or "",
                int(row["hosp_admsn_ms"]) if row.get("hosp_admsn_ms") is not None else "",
                int(row["hosp_disch_ms"]) if row.get("hosp_disch_ms") is not None else "",
                int(row["in_or_ms"])  if row.get("in_or_ms")  is not None else "",
                int(row["out_or_ms"]) if row.get("out_or_ms") is not None else "",
                int(row["an_start_ms"]) if row.get("an_start_ms") is not None else "",
                int(row["an_stop_ms"])  if row.get("an_stop_ms")  is not None else "",
                round(float(row["los_days"]), 2) if row.get("los_days") is not None else "",
                wave_start, wave_end,
                int(row["n_xml_files"]) if row.get("n_xml_files") is not None else "",
            ])
            n += 1
    summary = {"stage": "f_demographics",
               "ran_at_unix": int(time.time()),
               "elapsed_sec": round(time.time() - t0, 1),
               "n_rows": n, "output_csv": str(out_csv)}
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
