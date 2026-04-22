#!/usr/bin/env python3
"""
Stage F-2 - VitalDB demographics.csv.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import polars as pl

OUT_ROOT = "/opt/localdata100tb/physio_data/vitaldb"
COHORT_PARQUET = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/vitaldb/valid_cohort.parquet"
SUMMARY_JSON = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/vitaldb/stage_f_demographics_summary.json"


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
        w.writerow(["entity_id", "caseid", "subjectid",
                    "age", "sex", "height_cm", "weight_kg", "bmi",
                    "asa", "emop",
                    "department", "optype", "dx", "opname", "approach",
                    "position", "ane_type", "icu_days", "death_inhosp",
                    "anestart_s", "aneend_s", "opstart_s", "opend_s",
                    "adm_s", "dis_s",
                    "wave_start_ms", "wave_end_ms"])
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
                row["entity_id"], int(row["caseid"]), int(row.get("subjectid") or 0),
                row.get("age") or "", row.get("sex") or "",
                round(float(row["height_cm"]), 1) if row.get("height_cm") is not None else "",
                round(float(row["weight_kg"]), 2) if row.get("weight_kg") is not None else "",
                round(float(row["bmi"]), 2) if row.get("bmi") is not None else "",
                row.get("asa") or "", row.get("emop") or "",
                row.get("department") or "", row.get("optype") or "",
                row.get("dx") or "", row.get("opname") or "", row.get("approach") or "",
                row.get("position") or "", row.get("ane_type") or "",
                row.get("icu_days") or "", row.get("death_inhosp") or "",
                row.get("anestart_s") or "", row.get("aneend_s") or "",
                row.get("opstart_s") or "", row.get("opend_s") or "",
                row.get("adm_s") or "", row.get("dis_s") or "",
                wave_start, wave_end,
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
