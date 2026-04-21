#!/usr/bin/env python3
"""
Stage F-2 - MOVER/SIS demographics.csv.

One row per entity. All fields from patient_information.csv (already joined
in cohort parquet) plus wave bounds from meta.json.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import polars as pl

OUT_ROOT = "/opt/localdata100tb/physio_data/mover"
COHORT_PARQUET = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/mover/valid_cohort.parquet"
SUMMARY_JSON = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/mover/stage_f_demographics_summary.json"


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
        w.writerow(["entity_id", "pid",
                    "gender", "age", "height_cm", "weight_kg",
                    "procedure",
                    "or_start_ms", "or_end_ms",
                    "surgery_start_ms", "surgery_end_ms",
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
                row["entity_id"], row["pid"],
                row.get("gender") or "",
                int(row["age"]) if row.get("age") is not None else "",
                round(float(row["height_cm"]), 1) if row.get("height_cm") is not None else "",
                round(float(row["weight_kg"]), 2) if row.get("weight_kg") is not None else "",
                row.get("procedure") or "",
                int(row["or_start_ms"]) if row.get("or_start_ms") is not None else "",
                int(row["or_end_ms"])   if row.get("or_end_ms")   is not None else "",
                int(row["surgery_start_ms"]) if row.get("surgery_start_ms") is not None else "",
                int(row["surgery_end_ms"])   if row.get("surgery_end_ms")   is not None else "",
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
