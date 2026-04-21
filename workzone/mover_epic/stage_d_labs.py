#!/usr/bin/env python3
"""
Stage D - MOVER/EPIC labs from EPIC_EMR/EMR/patient_labs.csv (29 M rows).

Long format: LOG_ID, MRN, ENC_TYPE_NM, Lab Code, Lab Name, Observation Value,
Measurement Units, Reference Range, Abnormal Flag, Collection Datetime.

Maps `Lab Name` to var_registry IDs. Names calibrated against top-50
frequency list (see datasets/mover_epic/API.md).
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

RAW_ROOT = "/opt/localdata100tb/UNIPHY_Plus/raw_datasets/MOVER"
LABS_CSV = f"{RAW_ROOT}/EPIC_EMR/EMR/patient_labs.csv"

OUT_ROOT = "/opt/localdata100tb/physio_data/mover_epic"
COHORT_PARQUET = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/mover_epic/valid_cohort.parquet"
OUTPUTS_DIR = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/mover_epic"
COMBINED_PARQUET = f"{OUTPUTS_DIR}/stage_d_labs_combined.parquet"
SUMMARY_JSON = f"{OUTPUTS_DIR}/stage_d_summary.json"
LOG_DIR = "/labs/hulab/mxwang/Physio_Data/workzone/mover_epic/logs"

LA_TZ = "America/Los_Angeles"

LAB_NAME_TO_VAR_ID = {
    "Potassium": 0,
    "Calcium": 1,
    "Calcium.ionized": 1,
    "Sodium": 2,
    "Glucose": 3,
    "Creatinine": 5,
    "Bilirubin": 6,
    "Platelets": 7,
    "Hemoglobin": 9,
    "Coagulation tissue factor induced.INR": 10,
    "Urea nitrogen": 11,
    "Albumin": 12,
    "pH": 13,
    "Oxygen": 14,
    # 'Carbon dioxide' on EPIC = total CO2 in BMP (functionally HCO3);
    # 'Bicarbonate' = blood-gas HCO3 explicitly.
    "Carbon dioxide": 16,
    "Bicarbonate": 16,
    "Aspartate aminotransferase": 17,
    "Alanine aminotransferase": 18,
}
PHYSIO_RANGE = {
    0:  (1.0, 10.0),
    1:  (4.0, 15.0),
    2:  (100, 180),
    3:  (10,  1000),
    5:  (0.1, 25.0),
    6:  (0.1, 60.0),
    7:  (1,   1200),
    9:  (1.0, 25.0),
    10: (0.5, 15.0),
    11: (1.0, 250),
    12: (0.5, 6.5),
    13: (6.5, 8.0),
    14: (10,  700),
    16: (1.0, 60.0),
    17: (1,   50000),
    18: (1,   50000),
}
EVENT_DTYPE = np.dtype([
    ("time_ms", np.int64),
    ("var_id",  np.uint16),
    ("value",   np.float32),
])


def pt_local_to_utc_ms(col: str) -> pl.Expr:
    return (
        pl.col(col).cast(pl.Utf8)
        .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
        .dt.replace_time_zone(LA_TZ, ambiguous="earliest", non_existent="null")
        .dt.convert_time_zone("UTC")
        .dt.timestamp("ms")
    )


def phase1(cohort_log_ids: list[str]) -> dict:
    print(f"[D1] scanning {LABS_CSV}")
    t0 = time.time()
    lf = pl.scan_csv(LABS_CSV, low_memory=True, infer_schema_length=10000,
                     ignore_errors=True, null_values=["", "NA"])
    names = list(LAB_NAME_TO_VAR_ID.keys())
    df = (
        lf.filter(pl.col("LOG_ID").is_in(cohort_log_ids))
          .filter(pl.col("Lab Name").is_in(names))
          .select([
              pl.col("LOG_ID").cast(pl.Utf8).alias("log_id"),
              pt_local_to_utc_ms("Collection Datetime").alias("time_ms"),
              pl.col("Lab Name"),
              pl.col("Observation Value").cast(pl.Utf8).str.strip_chars()
                .str.strip_chars("<>=~%+ ")
                .cast(pl.Float64, strict=False).alias("value_f"),
          ])
          .with_columns([
              pl.col("Lab Name").replace_strict(LAB_NAME_TO_VAR_ID,
                                                return_dtype=pl.UInt16)
                .alias("var_id"),
          ])
          .filter(
              pl.col("time_ms").is_not_null()
              & pl.col("value_f").is_not_null()
              & pl.col("value_f").is_finite()
          )
    )
    rng_expr = None
    for vid, (lo, hi) in PHYSIO_RANGE.items():
        cond = (pl.col("var_id") == vid) & (pl.col("value_f") >= lo) & (pl.col("value_f") <= hi)
        rng_expr = cond if rng_expr is None else rng_expr | cond
    df = df.filter(rng_expr)
    out = (df.select(["log_id", "time_ms", "var_id",
                      pl.col("value_f").cast(pl.Float32).alias("value")])
             .collect(engine="streaming"))
    out.write_parquet(COMBINED_PARQUET)
    info = {"rows": out.height,
            "unique_log_ids": int(out["log_id"].n_unique()) if out.height else 0,
            "var_ids": sorted(int(v) for v in out["var_id"].unique()) if out.height else [],
            "elapsed_sec": round(time.time() - t0, 1)}
    print(f"[D1] done: {info}")
    return info


def phase2(cohort_df: pl.DataFrame, out_root: str, resume: bool) -> list[dict]:
    combined = pl.read_parquet(COMBINED_PARQUET)
    print(f"[D2] rows={combined.height:,}  unique_log_ids={combined['log_id'].n_unique()}")
    by_log = {k[0] if isinstance(k, tuple) else k: g
              for k, g in combined.partition_by("log_id", as_dict=True).items()}
    statuses: list[dict] = []
    for row in cohort_df.iter_rows(named=True):
        log_id = row["entity_id"]
        out_dir = Path(out_root) / log_id
        meta_path = out_dir / "meta.json"
        events_path = out_dir / "labs_events.npy"
        st = {"entity_id": log_id, "status": "pending", "n_events": 0}
        if not meta_path.exists():
            st["status"] = "no_stage_b"; statuses.append(st); continue
        meta = json.loads(meta_path.read_text())
        if resume and events_path.exists() and meta.get("stage_d_version", 0) >= 1:
            st["status"] = "resumed"
            st["n_events"] = int(meta.get("labs", {}).get("n_events", 0))
            statuses.append(st); continue
        sub = by_log.get(log_id)
        if sub is None or sub.height == 0:
            events = np.empty(0, dtype=EVENT_DTYPE)
        else:
            sub = sub.sort(["time_ms", "var_id"])
            events = np.empty(sub.height, dtype=EVENT_DTYPE)
            events["time_ms"] = sub["time_ms"].to_numpy()
            events["var_id"]  = sub["var_id"].to_numpy()
            events["value"]   = sub["value"].to_numpy()
        np.save(events_path, events)
        per_var = {}
        if events.shape[0] > 0:
            v, c = np.unique(events["var_id"], return_counts=True)
            per_var = {str(int(vv)): int(cc) for vv, cc in zip(v, c)}
        meta["labs"] = {"n_events": int(events.shape[0]),
                        "per_var_count": per_var,
                        "source": "EPIC_EMR/EMR/patient_labs.csv",
                        "time_convention": "Pacific->UTC ms"}
        meta["stage_d_version"] = 1
        meta_path.write_text(json.dumps(meta, indent=2, default=str))
        st["status"] = "ok" if events.shape[0] > 0 else "ok_empty"
        st["n_events"] = int(events.shape[0])
        statuses.append(st)
    return statuses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["1", "2", "all"], default="all")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--entity-id", default=None)
    ap.add_argument("--entities", default=None)
    ap.add_argument("--out-root", default=OUT_ROOT)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--redo-phase1", action="store_true")
    args = ap.parse_args()
    os.makedirs(LOG_DIR, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(f"{LOG_DIR}/stage_d_labs.log")])
    cohort = pl.read_parquet(COHORT_PARQUET).unique("entity_id", keep="first")
    cohort_log_ids = cohort["log_id"].to_list()
    print(f"[D] cohort entities: {cohort.height}")
    info1 = None
    if args.phase in ("1", "all"):
        if args.redo_phase1 or not os.path.exists(COMBINED_PARQUET):
            info1 = phase1(cohort_log_ids)
        else:
            print(f"[D1] skipping (exists)")
    if args.phase in ("2", "all"):
        if not os.path.exists(COMBINED_PARQUET):
            print("[D2] needs phase 1"); sys.exit(1)
        df = cohort
        if args.entity_id:
            df = df.filter(pl.col("entity_id") == args.entity_id)
        elif args.entities:
            ids = [s.strip() for s in args.entities.split(",") if s.strip()]
            df = df.filter(pl.col("entity_id").is_in(ids))
        elif args.limit:
            df = df.head(args.limit)
        t0 = time.time()
        statuses = phase2(df, args.out_root, resume=not args.no_resume)
        by = {}
        for s in statuses:
            by[s["status"]] = by.get(s["status"], 0) + 1
        summary = {"n_entities": len(statuses),
                   "elapsed_sec_phase2": round(time.time() - t0, 1),
                   "by_status": by,
                   "total_events": int(sum(s.get("n_events", 0) for s in statuses)),
                   "phase1": info1}
        with open(SUMMARY_JSON, "w") as f:
            json.dump({"summary": summary}, f, indent=2, default=str)
        print(f"\n=== Stage D summary ===\n{json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
