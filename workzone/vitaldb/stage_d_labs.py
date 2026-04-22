#!/usr/bin/env python3
"""
Stage D - VitalDB labs from lab_data.csv (long format).

Phase 1: polars scan of lab_data.csv (~928k rows, small) filtered to cohort
caseids; map `name` -> var_id; apply physio range; write combined parquet.

Phase 2: per-case partition -> {entity}/labs_events.npy.
Event timestamps: time_ms = (dtstart_s + dt_s) * 1000  where dtstart_s comes
from meta.json (set by Stage B).
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

RAW_ROOT = "/opt/localdata100tb/UNIPHY_Plus/raw_datasets/vitalDB"
LABS_CSV = f"{RAW_ROOT}/lab_data.csv"

OUT_ROOT = "/opt/localdata100tb/physio_data/vitaldb"
COHORT_PARQUET = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/vitaldb/valid_cohort.parquet"
OUTPUTS_DIR = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/vitaldb"
COMBINED_PARQUET = f"{OUTPUTS_DIR}/stage_d_labs_combined.parquet"
SUMMARY_JSON = f"{OUTPUTS_DIR}/stage_d_summary.json"
LOG_DIR = "/labs/hulab/mxwang/Physio_Data/workzone/vitaldb/logs"

# lab_data.name -> var_registry id. (abbreviations from lab_parameters.csv)
NAME_TO_VAR_ID = {
    "k":     0,    # Potassium
    "ica":   1,    # ionized Calcium (use as Calcium proxy)
    "na":    2,    # Sodium
    "gluc":  3,    # Glucose
    "lac":   4,    # Lactate
    "cr":    5,    # Creatinine
    "tbil":  6,    # Total bilirubin
    "plt":   7,    # Platelets
    "wbc":   8,    # WBC
    "hb":    9,    # Hemoglobin
    "ptinr": 10,   # INR
    "bun":   11,   # BUN
    "alb":   12,   # Albumin
    "ph":    13,   # Arterial pH
    "po2":   14,   # paO2
    "pco2":  15,   # paCO2
    "hco3":  16,   # HCO3
    "ast":   17,   # AST
    "alt":   18,   # ALT
}
PHYSIO_RANGE = {
    0:  (1.0, 10.0),
    1:  (0.5, 2.0),    # ionized Ca is ~1.0-1.35
    2:  (100, 180),
    3:  (10,  1000),
    4:  (0.1, 30.0),
    5:  (0.1, 25.0),
    6:  (0.1, 60.0),
    7:  (1,   1200),
    8:  (0.1, 100),
    9:  (1.0, 25.0),
    10: (0.5, 15.0),
    11: (1.0, 250),
    12: (0.5, 6.5),
    13: (6.5, 8.0),
    14: (10,  700),
    15: (5,   150),
    16: (1.0, 60.0),
    17: (1,   50000),
    18: (1,   50000),
}
EVENT_DTYPE = np.dtype([
    ("time_ms", np.int64),
    ("var_id",  np.uint16),
    ("value",   np.float32),
])


def phase1(cohort_caseids: list[int]) -> dict:
    print(f"[D1] scanning {LABS_CSV}")
    t0 = time.time()
    lf = pl.scan_csv(LABS_CSV, low_memory=True, infer_schema_length=10000,
                     ignore_errors=True)
    names = list(NAME_TO_VAR_ID.keys())
    df = (
        lf.filter(pl.col("caseid").cast(pl.Int64, strict=False).is_in(cohort_caseids))
          .filter(pl.col("name").is_in(names))
          .select([
              pl.col("caseid").cast(pl.Int64, strict=False).alias("caseid"),
              pl.col("dt").cast(pl.Float64, strict=False).alias("dt_s"),
              pl.col("name"),
              pl.col("result").cast(pl.Utf8).str.strip_chars()
                .cast(pl.Float64, strict=False).alias("value_f"),
          ])
          .with_columns([
              pl.col("name").replace_strict(NAME_TO_VAR_ID,
                                            return_dtype=pl.UInt16).alias("var_id"),
          ])
          .filter(
              pl.col("dt_s").is_not_null()
              & pl.col("value_f").is_not_null()
              & pl.col("value_f").is_finite()
          )
    )
    rng_expr = None
    for vid, (lo, hi) in PHYSIO_RANGE.items():
        cond = (pl.col("var_id") == vid) & (pl.col("value_f") >= lo) & (pl.col("value_f") <= hi)
        rng_expr = cond if rng_expr is None else rng_expr | cond
    df = df.filter(rng_expr)
    out = df.select(["caseid", "dt_s", "var_id",
                     pl.col("value_f").cast(pl.Float32).alias("value")]).collect(engine="streaming")
    out.write_parquet(COMBINED_PARQUET)
    info = {"rows": out.height,
            "unique_caseids": int(out["caseid"].n_unique()) if out.height else 0,
            "var_ids": sorted(int(v) for v in out["var_id"].unique()) if out.height else [],
            "elapsed_sec": round(time.time() - t0, 1)}
    print(f"[D1] done: {info}")
    return info


def phase2(cohort_df: pl.DataFrame, out_root: str, resume: bool) -> list[dict]:
    combined = pl.read_parquet(COMBINED_PARQUET)
    print(f"[D2] rows={combined.height:,}  unique_caseids={combined['caseid'].n_unique()}")
    by_case = {int(k[0]) if isinstance(k, tuple) else int(k): g
               for k, g in combined.partition_by("caseid", as_dict=True).items()}
    statuses: list[dict] = []
    for row in cohort_df.iter_rows(named=True):
        eid = row["entity_id"]
        case = int(row["caseid"])
        out_dir = Path(out_root) / eid
        meta_path = out_dir / "meta.json"
        events_path = out_dir / "labs_events.npy"
        st = {"entity_id": eid, "status": "pending", "n_events": 0}
        if not meta_path.exists():
            st["status"] = "no_stage_b"; statuses.append(st); continue
        meta = json.loads(meta_path.read_text())
        if resume and events_path.exists() and meta.get("stage_d_version", 0) >= 1:
            st["status"] = "resumed"
            st["n_events"] = int(meta.get("labs", {}).get("n_events", 0))
            statuses.append(st); continue
        sub = by_case.get(case)
        dtstart = float(meta.get("dtstart_s") or 0)
        if sub is None or sub.height == 0:
            events = np.empty(0, dtype=EVENT_DTYPE)
        else:
            sub = sub.sort(["dt_s", "var_id"])
            events = np.empty(sub.height, dtype=EVENT_DTYPE)
            events["time_ms"] = ((dtstart + sub["dt_s"].to_numpy()) * 1000).astype(np.int64)
            events["var_id"]  = sub["var_id"].to_numpy()
            events["value"]   = sub["value"].to_numpy()
            events = events[np.argsort(events["time_ms"], kind="stable")]
        np.save(events_path, events)
        per_var = {}
        if events.shape[0] > 0:
            v, c = np.unique(events["var_id"], return_counts=True)
            per_var = {str(int(vv)): int(cc) for vv, cc in zip(v, c)}
        meta["labs"] = {"n_events": int(events.shape[0]),
                        "per_var_count": per_var,
                        "source": "lab_data.csv (long-format, time_ms=(dtstart+dt)*1000)",
                        "time_convention": "absolute ms"}
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
    cohort_caseids = cohort["caseid"].to_list()
    print(f"[D] cohort entities: {cohort.height}")
    info1 = None
    if args.phase in ("1", "all"):
        if args.redo_phase1 or not os.path.exists(COMBINED_PARQUET):
            info1 = phase1(cohort_caseids)
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
