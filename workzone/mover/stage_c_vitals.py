#!/usr/bin/env python3
"""
Stage C - MOVER/SIS vitals from patient_vitals.csv (wide format).

Phase 1: polars scan + melt + time parse (Pacific naive -> UTC).
  Output: workzone/outputs/mover/stage_c_vitals_combined.parquet

Phase 2: per-PID partition -> {entity}/vitals_events.npy.
  Update meta.json with vitals section + stage_c_version=1.

Columns: PID, Obs_time, HRe, HRp, nSBP, nMAP, nDBP, SP02.
Mapping: HRe->100, nSBP->104, nDBP->105, nMAP->106, SP02->101.
HRp skipped (redundant with HRe on GE monitors).
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
VITALS_CSV = f"{RAW_ROOT}/EMR/patient_vitals.csv"

OUT_ROOT = "/opt/localdata100tb/physio_data/mover"
COHORT_PARQUET = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/mover/valid_cohort.parquet"
OUTPUTS_DIR = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/mover"
COMBINED_PARQUET = f"{OUTPUTS_DIR}/stage_c_vitals_combined.parquet"
SUMMARY_JSON = f"{OUTPUTS_DIR}/stage_c_summary.json"
LOG_DIR = "/labs/hulab/mxwang/Physio_Data/workzone/mover/logs"

LA_TZ = "America/Los_Angeles"

COL_TO_VAR_ID = {
    "HRe":  100,
    "SP02": 101,
    "nSBP": 104,
    "nDBP": 105,
    "nMAP": 106,
}
PHYSIO_RANGE = {
    100: (10,  300),
    101: (20,  100),
    104: (30,  300),
    105: (10,  200),
    106: (20,  250),
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


def phase1(cohort_pids: list[str]) -> dict:
    print(f"[C1] scanning {VITALS_CSV}")
    t0 = time.time()
    lf = pl.scan_csv(VITALS_CSV, low_memory=True, infer_schema_length=10000,
                     ignore_errors=True,
                     null_values=[r"\N", ""])
    sub = (
        lf.filter(pl.col("PID").is_in(cohort_pids))
          .select([
              pl.col("PID").cast(pl.Utf8).alias("pid"),
              pt_local_to_utc_ms("Obs_time").alias("time_ms"),
              *[pl.col(c).cast(pl.Float64, strict=False) for c in COL_TO_VAR_ID],
          ])
          .filter(pl.col("time_ms").is_not_null())
          .collect(engine="streaming")
    )
    long = sub.unpivot(index=["pid", "time_ms"],
                       on=list(COL_TO_VAR_ID.keys()),
                       variable_name="col",
                       value_name="value_f").filter(
        pl.col("value_f").is_not_null() & pl.col("value_f").is_finite()
    ).with_columns(
        pl.col("col").replace_strict(COL_TO_VAR_ID, return_dtype=pl.UInt16).alias("var_id")
    )
    # Physio range filter per var
    rng_expr = None
    for vid, (lo, hi) in PHYSIO_RANGE.items():
        cond = (pl.col("var_id") == vid) & (pl.col("value_f") >= lo) & (pl.col("value_f") <= hi)
        rng_expr = cond if rng_expr is None else rng_expr | cond
    long = long.filter(rng_expr)
    out = long.select(["pid", "time_ms", "var_id",
                       pl.col("value_f").cast(pl.Float32).alias("value")])
    out.write_parquet(COMBINED_PARQUET)
    info = {"rows": out.height, "unique_pids": int(out["pid"].n_unique()),
            "var_ids": sorted(int(v) for v in out["var_id"].unique()),
            "elapsed_sec": round(time.time() - t0, 1)}
    print(f"[C1] done: {info}")
    return info


def phase2(cohort_df: pl.DataFrame, out_root: str, resume: bool) -> list[dict]:
    combined = pl.read_parquet(COMBINED_PARQUET)
    print(f"[C2] rows={combined.height:,}  unique_pids={combined['pid'].n_unique()}")
    by_pid = {k[0] if isinstance(k, tuple) else k: g
              for k, g in combined.partition_by("pid", as_dict=True).items()}
    statuses: list[dict] = []
    for row in cohort_df.iter_rows(named=True):
        pid = row["entity_id"]
        out_dir = Path(out_root) / pid
        meta_path = out_dir / "meta.json"
        events_path = out_dir / "vitals_events.npy"
        st = {"entity_id": pid, "status": "pending", "n_events": 0}
        if not meta_path.exists():
            st["status"] = "no_stage_b"; statuses.append(st); continue
        meta = json.loads(meta_path.read_text())
        if resume and events_path.exists() and meta.get("stage_c_version", 0) >= 1:
            st["status"] = "resumed"
            st["n_events"] = int(meta.get("vitals", {}).get("n_events", 0))
            statuses.append(st); continue
        sub = by_pid.get(pid)
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
        meta["vitals"] = {"n_events": int(events.shape[0]),
                          "per_var_count": per_var,
                          "source": "patient_vitals.csv (wide-melted)",
                          "time_convention": "Pacific->UTC ms"}
        meta["stage_c_version"] = 1
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
                                  logging.FileHandler(f"{LOG_DIR}/stage_c_vitals.log")])

    cohort = pl.read_parquet(COHORT_PARQUET).unique("entity_id", keep="first")
    cohort_pids = cohort["pid"].to_list()
    print(f"[C] cohort entities: {cohort.height}")

    info1 = None
    if args.phase in ("1", "all"):
        if args.redo_phase1 or not os.path.exists(COMBINED_PARQUET):
            info1 = phase1(cohort_pids)
        else:
            print(f"[C1] skipping (exists)")

    if args.phase in ("2", "all"):
        if not os.path.exists(COMBINED_PARQUET):
            print("[C2] needs phase 1"); sys.exit(1)
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
        print(f"\n=== Stage C summary ===\n{json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
