#!/usr/bin/env python3
"""
Stage C — MC_MED vitals extraction from numerics.csv (2.6 GB).

Phase 1: Polars streaming scan of numerics.csv filtered to cohort CSNs.
  Map Measure -> var_id (see MEASURE_TO_VAR_ID). Drop non-numeric values.
  Write one combined parquet:
    workzone/outputs/mcmed/stage_c_vitals_combined.parquet

Phase 2: partition by CSN. For each entity with Stage B meta.json:
  Write {entity}/vitals_events.npy as structured (time_ms, var_id, value).
  Update meta with `vitals` section + stage_c_version=1.

Resume: skip entity if vitals_events.npy exists and stage_c_version>=1.
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

RAW_ROOT = "/opt/localdata100tb/UNIPHY_Plus/raw_datasets/physionet.org/files/mc-med/1.0.1/data"
NUMERICS_CSV = f"{RAW_ROOT}/numerics.csv"

OUT_ROOT = "/opt/localdata100tb/physio_data/mcmed"
COHORT_PARQUET = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/mcmed/valid_cohort.parquet"
OUTPUTS_DIR = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/mcmed"
COMBINED_PARQUET = f"{OUTPUTS_DIR}/stage_c_vitals_combined.parquet"
SUMMARY_JSON = f"{OUTPUTS_DIR}/stage_c_summary.json"
LOG_DIR = "/labs/hulab/mxwang/Physio_Data/workzone/mcmed/logs"

# numerics.Measure -> var_registry id. Only values in this map are kept.
MEASURE_TO_VAR_ID = {
    "HR":   100,
    "SpO2": 101,
    "RR":   102,
    "Temp": 103,
    "SBP":  104,
    "DBP":  105,
    "MAP":  106,
}
# Physiological bounds (lookup-free to avoid registry dep in worker)
PHYSIO_RANGE = {
    100: (10,  300),
    101: (20,  100),
    102: (1,    70),
    103: (25,   45),
    104: (30,  300),
    105: (10,  200),
    106: (20,  250),
}

EVENT_DTYPE = np.dtype([
    ("time_ms", np.int64),
    ("var_id",  np.uint16),
    ("value",   np.float32),
])


def iso_z_to_ms(col: str) -> pl.Expr:
    return (
        pl.col(col).cast(pl.Utf8)
        .str.strip_suffix("Z")
        .str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S%.f", strict=False)
        .dt.replace_time_zone("UTC")
        .dt.timestamp("ms")
    )


def phase1(cohort_csns: list[int]) -> dict:
    print(f"[C1] scanning {NUMERICS_CSV}")
    t0 = time.time()
    lf = pl.scan_csv(NUMERICS_CSV, low_memory=True, infer_schema_length=10000,
                     ignore_errors=True)
    measures = list(MEASURE_TO_VAR_ID.keys())
    df = (
        lf.filter(pl.col("CSN").cast(pl.Int64, strict=False).is_in(cohort_csns))
          .filter(pl.col("Measure").is_in(measures))
          .select([
              pl.col("CSN").cast(pl.Int64, strict=False).alias("csn"),
              iso_z_to_ms("Time").alias("time_ms"),
              pl.col("Measure"),
              pl.col("Value").cast(pl.Utf8).str.strip_chars()
                .cast(pl.Float64, strict=False).alias("value_f"),
          ])
          .with_columns([
              pl.col("Measure").replace_strict(MEASURE_TO_VAR_ID,
                                               return_dtype=pl.UInt16)
                .alias("var_id"),
          ])
          .filter(
              pl.col("time_ms").is_not_null()
              & pl.col("value_f").is_not_null()
              & pl.col("value_f").is_finite()
          )
    )
    # Apply physio range filter per var_id
    ranges_expr = None
    for vid, (lo, hi) in PHYSIO_RANGE.items():
        cond = (pl.col("var_id") == vid) & (pl.col("value_f") >= lo) & (pl.col("value_f") <= hi)
        ranges_expr = cond if ranges_expr is None else ranges_expr | cond
    df = df.filter(ranges_expr)
    out = (df.select([
        "csn", "time_ms", "var_id",
        pl.col("value_f").cast(pl.Float32).alias("value"),
    ])).collect(engine="streaming")
    out.write_parquet(COMBINED_PARQUET)
    info = {
        "rows": out.height,
        "unique_csns": int(out["csn"].n_unique()),
        "var_ids": sorted(int(v) for v in out["var_id"].unique()),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    print(f"[C1] done: {info}")
    return info


def phase2(cohort_df: pl.DataFrame, out_root: str, resume: bool) -> list[dict]:
    print(f"[C2] reading combined parquet")
    combined = pl.read_parquet(COMBINED_PARQUET)
    print(f"[C2] rows={combined.height:,}  unique_csns={combined['csn'].n_unique()}")

    by_csn = {int(k[0]) if isinstance(k, tuple) else int(k): g
              for k, g in combined.partition_by("csn", as_dict=True).items()}

    statuses: list[dict] = []
    for row in cohort_df.iter_rows(named=True):
        eid = row["entity_id"]
        csn = int(row["csn"])
        out_dir = Path(out_root) / eid
        meta_path = out_dir / "meta.json"
        events_path = out_dir / "vitals_events.npy"

        st = {"entity_id": eid, "csn": csn, "status": "pending", "n_events": 0}
        if not meta_path.exists():
            st["status"] = "no_stage_b"
            statuses.append(st); continue
        meta = json.loads(meta_path.read_text())
        if resume and events_path.exists() and meta.get("stage_c_version", 0) >= 1:
            st["status"] = "resumed"
            st["n_events"] = int(meta.get("vitals", {}).get("n_events", 0))
            statuses.append(st); continue

        sub = by_csn.get(csn)
        if sub is None or sub.height == 0:
            events = np.empty(0, dtype=EVENT_DTYPE)
        else:
            sub = sub.sort(["time_ms", "var_id"])
            events = np.empty(sub.height, dtype=EVENT_DTYPE)
            events["time_ms"] = sub["time_ms"].to_numpy()
            events["var_id"] = sub["var_id"].to_numpy()
            events["value"] = sub["value"].to_numpy()
        np.save(events_path, events)

        if events.shape[0] > 0:
            v, c = np.unique(events["var_id"], return_counts=True)
            per_var = {str(int(vv)): int(cc) for vv, cc in zip(v, c)}
        else:
            per_var = {}
        meta["vitals"] = {
            "n_events": int(events.shape[0]),
            "per_var_count": per_var,
            "source": "numerics.csv",
            "time_convention": "UTC ISO-8601 (random-shifted)",
        }
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
    ap.add_argument("--entity-id", type=str, default=None)
    ap.add_argument("--entities", type=str, default=None)
    ap.add_argument("--out-root", type=str, default=OUT_ROOT)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--redo-phase1", action="store_true")
    args = ap.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(f"{LOG_DIR}/stage_c_vitals.log")])

    cohort = pl.read_parquet(COHORT_PARQUET).unique("entity_id", keep="first")
    cohort_csns = cohort["csn"].to_list()
    print(f"[C] cohort entities: {cohort.height}")

    info1 = None
    if args.phase in ("1", "all"):
        if args.redo_phase1 or not os.path.exists(COMBINED_PARQUET):
            info1 = phase1(cohort_csns)
        else:
            print(f"[C1] skipping (combined exists: {COMBINED_PARQUET})")

    if args.phase in ("2", "all"):
        if not os.path.exists(COMBINED_PARQUET):
            print("[C2] needs phase 1 first"); sys.exit(1)
        df = cohort
        if args.entity_id:
            df = df.filter(pl.col("entity_id") == args.entity_id)
        elif args.entities:
            ids = [s.strip() for s in args.entities.split(",") if s.strip()]
            df = df.filter(pl.col("entity_id").is_in(ids))
        elif args.limit:
            df = df.head(args.limit)
        print(f"[C2] {df.height} entities  resume={not args.no_resume}")
        t0 = time.time()
        statuses = phase2(df, args.out_root, resume=not args.no_resume)
        elapsed = time.time() - t0
        by_status: dict[str, int] = {}
        for s in statuses:
            by_status[s["status"]] = by_status.get(s["status"], 0) + 1
        summary = {
            "n_entities": len(statuses),
            "elapsed_sec_phase2": round(elapsed, 1),
            "by_status": by_status,
            "total_events": int(sum(s.get("n_events", 0) for s in statuses)),
            "phase1": info1,
        }
        with open(SUMMARY_JSON, "w") as f:
            json.dump({"summary": summary}, f, indent=2, default=str)
        print(f"\n=== Stage C summary ===\n{json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
