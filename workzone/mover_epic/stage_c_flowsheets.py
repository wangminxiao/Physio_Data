#!/usr/bin/env python3
"""
Stage C - MOVER/EPIC vitals from flowsheets_cleaned/flowsheet_part{1..19}.csv.

Flowsheet format (long): LOG_ID, MRN, FLO_NAME, FLO_DISPLAY_NAME, RECORD_TYPE,
RECORDED_TIME, MEAS_VALUE, UNITS.

Maps (FLO_NAME, FLO_DISPLAY_NAME) pairs to var_registry IDs. We only keep the
"Vital Signs " group and the "Devices Testing Template" anesthesia-monitor
group - these cover HR/SpO2/RR/Temp/MAP/EtCO2.

Phase 1 streams all 19 parts through polars into a single combined parquet.
Phase 2 partitions by LOG_ID and writes per-entity vitals_events.npy.

BP (systolic/diastolic as "120/80" string) is NOT parsed in v1 - need a
separate extra stage for that.
"""
import argparse
import glob
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

RAW_ROOT = "/opt/localdata100tb/UNIPHY_Plus/raw_datasets/MOVER"
FLOWSHEET_GLOB = f"{RAW_ROOT}/flowsheets_cleaned/flowsheet_part*.csv"

OUT_ROOT = "/opt/localdata100tb/physio_data/mover_epic"
COHORT_PARQUET = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/mover_epic/valid_cohort.parquet"
OUTPUTS_DIR = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/mover_epic"
COMBINED_PARQUET = f"{OUTPUTS_DIR}/stage_c_vitals_combined.parquet"
SUMMARY_JSON = f"{OUTPUTS_DIR}/stage_c_summary.json"
LOG_DIR = "/labs/hulab/mxwang/Physio_Data/workzone/mover_epic/logs"

LA_TZ = "America/Los_Angeles"

# (FLO_NAME.strip(), FLO_DISPLAY_NAME.strip()) -> var_id. We strip whitespace
# because MC_MED-style flowsheet CSVs have trailing spaces in FLO_NAME.
FLO_TO_VAR_ID = {
    # Vital Signs group
    ("Vital Signs", "Pulse"):        100,
    ("Vital Signs", "SpO2"):         101,
    ("Vital Signs", "Resp"):         102,
    ("Vital Signs", "Temp"):         103,
    ("Vital Signs", "MAP (mmHg)"):   106,
    # Devices Testing Template (anesthesia monitor duplicates)
    ("Devices Testing Template", "Heart Rate"):    100,
    ("Devices Testing Template", "SpO2"):          101,
    ("Devices Testing Template", "Resp"):          102,
    ("Devices Testing Template", "ETCO2 (mmHg)"):  116,
    # ED Vitals (triage snapshot)
    ("ED Vitals", "Pulse"):   100,
    ("ED Vitals", "SpO2"):    101,
    ("ED Vitals", "Resp"):    102,
    ("ED Vitals", "Temp"):    103,
}
PHYSIO_RANGE = {
    100: (10,  300),
    101: (20,  100),
    102: (1,    70),
    103: (25,   45),
    106: (20,  250),
    116: (0,   120),
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
    parts = sorted(glob.glob(FLOWSHEET_GLOB))
    print(f"[C1] scanning {len(parts)} flowsheet parts (1.44B rows, 142 GB) ...")
    t0 = time.time()

    # Pre-compute the set of (FLO_NAME_stripped, FLO_DISPLAY_NAME_stripped) keys
    # and the corresponding var_id lookup. We do the mapping in-DF via when/then.
    var_id_expr = pl.lit(None, dtype=pl.UInt16)
    for (fn, fdn), vid in FLO_TO_VAR_ID.items():
        var_id_expr = (
            pl.when((pl.col("flo_name_s") == fn) & (pl.col("flo_display_name_s") == fdn))
              .then(pl.lit(vid, dtype=pl.UInt16))
              .otherwise(var_id_expr)
        )

    dfs = []
    for part in parts:
        print(f"  scanning {Path(part).name}")
        lf = pl.scan_csv(part, low_memory=True, infer_schema_length=10000,
                         ignore_errors=True, null_values=["", "NA"])
        sub = (
            lf.filter(pl.col("LOG_ID").is_in(cohort_log_ids))
              .select([
                  pl.col("LOG_ID").cast(pl.Utf8).alias("log_id"),
                  pl.col("FLO_NAME").cast(pl.Utf8).str.strip_chars().alias("flo_name_s"),
                  pl.col("FLO_DISPLAY_NAME").cast(pl.Utf8).str.strip_chars()
                    .alias("flo_display_name_s"),
                  pt_local_to_utc_ms("RECORDED_TIME").alias("time_ms"),
                  pl.col("MEAS_VALUE").cast(pl.Utf8).str.strip_chars()
                    .cast(pl.Float64, strict=False).alias("value_f"),
              ])
              .with_columns(var_id_expr.alias("var_id"))
              .filter(
                  pl.col("var_id").is_not_null()
                  & pl.col("time_ms").is_not_null()
                  & pl.col("value_f").is_not_null()
                  & pl.col("value_f").is_finite()
              )
              .select(["log_id", "time_ms", "var_id",
                       pl.col("value_f").cast(pl.Float32).alias("value")])
              .collect(engine="streaming")
        )
        print(f"    emitted rows: {sub.height}")
        dfs.append(sub)

    combined = pl.concat(dfs) if dfs else pl.DataFrame({"log_id": [],
                                                        "time_ms": [],
                                                        "var_id":  [],
                                                        "value":   []},
                                                       schema={"log_id": pl.Utf8,
                                                               "time_ms": pl.Int64,
                                                               "var_id":  pl.UInt16,
                                                               "value":   pl.Float32})

    # Apply physio range filter
    rng_expr = None
    for vid, (lo, hi) in PHYSIO_RANGE.items():
        cond = (pl.col("var_id") == vid) & (pl.col("value") >= lo) & (pl.col("value") <= hi)
        rng_expr = cond if rng_expr is None else rng_expr | cond
    combined = combined.filter(rng_expr)
    combined.write_parquet(COMBINED_PARQUET)
    info = {"rows": combined.height,
            "unique_log_ids": int(combined["log_id"].n_unique()) if combined.height else 0,
            "var_ids": sorted(int(v) for v in combined["var_id"].unique()) if combined.height else [],
            "elapsed_sec": round(time.time() - t0, 1)}
    print(f"[C1] done: {info}")
    return info


def phase2(cohort_df: pl.DataFrame, out_root: str, resume: bool) -> list[dict]:
    combined = pl.read_parquet(COMBINED_PARQUET)
    print(f"[C2] rows={combined.height:,}  unique_log_ids={combined['log_id'].n_unique()}")
    by_log = {k[0] if isinstance(k, tuple) else k: g
              for k, g in combined.partition_by("log_id", as_dict=True).items()}
    statuses: list[dict] = []
    for row in cohort_df.iter_rows(named=True):
        log_id = row["entity_id"]
        out_dir = Path(out_root) / log_id
        meta_path = out_dir / "meta.json"
        events_path = out_dir / "vitals_events.npy"
        st = {"entity_id": log_id, "status": "pending", "n_events": 0}
        if not meta_path.exists():
            st["status"] = "no_stage_b"; statuses.append(st); continue
        meta = json.loads(meta_path.read_text())
        if resume and events_path.exists() and meta.get("stage_c_version", 0) >= 1:
            st["status"] = "resumed"
            st["n_events"] = int(meta.get("vitals", {}).get("n_events", 0))
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
        meta["vitals"] = {"n_events": int(events.shape[0]),
                          "per_var_count": per_var,
                          "source": "flowsheets_cleaned/flowsheet_part*.csv",
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
                                  logging.FileHandler(f"{LOG_DIR}/stage_c_flowsheets.log")])
    cohort = pl.read_parquet(COHORT_PARQUET).unique("entity_id", keep="first")
    cohort_log_ids = cohort["log_id"].to_list()
    print(f"[C] cohort entities: {cohort.height}")
    info1 = None
    if args.phase in ("1", "all"):
        if args.redo_phase1 or not os.path.exists(COMBINED_PARQUET):
            info1 = phase1(cohort_log_ids)
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
