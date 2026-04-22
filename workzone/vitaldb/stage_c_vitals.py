#!/usr/bin/env python3
"""
Stage C - VitalDB 1-Hz numerics (Solar8000/*) -> vitals_events.npy per case.

Per case with Stage B meta on disk:
  1. Open .vital; read each target Solar8000/* track at 1 Hz.
  2. Convert to (time_ms, var_id, value) events (drop NaN).
  3. Apply physio range filter.
  4. Save vitals_events.npy; update meta with per-var counts + stage_c_version=1.
"""
import argparse
import json
import logging
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path

import numpy as np
import polars as pl
import vitaldb

OUT_ROOT = "/opt/localdata100tb/physio_data/vitaldb"
COHORT_PARQUET = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/vitaldb/valid_cohort.parquet"
SUMMARY_JSON = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/vitaldb/stage_c_summary.json"
LOG_DIR = "/labs/hulab/mxwang/Physio_Data/workzone/vitaldb/logs"

TRACK_TO_VAR_ID = {
    "Solar8000/HR":         100,
    "Solar8000/PLETH_SPO2": 101,
    "Solar8000/BT":         103,
    "Solar8000/NIBP_SBP":   104,
    "Solar8000/NIBP_DBP":   105,
    "Solar8000/NIBP_MBP":   106,
    "Solar8000/CVP":        107,
    "Solar8000/ART_SBP":    110,
    "Solar8000/ART_DBP":    111,
    "Solar8000/ART_MBP":    112,
    "Solar8000/ETCO2":      116,
}
PHYSIO_RANGE = {
    100: (10,  300),
    101: (20,  100),
    103: (25.0, 45.0),
    104: (30,  300),
    105: (10,  200),
    106: (20,  250),
    107: (-10, 40),
    110: (40,  300),
    111: (20,  200),
    112: (30,  250),
    116: (0,   120),
}
EVENT_DTYPE = np.dtype([
    ("time_ms", np.int64),
    ("var_id",  np.uint16),
    ("value",   np.float32),
])
DEFAULT_WORKERS = 16


def process_entity(row, out_root=OUT_ROOT, resume: bool = True) -> dict:
    eid = row["entity_id"]
    out_dir = Path(out_root) / eid
    meta_path = out_dir / "meta.json"
    events_path = out_dir / "vitals_events.npy"

    st = {"entity_id": eid, "status": "pending", "n_events": 0}
    if not meta_path.exists():
        st["status"] = "no_stage_b"; return st
    meta = json.loads(meta_path.read_text())
    if resume and events_path.exists() and meta.get("stage_c_version", 0) >= 1:
        st["status"] = "resumed"
        st["n_events"] = int(meta.get("vitals", {}).get("n_events", 0))
        return st

    try:
        vf = vitaldb.VitalFile(row["vital_file_path"])
    except Exception as e:
        st["status"] = "vital_parse_err"; st["error"] = str(e); return st

    dtstart = float(vf.dtstart)
    names = set(vf.get_track_names())
    wanted = [t for t in TRACK_TO_VAR_ID if t in names]
    if not wanted:
        # Still write an empty events array so Stage E can proceed
        events = np.empty(0, dtype=EVENT_DTYPE)
    else:
        arr = vf.to_numpy(wanted, interval=1.0)  # 1-Hz sampling
        n_samples, n_tracks = arr.shape
        # Build events by iterating columns
        parts = []
        for j, tname in enumerate(wanted):
            col = arr[:, j]
            mask = np.isfinite(col)
            if not mask.any():
                continue
            idx = np.where(mask)[0]
            vid = TRACK_TO_VAR_ID[tname]
            lo, hi = PHYSIO_RANGE.get(vid, (-1e9, 1e9))
            v = col[idx].astype(np.float32)
            rng_mask = (v >= lo) & (v <= hi)
            if not rng_mask.any():
                continue
            idx = idx[rng_mask]
            v = v[rng_mask]
            t_ms = ((dtstart + idx * 1.0) * 1000).astype(np.int64)
            e = np.empty(len(idx), dtype=EVENT_DTYPE)
            e["time_ms"] = t_ms
            e["var_id"]  = vid
            e["value"]   = v
            parts.append(e)
        if parts:
            events = np.concatenate(parts)
            events.sort(order=["time_ms", "var_id"])
        else:
            events = np.empty(0, dtype=EVENT_DTYPE)
    np.save(events_path, events)

    per_var = {}
    if events.shape[0] > 0:
        vals, counts = np.unique(events["var_id"], return_counts=True)
        per_var = {str(int(v)): int(c) for v, c in zip(vals, counts)}
    meta["vitals"] = {"n_events": int(events.shape[0]),
                      "per_var_count": per_var,
                      "source": "VitalDB .vital Solar8000/* tracks @ 1 Hz",
                      "time_convention": "absolute ms = (dtstart_s + track_idx) * 1000"}
    meta["stage_c_version"] = 1
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    st["status"] = "ok" if events.shape[0] > 0 else "ok_empty"
    st["n_events"] = int(events.shape[0])
    return st


def _worker(args):
    row, out_root, resume = args
    try:
        return process_entity(row, out_root, resume)
    except Exception as e:
        return {"entity_id": row.get("entity_id", "?"), "status": "error",
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()[-400:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--entity-id", default=None)
    ap.add_argument("--entities", default=None)
    ap.add_argument("--out-root", default=OUT_ROOT)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(f"{LOG_DIR}/stage_c_vitals.log")])
    log = logging.getLogger(__name__)

    cohort = pl.read_parquet(COHORT_PARQUET).unique("entity_id", keep="first")
    if args.entity_id:
        cohort = cohort.filter(pl.col("entity_id") == args.entity_id)
    elif args.entities:
        ids = [s.strip() for s in args.entities.split(",") if s.strip()]
        cohort = cohort.filter(pl.col("entity_id").is_in(ids))
    elif args.limit:
        cohort = cohort.head(args.limit)
    rows = cohort.to_dicts()
    log.info(f"Processing {len(rows)} entities  workers={args.workers}")

    t0 = time.time()
    results = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker,
                                                  [(row, args.out_root, not args.no_resume) for row in rows],
                                                  chunksize=1)):
            results.append(r)
            if (i + 1) % 100 == 0 or i + 1 == len(rows):
                st = {}
                for x in results:
                    st[x["status"]] = st.get(x["status"], 0) + 1
                log.info(f"  {i+1}/{len(rows)}  elapsed {time.time()-t0:.0f}s  {st}")

    by_status = {}
    for r in results:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    summary = {
        "n_entities": len(results),
        "elapsed_sec": round(time.time() - t0, 1),
        "by_status": by_status,
        "total_events": int(sum(r.get("n_events", 0) for r in results)),
        "workers": args.workers,
    }
    with open(SUMMARY_JSON, "w") as f:
        json.dump({"summary": summary,
                   "errors": [r for r in results if r["status"] == "error"][:30]},
                  f, indent=2, default=str)
    log.info(f"\n=== Stage C summary ===\n{json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
