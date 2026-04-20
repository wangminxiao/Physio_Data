#!/usr/bin/env python3
"""
Stage E — MC_MED 4-partition EHR assembler.

Per entity with Stage B (+C +D) on disk:
  1. Load time_ms.npy and meta.json.
  2. Load vitals_events.npy (Stage C) + labs_events.npy (Stage D) — 3-field dtype.
  3. Promote to EHR_EVENT_DTYPE, concat + sort.
  4. split_events(episode_start=arrival_ms, episode_end=departure_ms).
  5. Write ehr_{baseline,recent,events,future}.npy + meta update.

Episode bounds: MC_MED visits.Arrival_time / Departure_time (UTC, from
cohort parquet). For ED, departure_ms is typically within a few hours of
the last waveform segment.

Resume: skip entity if all four ehr_*.npy files exist and
ehr_layout_version == 2 and stage_e_version >= 1.
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from physio_data.ehr_trajectory import (  # noqa: E402
    EHR_EVENT_DTYPE,
    ALL_FNAMES,
    FNAME_BASELINE, FNAME_RECENT, FNAME_EVENTS, FNAME_FUTURE,
    CONTEXT_WINDOW_MS, BASELINE_CAP_MS, FUTURE_CAP_MS,
    split_events, validate_partition,
)

OUT_ROOT = "/opt/localdata100tb/physio_data/mcmed"
COHORT_PARQUET = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/mcmed/valid_cohort.parquet"
SUMMARY_JSON = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/mcmed/stage_e_summary.json"
LOG_DIR = "/labs/hulab/mxwang/Physio_Data/workzone/mcmed/logs"

MAX_WORKERS = 24
LAYOUT_VERSION = 2
WAVE_END_PAD_MS = 30 * 1000


def _load_simple(path: Path) -> np.ndarray:
    if not path.exists():
        return np.empty(0, dtype=EHR_EVENT_DTYPE)
    arr = np.load(path)
    if arr.dtype == EHR_EVENT_DTYPE:
        return arr
    if len(arr) == 0:
        return np.empty(0, dtype=EHR_EVENT_DTYPE)
    out = np.empty(len(arr), dtype=EHR_EVENT_DTYPE)
    out["time_ms"] = arr["time_ms"].astype(np.int64)
    out["seg_idx"] = 0
    out["var_id"]  = arr["var_id"].astype(np.uint16)
    out["value"]   = arr["value"].astype(np.float32)
    return out


def _concat_sort(parts: list[np.ndarray]) -> np.ndarray:
    nonempty = [p for p in parts if len(p)]
    if not nonempty:
        return np.empty(0, dtype=EHR_EVENT_DTYPE)
    out = nonempty[0].copy() if len(nonempty) == 1 else np.concatenate(nonempty)
    order = np.argsort(out["time_ms"], kind="stable")
    return out[order]


def process_entity(row: dict, out_root: str, resume: bool) -> dict:
    eid = row["entity_id"]
    out_dir = Path(out_root) / eid
    meta_path = out_dir / "meta.json"
    time_path = out_dir / "time_ms.npy"
    st = {"entity_id": eid, "status": "pending",
          "n_baseline": 0, "n_recent": 0, "n_events": 0, "n_future": 0,
          "n_source_vitals": 0, "n_source_labs": 0}
    try:
        if not meta_path.exists() or not time_path.exists():
            st["status"] = "no_stage_b"
            return st
        meta = json.loads(meta_path.read_text())
        outs = [out_dir / fn for fn in ALL_FNAMES]
        if (resume and all(p.exists() for p in outs)
                and meta.get("ehr_layout_version") == LAYOUT_VERSION
                and meta.get("stage_e_version", 0) >= 1):
            st["status"] = "resumed"
            st["n_events"] = int(meta.get("n_events", 0))
            st["n_baseline"] = int(meta.get("n_baseline", 0))
            st["n_recent"] = int(meta.get("n_recent", 0))
            st["n_future"] = int(meta.get("n_future", 0))
            return st

        time_ms = np.load(time_path)
        if len(time_ms) == 0:
            st["status"] = "empty_time"
            return st

        vitals = _load_simple(out_dir / "vitals_events.npy")
        labs   = _load_simple(out_dir / "labs_events.npy")
        st["n_source_vitals"] = int(len(vitals))
        st["n_source_labs"]   = int(len(labs))

        combined = _concat_sort([vitals, labs])

        arrival = row.get("arrival_ms")
        departure = row.get("departure_ms")
        adm_start = int(arrival) if arrival is not None else None
        adm_end = int(departure) if departure is not None else None

        partitions = split_events(
            combined, time_ms,
            episode_start_ms=adm_start,
            episode_end_ms=adm_end,
            wave_end_pad_ms=WAVE_END_PAD_MS,
        )

        n_seg = int(len(time_ms))
        errs: list[str] = []
        for kind in ("baseline", "recent", "events", "future"):
            errs.extend(validate_partition(partitions[kind], kind=kind, n_seg=n_seg))
        if errs:
            st["status"] = "error"
            st["error"] = "; ".join(errs)
            return st

        np.save(out_dir / FNAME_BASELINE, partitions["baseline"])
        np.save(out_dir / FNAME_RECENT,   partitions["recent"])
        np.save(out_dir / FNAME_EVENTS,   partitions["events"])
        np.save(out_dir / FNAME_FUTURE,   partitions["future"])

        def _n_vars(a):
            return int(len(np.unique(a["var_id"]))) if len(a) else 0
        meta.update({
            "n_events":        int(len(partitions["events"])),
            "n_baseline":      int(len(partitions["baseline"])),
            "n_recent":        int(len(partitions["recent"])),
            "n_future":        int(len(partitions["future"])),
            "n_events_vars":   _n_vars(partitions["events"]),
            "n_baseline_vars": _n_vars(partitions["baseline"]),
            "n_recent_vars":   _n_vars(partitions["recent"]),
            "n_future_vars":   _n_vars(partitions["future"]),
            "context_window_ms": CONTEXT_WINDOW_MS,
            "baseline_cap_ms":   BASELINE_CAP_MS,
            "future_cap_ms":     FUTURE_CAP_MS,
            "wave_end_pad_ms":   WAVE_END_PAD_MS,
            "admission_start_ms": adm_start,
            "admission_end_ms":   adm_end,
            "ehr_layout_version": LAYOUT_VERSION,
            "stage_e_version":    1,
        })
        meta_path.write_text(json.dumps(meta, indent=2, default=str))

        st["status"] = "ok"
        st["n_baseline"] = int(len(partitions["baseline"]))
        st["n_recent"]   = int(len(partitions["recent"]))
        st["n_events"]   = int(len(partitions["events"]))
        st["n_future"]   = int(len(partitions["future"]))
        return st
    except Exception as e:
        st["status"] = "error"
        st["error"] = f"{type(e).__name__}: {e}"
        st["traceback"] = traceback.format_exc()[-400:]
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
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--entity-id", type=str, default=None)
    ap.add_argument("--entities", type=str, default=None)
    ap.add_argument("--out-root", type=str, default=OUT_ROOT)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    if args.workers > MAX_WORKERS:
        args.workers = MAX_WORKERS

    os.makedirs(LOG_DIR, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(f"{LOG_DIR}/stage_e_assemble.log")])
    log = logging.getLogger(__name__)

    cohort = (pl.read_parquet(COHORT_PARQUET)
              .unique("entity_id", keep="first")
              .select(["entity_id", "arrival_ms", "departure_ms"]))
    log.info(f"cohort entities: {cohort.height}")

    have = [p.name for p in Path(args.out_root).iterdir()
            if p.is_dir() and (p / "meta.json").exists()] if Path(args.out_root).exists() else []
    log.info(f"entities with Stage B meta+time on disk: {len(have)}")
    cohort = cohort.filter(pl.col("entity_id").is_in(have))

    if args.entity_id:
        cohort = cohort.filter(pl.col("entity_id") == args.entity_id)
    elif args.entities:
        ids = [s.strip() for s in args.entities.split(",") if s.strip()]
        cohort = cohort.filter(pl.col("entity_id").is_in(ids))
    elif args.limit:
        cohort = cohort.head(args.limit)

    rows = cohort.to_dicts()
    log.info(f"Processing {len(rows)} entities  workers={args.workers}  resume={not args.no_resume}")
    t0 = time.time()
    statuses: list[dict] = []

    if args.workers <= 1:
        for i, r in enumerate(rows, 1):
            statuses.append(process_entity(r, args.out_root, not args.no_resume))
            if i % 500 == 0 or i == len(rows):
                by = {}
                for s in statuses:
                    by[s["status"]] = by.get(s["status"], 0) + 1
                log.info(f"  [{i}/{len(rows)}] {time.time()-t0:.1f}s  {by}")
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
            futs = {ex.submit(_worker, (r, args.out_root, not args.no_resume)): r["entity_id"]
                    for r in rows}
            done = 0
            for fut in as_completed(futs):
                eid = futs[fut]
                try:
                    s = fut.result()
                except BrokenProcessPool:
                    s = {"entity_id": eid, "status": "worker_killed"}
                except Exception as e:
                    s = {"entity_id": eid, "status": "error",
                         "error": f"{type(e).__name__}: {e}"}
                statuses.append(s); done += 1
                if done % 500 == 0 or done == len(rows):
                    by = {}
                    for s2 in statuses:
                        by[s2["status"]] = by.get(s2["status"], 0) + 1
                    log.info(f"  [{done}/{len(rows)}] {time.time()-t0:.1f}s  {by}")

    elapsed = time.time() - t0
    by_status: dict[str, int] = {}
    for s in statuses:
        by_status[s["status"]] = by_status.get(s["status"], 0) + 1
    totals = {k: int(sum(s.get(f"n_{k}", 0) for s in statuses))
              for k in ("baseline", "recent", "events", "future",
                        "source_vitals", "source_labs")}
    summary = {
        "n_entities": len(statuses),
        "elapsed_sec": round(elapsed, 1),
        "by_status": by_status,
        "totals": totals,
        "workers": args.workers,
    }
    with open(SUMMARY_JSON, "w") as f:
        json.dump({"summary": summary,
                   "errors": [s for s in statuses if s["status"] == "error"][:30]},
                  f, indent=2, default=str)
    log.info(f"\n=== Stage E summary ===\n{json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
