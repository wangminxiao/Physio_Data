#!/usr/bin/env python3
"""
Stage E — Emory 4-partition EHR trajectory assembler.

Per entity:
  1. Load `time_ms.npy` (wave bounds from Stage B).
  2. Load `vitals_events.npy` (Stage C) + `labs_events.npy` +
     `chart_vitals_events.npy` (Stage D), all in the simple 3-field dtype.
  3. Promote each to canonical EHR_EVENT_DTYPE (adds seg_idx=0 placeholder).
  4. Concat + stable-sort by time_ms (disjoint var_id bands: vitals 100-199,
     labs 0-99, chart 100-199 — no dedup needed across sources, but run the
     shared `merge_partition` logic for safety on within-source duplicates).
  5. Call `split_events` with admit_ms / discharge_ms from cohort parquet as
     episode bounds; defaults for context/baseline/future caps.
  6. Validate partitions; write `ehr_baseline.npy`, `ehr_recent.npy`,
     `ehr_events.npy`, `ehr_future.npy`; update meta with `stage_e_version=1`
     + layout metadata.

Resume: skip entity if all four `ehr_*.npy` files exist and
`meta.ehr_layout_version == 2`.

Single-threaded path is fastest for Emory (11k entities × ~ few ms each);
--workers>1 uses spawn if you want parallel.
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
import logging
import traceback
import multiprocessing as mp
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

OUT_ROOT = "/opt/localdata100tb/physio_data/emory"
COHORT_PARQUET = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/emory/valid_wave_window.parquet"
OUTPUTS_DIR = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/emory"
SUMMARY_JSON = f"{OUTPUTS_DIR}/stage_e_summary.json"
LOG_DIR = "/labs/hulab/mxwang/Physio_Data/workzone/emory/logs"

MAX_WORKERS = 24
LAYOUT_VERSION = 2
WAVE_END_PAD_MS = 30 * 1000  # 30 s: last-seg alignment for events 1-29 s after last seg start


def _load_simple(path: Path) -> np.ndarray:
    """Load a 3-field (time_ms, var_id, value) structured array, promote to
    EHR_EVENT_DTYPE with seg_idx=0 placeholder (split_events rewrites it)."""
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
    if len(nonempty) == 1:
        out = nonempty[0].copy()
    else:
        out = np.concatenate(nonempty)
    order = np.argsort(out["time_ms"], kind="stable")
    return out[order]


def process_entity(row: dict, out_root: str, resume: bool) -> dict:
    entity_id = row["entity_id"]
    out_dir = Path(out_root) / entity_id
    meta_path = out_dir / "meta.json"
    time_path = out_dir / "time_ms.npy"
    status = {
        "entity_id": entity_id,
        "status": "pending",
        "n_baseline": 0, "n_recent": 0, "n_events": 0, "n_future": 0,
        "n_source_vitals": 0, "n_source_labs": 0, "n_source_chart": 0,
    }
    try:
        if not meta_path.exists() or not time_path.exists():
            status["status"] = "no_stage_b"
            return status

        meta = json.loads(meta_path.read_text())

        outs = [out_dir / fn for fn in ALL_FNAMES]
        if resume and all(p.exists() for p in outs) and \
                meta.get("ehr_layout_version") == LAYOUT_VERSION and \
                meta.get("stage_e_version", 0) >= 1:
            status["status"] = "resumed"
            status["n_events"] = int(meta.get("n_events", 0))
            status["n_baseline"] = int(meta.get("n_baseline", 0))
            status["n_recent"] = int(meta.get("n_recent", 0))
            status["n_future"] = int(meta.get("n_future", 0))
            return status

        time_ms = np.load(time_path)
        if len(time_ms) == 0:
            status["status"] = "empty_time"
            return status

        vitals = _load_simple(out_dir / "vitals_events.npy")
        labs   = _load_simple(out_dir / "labs_events.npy")
        chart  = _load_simple(out_dir / "chart_vitals_events.npy")
        status["n_source_vitals"] = int(len(vitals))
        status["n_source_labs"]   = int(len(labs))
        status["n_source_chart"]  = int(len(chart))

        combined = _concat_sort([vitals, labs, chart])

        admit = row.get("admit_ms")
        disch = row.get("discharge_ms")
        adm_start = int(admit) if admit is not None else None
        adm_end = int(disch) if disch is not None else None

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
            status["status"] = "error"
            status["error"] = "; ".join(errs)
            return status

        np.save(out_dir / FNAME_BASELINE, partitions["baseline"])
        np.save(out_dir / FNAME_RECENT,   partitions["recent"])
        np.save(out_dir / FNAME_EVENTS,   partitions["events"])
        np.save(out_dir / FNAME_FUTURE,   partitions["future"])

        nb = int(len(partitions["baseline"]))
        nr = int(len(partitions["recent"]))
        ne = int(len(partitions["events"]))
        nf = int(len(partitions["future"]))

        def _n_vars(a: np.ndarray) -> int:
            return int(len(np.unique(a["var_id"]))) if len(a) else 0

        meta.update({
            "n_events": ne,
            "n_ehr_events": ne,
            "n_baseline": nb,
            "n_recent": nr,
            "n_future": nf,
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

        status["status"] = "ok"
        status["n_baseline"] = nb
        status["n_recent"]   = nr
        status["n_events"]   = ne
        status["n_future"]   = nf
        return status
    except Exception as e:
        status["status"] = "error"
        status["error"] = f"{type(e).__name__}: {e}"
        status["traceback"] = traceback.format_exc()[-400:]
        return status


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
    ap.add_argument("--entities", type=str, default=None,
                    help="comma-separated entity_ids")
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

    log.info(f"Loading cohort parquet: {COHORT_PARQUET}")
    cohort = (
        pl.read_parquet(COHORT_PARQUET)
          .unique("entity_id", keep="first")
          .select(["entity_id", "admit_ms", "discharge_ms",
                   "wave_start_ms", "wave_end_ms"])
    )
    log.info(f"  entities (unique): {cohort.height}")

    # Filter to entities that have Stage B (+ C + D) output on disk
    have_stage_b = []
    for p in Path(args.out_root).iterdir():
        if not p.is_dir():
            continue
        if (p / "meta.json").exists() and (p / "time_ms.npy").exists():
            have_stage_b.append(p.name)
    log.info(f"  entities with Stage B meta+time on disk: {len(have_stage_b)}")
    cohort = cohort.filter(pl.col("entity_id").is_in(have_stage_b))

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
                by: dict[str, int] = {}
                for s in statuses:
                    by[s["status"]] = by.get(s["status"], 0) + 1
                log.info(f"  [{i}/{len(rows)}] elapsed={time.time()-t0:.1f}s  {by}")
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
                statuses.append(s)
                done += 1
                if done % 500 == 0 or done == len(rows):
                    by: dict[str, int] = {}
                    for s2 in statuses:
                        by[s2["status"]] = by.get(s2["status"], 0) + 1
                    log.info(f"  [{done}/{len(rows)}] elapsed={time.time()-t0:.1f}s  {by}")

    elapsed = time.time() - t0
    by_status: dict[str, int] = {}
    for s in statuses:
        by_status[s["status"]] = by_status.get(s["status"], 0) + 1
    totals = {k: int(sum(s.get(f"n_{k}", 0) for s in statuses))
              for k in ("baseline", "recent", "events", "future",
                        "source_vitals", "source_labs", "source_chart")}
    summary = {
        "n_entities_processed": len(statuses),
        "elapsed_sec": round(elapsed, 1),
        "by_status": by_status,
        "totals": totals,
        "workers": args.workers,
    }
    with open(SUMMARY_JSON, "w") as f:
        json.dump({"summary": summary,
                   "errors": [s for s in statuses if s["status"] == "error"][:50]},
                  f, indent=2, default=str)
    log.info(f"\n=== Stage E summary ===\n{json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
