"""
Stage E - UCSF 4-partition EHR trajectory assembler.

Per entity:
  1. Read `labs_events.npy` + `vitals_events.npy` (simple 3-field dtype).
  2. Promote to canonical `EHR_EVENT_DTYPE` (adds `seg_idx=0` placeholder).
  3. Merge + dedupe on (time_ms, var_id, value) via `merge_partition`.
  4. Load `time_ms.npy` for wave bounds and admission bounds from meta/parquet.
  5. Call `split_events` to produce baseline/recent/events/future with correct
     seg_idx sentinels (baseline/recent/future) or searchsorted indices
     (events).
  6. Validate with `validate_partition`.
  7. Write `ehr_baseline.npy` / `ehr_recent.npy` / `ehr_events.npy` /
     `ehr_future.npy` and update `meta.json`.

Resume: if all four output files exist and ehr_layout_version == 2 in meta,
skip. `--no-resume` reprocesses everything.

Single-threaded: per-entity work is cheap (hundreds to thousands of events).
Adds --workers for parallel processing if needed.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np
import polars as pl
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from physio_data.ehr_trajectory import (  # noqa: E402
    EHR_EVENT_DTYPE,
    ALL_FNAMES,
    FNAME_BASELINE, FNAME_RECENT, FNAME_EVENTS, FNAME_FUTURE,
    CONTEXT_WINDOW_MS, BASELINE_CAP_MS, FUTURE_CAP_MS,
    SEG_IDX_BASELINE, SEG_IDX_RECENT, SEG_IDX_FUTURE,
    split_events, validate_partition,
)


def concat_sort_events(labs: np.ndarray, vitals: np.ndarray) -> np.ndarray:
    """Fast concat+sort without dedup. Safe when inputs have disjoint var_id
    ranges (labs 0-99 vs vitals 100-199), as in UCSF Stage D/C outputs."""
    if len(labs) == 0:
        return vitals.copy() if len(vitals) else np.empty(0, dtype=EHR_EVENT_DTYPE)
    if len(vitals) == 0:
        return labs.copy()
    combined = np.concatenate([labs, vitals])
    order = np.argsort(combined["time_ms"], kind="stable")
    return combined[order]

CONFIG_PATH = REPO_ROOT / "workzone" / "configs" / "server_paths.yaml"
MAX_WORKERS = 24
LAYOUT_VERSION = 2

# 30s seg duration pad so a lab recorded 1-29s after the last seg start still
# counts as in-waveform (matches MIMIC's stage3c convention).
WAVE_END_PAD_MS = 30 * 1000


def load_simple_events(path: Path) -> np.ndarray:
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
    out["var_id"] = arr["var_id"].astype(np.uint16)
    out["value"] = arr["value"].astype(np.float32)
    return out


def process_entity(entity_row: dict, output_dir_str: str,
                   resume: bool) -> dict:
    entity_id = entity_row["entity_id"]
    out_dir = Path(output_dir_str) / entity_id
    meta_path = out_dir / "meta.json"
    time_path = out_dir / "time_ms.npy"
    status = {"entity_id": entity_id, "status": "pending",
              "n_baseline": 0, "n_recent": 0, "n_events": 0, "n_future": 0,
              "n_source_labs": 0, "n_source_vitals": 0}
    try:
        if not meta_path.exists() or not time_path.exists():
            status["status"] = "no_meta_or_time"
            return status

        meta = json.loads(meta_path.read_text())

        outs = [out_dir / fn for fn in ALL_FNAMES]
        if resume and all(p.exists() for p in outs) and \
                meta.get("ehr_layout_version") == LAYOUT_VERSION:
            status["status"] = "already_done"
            return status

        time_ms = np.load(time_path)
        if len(time_ms) == 0:
            status["status"] = "empty_time"
            return status

        labs = load_simple_events(out_dir / "labs_events.npy")
        vitals = load_simple_events(out_dir / "vitals_events.npy")
        status["n_source_labs"] = int(len(labs))
        status["n_source_vitals"] = int(len(vitals))

        combined = concat_sort_events(labs, vitals)

        adm_start = int(entity_row["admission_start_ms"])
        adm_end = int(entity_row["admission_end_ms"])

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

        # Detect future-leakage categories
        fut = partitions["future"]
        if len(fut):
            fut_vids = np.unique(fut["var_id"])
            has_fut_actions = bool(np.any((fut_vids >= 200) & (fut_vids <= 299)))
            has_fut_sofa    = bool(np.any((fut_vids >= 300) & (fut_vids <= 306)))
            has_fut_onset   = bool(np.any(fut_vids == 307))
        else:
            has_fut_actions = has_fut_sofa = has_fut_onset = False

        meta.update({
            "n_events": ne,
            "n_ehr_events": ne,
            "n_baseline": nb,
            "n_recent": nr,
            "n_future": nf,
            "n_events_vars": _n_vars(partitions["events"]),
            "n_baseline_vars": _n_vars(partitions["baseline"]),
            "n_recent_vars": _n_vars(partitions["recent"]),
            "n_future_vars": _n_vars(partitions["future"]),
            "context_window_ms": CONTEXT_WINDOW_MS,
            "baseline_cap_ms":   BASELINE_CAP_MS,
            "future_cap_ms":     FUTURE_CAP_MS,
            "wave_end_pad_ms":   WAVE_END_PAD_MS,
            "admission_start_ms": adm_start,
            "admission_end_ms":   adm_end,
            "has_future_actions": has_fut_actions,
            "has_future_sofa":    has_fut_sofa,
            "has_future_sepsis_onset": has_fut_onset,
            "ehr_layout_version": LAYOUT_VERSION,
        })
        meta_path.write_text(json.dumps(meta, indent=2, default=str))

        status["status"] = "ok"
        status["n_baseline"] = nb
        status["n_recent"] = nr
        status["n_events"] = ne
        status["n_future"] = nf
        return status
    except Exception as e:
        status["status"] = "error"
        status["error"] = f"{type(e).__name__}: {e}"
        status["traceback"] = traceback.format_exc()[-400:]
        return status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--workers", type=int, default=8,
                    help=f"parallel workers (max {MAX_WORKERS})")
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = all; debug: process first N entities")
    ap.add_argument("--entities", default="",
                    help="comma-separated entity_ids (debug; overrides limit)")
    ap.add_argument("--no-resume", action="store_true",
                    help="reprocess entities even if layout already at v2")
    args = ap.parse_args()

    if args.workers > MAX_WORKERS:
        print(f"clamping workers {args.workers} -> {MAX_WORKERS}")
        args.workers = MAX_WORKERS

    cfg = yaml.safe_load(Path(args.config).read_text())["ucsf"]
    output_dir = Path(cfg["output_dir"])
    intermediate_dir = Path(cfg["intermediate_dir"])
    parquet = intermediate_dir / "valid_wave_window.parquet"

    print(f"output_dir = {output_dir}")
    print(f"parquet    = {parquet}")
    print(f"workers    = {args.workers}")

    entities_df = (
        pl.read_parquet(parquet)
        .unique("entity_id", keep="first")
        .select(["entity_id",
                 "admission_start_ms", "admission_end_ms"])
    )
    print(f"entities (unique): {entities_df.height}", flush=True)

    # Universe: entities with meta.json AND time_ms.npy
    universe = sorted(
        p.name for p in output_dir.iterdir()
        if p.is_dir() and (p / "meta.json").exists() and (p / "time_ms.npy").exists()
    )
    print(f"Stage B outputs present: {len(universe)} entities", flush=True)

    entities_df = entities_df.filter(pl.col("entity_id").is_in(universe))
    if args.entities:
        ids = [s.strip() for s in args.entities.split(",") if s.strip()]
        entities_df = entities_df.filter(pl.col("entity_id").is_in(ids))
        print(f"--entities filter -> {entities_df.height}")
    elif args.limit:
        entities_df = entities_df.head(args.limit)
        print(f"--limit {args.limit} -> {entities_df.height}")

    rows = entities_df.to_dicts()
    total = len(rows)
    print(f"processing: {total}", flush=True)

    statuses: list[dict] = []
    t0 = time.time()

    if args.workers <= 1:
        for i, r in enumerate(rows, 1):
            statuses.append(process_entity(r, str(output_dir), not args.no_resume))
            if i % 500 == 0 or i == total:
                print(f"  [{i}/{total}] elapsed={time.time()-t0:.1f}s "
                      f"last={statuses[-1]['status']}", flush=True)
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
            futs = {}
            for r in rows:
                fut = ex.submit(process_entity, r, str(output_dir),
                                not args.no_resume)
                futs[fut] = r["entity_id"]
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
                if done % 500 == 0 or done == total:
                    print(f"  [{done}/{total}] elapsed={time.time()-t0:.1f}s "
                          f"last={s['status']}", flush=True)

    elapsed = time.time() - t0
    by_status: dict[str, int] = {}
    for s in statuses:
        by_status[s["status"]] = by_status.get(s["status"], 0) + 1
    totals = {k: sum(s.get(f"n_{k}", 0) for s in statuses)
              for k in ("baseline", "recent", "events", "future",
                        "source_labs", "source_vitals")}
    summary = {
        "stage": "e_assemble",
        "ran_at_unix": int(time.time()),
        "elapsed_sec": round(elapsed, 1),
        "n_entities": total,
        "workers": args.workers,
        "by_status": by_status,
        "totals": totals,
        "output_dir": str(output_dir),
    }
    out_summary = intermediate_dir / "stage_e_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2))
    status_parquet = intermediate_dir / "stage_e_status.parquet"
    pl.DataFrame(statuses).write_parquet(status_parquet)

    print(f"wrote {out_summary}")
    print(f"wrote {status_parquet}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
