#!/usr/bin/env python3
"""
Stage B — MC_MED waveform extraction (raw WFDB -> canonical npy).

For each cohort CSN:
  1. Read all Pleth segments (CSN_{N}.hea + .dat) chronologically (by base_datetime).
  2. Resample Pleth 125 -> 40 Hz (up=8, down=25). Cut to 30 s non-overlap windows.
  3. For each Pleth window start time, attempt to load the matching II sample
     from any II segment that covers that window (resample 500 -> 120 Hz,
     up=6, down=25). NaN-fill windows with no II coverage.
  4. Drop windows with > 20% NaN in PLETH.
  5. Save PLETH40.npy (float16), II120.npy (float16), time_ms.npy (int64),
     meta.json. stage_b_version=1.

WFDB `base_datetime` is already UTC per README ("Random-shift date, keeping
season constant") — no +30y shift (that's Emory-specific).

Run modes:
  python stage_b_wave.py --limit 3 --workers 2     # smoke
  python stage_b_wave.py --entity-id 99370369
  python stage_b_wave.py                            # full run
"""
import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import time
import traceback
from datetime import timezone
from pathlib import Path

import numpy as np
import polars as pl
import wfdb
from scipy.signal import resample_poly

UTC = timezone.utc
RAW_ROOT = "/opt/localdata100tb/UNIPHY_Plus/raw_datasets/physionet.org/files/mc-med/1.0.1/data"
WAVE_DIR = f"{RAW_ROOT}/waveforms"
OUT_ROOT = "/opt/localdata100tb/physio_data/mcmed"
COHORT_PARQUET = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/mcmed/valid_cohort.parquet"
LOG_DIR = "/labs/hulab/mxwang/Physio_Data/workzone/mcmed/logs"
SUMMARY_JSON = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/mcmed/stage_b_summary.json"

SEG_SEC = 30
PLETH_FS = 40
II_FS = 120
PLETH_SRC_FS = 125
II_SRC_FS = 500
SEG_LEN_PLETH = SEG_SEC * PLETH_FS   # 1200
SEG_LEN_II = SEG_SEC * II_FS         # 3600
MAX_NAN_RATIO = 0.20
DEFAULT_WORKERS = 16


def csn_suffix(csn: str) -> str:
    """Last 3 digits of CSN determine the top-level waveform subdir."""
    return str(csn).zfill(4)[-3:]


def list_segments(csn_dir: Path, channel: str) -> list[tuple[int, Path]]:
    """Return (segment_number, hea_path) sorted ascending. Channel dir may be absent."""
    sub = csn_dir / channel
    if not sub.exists():
        return []
    out = []
    for p in sub.glob(f"*_*.hea"):
        # Filename: {CSN}_{N}.hea
        stem = p.stem
        try:
            n = int(stem.rsplit("_", 1)[1])
        except (ValueError, IndexError):
            continue
        out.append((n, p))
    out.sort(key=lambda x: x[0])
    return out


def read_wave_segment(hea_path: Path) -> tuple[np.ndarray, int, float, int] | None:
    """Read one .hea + .dat. Return (signal_1d_float32, base_ms, fs, sig_len). None on failure."""
    stem = str(hea_path.with_suffix(""))
    try:
        hdr = wfdb.rdheader(stem)
    except Exception:
        return None
    if hdr.base_datetime is None:
        return None
    # MC_MED base_datetime is already UTC (no +30y shift).
    base_dt = hdr.base_datetime
    if base_dt.tzinfo is None:
        base_dt = base_dt.replace(tzinfo=UTC)
    base_ms = int(base_dt.timestamp() * 1000)
    try:
        rec = wfdb.rdrecord(stem, physical=True, channels=[0])
    except Exception:
        return None
    sig = rec.p_signal[:, 0].astype(np.float32)
    return sig, base_ms, float(hdr.fs), int(hdr.sig_len)


def resample_pleth_to_windows(sig: np.ndarray, base_ms: int
                              ) -> tuple[np.ndarray, np.ndarray] | None:
    """Pleth 125 Hz -> 40 Hz -> 30 s windows aligned to this segment's base.

    Returns (blocks [n,1200] float32, time_ms [n] int64). None if no full window.
    """
    pleth40 = resample_poly(sig, 8, 25).astype(np.float32)
    n = len(pleth40) // SEG_LEN_PLETH
    if n == 0:
        return None
    blocks = pleth40[: n * SEG_LEN_PLETH].reshape(n, SEG_LEN_PLETH)
    time_ms = base_ms + np.arange(n, dtype=np.int64) * (SEG_SEC * 1000)
    return blocks, time_ms


def build_ii_aligned(pleth_time_ms: np.ndarray,
                     ii_segs: list[tuple[np.ndarray, int, int]]
                     ) -> tuple[np.ndarray, int]:
    """For each Pleth window start t, extract the matching 30 s of II samples
    across (potentially multiple) II segments, resample 500->120 Hz.

    ii_segs: list of (raw_500hz_signal, base_ms, n_samples). Segments may
    overlap in wall time; we pick the first one that fully covers [t, t+30 s].

    Returns (ii120 [n_windows, 3600] float32 NaN-filled where no coverage,
    n_windows_with_ii).
    """
    n = len(pleth_time_ms)
    out = np.full((n, SEG_LEN_II), np.nan, dtype=np.float32)
    if not ii_segs:
        return out, 0
    win_ms = SEG_SEC * 1000
    n_raw_per_win = SEG_SEC * II_SRC_FS       # 30 * 500 = 15000
    n_with = 0
    for i, t in enumerate(pleth_time_ms):
        t_int = int(t)
        t_end = t_int + win_ms
        for sig, base_ms, n_samp in ii_segs:
            end_ms = base_ms + int(n_samp * 1000 / II_SRC_FS)
            if t_int >= base_ms and t_end <= end_ms:
                start_idx = int((t_int - base_ms) * II_SRC_FS // 1000)
                chunk = sig[start_idx: start_idx + n_raw_per_win]
                if len(chunk) == n_raw_per_win:
                    resamp = resample_poly(chunk, 6, 25).astype(np.float32)
                    if len(resamp) >= SEG_LEN_II:
                        out[i] = resamp[:SEG_LEN_II]
                        n_with += 1
                        break
    return out, n_with


def process_entity(row: dict, out_root: str = OUT_ROOT) -> dict:
    entity_id = row["entity_id"]
    csn = str(row.get("csn") or entity_id)
    out_dir = Path(out_root) / entity_id
    meta_path = out_dir / "meta.json"

    # Resume
    required = ["PLETH40.npy", "II120.npy", "time_ms.npy", "meta.json"]
    if all((out_dir / f).exists() for f in required):
        try:
            m = json.loads(meta_path.read_text())
            if m.get("stage_b_version", 0) >= 1:
                return {"entity_id": entity_id, "status": "resumed",
                        "n_seg": int(m.get("n_segments", 0))}
        except Exception:
            pass

    csn_dir = Path(WAVE_DIR) / csn_suffix(csn) / csn
    if not csn_dir.exists():
        return {"entity_id": entity_id, "status": "no_wave_dir",
                "path": str(csn_dir)}

    pleth_segs = list_segments(csn_dir, "Pleth")
    ii_segs = list_segments(csn_dir, "II")
    if not pleth_segs:
        return {"entity_id": entity_id, "status": "no_pleth"}

    # Read Pleth segments -> 30 s windows aligned to each seg's base_datetime
    pleth_blocks_list = []
    pleth_times_list = []
    n_pleth_read = n_pleth_fail = 0
    for _, hea_p in pleth_segs:
        r = read_wave_segment(hea_p)
        if r is None:
            n_pleth_fail += 1
            continue
        sig, base_ms, fs, _ = r
        if fs != PLETH_SRC_FS:
            n_pleth_fail += 1
            continue
        w = resample_pleth_to_windows(sig, base_ms)
        if w is None:
            continue
        pleth_blocks_list.append(w[0])
        pleth_times_list.append(w[1])
        n_pleth_read += 1

    if not pleth_blocks_list:
        return {"entity_id": entity_id, "status": "no_valid_pleth",
                "n_pleth_segs": len(pleth_segs), "n_pleth_fail": n_pleth_fail}

    # Read II segments as raw 500 Hz streams (defer resampling to window-level)
    ii_streams: list[tuple[np.ndarray, int, int]] = []
    n_ii_read = n_ii_fail = 0
    for _, hea_p in ii_segs:
        r = read_wave_segment(hea_p)
        if r is None:
            n_ii_fail += 1
            continue
        sig, base_ms, fs, n_samp = r
        if fs != II_SRC_FS:
            n_ii_fail += 1
            continue
        ii_streams.append((sig, base_ms, n_samp))
        n_ii_read += 1

    pleth40 = np.vstack(pleth_blocks_list)
    pleth_time_ms = np.concatenate(pleth_times_list).astype(np.int64)

    order = np.argsort(pleth_time_ms, kind="stable")
    pleth40 = pleth40[order]
    pleth_time_ms = pleth_time_ms[order]
    if len(pleth_time_ms) > 1:
        keep = np.concatenate([[True], np.diff(pleth_time_ms) > 0])
        pleth40 = pleth40[keep]
        pleth_time_ms = pleth_time_ms[keep]

    ii120, n_with_ii = build_ii_aligned(pleth_time_ms, ii_streams)

    # NaN filter on PLETH
    nan_frac = np.isnan(pleth40).mean(axis=1)
    keep = nan_frac <= MAX_NAN_RATIO
    n_dropped = int((~keep).sum())
    if not keep.any():
        return {"entity_id": entity_id, "status": "all_nan",
                "n_pleth_windows": int(len(pleth40))}
    pleth40 = pleth40[keep]
    ii120 = ii120[keep]
    pleth_time_ms = pleth_time_ms[keep]

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    pleth40_f16 = np.ascontiguousarray(pleth40.astype(np.float16))
    ii120_f16 = np.ascontiguousarray(ii120.astype(np.float16))
    assert pleth40_f16.flags["C_CONTIGUOUS"] and ii120_f16.flags["C_CONTIGUOUS"]
    assert pleth40_f16.shape[0] == ii120_f16.shape[0] == len(pleth_time_ms)
    assert pleth40_f16.shape[1] == SEG_LEN_PLETH and ii120_f16.shape[1] == SEG_LEN_II
    assert len(pleth_time_ms) == 1 or np.all(np.diff(pleth_time_ms) > 0)

    np.save(out_dir / "PLETH40.npy", pleth40_f16)
    np.save(out_dir / "II120.npy", ii120_f16)
    np.save(out_dir / "time_ms.npy", pleth_time_ms.astype(np.int64))

    meta = {
        "entity_id": entity_id,
        "csn": int(csn),
        "mrn": int(row.get("mrn") or 0),
        "source_dataset": "mcmed",
        "n_segments": int(pleth40_f16.shape[0]),
        "segment_duration_sec": SEG_SEC,
        "total_duration_hours": round(int(pleth40_f16.shape[0]) * SEG_SEC / 3600, 2),
        "wave_start_ms": int(pleth_time_ms[0]),
        "wave_end_ms": int(pleth_time_ms[-1] + SEG_SEC * 1000),
        "channels": {
            "PLETH40": {"sample_rate_hz": PLETH_FS, "shape": list(pleth40_f16.shape),
                        "dtype": "float16",
                        "source": f"MC_MED Pleth/*.dat @ {PLETH_SRC_FS} Hz, resample_poly(8,25)"},
            "II120":   {"sample_rate_hz": II_FS, "shape": list(ii120_f16.shape),
                        "dtype": "float16",
                        "source": f"MC_MED II/*.dat @ {II_SRC_FS} Hz, resample_poly(6,25); NaN-filled when absent"},
        },
        "n_pleth_segs_listed": len(pleth_segs),
        "n_pleth_segs_used":   n_pleth_read,
        "n_pleth_segs_failed": n_pleth_fail,
        "n_ii_segs_listed":    len(ii_segs),
        "n_ii_segs_used":      n_ii_read,
        "n_ii_segs_failed":    n_ii_fail,
        "n_windows_with_ii":   int(n_with_ii),
        "n_windows_dropped_nan": n_dropped,
        "max_nan_ratio": MAX_NAN_RATIO,
        "arrival_ms":   int(row.get("arrival_ms") or 0) or None,
        "departure_ms": int(row.get("departure_ms") or 0) or None,
        "stage_b_version": 1,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    return {"entity_id": entity_id, "status": "ok",
            "n_seg": int(pleth40_f16.shape[0]),
            "n_pleth_segs": n_pleth_read, "n_ii_segs": n_ii_read,
            "n_dropped": n_dropped}


def _worker(args):
    row, out_root = args
    try:
        return process_entity(row, out_root=out_root)
    except Exception as e:
        return {"entity_id": row.get("entity_id", "?"), "status": "error",
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()[-400:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--entity-id", type=str, default=None)
    ap.add_argument("--entities", type=str, default=None)
    ap.add_argument("--out-root", type=str, default=OUT_ROOT)
    args = ap.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(f"{LOG_DIR}/stage_b_wave.log")])
    log = logging.getLogger(__name__)
    log.info(f"Loading cohort parquet: {COHORT_PARQUET}")

    df = pl.read_parquet(COHORT_PARQUET)
    if args.entity_id:
        df = df.filter(pl.col("entity_id") == args.entity_id)
    elif args.entities:
        ids = [s.strip() for s in args.entities.split(",") if s.strip()]
        df = df.filter(pl.col("entity_id").is_in(ids))
    elif args.limit:
        df = df.head(args.limit)

    rows = df.to_dicts()
    log.info(f"Entities to process: {len(rows)}  workers: {args.workers}")

    worker_args = [(r, args.out_root) for r in rows]
    t0 = time.time()
    results = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, worker_args, chunksize=1)):
            results.append(r)
            if (i + 1) % 50 == 0 or i + 1 == len(rows):
                statuses = {}
                for x in results:
                    statuses[x["status"]] = statuses.get(x["status"], 0) + 1
                log.info(f"  {i+1}/{len(rows)}  elapsed {time.time()-t0:.0f}s  {statuses}")

    elapsed = time.time() - t0
    by = {}
    for r in results:
        by.setdefault(r["status"], []).append(r)
    summary = {
        "n_entities_processed": len(results),
        "elapsed_sec": round(elapsed, 1),
        "by_status": {s: len(v) for s, v in by.items()},
        "ok_total_segments": sum(r.get("n_seg", 0) for r in by.get("ok", [])),
        "workers": args.workers,
    }
    with open(SUMMARY_JSON, "w") as f:
        json.dump({"summary": summary,
                   "errors": [r for r in results if r["status"] == "error"][:30]},
                  f, indent=2, default=str)
    log.info(f"\n=== Stage B summary ===\n{json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
