#!/usr/bin/env python3
"""
Stage B — Emory waveform extraction (raw WFDB → canonical).

For each entity row in valid_wave_window.parquet:
  1. Iterate all wfdb_records_all (chronological)
  2. For each record, read top-level .hea, enumerate real segments
     (skip _layout virtual entry and '~' gap markers)
  3. For each segment, read only SPO2 + II channels via wfdb.rdrecord(physical=True, channels=[...])
  4. Resample SPO2 240 -> 40 Hz  = PLETH40 (resample_poly(1, 6))
     Resample II   240 -> 120 Hz = II120   (resample_poly(1, 2); NaN-fill if absent)
  5. Cut into 30s non-overlapping windows (1200 samples @ 40 Hz, 3600 @ 120 Hz)
  6. Drop windows with > 20% NaN in PLETH (PLETH-anchored)
  7. Accumulate across segments / records; sort by time_ms; dedup by time_ms
  8. Save PLETH40.npy, II120.npy (float16 C-contiguous), time_ms.npy (int64)
     + meta.json with channel info + n_segments breakdown
  9. Resume: skip if all four output files already exist

Cohort parquet: /labs/hulab/mxwang/Physio_Data/workzone/outputs/emory/valid_wave_window.parquet
Output root:    /opt/localdata100tb/physio_data/emory/{empi}_{enc}/

Run modes:
  python stage_b_wave.py --limit 5 --workers 4          # smoke test
  python stage_b_wave.py --entity-id 1827183_359559206  # single entity
  python stage_b_wave.py                                 # full run (~5 h @ 24 workers)
"""
import os
import sys
import json
import time
import argparse
import logging
import traceback
import multiprocessing as mp
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta

import numpy as np
import polars as pl
import wfdb
from scipy.signal import resample_poly

UTC = timezone.utc
WFDB_ROOT = "/labs/collab/Waveform_Data/Waveform_Data"
OUT_ROOT = "/opt/localdata100tb/physio_data/emory"
COHORT_PARQUET = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/emory/valid_wave_window.parquet"
LOG_DIR = "/labs/hulab/mxwang/Physio_Data/workzone/emory/logs"
SUMMARY_JSON = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/emory/stage_b_summary.json"

SEG_SEC = 30
PLETH_FS = 40
II_FS = 120
WAVE_SRC_FS = 240
SEG_LEN_PLETH = SEG_SEC * PLETH_FS   # 1200
SEG_LEN_II = SEG_SEC * II_FS         # 3600
MAX_NAN_RATIO = 0.20
DEFAULT_WORKERS = 24   # half of 48 cores (shared cluster cap)

# Canonical channel names (case-sensitive, per WFDB .hea sig_name conventions)
PLETH_NAMES = {"SPO2", "Pleth", "PLETH", "pleth", "PlethT", "SpO2_l"}
II_NAMES = {"II"}


def find_ch(sig_name, candidates):
    for i, n in enumerate(sig_name):
        if n in candidates:
            return i
    return None


def load_segment(rec_dir, seg_name):
    """Read one _0XXX segment; return (pleth_raw_240, ii_raw_240_or_None, base_ms, fs, sig_len) or None."""
    seg_path = f"{rec_dir}/{seg_name}"
    try:
        hdr = wfdb.rdheader(seg_path)
    except Exception:
        return None
    i_pleth = find_ch(hdr.sig_name, PLETH_NAMES)
    if i_pleth is None:
        return None
    i_ii = find_ch(hdr.sig_name, II_NAMES)

    channels = [i_pleth] if i_ii is None else [i_pleth, i_ii]
    try:
        rec = wfdb.rdrecord(seg_path, physical=True, channels=channels)
    except Exception:
        return None

    spo2 = rec.p_signal[:, 0].astype(np.float32)
    ii = rec.p_signal[:, 1].astype(np.float32) if i_ii is not None else None

    if hdr.base_datetime is None:
        return None
    base_dt = hdr.base_datetime + relativedelta(years=30)
    base_ms = int(base_dt.replace(tzinfo=UTC).timestamp() * 1000)
    return spo2, ii, base_ms, hdr.fs, hdr.sig_len


def segment_to_blocks(spo2, ii_or_none, base_ms, src_fs):
    """Resample + cut to 30s windows; return (pleth40_blocks, ii120_blocks, time_ms) after NaN filter."""
    # Resample src_fs -> target; only use resample_poly for integer-ratio cases
    if src_fs == WAVE_SRC_FS:
        up_p, down_p = 1, 6
        up_i, down_i = 1, 2
    else:
        # Generic ratio
        from math import gcd
        g_p = gcd(int(src_fs), PLETH_FS)
        up_p, down_p = PLETH_FS // g_p, int(src_fs) // g_p
        g_i = gcd(int(src_fs), II_FS)
        up_i, down_i = II_FS // g_i, int(src_fs) // g_i

    pleth40 = resample_poly(spo2, up_p, down_p).astype(np.float32)
    if ii_or_none is None:
        ii120 = np.full(len(pleth40) * 3, np.nan, dtype=np.float32)
    else:
        ii120 = resample_poly(ii_or_none, up_i, down_i).astype(np.float32)

    # Align II length to 3x PLETH (should already be)
    if len(ii120) != 3 * len(pleth40):
        min3 = 3 * (min(len(ii120) // 3, len(pleth40)))
        pleth40 = pleth40[: min3 // 3]
        ii120 = ii120[:min3]

    n_blocks = len(pleth40) // SEG_LEN_PLETH
    if n_blocks == 0:
        return None
    pleth40 = pleth40[: n_blocks * SEG_LEN_PLETH].reshape(n_blocks, SEG_LEN_PLETH)
    ii120 = ii120[: n_blocks * SEG_LEN_II].reshape(n_blocks, SEG_LEN_II)

    # PLETH-anchored: drop windows with too much NaN in PLETH
    nan_frac = np.isnan(pleth40).mean(axis=1)
    keep = nan_frac <= MAX_NAN_RATIO
    if not keep.any():
        return None
    pleth40 = pleth40[keep]
    ii120 = ii120[keep]
    block_idx = np.where(keep)[0].astype(np.int64)
    time_ms = base_ms + block_idx * SEG_SEC * 1000
    return pleth40, ii120, time_ms


def process_entity(row, out_root=OUT_ROOT):
    entity_id = row["entity_id"]
    rec_ids = list(row["wfdb_records_all"]) if row["wfdb_records_all"] is not None else []
    out_dir = f"{out_root}/{entity_id}"

    # Resume check
    required = ["PLETH40.npy", "II120.npy", "time_ms.npy", "meta.json"]
    if all(os.path.exists(f"{out_dir}/{f}") for f in required):
        try:
            with open(f"{out_dir}/meta.json") as f:
                m = json.load(f)
            if m.get("stage_b_version", 0) >= 1:
                return {"entity_id": entity_id, "status": "resumed", "n_seg": m.get("n_segments", 0)}
        except Exception:
            pass

    os.makedirs(out_dir, exist_ok=True)

    pleth_parts, ii_parts, time_parts = [], [], []
    n_scanned = n_used = n_no_pleth = n_no_ii = n_blocks_dropped_nan = 0
    records_meta = []

    for rec_id in rec_ids:
        cohort_prefix = rec_id.split("-")[0]
        rec_dir = f"{WFDB_ROOT}/{cohort_prefix}/{rec_id}"
        try:
            top_hdr = wfdb.rdheader(f"{rec_dir}/{rec_id}")
        except Exception as e:
            records_meta.append({"record": rec_id, "error": f"top_hdr: {type(e).__name__}: {e}"})
            continue
        seg_names = [s for s in (top_hdr.seg_name or []) if s and s != "~" and not s.endswith("_layout")]
        n_scanned += len(seg_names)

        for seg in seg_names:
            result = load_segment(rec_dir, seg)
            if result is None:
                n_no_pleth += 1
                continue
            spo2, ii, base_ms, src_fs, src_len = result
            if ii is None:
                n_no_ii += 1
            blocks = segment_to_blocks(spo2, ii, base_ms, src_fs)
            if blocks is None:
                continue
            p_b, i_b, t_b = blocks
            n_used += 1
            pleth_parts.append(p_b)
            ii_parts.append(i_b)
            time_parts.append(t_b)

        records_meta.append({"record": rec_id, "n_seg_listed": len(seg_names)})

    if not pleth_parts:
        return {"entity_id": entity_id, "status": "skip_no_valid_pleth",
                "n_records": len(rec_ids), "n_scanned": n_scanned,
                "n_no_pleth": n_no_pleth}

    # Concat
    pleth40 = np.vstack(pleth_parts).astype(np.float16)
    ii120 = np.vstack(ii_parts).astype(np.float16)
    time_ms = np.concatenate(time_parts).astype(np.int64)

    # Sort by time
    order = np.argsort(time_ms, kind="stable")
    time_ms = time_ms[order]
    pleth40 = pleth40[order]
    ii120 = ii120[order]

    # Dedup exact duplicate timestamps (can occur if two records overlap)
    if len(time_ms) > 1:
        uniq_mask = np.concatenate([[True], np.diff(time_ms) > 0])
        time_ms = time_ms[uniq_mask]
        pleth40 = pleth40[uniq_mask]
        ii120 = ii120[uniq_mask]

    pleth40 = np.ascontiguousarray(pleth40)
    ii120 = np.ascontiguousarray(ii120)

    assert pleth40.flags["C_CONTIGUOUS"] and ii120.flags["C_CONTIGUOUS"]
    assert pleth40.shape[0] == ii120.shape[0] == len(time_ms)
    assert np.all(np.diff(time_ms) > 0), f"time_ms not strictly monotonic for {entity_id}"

    np.save(f"{out_dir}/PLETH40.npy", pleth40)
    np.save(f"{out_dir}/II120.npy", ii120)
    np.save(f"{out_dir}/time_ms.npy", time_ms)

    meta = {
        "entity_id": entity_id,
        "empi_nbr": int(row["empi_nbr"]),
        "encounter_nbr": int(row["encounter_nbr"]),
        "pat_id": row.get("pat_id"),
        "source_dataset": "emory_sepsis",
        "n_segments": int(len(pleth40)),
        "segment_duration_sec": SEG_SEC,
        "total_duration_hours": round(len(pleth40) * SEG_SEC / 3600, 2),
        "wave_start_ms": int(time_ms[0]),
        "wave_end_ms": int(time_ms[-1] + SEG_SEC * 1000),
        "channels": {
            "PLETH40": {"sample_rate_hz": PLETH_FS, "shape": list(pleth40.shape), "dtype": "float16",
                        "source": "WFDB PLETH candidates (SPO2/Pleth/PLETH/pleth/PlethT) @ 240 Hz, resample_poly(1,6)"},
            "II120":   {"sample_rate_hz": II_FS,    "shape": list(ii120.shape),   "dtype": "float16",
                        "source": "WFDB II @ 240 Hz, resample_poly(1,2); NaN-filled when absent"},
        },
        "n_wfdb_records": len(rec_ids),
        "wfdb_records": rec_ids,
        "n_wave_segments_scanned": n_scanned,
        "n_wave_segments_used": n_used,
        "n_wave_segments_no_pleth": n_no_pleth,
        "n_wave_segments_no_ii": n_no_ii,
        "max_nan_ratio": MAX_NAN_RATIO,
        "records_meta": records_meta,
        "stage_b_version": 1,
    }
    with open(f"{out_dir}/meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    return {
        "entity_id": entity_id,
        "status": "ok",
        "n_seg": int(len(pleth40)),
        "n_records": len(rec_ids),
        "n_scanned": n_scanned,
        "n_used": n_used,
        "n_no_pleth": n_no_pleth,
        "n_no_ii": n_no_ii,
    }


def _worker(args):
    row, out_root = args
    try:
        return process_entity(row, out_root=out_root)
    except Exception as e:
        return {"entity_id": row.get("entity_id", "?"), "status": "error",
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(limit=3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--entity-id", type=str, default=None)
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
    if args.limit:
        df = df.head(args.limit)
    log.info(f"Entities to process: {df.height}  workers: {args.workers}")

    rows = df.to_dicts()
    worker_args = [(r, args.out_root) for r in rows]
    t0 = time.time()
    results = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, worker_args, chunksize=1)):
            results.append(r)
            if (i + 1) % 20 == 0 or i + 1 == len(rows):
                statuses = {}
                for x in results:
                    statuses[x["status"]] = statuses.get(x["status"], 0) + 1
                log.info(f"  {i+1}/{len(rows)}  elapsed {time.time()-t0:.0f}s  {statuses}")

    elapsed = time.time() - t0
    by_status = {}
    for r in results:
        s = r["status"]
        by_status.setdefault(s, []).append(r)
    summary = {
        "n_entities_processed": len(results),
        "elapsed_sec": round(elapsed, 1),
        "by_status": {s: len(v) for s, v in by_status.items()},
        "ok_total_segments": sum(r.get("n_seg", 0) for r in by_status.get("ok", [])),
        "workers": args.workers,
    }
    with open(SUMMARY_JSON, "w") as f:
        json.dump({"summary": summary, "errors": [r for r in results if r["status"] == "error"][:50]},
                  f, indent=2, default=str)
    log.info(f"\n=== Stage B summary ===\n{json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
