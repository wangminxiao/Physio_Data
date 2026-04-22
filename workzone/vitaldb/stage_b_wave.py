#!/usr/bin/env python3
"""
Stage B - VitalDB waveform extraction (.vital -> canonical npy).

Per case:
  1. Open the .vital file via `vitaldb.VitalFile`.
  2. Read tracks SNUADC/PLETH (500 Hz) and SNUADC/ECG_II (500 Hz) as dense
     numpy arrays covering [dtstart, dtend].
  3. Resample PLETH 500 -> 40 Hz via resample_poly(2, 25).
     Resample ECG_II 500 -> 120 Hz via resample_poly(6, 25).
  4. Cut to 30 s non-overlap windows aligned to dtstart.
  5. Require MIN_SECONDS_PRESENT=30 (strict; VitalDB is clean so yield is high).
     Drop windows whose PLETH is >0% NaN in the 30 s span.
  6. Save PLETH40.npy, II120.npy (float16 C-contig), time_ms.npy (int64).
     time_ms = (dtstart + k*30) * 1000 for window k.

Resume: skip when PLETH40.npy + II120.npy + time_ms.npy + meta.json exist
and meta.stage_b_version >= 1.
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
from scipy.signal import resample_poly

OUT_ROOT = "/opt/localdata100tb/physio_data/vitaldb"
COHORT_PARQUET = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/vitaldb/valid_cohort.parquet"
LOG_DIR = "/labs/hulab/mxwang/Physio_Data/workzone/vitaldb/logs"
SUMMARY_JSON = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/vitaldb/stage_b_summary.json"

SEG_SEC = 30
PLETH_FS = 40
II_FS = 120
SRC_FS = 500              # all SNUADC/ tracks in VitalDB are 500 Hz
SEG_LEN_PLETH = SEG_SEC * PLETH_FS    # 1200
SEG_LEN_II = SEG_SEC * II_FS          # 3600
SEG_LEN_SRC = SEG_SEC * SRC_FS        # 15000 samples per 30 s at source rate
MIN_SECONDS_PRESENT = 30   # strict: no NaN allowed in a kept window
MAX_NAN_RATIO = 0.0
DEFAULT_WORKERS = 16

PLETH_TRACK = "SNUADC/PLETH"
ECG_TRACK   = "SNUADC/ECG_II"


def _extract_source_streams(vf: vitaldb.VitalFile):
    """Return (pleth_500hz [N], ecg_500hz [N] or None, dtstart_s)."""
    names = set(vf.get_track_names())
    if PLETH_TRACK not in names:
        return None, None, None
    pleth = vf.to_numpy([PLETH_TRACK], interval=1.0 / SRC_FS).ravel().astype(np.float32)
    if ECG_TRACK in names:
        ecg = vf.to_numpy([ECG_TRACK], interval=1.0 / SRC_FS).ravel().astype(np.float32)
    else:
        ecg = None
    return pleth, ecg, float(vf.dtstart)


def _window_and_resample(pleth: np.ndarray, ecg: np.ndarray | None,
                         dtstart_s: float):
    """Cut into 30 s windows. A window is kept only if PLETH 30 s span has
    zero NaN AFTER source-rate check. Returns (pleth40 [n,1200], ii120 [n,3600],
    time_ms [n], n_dropped)."""
    n_total = len(pleth) // SEG_LEN_SRC
    if n_total == 0:
        return None, None, None, 0
    pleth_trunc = pleth[: n_total * SEG_LEN_SRC].reshape(n_total, SEG_LEN_SRC)
    if ecg is not None:
        # Trim ecg to same n_total if long enough; else pad with NaN
        if len(ecg) >= n_total * SEG_LEN_SRC:
            ecg_trunc = ecg[: n_total * SEG_LEN_SRC].reshape(n_total, SEG_LEN_SRC)
        else:
            ecg_padded = np.concatenate([ecg,
                                         np.full(n_total * SEG_LEN_SRC - len(ecg),
                                                 np.nan, dtype=np.float32)])
            ecg_trunc = ecg_padded.reshape(n_total, SEG_LEN_SRC)
    else:
        ecg_trunc = np.full_like(pleth_trunc, np.nan)

    keep = ~np.isnan(pleth_trunc).any(axis=1)
    n_dropped = int((~keep).sum())
    if not keep.any():
        return None, None, None, n_dropped
    p = pleth_trunc[keep]
    e = ecg_trunc[keep]
    time_ms = ((dtstart_s + np.where(keep)[0] * SEG_SEC) * 1000).astype(np.int64)

    # Resample each window
    pleth40 = np.zeros((p.shape[0], SEG_LEN_PLETH), dtype=np.float32)
    ii120   = np.full ((p.shape[0], SEG_LEN_II),   np.nan, dtype=np.float32)
    for i in range(p.shape[0]):
        pleth40[i] = resample_poly(p[i], 2, 25)[:SEG_LEN_PLETH]
        if not np.isnan(e[i]).any():
            ii120[i] = resample_poly(e[i], 6, 25)[:SEG_LEN_II]
    return pleth40, ii120, time_ms, n_dropped


def process_entity(row, out_root=OUT_ROOT) -> dict:
    eid = row["entity_id"]
    out_dir = Path(out_root) / eid
    meta_path = out_dir / "meta.json"

    required = ["PLETH40.npy", "II120.npy", "time_ms.npy", "meta.json"]
    if all((out_dir / f).exists() for f in required):
        try:
            m = json.loads(meta_path.read_text())
            if m.get("stage_b_version", 0) >= 1:
                return {"entity_id": eid, "status": "resumed",
                        "n_seg": int(m.get("n_segments", 0))}
        except Exception:
            pass

    vpath = row.get("vital_file_path")
    if not vpath or not Path(vpath).exists():
        return {"entity_id": eid, "status": "no_vital_file"}

    try:
        vf = vitaldb.VitalFile(vpath)
    except Exception as e:
        return {"entity_id": eid, "status": "vital_parse_err",
                "error": f"{type(e).__name__}: {e}"}

    pleth, ecg, dtstart = _extract_source_streams(vf)
    if pleth is None:
        return {"entity_id": eid, "status": "no_pleth_track"}
    pleth40, ii120, time_ms, n_dropped = _window_and_resample(pleth, ecg, dtstart)
    if pleth40 is None:
        return {"entity_id": eid, "status": "no_valid_windows",
                "n_dropped_nan": n_dropped}

    p_f16 = np.ascontiguousarray(pleth40.astype(np.float16))
    i_f16 = np.ascontiguousarray(ii120.astype(np.float16))
    assert p_f16.flags["C_CONTIGUOUS"] and i_f16.flags["C_CONTIGUOUS"]
    assert p_f16.shape == (len(time_ms), SEG_LEN_PLETH)
    assert i_f16.shape == (len(time_ms), SEG_LEN_II)
    assert len(time_ms) == 1 or np.all(np.diff(time_ms) > 0)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "PLETH40.npy", p_f16)
    np.save(out_dir / "II120.npy",    i_f16)
    np.save(out_dir / "time_ms.npy",  time_ms)

    n_ii_with_data = int(np.sum(~np.all(np.isnan(i_f16), axis=1)))
    meta = {
        "entity_id": eid,
        "caseid": int(row.get("caseid") or 0),
        "subjectid": int(row.get("subjectid") or 0),
        "source_dataset": "vitaldb",
        "n_segments": int(p_f16.shape[0]),
        "segment_duration_sec": SEG_SEC,
        "total_duration_hours": round(int(p_f16.shape[0]) * SEG_SEC / 3600, 2),
        "wave_start_ms": int(time_ms[0]),
        "wave_end_ms": int(time_ms[-1] + SEG_SEC * 1000),
        "channels": {
            "PLETH40": {"sample_rate_hz": PLETH_FS, "shape": list(p_f16.shape),
                        "dtype": "float16",
                        "source": f"{PLETH_TRACK} @ {SRC_FS} Hz, resample_poly(2,25); strict 0% NaN anchor"},
            "II120":   {"sample_rate_hz": II_FS,   "shape": list(i_f16.shape),
                        "dtype": "float16",
                        "source": f"{ECG_TRACK} @ {SRC_FS} Hz, resample_poly(6,25); NaN when ECG_II absent or has any gap"},
        },
        "dtstart_s":  float(dtstart),
        "n_windows_dropped_nan": int(n_dropped),
        "n_windows_with_ii":     n_ii_with_data,
        "has_ii":                n_ii_with_data > 0,
        "min_seconds_present":   MIN_SECONDS_PRESENT,
        "max_nan_ratio":         MAX_NAN_RATIO,
        "anestart_ms": int((dtstart + float(row.get("anestart_s") or 0)) * 1000)
                       if row.get("anestart_s") is not None else None,
        "aneend_ms":   int((dtstart + float(row.get("aneend_s")   or 0)) * 1000)
                       if row.get("aneend_s")   is not None else None,
        "stage_b_version": 1,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    return {"entity_id": eid, "status": "ok",
            "n_seg": int(p_f16.shape[0]),
            "n_windows_with_ii": n_ii_with_data,
            "n_dropped_nan": n_dropped}


def _worker(args):
    row, out_root = args
    try:
        return process_entity(row, out_root)
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
    ap.add_argument("--out-root", default=OUT_ROOT)
    args = ap.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(f"{LOG_DIR}/stage_b_wave.log")])
    log = logging.getLogger(__name__)
    log.info(f"Loading cohort: {COHORT_PARQUET}")
    df = pl.read_parquet(COHORT_PARQUET)
    if args.entity_id:
        df = df.filter(pl.col("entity_id") == args.entity_id)
    elif args.entities:
        ids = [s.strip() for s in args.entities.split(",") if s.strip()]
        df = df.filter(pl.col("entity_id").is_in(ids))
    elif args.limit:
        df = df.head(args.limit)
    rows = df.to_dicts()
    log.info(f"Processing {len(rows)} entities  workers={args.workers}")

    t0 = time.time()
    results = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker,
                                                  [(row, args.out_root) for row in rows],
                                                  chunksize=1)):
            results.append(r)
            if (i + 1) % 50 == 0 or i + 1 == len(rows):
                st = {}
                for x in results:
                    st[x["status"]] = st.get(x["status"], 0) + 1
                log.info(f"  {i+1}/{len(rows)}  elapsed {time.time()-t0:.0f}s  {st}")

    by = {}
    for r in results:
        by.setdefault(r["status"], []).append(r)
    summary = {
        "n_entities_processed": len(results),
        "elapsed_sec": round(time.time() - t0, 1),
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
