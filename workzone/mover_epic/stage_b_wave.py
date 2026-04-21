#!/usr/bin/env python3
"""
Stage B - MOVER/EPIC waveform extraction (same XML schema as SIS).

Per LOG_ID:
  - Read xml_paths list from cohort parquet.
  - Decode each XML (same cpcArchive -> cpc -> mg structure as SIS).
  - Build per-second PLETH + ECG1 maps across all XMLs.
  - Window into 30 s blocks aligned to second, resample, NaN-filter.
  - Save PLETH40.npy, II120.npy, time_ms.npy, meta.json.

Handles DATADOWN placeholder XMLs (empty <measurements/>) gracefully - they
yield 0 blocks and the LOG_ID is skipped if no PLETH data is recovered.

Resume: skip LOG_ID when PLETH40.npy/II120.npy/time_ms.npy/meta.json exist
and meta.stage_b_version>=1.
"""
import argparse
import base64
import json
import logging
import multiprocessing as mp
import os
import time
import traceback
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
from scipy.signal import resample_poly

UTC = timezone.utc
OUT_ROOT = "/opt/localdata100tb/physio_data/mover_epic"
COHORT_PARQUET = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/mover_epic/valid_cohort.parquet"
LOG_DIR = "/labs/hulab/mxwang/Physio_Data/workzone/mover_epic/logs"
SUMMARY_JSON = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/mover_epic/stage_b_summary.json"

SEG_SEC = 30
PLETH_FS = 40
II_FS = 120
PLETH_SRC_FS = 100
II_SRC_FS = 300
SEG_LEN_PLETH = SEG_SEC * PLETH_FS   # 1200
SEG_LEN_II = SEG_SEC * II_FS         # 3600
MIN_SECONDS_PRESENT = 24
MAX_NAN_RATIO = 0.20
DEFAULT_WORKERS = 16

WANTED_CHANNELS = {"PLETH", "ECG1"}


def parse_dt_z(s: str) -> int:
    s = s.rstrip("Z")
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            dt = datetime.strptime(s, fmt).replace(tzinfo=UTC)
            return int(dt.timestamp() * 1000)
        except ValueError:
            continue
    raise ValueError(f"bad datetime: {s!r}")


def decode_wave(b64_str: str, gain: float, offset: float,
                vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    raw = base64.b64decode(b64_str)
    raw_i16 = np.frombuffer(raw, dtype="<i2")
    sentinel = (raw_i16 <= -32767) | (raw_i16 >= 32767)
    arr = raw_i16.astype(np.float32) * gain + offset
    if vmin is not None and vmax is not None:
        out_of_range = (arr < vmin) | (arr > vmax)
        arr[out_of_range] = np.nan
    arr[sentinel] = np.nan
    return arr


def parse_xml_file(path: Path) -> dict:
    out = {"PLETH": {}, "ECG1": {}}
    try:
        ctx = ET.iterparse(str(path), events=("start", "end"))
    except Exception:
        return out
    cur_cpc_ms = None
    for event, elem in ctx:
        if event == "start" and elem.tag == "cpc":
            dt_s = elem.attrib.get("datetime")
            if dt_s:
                try:
                    cur_cpc_ms = parse_dt_z(dt_s)
                except ValueError:
                    cur_cpc_ms = None
            continue
        if event != "end":
            continue
        if elem.tag == "cpc":
            elem.clear()
        elif elem.tag == "mg":
            name = elem.get("name")
            if name in WANTED_CHANNELS and cur_cpc_ms is not None:
                wave = offset = gain = None
                points = vmin = vmax = None
                for m in elem.findall("m"):
                    n = m.attrib.get("name")
                    if n == "Wave":
                        wave = m.text
                    elif n == "Gain":
                        try: gain = float(m.text)
                        except (TypeError, ValueError): gain = None
                    elif n == "Offset":
                        try: offset = float(m.text)
                        except (TypeError, ValueError): offset = None
                    elif n == "Points":
                        try: points = int(m.text)
                        except (TypeError, ValueError): points = None
                    elif n == "Min":
                        try: vmin = float(m.text)
                        except (TypeError, ValueError): vmin = None
                    elif n == "Max":
                        try: vmax = float(m.text)
                        except (TypeError, ValueError): vmax = None
                if wave and gain is not None and offset is not None and points:
                    try:
                        samples = decode_wave(wave, gain, offset, vmin, vmax)
                        if len(samples) == points:
                            out[name][cur_cpc_ms] = samples
                    except Exception:
                        pass
            elem.clear()
    return out


def _align_window(per_sec_map: dict, t_start_ms: int,
                  src_fs: int, target_len: int) -> tuple[np.ndarray, int]:
    from math import gcd
    target_fs = PLETH_FS if target_len == SEG_LEN_PLETH else II_FS
    g = gcd(src_fs, target_fs)
    up = target_fs // g
    down = src_fs // g
    n_raw_per_sec = src_fs
    raw = np.full(SEG_SEC * n_raw_per_sec, np.nan, dtype=np.float32)
    presence = np.zeros(SEG_SEC, dtype=bool)
    for i in range(SEG_SEC):
        t = t_start_ms + i * 1000
        s = per_sec_map.get(t)
        if s is None or len(s) != n_raw_per_sec:
            continue
        if float(np.isnan(s).mean()) >= 0.5:
            continue
        raw[i * n_raw_per_sec:(i + 1) * n_raw_per_sec] = s
        presence[i] = True
    n_present = int(presence.sum())
    if n_present == 0:
        return np.full(target_len, np.nan, dtype=np.float32), 0
    filled = np.nan_to_num(raw, nan=0.0)
    resamp = resample_poly(filled, up, down).astype(np.float32)
    if len(resamp) < target_len:
        resamp = np.concatenate([resamp, np.full(target_len - len(resamp),
                                                 np.nan, dtype=np.float32)])
    elif len(resamp) > target_len:
        resamp = resamp[:target_len]
    if not presence.all():
        per_sec_target = target_len // SEG_SEC
        out_sig = resamp.copy()
        for i in np.where(~presence)[0]:
            out_sig[i * per_sec_target:(i + 1) * per_sec_target] = np.nan
        return out_sig, n_present
    return resamp, n_present


def process_entity(row: dict, out_root: str = OUT_ROOT) -> dict:
    log_id = row["entity_id"]
    out_dir = Path(out_root) / log_id
    meta_path = out_dir / "meta.json"

    required = ["PLETH40.npy", "II120.npy", "time_ms.npy", "meta.json"]
    if all((out_dir / f).exists() for f in required):
        try:
            m = json.loads(meta_path.read_text())
            if m.get("stage_b_version", 0) >= 1:
                return {"entity_id": log_id, "status": "resumed",
                        "n_seg": int(m.get("n_segments", 0))}
        except Exception:
            pass

    xml_paths = list(row.get("xml_paths") or [])
    if not xml_paths:
        return {"entity_id": log_id, "status": "no_xmls"}

    pleth_map: dict[int, np.ndarray] = {}
    ecg_map: dict[int, np.ndarray] = {}
    n_xml_parsed = n_xml_fail = 0
    for xpath in xml_paths:
        try:
            parsed = parse_xml_file(Path(xpath))
            pleth_map.update(parsed["PLETH"])
            ecg_map.update(parsed["ECG1"])
            n_xml_parsed += 1
        except Exception:
            n_xml_fail += 1

    if not pleth_map:
        return {"entity_id": log_id, "status": "no_pleth_blocks",
                "n_xmls": len(xml_paths), "n_xml_fail": n_xml_fail}

    all_secs = sorted(pleth_map.keys())
    first_ms = all_secs[0]
    last_ms = all_secs[-1]
    win_starts = list(range(first_ms, last_ms + 1, SEG_SEC * 1000))

    pleth_blocks = []
    ii_blocks = []
    time_ms_list = []
    n_dropped_coverage = 0
    n_dropped_nan = 0

    for t_start in win_starts:
        p_win, p_sec = _align_window(pleth_map, t_start, PLETH_SRC_FS, SEG_LEN_PLETH)
        if p_sec < MIN_SECONDS_PRESENT:
            n_dropped_coverage += 1
            continue
        nan_frac = float(np.isnan(p_win).mean())
        if nan_frac > MAX_NAN_RATIO:
            n_dropped_nan += 1
            continue
        ii_win, i_sec = _align_window(ecg_map, t_start, II_SRC_FS, SEG_LEN_II)
        if i_sec < MIN_SECONDS_PRESENT:
            ii_win = np.full(SEG_LEN_II, np.nan, dtype=np.float32)
        pleth_blocks.append(p_win)
        ii_blocks.append(ii_win)
        time_ms_list.append(t_start)

    if not pleth_blocks:
        return {"entity_id": log_id, "status": "no_valid_windows",
                "n_xmls_parsed": n_xml_parsed,
                "n_dropped_coverage": n_dropped_coverage,
                "n_dropped_nan": n_dropped_nan}

    pleth40 = np.ascontiguousarray(np.vstack(pleth_blocks).astype(np.float16))
    ii120 = np.ascontiguousarray(np.vstack(ii_blocks).astype(np.float16))
    time_ms = np.asarray(time_ms_list, dtype=np.int64)
    assert pleth40.flags["C_CONTIGUOUS"] and ii120.flags["C_CONTIGUOUS"]
    assert pleth40.shape[0] == ii120.shape[0] == len(time_ms)
    assert pleth40.shape[1] == SEG_LEN_PLETH and ii120.shape[1] == SEG_LEN_II
    assert len(time_ms) == 1 or np.all(np.diff(time_ms) > 0)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "PLETH40.npy", pleth40)
    np.save(out_dir / "II120.npy", ii120)
    np.save(out_dir / "time_ms.npy", time_ms)

    n_ii_with_data = int(np.sum(~np.all(np.isnan(ii120), axis=1)))
    meta = {
        "entity_id": log_id,
        "log_id": log_id,
        "mrn": row.get("mrn"),
        "source_dataset": "mover_epic",
        "n_segments": int(pleth40.shape[0]),
        "segment_duration_sec": SEG_SEC,
        "total_duration_hours": round(int(pleth40.shape[0]) * SEG_SEC / 3600, 2),
        "wave_start_ms": int(time_ms[0]),
        "wave_end_ms": int(time_ms[-1] + SEG_SEC * 1000),
        "channels": {
            "PLETH40": {"sample_rate_hz": PLETH_FS, "shape": list(pleth40.shape),
                        "dtype": "float16",
                        "source": f"EPIC XML PLETH @ {PLETH_SRC_FS} Hz, resample_poly(2,5)"},
            "II120":   {"sample_rate_hz": II_FS, "shape": list(ii120.shape),
                        "dtype": "float16",
                        "source": f"EPIC XML ECG1 @ {II_SRC_FS} Hz, resample_poly(2,5); NaN when coverage<80%"},
        },
        "n_xml_files_listed": len(xml_paths),
        "n_xml_files_parsed": n_xml_parsed,
        "n_xml_files_failed": n_xml_fail,
        "n_windows_dropped_coverage": n_dropped_coverage,
        "n_windows_dropped_nan":      n_dropped_nan,
        "n_windows_with_ii":          n_ii_with_data,
        "max_nan_ratio":     MAX_NAN_RATIO,
        "min_seconds_present": MIN_SECONDS_PRESENT,
        "an_start_ms": int(row["an_start_ms"]) if row.get("an_start_ms") is not None else None,
        "an_stop_ms":  int(row["an_stop_ms"])  if row.get("an_stop_ms")  is not None else None,
        "stage_b_version": 1,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return {"entity_id": log_id, "status": "ok", "n_seg": int(pleth40.shape[0]),
            "n_xmls": n_xml_parsed, "n_windows_with_ii": n_ii_with_data,
            "n_dropped_coverage": n_dropped_coverage,
            "n_dropped_nan": n_dropped_nan}


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
