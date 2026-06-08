#!/usr/bin/env python3
"""
Stage B - MOVER/EPIC waveform extraction (XML -> canonical npy).

v4: drift-aware cpc-timestamp handling + CB-only filter, matching SIS v4.

Differences vs SIS v4:
  - EPIC XMLs are attributed per LOG_ID via Stage A's cohort parquet
    (`xml_paths` column). Stage B iterates that list rather than globbing
    a PID dir.
  - XML filenames use a `{PAT_ID}{CB|IP}-{datetime}.xml` pattern. Only *CB*
    XMLs carry Stream A (Bernoulli Pollster: PLETH/ECG1/INVP1 + POLLTIME).
    IP XMLs are Stream B (Monitor: GE_ECG/GE_ART only, fractional cpc ts,
    no PLETH) and are SKIPPED at the xml_paths filter step.
  - Wave 1 and wave 2 CB XMLs are ~100 % DATADOWN placeholders (VitalSigns
    device with empty <measurements/>). They yield 0 blocks and fall
    through naturally. Wave 3 CB is where ~10-15 % of XMLs carry PLETH.

Drift rules (identical to SIS v4):
  - delta == 0 (dup): bytewise-compare <m name="Wave">. Identical -> drop
    (retransmit). Different -> continuation, assign prev_assigned+1000.
  - delta 1-2 s: assigned = max(raw_ms, prev_assigned + 1000).
  - delta > 2 s OR WaveDataGap=TRUE: re-anchor to raw_ms.

Entity-level filter: require >=MIN_CLEAN_WINDOWS fully-clean (30/30) windows.

Resume: skip LOG_ID when all outputs exist and meta.stage_b_version>=4.
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
II_SRC_FS = 300       # ECG1 (Bernoulli Pollster) native rate
SEG_LEN_PLETH = SEG_SEC * PLETH_FS   # 1200
SEG_LEN_II = SEG_SEC * II_FS         # 3600

MIN_SECONDS_PRESENT = 1        # window-level: keep any window with any PLETH
MIN_CLEAN_WINDOWS = 5          # entity-level: require >=5 fully-clean (30/30) windows
MAX_NAN_RATIO = 1.0            # permissive at storage; training filters via coverage_s
DEFAULT_WORKERS = 16

# CB-only Stream A: PLETH + ECG1 from Bernoulli Pollster.
WANTED_CHANNELS = {"PLETH", "ECG1"}
# UCI waveform_decode.py gains (dormant for PLETH/ECG1, future-proof for ABP).
CHANNEL_GAIN_OVERRIDE = {"INVP1": 0.01, "GE_ART": 0.25}


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
    """base64 -> little-endian int16 -> float32 physical values.

    GE monitors use int16 <=-32767 / >=32767 as "no data" sentinels.
    XML-provided Min/Max is applied as an additional validity range.
    """
    raw = base64.b64decode(b64_str)
    raw_i16 = np.frombuffer(raw, dtype="<i2")
    sentinel = (raw_i16 <= -32767) | (raw_i16 >= 32767)
    arr = raw_i16.astype(np.float32) * gain + offset
    if vmin is not None and vmax is not None:
        out_of_range = (arr < vmin) | (arr > vmax)
        arr[out_of_range] = np.nan
    arr[sentinel] = np.nan
    return arr


def parse_xml_file(path: Path) -> tuple[dict, dict]:
    """Return ({'PLETH': {assigned_ms: samples}, 'ECG1': {assigned_ms: samples}}, stats).

    Same drift-aware logic as SIS v4. Only Stream A (POLLTIME present) is parsed.
    """
    out = {"PLETH": {}, "ECG1": {}}
    stats = {
        "n_measurements_seen": 0,
        "n_retransmit_dropped": 0,
        "n_continuation_renumbered": 0,
        "n_cpc_normal": 0,
        "n_cpc_skip_1s": 0,
        "n_cpc_big_gap": 0,
        "n_wave_data_gap_true": 0,
    }
    try:
        ctx = ET.iterparse(str(path), events=("start", "end"))
    except Exception:
        return out, stats
    cur_cpc_ms = None
    prev_raw_ms = None
    prev_assigned_ms = None
    prev_pleth_wave = None
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
            elem.clear(); continue
        if elem.tag != "measurements":
            continue
        polltime_present = False
        wdg_true = False
        for m in elem.findall("m"):
            n = m.attrib.get("name")
            if n == "POLLTIME" and m.text:
                polltime_present = True
            elif n == "WaveDataGap" and m.text and m.text.strip().upper() == "TRUE":
                wdg_true = True
        if not polltime_present or cur_cpc_ms is None:
            elem.clear(); continue
        channels: dict[str, dict] = {}
        for mg in elem.findall("mg"):
            name = mg.get("name")
            if name not in WANTED_CHANNELS:
                continue
            p = {"wave": None, "gain": None, "offset": None, "points": None,
                 "vmin": None, "vmax": None}
            for m in mg.findall("m"):
                n = m.attrib.get("name")
                txt = m.text
                if n == "Wave":
                    p["wave"] = txt
                elif n == "Gain":
                    try: p["gain"] = float(txt)
                    except (TypeError, ValueError): pass
                elif n == "Offset":
                    try: p["offset"] = float(txt)
                    except (TypeError, ValueError): pass
                elif n == "Points":
                    try: p["points"] = int(txt)
                    except (TypeError, ValueError): pass
                elif n == "Min":
                    try: p["vmin"] = float(txt)
                    except (TypeError, ValueError): pass
                elif n == "Max":
                    try: p["vmax"] = float(txt)
                    except (TypeError, ValueError): pass
            channels[name] = p

        if not any(channels.get(c) for c in WANTED_CHANNELS):
            elem.clear(); continue

        stats["n_measurements_seen"] += 1
        if wdg_true:
            stats["n_wave_data_gap_true"] += 1

        pleth_wave_str = (channels.get("PLETH") or {}).get("wave")
        raw_ms = cur_cpc_ms

        if prev_raw_ms is None:
            assigned_ms = raw_ms
        else:
            delta = raw_ms - prev_raw_ms
            if delta == 0:
                if pleth_wave_str is not None and pleth_wave_str == prev_pleth_wave:
                    stats["n_retransmit_dropped"] += 1
                    elem.clear(); continue
                assigned_ms = prev_assigned_ms + 1000
                stats["n_continuation_renumbered"] += 1
            elif wdg_true or delta > 2000:
                assigned_ms = max(raw_ms, prev_assigned_ms + 1000)
                if delta > 2000:
                    stats["n_cpc_big_gap"] += 1
            elif delta == 1000:
                assigned_ms = max(raw_ms, prev_assigned_ms + 1000)
                stats["n_cpc_normal"] += 1
            elif delta == 2000:
                assigned_ms = max(raw_ms, prev_assigned_ms + 1000)
                stats["n_cpc_skip_1s"] += 1
            else:
                assigned_ms = prev_assigned_ms + 1000

        for name, p in channels.items():
            if not (p["wave"] and p["gain"] is not None and p["offset"] is not None
                    and p["points"]):
                continue
            gain = CHANNEL_GAIN_OVERRIDE.get(name, p["gain"])
            try:
                samples = decode_wave(p["wave"], gain, p["offset"], p["vmin"], p["vmax"])
                if len(samples) == p["points"]:
                    out[name][assigned_ms] = samples
            except Exception:
                pass

        prev_raw_ms = raw_ms
        prev_assigned_ms = assigned_ms
        prev_pleth_wave = pleth_wave_str
        elem.clear()
    return out, stats


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

    required = ["PLETH40.npy", "II120.npy", "time_ms.npy", "meta.json",
                "coverage_s.npy"]
    if all((out_dir / f).exists() for f in required):
        try:
            m = json.loads(meta_path.read_text())
            if m.get("stage_b_version", 0) >= 4:
                return {"entity_id": log_id, "status": "resumed",
                        "n_seg": int(m.get("n_segments", 0))}
        except Exception:
            pass

    # Filter Stage A's xml_paths to CB-only (Stream A carrier).
    # IP XMLs are Monitor / Stream B (GE_ECG/GE_ART only, fractional cpc ts,
    # no PLETH) — skipping them at the path level avoids wasted parse work.
    all_xml_paths = list(row.get("xml_paths") or [])
    xml_paths = [p for p in all_xml_paths if "CB" in Path(p).name]
    n_ip_skipped = len(all_xml_paths) - len(xml_paths)
    if not xml_paths:
        return {"entity_id": log_id, "status": "no_cb_xmls",
                "n_xmls_all": len(all_xml_paths),
                "n_ip_skipped": n_ip_skipped}

    pleth_map: dict[int, np.ndarray] = {}
    ecg1_map: dict[int, np.ndarray] = {}
    n_xml_parsed = n_xml_fail = 0
    drift_stats = {
        "n_measurements_seen": 0,
        "n_retransmit_dropped": 0,
        "n_continuation_renumbered": 0,
        "n_cpc_normal": 0,
        "n_cpc_skip_1s": 0,
        "n_cpc_big_gap": 0,
        "n_wave_data_gap_true": 0,
    }
    for xpath in xml_paths:
        try:
            parsed, s = parse_xml_file(Path(xpath))
            pleth_map.update(parsed["PLETH"])
            ecg1_map.update(parsed["ECG1"])
            for k, v in s.items():
                drift_stats[k] = drift_stats.get(k, 0) + v
            n_xml_parsed += 1
        except Exception:
            n_xml_fail += 1

    if not pleth_map:
        return {"entity_id": log_id, "status": "no_pleth_blocks",
                "n_xmls_all": len(all_xml_paths),
                "n_cb_xmls": len(xml_paths),
                "n_xml_fail": n_xml_fail}

    all_secs = sorted(pleth_map.keys())
    first_ms = all_secs[0]
    last_ms = all_secs[-1]
    win_starts = list(range(first_ms, last_ms + 1, SEG_SEC * 1000))

    pleth_blocks = []
    ii_blocks = []
    time_ms_list: list[int] = []
    coverage_s_list: list[int] = []
    ii_coverage_s_list: list[int] = []
    n_dropped_empty = 0
    n_ii_with_any = 0

    for t_start in win_starts:
        p_win, p_sec = _align_window(pleth_map, t_start, PLETH_SRC_FS, SEG_LEN_PLETH)
        if p_sec < MIN_SECONDS_PRESENT:
            n_dropped_empty += 1
            continue
        ii_win, i_sec = _align_window(ecg1_map, t_start, II_SRC_FS, SEG_LEN_II)
        if i_sec >= 1:
            n_ii_with_any += 1
        pleth_blocks.append(p_win)
        ii_blocks.append(ii_win)
        time_ms_list.append(t_start)
        coverage_s_list.append(p_sec)
        ii_coverage_s_list.append(i_sec)

    if not pleth_blocks:
        return {"entity_id": log_id, "status": "no_valid_windows",
                "n_xmls_parsed": n_xml_parsed,
                "n_dropped_empty": n_dropped_empty}

    coverage_s = np.asarray(coverage_s_list, dtype=np.uint8)
    ii_coverage_s = np.asarray(ii_coverage_s_list, dtype=np.uint8)
    n_clean_windows = int((coverage_s == SEG_SEC).sum())
    if n_clean_windows < MIN_CLEAN_WINDOWS:
        return {"entity_id": log_id, "status": "too_few_clean_windows",
                "n_xmls_parsed": n_xml_parsed,
                "n_windows_kept": int(len(coverage_s)),
                "n_clean_windows": n_clean_windows}

    pleth40 = np.ascontiguousarray(np.vstack(pleth_blocks).astype(np.float16))
    ii120 = np.ascontiguousarray(np.vstack(ii_blocks).astype(np.float16))
    time_ms = np.asarray(time_ms_list, dtype=np.int64)
    assert pleth40.flags["C_CONTIGUOUS"] and ii120.flags["C_CONTIGUOUS"]
    assert pleth40.shape[0] == ii120.shape[0] == len(time_ms) == len(coverage_s)
    assert pleth40.shape[1] == SEG_LEN_PLETH and ii120.shape[1] == SEG_LEN_II
    assert len(time_ms) == 1 or np.all(np.diff(time_ms) > 0)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "PLETH40.npy", pleth40)
    np.save(out_dir / "II120.npy", ii120)
    np.save(out_dir / "time_ms.npy", time_ms)
    np.save(out_dir / "coverage_s.npy", coverage_s)
    np.save(out_dir / "ii_coverage_s.npy", ii_coverage_s)

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
                        "source": f"EPIC XML PLETH (Stream-A CB-only via POLLTIME filter) @ {PLETH_SRC_FS} Hz, resample_poly(2,5); NaN-filled per missing-second"},
            "II120":   {"sample_rate_hz": II_FS, "shape": list(ii120.shape),
                        "dtype": "float16",
                        "source": f"EPIC XML ECG1 (Stream-A CB-only) @ {II_SRC_FS} Hz, resample_poly(2,5); NaN-filled per missing-second"},
        },
        "n_xml_files_listed_all":  len(all_xml_paths),
        "n_xml_files_listed_cb":   len(xml_paths),
        "n_xml_files_ip_skipped":  n_ip_skipped,
        "n_xml_files_parsed":      n_xml_parsed,
        "n_xml_files_failed":      n_xml_fail,
        "n_windows_dropped_empty": n_dropped_empty,
        "n_windows_with_ii_any":   n_ii_with_any,
        "n_windows_clean_pleth":   n_clean_windows,
        "n_windows_clean_ii":      int((ii_coverage_s == SEG_SEC).sum()),
        "has_ii":          bool(n_ii_with_any > 0),
        "coverage_file":   "coverage_s.npy",
        "ii_coverage_file": "ii_coverage_s.npy",
        "min_seconds_present": MIN_SECONDS_PRESENT,
        "min_clean_windows":   MIN_CLEAN_WINDOWS,
        "max_nan_ratio":   MAX_NAN_RATIO,
        "stream_filter":   "polltime_only",
        "drift_stats":     drift_stats,
        "an_start_ms":     int(row["an_start_ms"]) if row.get("an_start_ms") is not None else None,
        "an_stop_ms":      int(row["an_stop_ms"])  if row.get("an_stop_ms")  is not None else None,
        "stage_b_version": 4,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return {"entity_id": log_id, "status": "ok", "n_seg": int(pleth40.shape[0]),
            "n_xmls": n_xml_parsed, "n_windows_clean_pleth": n_clean_windows,
            "n_windows_with_ii_any": n_ii_with_any,
            "n_dropped_empty": n_dropped_empty}


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
