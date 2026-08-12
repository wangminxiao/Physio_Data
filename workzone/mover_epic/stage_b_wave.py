#!/usr/bin/env python3
"""
Stage B - MOVER/EPIC waveform extraction (XML -> canonical npy).

v4: drift-aware cpc-timestamp handling + CB-only filter, matching SIS v4.

Parameterized by a **channel list** and **seg_sec** so one script produces any
UNIPHY FM variant, following the MC-MED Stage B scaffolding (ChannelSpec /
ExtractConfig / resample_factors / _resample / parse_channels). A channel =
`name:source:target_fs[:src_fs]`; multiple channels may share a source (parsed
once, resampled to each target). `src_fs` is inferred from `SOURCE_FS` when
omitted. The `anchor` source defines the 30-s window grid.

Defaults reproduce the original hardcoded behavior exactly (PLETH40 @40 from the
PLETH source @100 Hz, II120 @120 from the ECG1 source @300 Hz, 30-s windows,
coverage_s.npy / ii_coverage_s.npy, stage_b_version=4).

Run modes:
  python stage_b_wave.py --limit 3 --workers 2                     # smoke (default 40/120/30s)
  python stage_b_wave.py --entity-id <LOG_ID>
  # variant: 240 s @ PLETH50
  python stage_b_wave.py --seg-sec 240 --channels PLETH50:PLETH:50 \\
      --out-root /opt/localdata100tb/physio_data/mover_epic_seg240

Differences vs SIS v4 (EPIC-specific, preserved):
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
  - delta == 0 (dup): bytewise-compare the anchor <m name="Wave">. Identical ->
    drop (retransmit). Different -> continuation, assign prev_assigned+1000.
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
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from math import gcd
from pathlib import Path

import numpy as np
import polars as pl
from scipy.signal import resample_poly

UTC = timezone.utc
OUT_ROOT = "/opt/localdata100tb/physio_data/mover_epic"
COHORT_PARQUET = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/mover_epic/valid_cohort.parquet"
LOG_DIR = "/labs/hulab/mxwang/Physio_Data/workzone/mover_epic/logs"
SUMMARY_JSON = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/mover_epic/stage_b_summary.json"

# Native rate of each XML source channel (used when a channel spec omits src_fs).
#   PLETH source @ 100 Hz, ECG1 source (Bernoulli Pollster) @ 300 Hz.
SOURCE_FS = {"PLETH": 100, "ECG1": 300}

MIN_SECONDS_PRESENT = 1        # window-level: keep any window with any anchor data
MIN_CLEAN_WINDOWS = 5          # entity-level: require >=5 fully-clean (30/30) windows
MAX_NAN_RATIO = 1.0            # permissive at storage; training filters via coverage_s
DEFAULT_WORKERS = 16

# UCI waveform_decode.py gains (dormant for PLETH/ECG1, future-proof for ABP).
CHANNEL_GAIN_OVERRIDE = {"INVP1": 0.01, "GE_ART": 0.25}


@dataclass(frozen=True)
class ChannelSpec:
    name: str                # canonical output name, e.g. "PLETH40"
    source: str              # XML <mg name> source channel, e.g. "PLETH"
    target_fs: int           # output sample rate, e.g. 40
    src_fs: int              # native rate of `source` in the XML, e.g. 100
    coverage_file: str = ""  # per-channel coverage_s filename ("" -> auto)


@dataclass(frozen=True)
class ExtractConfig:
    seg_sec: int
    anchor: str                       # source that defines the grid; must be present
    channels: tuple[ChannelSpec, ...]
    max_nan_ratio: float = MAX_NAN_RATIO


# Default = the original hardcoded behavior (byte-identical output).
DEFAULT_CFG = ExtractConfig(
    seg_sec=30, anchor="PLETH",
    channels=(
        ChannelSpec("PLETH40", "PLETH", 40, 100, "coverage_s.npy"),
        ChannelSpec("II120",   "ECG1",  120, 300, "ii_coverage_s.npy"),
    ),
)


def resample_factors(target_fs: int, src_fs: int) -> tuple[int, int]:
    """(up, down) for scipy.resample_poly. gcd-reduced."""
    g = gcd(int(target_fs), int(src_fs))
    return int(target_fs) // g, int(src_fs) // g


def _resample(sig: np.ndarray, target_fs: int, src_fs: int) -> np.ndarray:
    """Resample to target_fs; no-op (no FIR pass) when rates already match."""
    if int(target_fs) == int(src_fs):
        return np.asarray(sig, dtype=np.float32)
    up, down = resample_factors(target_fs, src_fs)
    return resample_poly(sig, up, down).astype(np.float32)


def parse_channels(spec: str) -> tuple[ChannelSpec, ...]:
    """Parse `name:source:target_fs[:src_fs]` comma-separated list.
    src_fs is inferred from SOURCE_FS when omitted."""
    chans = []
    for tok in (t.strip() for t in spec.split(",") if t.strip()):
        parts = tok.split(":")
        if len(parts) not in (3, 4):
            raise ValueError(f"bad channel spec {tok!r}; want name:source:target_fs[:src_fs]")
        name, source, target_fs = parts[0], parts[1], int(parts[2])
        if len(parts) == 4:
            src_fs = int(parts[3])
        elif source in SOURCE_FS:
            src_fs = SOURCE_FS[source]
        else:
            raise ValueError(f"src_fs for source {source!r} unknown; give it explicitly "
                             f"(known: {sorted(SOURCE_FS)})")
        chans.append(ChannelSpec(name, source, target_fs, src_fs))
    return tuple(chans)


def _assign_coverage_files(channels: tuple[ChannelSpec, ...], anchor: str
                           ) -> tuple[ChannelSpec, ...]:
    """Fill each spec's coverage_file when unset. The anchor source's first
    channel keeps the canonical `coverage_s.npy`; every other channel gets
    `<name-lower>_coverage_s.npy`. Specs with an explicit coverage_file are
    left untouched (so DEFAULT_CFG's `ii_coverage_s.npy` survives)."""
    out = []
    anchor_used = False
    for c in channels:
        if c.coverage_file:
            out.append(c)
            if c.source == anchor:
                anchor_used = True
            continue
        if c.source == anchor and not anchor_used:
            cf = "coverage_s.npy"
            anchor_used = True
        else:
            cf = f"{c.name.lower()}_coverage_s.npy"
        out.append(replace(c, coverage_file=cf))
    return tuple(out)


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


def parse_xml_file(path: Path, wanted: set[str], anchor: str) -> tuple[dict, dict]:
    """Return ({source: {assigned_ms: samples}} for source in `wanted`, stats).

    Same drift-aware logic as SIS v4. Only Stream A (POLLTIME present) is parsed.
    The `anchor` source's Wave string drives the delta==0 retransmit/continuation
    disambiguation.
    """
    out = {name: {} for name in wanted}
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
    prev_anchor_wave = None
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
            if name not in wanted:
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

        if not any(channels.get(c) for c in wanted):
            elem.clear(); continue

        stats["n_measurements_seen"] += 1
        if wdg_true:
            stats["n_wave_data_gap_true"] += 1

        anchor_wave_str = (channels.get(anchor) or {}).get("wave")
        raw_ms = cur_cpc_ms

        if prev_raw_ms is None:
            assigned_ms = raw_ms
        else:
            delta = raw_ms - prev_raw_ms
            if delta == 0:
                if anchor_wave_str is not None and anchor_wave_str == prev_anchor_wave:
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
        prev_anchor_wave = anchor_wave_str
        elem.clear()
    return out, stats


def _align_window(per_sec_map: dict, t_start_ms: int, src_fs: int,
                  target_fs: int, seg_sec: int) -> tuple[np.ndarray, int]:
    """Build one `seg_sec`-second window at `target_fs` from per-second blocks.

    per_sec_map: {assigned_ms_aligned_to_second: samples_at_src_fs_for_1s}.
    Missing / mostly-NaN seconds are NaN-filled. Returns
    (window float32 length=seg_sec*target_fs, n_seconds_present 0..seg_sec).
    """
    target_len = seg_sec * target_fs
    n_raw_per_sec = src_fs
    raw = np.full(seg_sec * n_raw_per_sec, np.nan, dtype=np.float32)
    presence = np.zeros(seg_sec, dtype=bool)
    for i in range(seg_sec):
        t = t_start_ms + i * 1000
        s = per_sec_map.get(t)
        if s is None or len(s) != n_raw_per_sec:
            continue
        # A 1-second block counts as present only if <50% of its samples were
        # sentinel-masked to NaN by decode_wave; mostly-NaN blocks would pollute
        # resample output via the nan_to_num zero-fill.
        if float(np.isnan(s).mean()) >= 0.5:
            continue
        raw[i * n_raw_per_sec:(i + 1) * n_raw_per_sec] = s
        presence[i] = True
    n_present = int(presence.sum())
    if n_present == 0:
        return np.full(target_len, np.nan, dtype=np.float32), 0
    filled = np.nan_to_num(raw, nan=0.0)
    resamp = _resample(filled, target_fs, src_fs)
    if len(resamp) < target_len:
        resamp = np.concatenate([resamp, np.full(target_len - len(resamp),
                                                 np.nan, dtype=np.float32)])
    elif len(resamp) > target_len:
        resamp = resamp[:target_len]
    if not presence.all():
        per_sec_target = target_len // seg_sec
        out_sig = resamp.copy()
        for i in np.where(~presence)[0]:
            out_sig[i * per_sec_target:(i + 1) * per_sec_target] = np.nan
        return out_sig, n_present
    return resamp, n_present


def process_entity(row: dict, cfg: ExtractConfig = DEFAULT_CFG,
                   out_root: str = OUT_ROOT) -> dict:
    log_id = row["entity_id"]
    out_dir = Path(out_root) / log_id
    meta_path = out_dir / "meta.json"
    seg_sec = cfg.seg_sec

    by_source: dict[str, list[ChannelSpec]] = {}
    for c in cfg.channels:
        by_source.setdefault(c.source, []).append(c)
    if cfg.anchor not in by_source:
        raise ValueError(f"anchor {cfg.anchor!r} not among channel sources {sorted(by_source)}")
    wanted = set(by_source)
    anchor_chan = by_source[cfg.anchor][0]

    required = ([f"{c.name}.npy" for c in cfg.channels]
                + ["time_ms.npy", "meta.json", anchor_chan.coverage_file])
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

    source_maps: dict[str, dict[int, np.ndarray]] = {s: {} for s in wanted}
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
            parsed, s = parse_xml_file(Path(xpath), wanted, cfg.anchor)
            for src in wanted:
                source_maps[src].update(parsed[src])
            for k, v in s.items():
                drift_stats[k] = drift_stats.get(k, 0) + v
            n_xml_parsed += 1
        except Exception:
            n_xml_fail += 1

    anchor_map = source_maps[cfg.anchor]
    if not anchor_map:
        return {"entity_id": log_id, "status": "no_pleth_blocks",
                "n_xmls_all": len(all_xml_paths),
                "n_cb_xmls": len(xml_paths),
                "n_xml_fail": n_xml_fail}

    all_secs = sorted(anchor_map.keys())
    first_ms = all_secs[0]
    last_ms = all_secs[-1]
    win_starts = list(range(first_ms, last_ms + 1, seg_sec * 1000))

    block_lists: dict[str, list] = {c.name: [] for c in cfg.channels}
    coverage_lists: dict[str, list] = {c.name: [] for c in cfg.channels}
    time_ms_list: list[int] = []
    n_dropped_empty = 0
    non_anchor = [c for c in cfg.channels if c.source != cfg.anchor]
    n_with_any = {c.name: 0 for c in non_anchor}

    for t_start in win_starts:
        # Anchor source: align each anchor channel; gate the window on its
        # per-second coverage (identical across anchor channels — same source).
        anchor_windows: dict[str, tuple[np.ndarray, int]] = {}
        gate_sec = 0
        for c in by_source[cfg.anchor]:
            anchor_windows[c.name] = _align_window(anchor_map, t_start, c.src_fs,
                                                   c.target_fs, seg_sec)
            gate_sec = anchor_windows[c.name][1]
        if gate_sec < MIN_SECONDS_PRESENT:
            n_dropped_empty += 1
            continue
        time_ms_list.append(t_start)
        for c in by_source[cfg.anchor]:
            w, sec = anchor_windows[c.name]
            block_lists[c.name].append(w)
            coverage_lists[c.name].append(sec)
        for c in non_anchor:
            w, sec = _align_window(source_maps[c.source], t_start, c.src_fs,
                                   c.target_fs, seg_sec)
            block_lists[c.name].append(w)
            coverage_lists[c.name].append(sec)
            if sec >= 1:
                n_with_any[c.name] += 1

    if not time_ms_list:
        return {"entity_id": log_id, "status": "no_valid_windows",
                "n_xmls_parsed": n_xml_parsed,
                "n_dropped_empty": n_dropped_empty}

    coverage_arrays = {name: np.asarray(coverage_lists[name], dtype=np.uint8)
                       for name in coverage_lists}
    anchor_cov = coverage_arrays[anchor_chan.name]
    n_clean_windows = int((anchor_cov == seg_sec).sum())
    if n_clean_windows < MIN_CLEAN_WINDOWS:
        return {"entity_id": log_id, "status": "too_few_clean_windows",
                "n_xmls_parsed": n_xml_parsed,
                "n_windows_kept": int(len(anchor_cov)),
                "n_clean_windows": n_clean_windows}

    arrays: dict[str, np.ndarray] = {}
    for c in cfg.channels:
        arr = np.ascontiguousarray(np.vstack(block_lists[c.name]).astype(np.float16))
        assert arr.flags["C_CONTIGUOUS"]
        assert arr.shape[1] == seg_sec * c.target_fs
        arrays[c.name] = arr
    time_ms = np.asarray(time_ms_list, dtype=np.int64)
    n_seg = int(time_ms.shape[0])
    for c in cfg.channels:
        assert arrays[c.name].shape[0] == n_seg == len(coverage_arrays[c.name])
    assert n_seg == 1 or np.all(np.diff(time_ms) > 0)

    out_dir.mkdir(parents=True, exist_ok=True)
    for c in cfg.channels:
        np.save(out_dir / f"{c.name}.npy", arrays[c.name])
    np.save(out_dir / "time_ms.npy", time_ms)
    for c in cfg.channels:
        np.save(out_dir / c.coverage_file, coverage_arrays[c.name])

    first_other = non_anchor[0] if non_anchor else None
    ch_meta = {}
    for c in cfg.channels:
        up, down = resample_factors(c.target_fs, c.src_fs)
        if c.source == cfg.anchor:
            src = (f"EPIC XML {c.source} (Stream-A CB-only via POLLTIME filter) "
                   f"@ {c.src_fs} Hz, resample_poly({up},{down}); NaN-filled per missing-second")
        else:
            src = (f"EPIC XML {c.source} (Stream-A CB-only) "
                   f"@ {c.src_fs} Hz, resample_poly({up},{down}); NaN-filled per missing-second")
        ch_meta[c.name] = {"sample_rate_hz": c.target_fs,
                           "shape": list(arrays[c.name].shape),
                           "dtype": "float16", "source": src}

    meta = {
        "entity_id": log_id,
        "log_id": log_id,
        "mrn": row.get("mrn"),
        "source_dataset": "mover_epic",
        "n_segments": n_seg,
        "segment_duration_sec": seg_sec,
        "total_duration_hours": round(n_seg * seg_sec / 3600, 2),
        "wave_start_ms": int(time_ms[0]),
        "wave_end_ms": int(time_ms[-1] + seg_sec * 1000),
        "channels": ch_meta,
        "n_xml_files_listed_all":  len(all_xml_paths),
        "n_xml_files_listed_cb":   len(xml_paths),
        "n_xml_files_ip_skipped":  n_ip_skipped,
        "n_xml_files_parsed":      n_xml_parsed,
        "n_xml_files_failed":      n_xml_fail,
        "n_windows_dropped_empty": n_dropped_empty,
        "n_windows_with_ii_any":   n_with_any[first_other.name] if first_other else 0,
        "n_windows_clean_pleth":   n_clean_windows,
        "n_windows_clean_ii":      (int((coverage_arrays[first_other.name] == seg_sec).sum())
                                    if first_other else 0),
        "has_ii":          bool((n_with_any[first_other.name] if first_other else 0) > 0),
        "coverage_file":   anchor_chan.coverage_file,
        "ii_coverage_file": first_other.coverage_file if first_other else None,
        "min_seconds_present": MIN_SECONDS_PRESENT,
        "min_clean_windows":   MIN_CLEAN_WINDOWS,
        "max_nan_ratio":   cfg.max_nan_ratio,
        "stream_filter":   "polltime_only",
        "drift_stats":     drift_stats,
        "an_start_ms":     int(row["an_start_ms"]) if row.get("an_start_ms") is not None else None,
        "an_stop_ms":      int(row["an_stop_ms"])  if row.get("an_stop_ms")  is not None else None,
        "stage_b_version": 4,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return {"entity_id": log_id, "status": "ok", "n_seg": n_seg,
            "n_xmls": n_xml_parsed, "n_windows_clean_pleth": n_clean_windows,
            "n_windows_with_ii_any": (n_with_any[first_other.name] if first_other else 0),
            "n_dropped_empty": n_dropped_empty}


def _worker(args):
    row, cfg, out_root = args
    try:
        return process_entity(row, cfg=cfg, out_root=out_root)
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
    ap.add_argument("--seg-sec", type=int, default=None,
                    help="Segment duration (default 30 = the canonical variant)")
    ap.add_argument("--anchor", type=str, default=None,
                    help="Anchor XML source that defines the grid (default PLETH)")
    ap.add_argument("--channels", type=str, default=None,
                    help="name:source:target_fs[:src_fs] comma list. Default: PLETH40+II120")
    ap.add_argument("--max-nan-ratio", type=float, default=None)
    args = ap.parse_args()

    if args.channels or args.seg_sec is not None or args.anchor or args.max_nan_ratio is not None:
        anchor = args.anchor or DEFAULT_CFG.anchor
        channels = parse_channels(args.channels) if args.channels else DEFAULT_CFG.channels
        channels = _assign_coverage_files(channels, anchor)
        cfg = ExtractConfig(
            seg_sec=args.seg_sec if args.seg_sec is not None else DEFAULT_CFG.seg_sec,
            anchor=anchor,
            channels=channels,
            max_nan_ratio=args.max_nan_ratio if args.max_nan_ratio is not None else MAX_NAN_RATIO,
        )
    else:
        cfg = DEFAULT_CFG

    os.makedirs(LOG_DIR, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(f"{LOG_DIR}/stage_b_wave.log")])
    log = logging.getLogger(__name__)
    log.info(f"Loading cohort: {COHORT_PARQUET}")
    log.info(f"Config: seg_sec={cfg.seg_sec} anchor={cfg.anchor} out_root={args.out_root} "
             f"channels={[(c.name, c.source, c.target_fs, c.src_fs) for c in cfg.channels]}")
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
                                                  [(row, cfg, args.out_root) for row in rows],
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
        "seg_sec": cfg.seg_sec,
        "channels": [c.name for c in cfg.channels],
        "out_root": args.out_root,
        "workers": args.workers,
    }
    with open(SUMMARY_JSON, "w") as f:
        json.dump({"summary": summary,
                   "errors": [r for r in results if r["status"] == "error"][:30]},
                  f, indent=2, default=str)
    log.info(f"\n=== Stage B summary ===\n{json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
