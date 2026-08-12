#!/usr/bin/env python3
"""
Stage B — MOVER/SIS waveform extraction (raw XML -> canonical npy).

Anchor-driven, per-second XML alignment. Each raw `<mg name>` channel (source) is
decoded into a per-second sample map {cpc_ms: samples_at_src_fs}. The anchor source
(default `PLETH`) enumerates seg_sec-second non-overlap windows aligned to its first
cpc second and gates each entity via MIN_CLEAN_WINDOWS. Every requested channel is
then aligned onto that window grid via `_align_window` (concat per-second blocks,
NaN-fill missing seconds, resample_poly to the channel's target rate); each channel
also records a per-window seconds-present coverage array.

Parameterized by a **channel list** and **seg_sec** (mirrors mcmed/stage_b_wave.py)
so one script produces any UNIPHY FM variant. A channel = `name:source:target_fs
[:src_fs]`; multiple channels may share a source (read once from the same per-second
map, resampled to each target). `src_fs` is inferred from `SOURCE_FS` when omitted.

Defaults reproduce the original exactly: PLETH40 <- PLETH @40 (src 100), II120 <-
ECG1 @120 (src 300), seg_sec=30, coverage_s.npy (anchor) + ii_coverage_s.npy (ECG1),
MIN_CLEAN_WINDOWS=5, Stream-A POLLTIME filter, stage_b_version=3.

Run modes:
  python stage_b_wave.py --limit 3 --workers 2          # smoke (default 40/120/30s)
  python stage_b_wave.py --entity-id <PID>
  # single-channel variant: 240 s @ PLETH50
  python stage_b_wave.py --seg-sec 240 --anchor PLETH \\
      --channels PLETH50:PLETH:50 \\
      --out-root /opt/localdata100tb/physio_data/mover_seg240
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
from dataclasses import dataclass
from datetime import datetime, timezone
from math import gcd
from pathlib import Path

import numpy as np
import polars as pl
from scipy.signal import resample_poly

UTC = timezone.utc
RAW_ROOT = "/opt/localdata100tb/UNIPHY_Plus/raw_datasets/MOVER"
SIS_WAVE_ROOT = f"{RAW_ROOT}/sis_wave_v2/UCI_deidentified_part3_SIS_11_07/Waveforms"
OUT_ROOT = "/opt/localdata100tb/physio_data/mover"
COHORT_PARQUET = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/mover/valid_cohort.parquet"
LOG_DIR = "/labs/hulab/mxwang/Physio_Data/workzone/mover/logs"
SUMMARY_JSON = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/mover/stage_b_summary.json"

SEG_SEC = 30

# Native rate of each raw XML `<mg name>` channel (used when a channel spec omits
# src_fs). PLETH is 100 Hz (100 samples/cpc); ECG1 is 300 Hz (300 samples/cpc).
SOURCE_FS = {"PLETH": 100, "ECG1": 300}

# v3 windowing policy: keep every window with >=1 s of real anchor signal; NaN-fill
# any missing-second region in every channel. The per-window second count is saved
# as a coverage array so training can pick its own quality bar.
MIN_SECONDS_PRESENT = 1        # window-level: keep anything with any data
MIN_CLEAN_WINDOWS   = 5        # entity-level: require >=5 fully-clean windows on anchor
MAX_NAN_RATIO = 1.0            # permissive at storage time; training filters via coverage
DEFAULT_WORKERS = 16

# v3 channel policy: Stream-A filter (see parse_xml_file). Only Stream-A `<mg>`
# blocks (those carrying a POLLTIME) are used; Stream B's fractional-second cpc
# timestamps never matched our whole-second per_sec_map. The set of raw sources
# to parse is derived from the active config (only parse what a channel needs).

# Per UCI's waveform_decode.py, INVP1 / GE_ART gains in the XML are incorrect
# and must be overridden. Dormant for now (we don't extract those channels) but
# retained so an ABP125 extension won't silently mis-scale.
CHANNEL_GAIN_OVERRIDE = {"INVP1": 0.01, "GE_ART": 0.25}


@dataclass(frozen=True)
class ChannelSpec:
    name: str          # canonical output name, e.g. "PLETH40"
    source: str        # raw XML <mg name> channel, e.g. "PLETH"
    target_fs: int     # output sample rate, e.g. 40
    src_fs: int        # expected native rate of `source`, e.g. 100


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
        ChannelSpec("PLETH40", "PLETH", 40, 100),
        ChannelSpec("II120",   "ECG1",  120, 300),
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


def pid_suffix(pid: str) -> str:
    return pid[:2]


def parse_dt_z(s: str) -> int:
    """'2017-09-08T17:30:01Z' -> ms int."""
    # strip Z or anything after; assume seconds precision
    s = s.rstrip("Z")
    # some SIS XMLs may have fractional, try both
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

    GE monitors use -32768 / -32767 / 32767 as "no data" sentinels. We also
    apply the XML-provided Min/Max as a validity range if present (e.g. PLETH
    has Min=-500, Max=500) — samples outside that range are set to NaN.
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


def parse_xml_file(path: Path, wanted: set[str]) -> dict:
    """Return {source: {cpc_ms: samples}} for each source in `wanted`.

    `wanted` is the set of raw `<mg name>` sources the active config needs
    (e.g. {"PLETH", "ECG1"}). v3 only processes Stream A: `<measurements>`
    blocks that contain a `<m name="POLLTIME">` child. Stream B (GE native,
    fractional cpc timestamps, no POLLTIME) is skipped explicitly. Uses
    iterparse + element clearing so 6 MB XMLs don't blow memory.
    """
    out = {name: {} for name in wanted}
    try:
        ctx = ET.iterparse(str(path), events=("start", "end"))
    except Exception:
        return out
    cur_cpc_ms = None
    for event, elem in ctx:
        # Capture cpc datetime on START event — end events fire bottom-up so
        # <measurements> ends before its parent <cpc>; if we read datetime on
        # cpc-end we'd only have it after children already processed.
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
            continue
        if elem.tag != "measurements":
            continue
        # Stream-A filter: only process measurements blocks that have POLLTIME.
        # Stream B (GE_ECG / GE_ART, fractional-second cpc timestamps) carries
        # no POLLTIME and isn't useful in the per-second-keyed alignment.
        polltime_present = False
        for m in elem.findall("m"):
            if m.attrib.get("name") == "POLLTIME" and m.text:
                polltime_present = True
                break
        if not polltime_present or cur_cpc_ms is None:
            elem.clear()
            continue
        # Extract all wanted <mg> children of this measurements block.
        for mg in elem.findall("mg"):
            name = mg.get("name")
            if name not in wanted:
                continue
            wave = offset = gain = None
            points = None
            vmin = vmax = None
            for m in mg.findall("m"):
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
            # Apply UCI's per-channel gain overrides where applicable (dormant
            # for PLETH/ECG1 but future-proof for ABP extraction).
            if name in CHANNEL_GAIN_OVERRIDE:
                gain = CHANNEL_GAIN_OVERRIDE[name]
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
                  src_fs: int, target_fs: int,
                  seg_sec: int = SEG_SEC) -> tuple[np.ndarray, int]:
    """Build one seg_sec-second window at target_fs via resample_poly.

    per_sec_map: {cpc_ms_aligned_to_second: samples_at_src_fs_for_1s}
    Returns (window float32 length=seg_sec*target_fs, n_seconds_present 0..seg_sec).
    """
    target_len = seg_sec * target_fs
    up, down = resample_factors(target_fs, src_fs)
    n_raw_per_sec = src_fs
    raw = np.full(seg_sec * n_raw_per_sec, np.nan, dtype=np.float32)
    presence = np.zeros(seg_sec, dtype=bool)
    for i in range(seg_sec):
        t = t_start_ms + i * 1000
        s = per_sec_map.get(t)
        if s is None or len(s) != n_raw_per_sec:
            continue
        # A 1-second block is "present" only if <50% of its samples were
        # sentinel-masked to NaN by decode_wave. Blocks that are mostly NaN
        # would pollute resample output via nan_to_num zero-fill.
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
    # Re-apply NaN on absent seconds so downstream sees NaN, not 0-filled noise.
    if not presence.all():
        per_sec_target = target_len // seg_sec
        out_sig = resamp.copy()
        for i in np.where(~presence)[0]:
            out_sig[i * per_sec_target:(i + 1) * per_sec_target] = np.nan
        return out_sig, n_present
    return resamp, n_present


def _coverage_filename(chan: ChannelSpec, anchor_primary_name: str,
                       is_default: bool) -> str:
    """Per-channel coverage filename. Anchor primary -> `coverage_s.npy`. The
    single non-anchor channel of the default config keeps the legacy name
    `ii_coverage_s.npy`; every other non-anchor channel gets `{name}_coverage_s.npy`.
    """
    if chan.name == anchor_primary_name:
        return "coverage_s.npy"
    if is_default:
        return "ii_coverage_s.npy"
    return f"{chan.name}_coverage_s.npy"


def process_entity(row: dict, cfg: ExtractConfig = DEFAULT_CFG,
                   out_root: str = OUT_ROOT) -> dict:
    pid = str(row["pid"])
    out_dir = Path(out_root) / pid
    meta_path = out_dir / "meta.json"
    seg_sec = cfg.seg_sec

    by_source: dict[str, list[ChannelSpec]] = {}
    for c in cfg.channels:
        by_source.setdefault(c.source, []).append(c)
    if cfg.anchor not in by_source:
        raise ValueError(f"anchor {cfg.anchor!r} not among channel sources {sorted(by_source)}")
    anchor_chans = by_source[cfg.anchor]
    anchor0 = anchor_chans[0]                 # primary anchor channel: drives grid + gate

    is_default = (cfg == DEFAULT_CFG)
    cov_files = {c.name: _coverage_filename(c, anchor0.name, is_default)
                 for c in cfg.channels}

    # Resume: keyed on the config's channel + coverage files + version >= 3.
    required = ([f"{c.name}.npy" for c in cfg.channels]
                + ["time_ms.npy", "meta.json"]
                + list(cov_files.values()))
    if all((out_dir / f).exists() for f in required):
        try:
            m = json.loads(meta_path.read_text())
            if m.get("stage_b_version", 0) >= 3:
                return {"entity_id": pid, "status": "resumed",
                        "n_seg": int(m.get("n_segments", 0))}
        except Exception:
            pass

    pid_dir = Path(SIS_WAVE_ROOT) / pid_suffix(pid) / pid
    if not pid_dir.exists():
        return {"entity_id": pid, "status": "no_wave_dir"}
    xml_paths = sorted(pid_dir.glob("*.xml"))
    if not xml_paths:
        return {"entity_id": pid, "status": "no_xmls"}

    # Only parse the raw sources this config actually needs.
    wanted = {c.source for c in cfg.channels}
    source_maps: dict[str, dict[int, np.ndarray]] = {s: {} for s in wanted}
    n_xml_parsed = n_xml_fail = 0
    for p in xml_paths:
        try:
            parsed = parse_xml_file(p, wanted)
            for s in wanted:
                source_maps[s].update(parsed[s])
            n_xml_parsed += 1
        except Exception:
            n_xml_fail += 1

    anchor_map = source_maps[cfg.anchor]
    if not anchor_map:
        return {"entity_id": pid, "status": "no_anchor_blocks",
                "n_xmls": len(xml_paths), "n_xml_fail": n_xml_fail}

    # Enumerate seg_sec windows aligned to the first anchor cpc second.
    all_secs = sorted(anchor_map.keys())
    first_ms = all_secs[0]
    last_ms = all_secs[-1]
    win_starts = list(range(first_ms, last_ms + 1, seg_sec * 1000))

    blocks: dict[str, list] = {c.name: [] for c in cfg.channels}
    coverage: dict[str, list] = {c.name: [] for c in cfg.channels}  # per-window seconds present
    time_ms_list: list[int] = []
    n_dropped_empty = 0
    # Windows-with-any-signal count, only for non-anchor-source channels.
    n_with_any: dict[str, int] = {c.name: 0 for c in cfg.channels
                                  if c.source != cfg.anchor}

    for t_start in win_starts:
        a_win, a_sec = _align_window(anchor_map, t_start, anchor0.src_fs,
                                     anchor0.target_fs, seg_sec)
        if a_sec < MIN_SECONDS_PRESENT:
            n_dropped_empty += 1
            continue
        win_by_ch = {anchor0.name: a_win}
        sec_by_ch = {anchor0.name: a_sec}
        for c in cfg.channels:
            if c.name == anchor0.name:
                continue
            w, sec = _align_window(source_maps[c.source], t_start, c.src_fs,
                                   c.target_fs, seg_sec)
            win_by_ch[c.name] = w
            sec_by_ch[c.name] = sec
            # _align_window returns all-NaN when sec == 0; keep as-is.
            if c.source != cfg.anchor and sec >= 1:
                n_with_any[c.name] += 1
        for c in cfg.channels:
            blocks[c.name].append(win_by_ch[c.name])
            coverage[c.name].append(sec_by_ch[c.name])
        time_ms_list.append(t_start)

    if not time_ms_list:
        return {"entity_id": pid, "status": "no_valid_windows",
                "n_xmls_parsed": n_xml_parsed,
                "n_dropped_empty": n_dropped_empty}

    cov_arrays = {c.name: np.asarray(coverage[c.name], dtype=np.uint8)
                  for c in cfg.channels}
    anchor_cov = cov_arrays[anchor0.name]
    n_clean_windows = int((anchor_cov == seg_sec).sum())
    if n_clean_windows < MIN_CLEAN_WINDOWS:
        return {"entity_id": pid, "status": "too_few_clean_windows",
                "n_xmls_parsed": n_xml_parsed,
                "n_windows_kept": int(len(anchor_cov)),
                "n_clean_windows": n_clean_windows}

    arrays = {}
    for c in cfg.channels:
        arr = np.ascontiguousarray(np.vstack(blocks[c.name]).astype(np.float16))
        assert arr.flags["C_CONTIGUOUS"]
        assert arr.shape[1] == seg_sec * c.target_fs
        arrays[c.name] = arr
    time_ms = np.asarray(time_ms_list, dtype=np.int64)
    n_seg = int(time_ms.shape[0])
    for c in cfg.channels:
        assert arrays[c.name].shape[0] == n_seg == len(cov_arrays[c.name])
    assert n_seg == 1 or np.all(np.diff(time_ms) > 0)

    out_dir.mkdir(parents=True, exist_ok=True)
    for c in cfg.channels:
        np.save(out_dir / f"{c.name}.npy", arrays[c.name])
    np.save(out_dir / "time_ms.npy", time_ms)
    for c in cfg.channels:
        np.save(out_dir / cov_files[c.name], cov_arrays[c.name])

    ch_meta = {}
    for c in cfg.channels:
        up, down = resample_factors(c.target_fs, c.src_fs)
        if c.source == cfg.anchor:
            src = (f"SIS XML {c.source} (Stream-A only via POLLTIME filter) @ "
                   f"{c.src_fs} Hz, resample_poly({up},{down}); NaN-filled per missing-second")
        else:
            src = (f"SIS XML {c.source} (Stream-A only) @ {c.src_fs} Hz, "
                   f"resample_poly({up},{down}); NaN-filled per missing-second")
        ch_meta[c.name] = {"sample_rate_hz": c.target_fs,
                           "shape": list(arrays[c.name].shape),
                           "dtype": "float16", "source": src,
                           "coverage_file": cov_files[c.name]}

    meta = {
        "entity_id": pid,
        "pid": pid,
        "source_dataset": "mover_sis",
        "n_segments": n_seg,
        "segment_duration_sec": seg_sec,
        "total_duration_hours": round(n_seg * seg_sec / 3600, 2),
        "wave_start_ms": int(time_ms[0]),
        "wave_end_ms": int(time_ms[-1] + seg_sec * 1000),
        "anchor": cfg.anchor,
        "channels": ch_meta,
        "n_xml_files_listed":  len(xml_paths),
        "n_xml_files_parsed":  n_xml_parsed,
        "n_xml_files_failed":  n_xml_fail,
        "n_windows_dropped_empty":  n_dropped_empty,
        "n_windows_with":      n_with_any,   # {non-anchor channel: windows with >=1 s}
        "n_windows_clean":     {c.name: int((cov_arrays[c.name] == seg_sec).sum())
                                for c in cfg.channels},
        "n_windows_clean_anchor": n_clean_windows,
        "coverage_file":   "coverage_s.npy",       # anchor: [N_seg] uint8, seconds present per window
        "min_seconds_present": MIN_SECONDS_PRESENT,
        "min_clean_windows":   MIN_CLEAN_WINDOWS,
        "max_nan_ratio":   cfg.max_nan_ratio,
        "stream_filter":   "polltime_only",
        "or_start_ms":     int(row["or_start_ms"]) if row.get("or_start_ms") is not None else None,
        "or_end_ms":       int(row["or_end_ms"])   if row.get("or_end_ms")   is not None else None,
        "stage_b_version": 3,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return {"entity_id": pid, "status": "ok", "n_seg": n_seg,
            "n_xmls": n_xml_parsed, "n_windows_clean_anchor": n_clean_windows,
            "n_windows_with": n_with_any,
            "n_dropped_empty": n_dropped_empty}


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
                    help="Anchor source (XML <mg name>) that defines the grid (default PLETH)")
    ap.add_argument("--channels", type=str, default=None,
                    help="name:source:target_fs[:src_fs] comma list. Default: PLETH40+II120")
    ap.add_argument("--max-nan-ratio", type=float, default=None)
    args = ap.parse_args()

    if args.channels or args.seg_sec is not None or args.anchor or args.max_nan_ratio is not None:
        cfg = ExtractConfig(
            seg_sec=args.seg_sec if args.seg_sec is not None else DEFAULT_CFG.seg_sec,
            anchor=args.anchor or DEFAULT_CFG.anchor,
            channels=parse_channels(args.channels) if args.channels else DEFAULT_CFG.channels,
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

    elapsed = time.time() - t0
    by = {}
    for r in results:
        by.setdefault(r["status"], []).append(r)
    summary = {
        "n_entities_processed": len(results),
        "elapsed_sec": round(elapsed, 1),
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
