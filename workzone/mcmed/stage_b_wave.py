#!/usr/bin/env python3
"""
Stage B — MC_MED waveform extraction (raw WFDB -> canonical npy).

PLETH-anchored: the anchor source (default `Pleth`) is read at its native rate,
tiled into `seg_sec`-second non-overlap windows aligned to each segment's
base_datetime — this defines the `time_ms` grid. Every requested channel is then
produced onto that grid; non-anchor sources (e.g. `II`) are aligned per-window and
NaN-filled where absent. Windows with > `max_nan_ratio` NaN in the first anchor
channel are dropped.

Parameterized by a **channel list** and **seg_sec** so one script produces any
UNIPHY FM variant. A channel = `name:source:target_fs[:src_fs]`; multiple channels
may share a source (read once, resampled to each target). `src_fs` is inferred from
`SOURCE_FS` when omitted.

Defaults reproduce the original (PLETH40 @40, II120 @120, 30 s) output exactly.

Run modes:
  python stage_b_wave.py --limit 3 --workers 2                     # smoke (default 40/120/30s)
  python stage_b_wave.py --entity-id 99370369
  # baseline variant: 10 s @ PLETH125 + PLETH50 + II500
  python stage_b_wave.py --seg-sec 10 --anchor Pleth \\
      --channels PLETH125:Pleth:125,PLETH50:Pleth:50,II500:II:500 \\
      --out-root /opt/localdata100tb/physio_data/mcmed_seg10
"""
import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import timezone
from math import gcd
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

# Native rate of each raw WFDB channel dir (used when a channel spec omits src_fs).
SOURCE_FS = {"Pleth": 125, "II": 500}
MAX_NAN_RATIO = 0.20
DEFAULT_WORKERS = 16


@dataclass(frozen=True)
class ChannelSpec:
    name: str          # canonical output name, e.g. "PLETH40"
    source: str        # raw WFDB channel dir, e.g. "Pleth"
    target_fs: int     # output sample rate, e.g. 40
    src_fs: int        # expected native rate of `source`, e.g. 125


@dataclass(frozen=True)
class ExtractConfig:
    seg_sec: int
    anchor: str                       # source that defines the grid; must be present
    channels: tuple[ChannelSpec, ...]
    max_nan_ratio: float = MAX_NAN_RATIO


# Default = the original hardcoded behavior (byte-identical output).
DEFAULT_CFG = ExtractConfig(
    seg_sec=30, anchor="Pleth",
    channels=(
        ChannelSpec("PLETH40", "Pleth", 40, 125),
        ChannelSpec("II120",   "II",    120, 500),
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


def csn_suffix(csn: str) -> str:
    """Last 3 digits of CSN determine the top-level waveform subdir."""
    return str(csn).zfill(4)[-3:]


def list_segments(csn_dir: Path, channel: str) -> list[tuple[int, Path]]:
    """Return (segment_number, hea_path) sorted ascending. Channel dir may be absent."""
    sub = csn_dir / channel
    if not sub.exists():
        return []
    out = []
    for p in sub.glob("*_*.hea"):
        try:
            n = int(p.stem.rsplit("_", 1)[1])
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


def build_aligned(grid_time_ms: np.ndarray,
                  streams: list[tuple[np.ndarray, int, int]],
                  seg_sec: int, src_fs: int, target_fs: int
                  ) -> tuple[np.ndarray, int]:
    """Align a non-anchor source onto the anchor grid.

    For each grid window start t, find the first stream fully covering [t, t+seg],
    extract the native samples, resample to target_fs. NaN-filled where no coverage.
    Returns (arr [n, seg_sec*target_fs] float32, n_windows_with_coverage).
    """
    n = len(grid_time_ms)
    L = seg_sec * target_fs
    out = np.full((n, L), np.nan, dtype=np.float32)
    if not streams:
        return out, 0
    win_ms = seg_sec * 1000
    n_raw_per_win = seg_sec * src_fs
    n_with = 0
    for i, t in enumerate(grid_time_ms):
        t_int = int(t)
        t_end = t_int + win_ms
        for sig, base_ms, n_samp in streams:
            end_ms = base_ms + int(n_samp * 1000 / src_fs)
            if t_int >= base_ms and t_end <= end_ms:
                start_idx = int((t_int - base_ms) * src_fs // 1000)
                chunk = sig[start_idx: start_idx + n_raw_per_win]
                if len(chunk) == n_raw_per_win:
                    rs = _resample(chunk, target_fs, src_fs)
                    if len(rs) >= L:
                        out[i] = rs[:L]
                        n_with += 1
                        break
    return out, n_with


def process_entity(row: dict, cfg: ExtractConfig = DEFAULT_CFG, out_root: str = OUT_ROOT) -> dict:
    entity_id = row["entity_id"]
    csn = str(row.get("csn") or entity_id)
    out_dir = Path(out_root) / entity_id
    seg_sec = cfg.seg_sec

    by_source: dict[str, list[ChannelSpec]] = {}
    for c in cfg.channels:
        by_source.setdefault(c.source, []).append(c)
    if cfg.anchor not in by_source:
        raise ValueError(f"anchor {cfg.anchor!r} not among channel sources {sorted(by_source)}")

    # Resume
    required = [f"{c.name}.npy" for c in cfg.channels] + ["time_ms.npy", "meta.json"]
    if all((out_dir / f).exists() for f in required):
        try:
            m = json.loads((out_dir / "meta.json").read_text())
            if m.get("stage_b_version", 0) >= 1:
                return {"entity_id": entity_id, "status": "resumed",
                        "n_seg": int(m.get("n_segments", 0))}
        except Exception:
            pass

    csn_dir = Path(WAVE_DIR) / csn_suffix(csn) / csn
    if not csn_dir.exists():
        return {"entity_id": entity_id, "status": "no_wave_dir", "path": str(csn_dir)}

    src_stats = {s: {"listed": len(list_segments(csn_dir, s)), "used": 0, "failed": 0}
                 for s in by_source}
    blocks: dict[str, list | np.ndarray] = {c.name: [] for c in cfg.channels}

    # --- anchor: read native segments -> grid + anchor-channel blocks ---
    anchor_fs = by_source[cfg.anchor][0].src_fs
    anchor_chans = by_source[cfg.anchor]
    times_list: list[np.ndarray] = []
    for _, hea_p in list_segments(csn_dir, cfg.anchor):
        r = read_wave_segment(hea_p)
        if r is None:
            src_stats[cfg.anchor]["failed"] += 1
            continue
        sig, base_ms, fs, n_samp = r
        if int(round(fs)) != anchor_fs:
            src_stats[cfg.anchor]["failed"] += 1
            continue
        # Resample every anchor channel; window count = min over channels of
        # len(resampled)//L so all anchor channels share the same grid. For the
        # single-channel default this equals the original len(resample)//1200.
        rs_by_ch = {c.name: _resample(sig, c.target_fs, anchor_fs) for c in anchor_chans}
        n_win = min(len(rs_by_ch[c.name]) // (seg_sec * c.target_fs) for c in anchor_chans)
        if n_win == 0:
            continue
        times_list.append(base_ms + np.arange(n_win, dtype=np.int64) * (seg_sec * 1000))
        for c in anchor_chans:
            L = seg_sec * c.target_fs
            blocks[c.name].append(rs_by_ch[c.name][: n_win * L].reshape(n_win, L))
        src_stats[cfg.anchor]["used"] += 1

    if not times_list:
        return {"entity_id": entity_id, "status": "no_valid_anchor",
                "anchor": cfg.anchor, **{f"n_{cfg.anchor}_fail": src_stats[cfg.anchor]["failed"]}}

    time_ms = np.concatenate(times_list).astype(np.int64)
    for c in anchor_chans:
        blocks[c.name] = np.vstack(blocks[c.name])

    # chronological sort + drop duplicate window starts
    order = np.argsort(time_ms, kind="stable")
    time_ms = time_ms[order]
    for c in anchor_chans:
        blocks[c.name] = blocks[c.name][order]
    if len(time_ms) > 1:
        keep = np.concatenate([[True], np.diff(time_ms) > 0])
        time_ms = time_ms[keep]
        for c in anchor_chans:
            blocks[c.name] = blocks[c.name][keep]

    # --- non-anchor sources aligned to the grid ---
    n_with: dict[str, int] = {}
    for s, chans in by_source.items():
        if s == cfg.anchor:
            continue
        src_fs = chans[0].src_fs
        streams: list[tuple[np.ndarray, int, int]] = []
        for _, hea_p in list_segments(csn_dir, s):
            r = read_wave_segment(hea_p)
            if r is None:
                src_stats[s]["failed"] += 1
                continue
            sig, base_ms, fs, n_samp = r
            if int(round(fs)) != src_fs:
                src_stats[s]["failed"] += 1
                continue
            streams.append((sig, base_ms, n_samp))
            src_stats[s]["used"] += 1
        for c in chans:
            arr, nw = build_aligned(time_ms, streams, seg_sec, src_fs, c.target_fs)
            blocks[c.name] = arr
            n_with[c.name] = nw

    # --- NaN filter on the first anchor channel ---
    anchor0 = anchor_chans[0].name
    nan_frac = np.isnan(blocks[anchor0]).mean(axis=1)
    keep = nan_frac <= cfg.max_nan_ratio
    n_dropped = int((~keep).sum())
    if not keep.any():
        return {"entity_id": entity_id, "status": "all_nan",
                "n_windows": int(len(time_ms))}
    for c in cfg.channels:
        blocks[c.name] = blocks[c.name][keep]
    time_ms = time_ms[keep]

    # --- save ---
    out_dir.mkdir(parents=True, exist_ok=True)
    for c in cfg.channels:
        arr = np.ascontiguousarray(blocks[c.name].astype(np.float16))
        assert arr.flags["C_CONTIGUOUS"]
        assert arr.shape[0] == len(time_ms)
        assert arr.shape[1] == seg_sec * c.target_fs
        np.save(out_dir / f"{c.name}.npy", arr)
    assert len(time_ms) == 1 or np.all(np.diff(time_ms) > 0)
    np.save(out_dir / "time_ms.npy", time_ms.astype(np.int64))

    n_seg = int(len(time_ms))
    ch_meta = {}
    for c in cfg.channels:
        up, down = resample_factors(c.target_fs, c.src_fs)
        src = (f"MC_MED {c.source}/*.dat @ {c.src_fs} Hz -> {c.target_fs} Hz "
               f"(resample_poly({up},{down}))")
        if c.source != cfg.anchor:
            src += "; NaN-filled when absent"
        ch_meta[c.name] = {"sample_rate_hz": c.target_fs,
                           "shape": [n_seg, seg_sec * c.target_fs],
                           "dtype": "float16", "source": src}

    meta = {
        "entity_id": entity_id,
        "csn": int(csn),
        "mrn": int(row.get("mrn") or 0),
        "source_dataset": "mcmed",
        "n_segments": n_seg,
        "segment_duration_sec": seg_sec,
        "total_duration_hours": round(n_seg * seg_sec / 3600, 2),
        "wave_start_ms": int(time_ms[0]),
        "wave_end_ms": int(time_ms[-1] + seg_sec * 1000),
        "anchor": cfg.anchor,
        "channels": ch_meta,
        "source_stats": src_stats,
        "n_windows_with": n_with,
        "n_windows_dropped_nan": n_dropped,
        "max_nan_ratio": cfg.max_nan_ratio,
        "arrival_ms": int(row.get("arrival_ms") or 0) or None,
        "departure_ms": int(row.get("departure_ms") or 0) or None,
        "stage_b_version": 1,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    return {"entity_id": entity_id, "status": "ok", "n_seg": n_seg,
            "n_dropped": n_dropped}


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
    ap.add_argument("--out-root", type=str, default=OUT_ROOT)
    ap.add_argument("--seg-sec", type=int, default=None,
                    help="Segment duration (default 30 = the canonical variant)")
    ap.add_argument("--anchor", type=str, default=None,
                    help="Anchor source dir that defines the grid (default Pleth)")
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
    log.info(f"Loading cohort parquet: {COHORT_PARQUET}")
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
    log.info(f"Entities to process: {len(rows)}  workers: {args.workers}")

    worker_args = [(r, cfg, args.out_root) for r in rows]
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
