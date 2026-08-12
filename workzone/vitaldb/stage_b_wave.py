#!/usr/bin/env python3
"""
Stage B - VitalDB waveform extraction (.vital -> canonical npy).

PLETH-anchored: the anchor track (default `SNUADC/PLETH`) is read at its native
rate over [dtstart, dtend], tiled into `seg_sec`-second non-overlap windows aligned
to dtstart — this defines the `time_ms` grid. Every requested channel is produced
onto that grid by per-window `resample_poly`; non-anchor tracks (e.g. `SNUADC/ECG_II`)
are read densely (same dtstart) and NaN-filled per window where absent/gapped.
Windows whose anchor span has > `max_nan_ratio` NaN are dropped (strict = 0%).

Parameterized by a **channel list** and **seg_sec** so one script produces any
UNIPHY FM variant. A channel = `name:source:target_fs[:src_fs]`; multiple channels
may share a source track (read once, resampled to each target). `src_fs` is inferred
from `SOURCE_FS` when omitted.

Defaults reproduce the original (PLETH40 @40, II120 @120, 30 s) output exactly.

Run modes:
  python stage_b_wave.py --limit 3 --workers 2                     # smoke (default 40/120/30s)
  python stage_b_wave.py --entity-id <eid>
  # baseline variant: 10 s @ PLETH125 + PLETH50 + II500
  python stage_b_wave.py --seg-sec 10 --anchor SNUADC/PLETH \\
      --channels PLETH125:SNUADC/PLETH:125,PLETH50:SNUADC/PLETH:50,II500:SNUADC/ECG_II:500 \\
      --out-root /opt/localdata100tb/physio_data/vitaldb_seg10
"""
import argparse
import json
import logging
import multiprocessing as mp
import os
import time
import traceback
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
import polars as pl
import vitaldb
from scipy.signal import resample_poly

OUT_ROOT = "/opt/localdata100tb/physio_data/vitaldb"
COHORT_PARQUET = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/vitaldb/valid_cohort.parquet"
LOG_DIR = "/labs/hulab/mxwang/Physio_Data/workzone/vitaldb/logs"
SUMMARY_JSON = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/vitaldb/stage_b_summary.json"

# Native rate of each raw .vital track (used when a channel spec omits src_fs).
SOURCE_FS = {"SNUADC/PLETH": 500, "SNUADC/ECG_II": 500}
MAX_NAN_RATIO = 0.0        # strict: no NaN allowed in a kept anchor window
DEFAULT_WORKERS = 16


@dataclass(frozen=True)
class ChannelSpec:
    name: str          # canonical output name, e.g. "PLETH40"
    source: str        # raw .vital track name, e.g. "SNUADC/PLETH"
    target_fs: int     # output sample rate, e.g. 40
    src_fs: int        # expected native rate of `source`, e.g. 500


@dataclass(frozen=True)
class ExtractConfig:
    seg_sec: int
    anchor: str                       # source that defines the grid; must be present
    channels: tuple[ChannelSpec, ...]
    max_nan_ratio: float = MAX_NAN_RATIO


# Default = the original hardcoded behavior (byte-identical output).
DEFAULT_CFG = ExtractConfig(
    seg_sec=30, anchor="SNUADC/PLETH",
    channels=(
        ChannelSpec("PLETH40", "SNUADC/PLETH",  40, 500),
        ChannelSpec("II120",   "SNUADC/ECG_II", 120, 500),
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


def _extract_source_streams(vf: "vitaldb.VitalFile", cfg: ExtractConfig):
    """Read each distinct source track once as a dense array at its src_fs.

    Returns (streams: dict[source -> np.ndarray | None], dtstart_s). Absent tracks
    map to None. If the anchor track is absent, returns (None, None).
    """
    names = set(vf.get_track_names())
    if cfg.anchor not in names:
        return None, None
    by_source: dict[str, int] = {}
    for c in cfg.channels:
        by_source.setdefault(c.source, c.src_fs)
    streams: dict[str, np.ndarray | None] = {}
    for source, src_fs in by_source.items():
        if source in names:
            streams[source] = vf.to_numpy([source], interval=1.0 / src_fs).ravel().astype(np.float32)
        else:
            streams[source] = None
    return streams, float(vf.dtstart)


def _window_and_resample(streams: dict, dtstart_s: float, cfg: ExtractConfig):
    """Cut each source into `seg_sec`-second windows aligned to dtstart.

    The anchor source defines the grid: `n_total = len(anchor)//(seg_sec*src_fs)`.
    An anchor window is kept only if its NaN ratio <= cfg.max_nan_ratio (strict 0%).
    Each channel is per-window `resample_poly`d onto its target rate; non-anchor
    channels get NaN for any window that has NaN in its source span.

    Returns (blocks: dict[name -> arr [n, seg_sec*target_fs] float32], time_ms [n],
    n_dropped). Returns (None, None, n_dropped) when no anchor window survives.
    """
    by_source: dict[str, list[ChannelSpec]] = {}
    for c in cfg.channels:
        by_source.setdefault(c.source, []).append(c)
    seg_sec = cfg.seg_sec
    anchor = cfg.anchor
    anchor_src_fs = by_source[anchor][0].src_fs
    anchor_seg_len_src = seg_sec * anchor_src_fs

    anchor_sig = streams[anchor]
    n_total = len(anchor_sig) // anchor_seg_len_src
    if n_total == 0:
        return None, None, 0
    anchor_trunc = anchor_sig[: n_total * anchor_seg_len_src].reshape(n_total, anchor_seg_len_src)

    # keep mask on the anchor span (strict via max_nan_ratio)
    nan_frac = np.isnan(anchor_trunc).mean(axis=1)
    keep = nan_frac <= cfg.max_nan_ratio
    n_dropped = int((~keep).sum())
    if not keep.any():
        return None, None, n_dropped
    kept_idx = np.where(keep)[0]
    time_ms = ((dtstart_s + kept_idx * seg_sec) * 1000).astype(np.int64)

    blocks: dict[str, np.ndarray] = {}
    for source, chans in by_source.items():
        src_fs = chans[0].src_fs
        seg_len_src = seg_sec * src_fs
        sig = streams.get(source)
        # tile this source to n_total windows (trim if long, NaN-pad if short/absent)
        if sig is not None and len(sig) >= n_total * seg_len_src:
            src_trunc = sig[: n_total * seg_len_src].reshape(n_total, seg_len_src)
        elif sig is not None:
            padded = np.concatenate([sig, np.full(n_total * seg_len_src - len(sig),
                                                  np.nan, dtype=np.float32)])
            src_trunc = padded.reshape(n_total, seg_len_src)
        else:
            src_trunc = np.full((n_total, seg_len_src), np.nan, dtype=np.float32)
        src_kept = src_trunc[keep]
        is_anchor = (source == anchor)
        for c in chans:
            L = seg_sec * c.target_fs
            if is_anchor:
                # kept anchor windows are guaranteed within max_nan_ratio -> resample all
                out = np.zeros((src_kept.shape[0], L), dtype=np.float32)
                for i in range(src_kept.shape[0]):
                    out[i] = _resample(src_kept[i], c.target_fs, c.src_fs)[:L]
            else:
                out = np.full((src_kept.shape[0], L), np.nan, dtype=np.float32)
                for i in range(src_kept.shape[0]):
                    if not np.isnan(src_kept[i]).any():
                        out[i] = _resample(src_kept[i], c.target_fs, c.src_fs)[:L]
            blocks[c.name] = out
    return blocks, time_ms, n_dropped


def process_entity(row, cfg: ExtractConfig = DEFAULT_CFG, out_root: str = OUT_ROOT) -> dict:
    eid = row["entity_id"]
    out_dir = Path(out_root) / eid
    meta_path = out_dir / "meta.json"
    seg_sec = cfg.seg_sec

    by_source: dict[str, list[ChannelSpec]] = {}
    for c in cfg.channels:
        by_source.setdefault(c.source, []).append(c)
    if cfg.anchor not in by_source:
        raise ValueError(f"anchor {cfg.anchor!r} not among channel sources {sorted(by_source)}")

    required = [f"{c.name}.npy" for c in cfg.channels] + ["time_ms.npy", "meta.json"]
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

    streams, dtstart = _extract_source_streams(vf, cfg)
    if streams is None:
        return {"entity_id": eid, "status": "no_anchor_track", "anchor": cfg.anchor}
    blocks, time_ms, n_dropped = _window_and_resample(streams, dtstart, cfg)
    if blocks is None:
        return {"entity_id": eid, "status": "no_valid_windows",
                "n_dropped_nan": n_dropped}

    out_dir.mkdir(parents=True, exist_ok=True)
    f16: dict[str, np.ndarray] = {}
    for c in cfg.channels:
        arr = np.ascontiguousarray(blocks[c.name].astype(np.float16))
        assert arr.flags["C_CONTIGUOUS"]
        assert arr.shape == (len(time_ms), seg_sec * c.target_fs)
        f16[c.name] = arr
        np.save(out_dir / f"{c.name}.npy", arr)
    assert len(time_ms) == 1 or np.all(np.diff(time_ms) > 0)
    np.save(out_dir / "time_ms.npy", time_ms.astype(np.int64))

    n_seg = int(len(time_ms))
    ch_meta = {}
    n_with = {}
    for c in cfg.channels:
        up, down = resample_factors(c.target_fs, c.src_fs)
        if c.source == cfg.anchor:
            src = (f"{c.source} @ {c.src_fs} Hz -> {c.target_fs} Hz "
                   f"(resample_poly({up},{down})); strict {cfg.max_nan_ratio:.0%} NaN anchor")
        else:
            src = (f"{c.source} @ {c.src_fs} Hz -> {c.target_fs} Hz "
                   f"(resample_poly({up},{down})); NaN when {c.source} absent or has any gap")
        n_with[c.name] = int(np.sum(~np.all(np.isnan(f16[c.name]), axis=1)))
        ch_meta[c.name] = {"sample_rate_hz": c.target_fs,
                           "shape": list(f16[c.name].shape),
                           "dtype": "float16", "source": src}

    meta = {
        "entity_id": eid,
        "caseid": int(row.get("caseid") or 0),
        "subjectid": int(row.get("subjectid") or 0),
        "source_dataset": "vitaldb",
        "n_segments": n_seg,
        "segment_duration_sec": seg_sec,
        "total_duration_hours": round(n_seg * seg_sec / 3600, 2),
        "wave_start_ms": int(time_ms[0]),
        "wave_end_ms": int(time_ms[-1] + seg_sec * 1000),
        "anchor": cfg.anchor,
        "channels": ch_meta,
        "dtstart_s": float(dtstart),
        "n_windows_with": n_with,
        "n_windows_dropped_nan": int(n_dropped),
        "min_seconds_present": seg_sec,
        "max_nan_ratio": cfg.max_nan_ratio,
        "anestart_ms": int((dtstart + float(row.get("anestart_s") or 0)) * 1000)
                       if row.get("anestart_s") is not None else None,
        "aneend_ms":   int((dtstart + float(row.get("aneend_s")   or 0)) * 1000)
                       if row.get("aneend_s")   is not None else None,
        "stage_b_version": 1,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    return {"entity_id": eid, "status": "ok", "n_seg": n_seg,
            "n_dropped_nan": n_dropped}


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
                    help="Anchor track that defines the grid (default SNUADC/PLETH)")
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

    worker_args = [(row, cfg, args.out_root) for row in rows]
    t0 = time.time()
    results = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, worker_args, chunksize=1)):
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
