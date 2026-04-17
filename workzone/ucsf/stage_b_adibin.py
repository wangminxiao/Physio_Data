"""
Stage B - UCSF .adibin extraction.

For each entity in valid_wave_window.parquet:
  1. List `.adibin` files matching DE{pid}_*_{wave_cycle_uid}.adibin under
     {raw_waveform_dir}/{wynton_folder}/DE{patient_id_ge}/{bed_subdir}/
  2. Read II (240 Hz, mV) and SPO2 (240 Hz, %) channels.
  3. Resample to PLETH40 (40 Hz) and II120 (120 Hz).
  4. Place into a 30-second segment grid anchored to episode_start_ms.
  5. Write {output_dir}/{entity_id}/{PLETH40,II120,time_ms}.npy + meta.json.

Out-of-window samples are dropped. Gaps between .adibin files stay as NaN.
Entities with episode duration < min_duration_sec (default 300) are skipped.
Entities longer than max_duration_sec (default 14 days) are truncated.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from scipy.signal import resample_poly

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "workzone" / "ucsf"))
from readers.binfilepy import binfile  # noqa: E402
from readers.binfilepy import constant as binconst  # noqa: E402

CONFIG_PATH = REPO_ROOT / "workzone" / "configs" / "server_paths.yaml"

SRC_RATE = 240
PLETH_TARGET_RATE = 40
II_TARGET_RATE = 120
SEG_SEC = 30
PLETH_SAMPLES_PER_SEG = PLETH_TARGET_RATE * SEG_SEC  # 1200
II_SAMPLES_PER_SEG = II_TARGET_RATE * SEG_SEC        # 3600
PLETH_DOWN = SRC_RATE // PLETH_TARGET_RATE  # 6
II_DOWN = SRC_RATE // II_TARGET_RATE        # 2
MAX_WORKERS = 24


def adibin_start_ms(h) -> int:
    sec_int = int(h.Second)
    sec_frac = h.Second - sec_int
    dt = datetime(h.Year, h.Month, h.Day, h.Hour, h.Minute, sec_int,
                  tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000) + int(round(sec_frac * 1000))


def list_adibin_files(raw_dir: Path, wynton_folder: str, patient_id_ge: str,
                      bed_subdir: str, wave_cycle_uid: str) -> list[Path]:
    bed_dir = raw_dir / wynton_folder / f"DE{patient_id_ge}" / bed_subdir
    if not bed_dir.is_dir():
        return []
    suffix = f"_{wave_cycle_uid}.adibin"
    return sorted(p for p in bed_dir.iterdir() if p.name.endswith(suffix))


GAP_SHORT_VALUES = (-32767, -32768)  # binfilepy raw int16 gap sentinels


_DATA_FMT_DTYPE = {
    binconst.FORMAT_DOUBLE: np.float64,
    binconst.FORMAT_FLOAT: np.float32,
    binconst.FORMAT_SHORT: np.int16,
}


def read_adibin_channels(path: Path, want_titles: tuple[str, ...]):
    """Return (start_ms, dur_ms, {title: float32 array @ 240Hz} or None).

    Memory-efficient reader: bypasses binfile.readChannelData_new which calls
    .tolist() at the end and blows up int16 data to ~15x its size as Python
    ints. Here we use np.memmap to page the interleaved int16 matrix from disk
    and slice only the columns we need, then scale + NaN-mark gaps ourselves.

    Peak RAM per file: ~4x bytes_per_sample * n_samp * len(want_titles).
    For a 14-day 240 Hz int16 file with 2 wanted channels: ~2.5 GB vs ~26 GB.
    """
    bf = binfile.BinFile(str(path), "r")
    bf.open()
    bf.readHeader()
    h = bf.header
    start_ms = adibin_start_ms(h)
    n_samp = h.SamplesPerChannel
    n_chan = h.NChannels
    dur_ms = int(round(n_samp * h.secsPerTick * 1000))
    titles = [ch.Title for ch in bf.channels]
    scales = [(ch.scale, ch.offset) for ch in bf.channels]
    data_fmt = h.DataFormat
    bf.close()  # we only needed the header

    if not all(t in titles for t in want_titles):
        return start_ms, dur_ms, None
    if n_samp <= 0:
        return start_ms, dur_ms, None

    sample_dtype = _DATA_FMT_DTYPE.get(data_fmt)
    if sample_dtype is None:
        return start_ms, dur_ms, None

    header_bytes = binconst.CFWB_SIZE + binconst.CHANNEL_SIZE * n_chan
    # Memmap as (n_samp, n_chan) so per-channel slice is a strided column view.
    mm = np.memmap(str(path), dtype=sample_dtype, mode="r",
                   offset=header_bytes, shape=(n_samp, n_chan))

    out = {}
    try:
        for t in want_titles:
            idx = titles.index(t)
            # np.ascontiguousarray materializes just this one channel, no
            # cross-column overhead. For int16 it's n_samp*2 bytes.
            raw = np.ascontiguousarray(mm[:, idx])
            if sample_dtype == np.int16:
                gap_mask = (raw == -32767) | (raw == -32768)
            else:
                gap_mask = ~np.isfinite(raw)
            scale, offset = scales[idx]
            arr = (scale * (raw.astype(np.float32) + offset)).astype(np.float32)
            if gap_mask.any():
                arr[gap_mask] = np.nan
            out[t] = arr
            del raw
    finally:
        # Release the memmap before the file handle goes out of scope.
        del mm
    return start_ms, dur_ms, out


def resample_to(arr: np.ndarray, down: int) -> np.ndarray:
    nan_mask = ~np.isfinite(arr)
    if nan_mask.any():
        arr = arr.copy()
        arr[nan_mask] = 0.0
    out = resample_poly(arr, 1, down).astype(np.float32)
    if nan_mask.any():
        marker = resample_poly(nan_mask.astype(np.uint8), 1, down)
        out[marker > 0.01] = np.nan
    return out


def _place(grid: np.ndarray, src: np.ndarray, grid_start: int) -> None:
    grid_end = grid_start + len(src)
    src_lo, src_hi = 0, len(src)
    if grid_start < 0:
        src_lo = -grid_start
        grid_start = 0
    if grid_end > len(grid):
        src_hi -= grid_end - len(grid)
        grid_end = len(grid)
    if src_hi > src_lo:
        grid[grid_start:grid_end] = src[src_lo:src_hi].astype(grid.dtype)


def process_entity(row: dict, raw_dir: str, output_dir: str,
                   min_dur: int, max_dur: int) -> dict:
    entity_id = row["entity_id"]
    out_dir = Path(output_dir) / entity_id
    status = {"entity_id": entity_id, "status": "pending", "n_seg": 0,
              "n_adibin_files": 0, "n_files_ok": 0, "n_files_no_chans": 0,
              "n_files_skipped_window": 0}
    try:
        ep_start = int(row["episode_start_ms"])
        ep_end = int(row["episode_end_ms"])
        dur_sec = (ep_end - ep_start) / 1000.0
        if dur_sec < min_dur:
            status["status"] = "skip_short"
            return status
        if dur_sec > max_dur:
            ep_end = ep_start + max_dur * 1000
            dur_sec = max_dur
            status["truncated"] = True

        files = list_adibin_files(Path(raw_dir), row["wynton_folder"],
                                  row["patient_id_ge"], row["bed_subdir"],
                                  row["wave_cycle_uid"])
        status["n_adibin_files"] = len(files)
        if not files:
            status["status"] = "no_adibin"
            return status

        n_seg = int(dur_sec) // SEG_SEC
        if n_seg < 1:
            status["status"] = "skip_short"
            return status

        pleth_grid = np.full(n_seg * PLETH_SAMPLES_PER_SEG, np.nan,
                             dtype=np.float16)
        ii_grid = np.full(n_seg * II_SAMPLES_PER_SEG, np.nan,
                          dtype=np.float16)
        time_ms = ep_start + 1000 * SEG_SEC * np.arange(n_seg, dtype=np.int64)

        for fp in files:
            try:
                f_start_ms, f_dur_ms, chans = read_adibin_channels(
                    fp, ("II", "SPO2"))
            except Exception as e:
                status[f"err_{fp.name}"] = str(e)[:120]
                continue
            if chans is None:
                status["n_files_no_chans"] += 1
                continue
            f_end_ms = f_start_ms + f_dur_ms
            if f_end_ms <= ep_start or f_start_ms >= ep_end:
                status["n_files_skipped_window"] += 1
                continue

            pleth_40 = resample_to(chans["SPO2"], PLETH_DOWN)
            ii_120 = resample_to(chans["II"], II_DOWN)
            f_offset_ms = f_start_ms - ep_start
            _place(pleth_grid, pleth_40,
                   int(round(f_offset_ms * PLETH_TARGET_RATE / 1000)))
            _place(ii_grid, ii_120,
                   int(round(f_offset_ms * II_TARGET_RATE / 1000)))
            status["n_files_ok"] += 1

        pleth_seg = pleth_grid.reshape(n_seg, PLETH_SAMPLES_PER_SEG)
        ii_seg = ii_grid.reshape(n_seg, II_SAMPLES_PER_SEG)

        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "PLETH40.npy", pleth_seg)
        np.save(out_dir / "II120.npy", ii_seg)
        np.save(out_dir / "time_ms.npy", time_ms)

        meta = {
            "entity_id": entity_id,
            "patient_id_ge": row["patient_id_ge"],
            "wave_cycle_uid": row["wave_cycle_uid"],
            "wynton_folder": row["wynton_folder"],
            "bed_subdir": row["bed_subdir"],
            "episode_start_ms": ep_start,
            "episode_end_ms": ep_end,
            "episode_duration_sec": int(dur_sec),
            "n_seg": n_seg,
            "seg_duration_sec": SEG_SEC,
            "channels": {
                "PLETH40": {"rate_hz": PLETH_TARGET_RATE,
                            "samples_per_seg": PLETH_SAMPLES_PER_SEG,
                            "source_title": "SPO2"},
                "II120":   {"rate_hz": II_TARGET_RATE,
                            "samples_per_seg": II_SAMPLES_PER_SEG,
                            "source_title": "II"},
            },
            "n_adibin_files": status["n_adibin_files"],
            "n_files_ok": status["n_files_ok"],
            "has_ca": int(row["has_ca"]),
            "event_time_raw": row["event_time_raw"],
            "encounter_id": row.get("encounter_id"),
            "offset_days": row.get("offset_days"),
            "offset_ge_days": row.get("offset_ge_days"),
            "n_candidate_encounters": row.get("n_candidate_encounters"),
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2,
                                                     default=str))

        status["status"] = "ok"
        status["n_seg"] = n_seg
        status["pleth_nan_ratio"] = round(float(np.isnan(pleth_seg).mean()), 4)
        status["ii_nan_ratio"] = round(float(np.isnan(ii_seg).mean()), 4)
        return status
    except Exception as e:
        status["status"] = "error"
        status["error"] = f"{type(e).__name__}: {e}"
        status["traceback"] = traceback.format_exc()[-500:]
        return status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--min-duration-sec", type=int, default=300)
    ap.add_argument("--max-duration-sec", type=int, default=14 * 24 * 3600)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--workers", type=int, default=12,
                    help=f"max {MAX_WORKERS} (shared cluster cap)")
    ap.add_argument("--entities", default="",
                    help="comma-separated entity_ids; overrides limit")
    ap.add_argument("--no-resume", action="store_true",
                    help="do not skip entities that already have meta.json")
    ap.add_argument("--batch-size", type=int, default=100,
                    help="recycle pool every N entities (bounds OOM blast radius)")
    args = ap.parse_args()

    if args.workers > MAX_WORKERS:
        print(f"clamping workers {args.workers} -> {MAX_WORKERS}")
        args.workers = MAX_WORKERS

    cfg = yaml.safe_load(Path(args.config).read_text())["ucsf"]
    raw_dir = cfg["raw_waveform_dir"]
    output_dir = cfg["output_dir"]
    intermediate_dir = Path(cfg["intermediate_dir"])
    parquet = intermediate_dir / "valid_wave_window.parquet"

    print(f"raw_waveform_dir = {raw_dir}")
    print(f"output_dir       = {output_dir}")
    print(f"parquet          = {parquet}")
    print(f"workers          = {args.workers}")

    df = pl.read_parquet(parquet)
    df = df.unique(subset=["entity_id"], keep="first")

    if args.entities:
        ids = [s.strip() for s in args.entities.split(",") if s.strip()]
        df = df.filter(pl.col("entity_id").is_in(ids))
    elif args.limit:
        df = df.head(args.limit)

    all_rows = df.to_dicts()
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Resume: skip entities whose meta.json already exists on disk.
    statuses = []
    if not args.no_resume:
        rows = []
        n_skipped = 0
        for r in all_rows:
            if (Path(output_dir) / r["entity_id"] / "meta.json").exists():
                statuses.append({"entity_id": r["entity_id"],
                                 "status": "already_done", "n_seg": 0,
                                 "n_adibin_files": 0, "n_files_ok": 0,
                                 "n_files_no_chans": 0,
                                 "n_files_skipped_window": 0})
                n_skipped += 1
            else:
                rows.append(r)
        print(f"resume: skipping {n_skipped} entities with existing meta.json")
    else:
        rows = all_rows

    print(f"processing {len(rows)} entities "
          f"(of {len(all_rows)} total) workers={args.workers} "
          f"batch={args.batch_size}")

    out_status = intermediate_dir / "stage_b_status.parquet"

    def _flush_status():
        try:
            pl.DataFrame(statuses).write_parquet(out_status)
        except Exception as e:
            print(f"  (warning: could not write status parquet: {e})")

    t0 = time.time()
    total = len(rows)
    done = 0

    if args.workers <= 1:
        for i, row in enumerate(rows):
            s = process_entity(row, raw_dir, output_dir,
                               args.min_duration_sec, args.max_duration_sec)
            statuses.append(s)
            done += 1
            if done % 25 == 0 or done == total:
                print(f"  [{done}/{total}] elapsed={time.time()-t0:.1f}s "
                      f"last={s['status']}", flush=True)
                _flush_status()
    else:
        batch_size = max(1, args.batch_size)
        for batch_start in range(0, total, batch_size):
            batch = rows[batch_start:batch_start + batch_size]
            try:
                with ProcessPoolExecutor(max_workers=args.workers) as ex:
                    futs = {ex.submit(process_entity, r, raw_dir, output_dir,
                                      args.min_duration_sec,
                                      args.max_duration_sec): r["entity_id"]
                            for r in batch}
                    for fut in as_completed(futs):
                        eid = futs[fut]
                        try:
                            s = fut.result()
                        except BrokenProcessPool:
                            s = {"entity_id": eid, "status": "worker_killed",
                                 "n_seg": 0, "n_adibin_files": 0,
                                 "n_files_ok": 0, "n_files_no_chans": 0,
                                 "n_files_skipped_window": 0,
                                 "error": "BrokenProcessPool (likely OOM)"}
                        except Exception as e:
                            s = {"entity_id": eid, "status": "error",
                                 "n_seg": 0, "n_adibin_files": 0,
                                 "n_files_ok": 0, "n_files_no_chans": 0,
                                 "n_files_skipped_window": 0,
                                 "error": f"{type(e).__name__}: {str(e)[:200]}"}
                        statuses.append(s)
                        done += 1
                        if done % 25 == 0 or done == total:
                            print(f"  [{done}/{total}] "
                                  f"elapsed={time.time()-t0:.1f}s "
                                  f"last={s['status']}", flush=True)
            except BrokenProcessPool as e:
                # Pool died before as_completed could finish draining.
                finished_ids = {s["entity_id"] for s in statuses}
                for r in batch:
                    if r["entity_id"] not in finished_ids:
                        statuses.append({"entity_id": r["entity_id"],
                                         "status": "worker_killed",
                                         "n_seg": 0, "n_adibin_files": 0,
                                         "n_files_ok": 0,
                                         "n_files_no_chans": 0,
                                         "n_files_skipped_window": 0,
                                         "error": "BrokenProcessPool "
                                                  "(whole-batch; likely OOM)"})
                        done += 1
                print(f"  BATCH {batch_start}: pool broken, marked batch as "
                      f"worker_killed ({e})", flush=True)
            _flush_status()

    elapsed = time.time() - t0

    by_status = {}
    for s in statuses:
        by_status[s["status"]] = by_status.get(s["status"], 0) + 1
    n_seg_total = sum(s["n_seg"] for s in statuses)
    summary = {
        "stage": "b_adibin",
        "ran_at_unix": int(time.time()),
        "elapsed_sec": round(elapsed, 1),
        "n_entities_input": len(rows),
        "by_status": by_status,
        "n_segments_total": n_seg_total,
        "min_duration_sec": args.min_duration_sec,
        "max_duration_sec": args.max_duration_sec,
        "workers": args.workers,
        "output_dir": output_dir,
    }
    out_summary = intermediate_dir / "stage_b_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2))

    pl.DataFrame(statuses).write_parquet(out_status)

    print(f"wrote {out_summary}")
    print(f"wrote {out_status}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
