"""
Stage C - UCSF .vital extraction.

For each entity with a Stage B `meta.json` on disk:
  1. List `.vital` files matching DE{pid}_*_{wave_cycle_uid}_{suffix}.vital under
     {raw_waveform_dir}/{wynton_folder}/DE{patient_id_ge}/{bed_subdir}/.
  2. For each file whose suffix maps to a var_id in var_registry.json
     (`ucsf_vital_suffixes`), read all samples (value, offset_sec, low, high).
  3. Compute absolute time_ms, clip to [episode_start_ms, episode_end_ms],
     drop invalid values.
  4. Concatenate per entity, sort by time_ms, write
     {output_dir}/{entity_id}/vitals_events.npy as a structured array
     with dtype (time_ms: int64, var_id: uint16, value: float32).
  5. Update meta.json in place with the per-var count and file list.

Same orchestration pattern as Stage B (resume, batched pool, per-future
BrokenProcessPool catch, incremental status parquet).
"""
from __future__ import annotations

import argparse
import json
import struct
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

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "workzone" / "ucsf"))
from readers.vitalfilepy import constant as vconst  # noqa: E402

CONFIG_PATH = REPO_ROOT / "workzone" / "configs" / "server_paths.yaml"
VAR_REGISTRY_PATH = REPO_ROOT / "indices" / "var_registry.json"

MAX_WORKERS = 24

EVENT_DTYPE = np.dtype([
    ("time_ms", np.int64),
    ("var_id", np.uint16),
    ("value", np.float32),
])

# Invalid-value sentinels seen in .vital payloads (very negative placeholders,
# low/high fields leaking into value, etc). Keep liberal bounds; downstream
# applies physio_min/physio_max.
VALUE_MIN, VALUE_MAX = -1e6, 1e6


def build_suffix_map(registry_path: Path) -> dict[str, int]:
    reg = json.loads(registry_path.read_text())
    entries = reg["variables"] if isinstance(reg, dict) else reg
    out: dict[str, int] = {}
    for entry in entries:
        suffixes = entry.get("ucsf_vital_suffixes")
        if not suffixes:
            continue
        for s in suffixes:
            out[s] = int(entry["id"])
    return out


def vital_start_ms(h) -> int:
    sec_int = int(h.Second)
    sec_frac = h.Second - sec_int
    dt = datetime(h.Year, h.Month, h.Day, h.Hour, h.Minute, sec_int,
                  tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000) + int(round(sec_frac * 1000))


def read_vital_file(path: Path):
    """Return (start_ms, values_f64, offsets_f64) or None on parse failure."""
    size = path.stat().st_size
    if size < vconst.VITALHEADER_SIZE:
        return None
    n_samp = (size - vconst.VITALHEADER_SIZE) // (vconst.DOUBLE_SIZE * 4)
    if n_samp <= 0:
        return None

    with open(path, "rb") as f:
        header_bytes = f.read(vconst.VITALHEADER_SIZE)
    # Parse header manually (much faster than VitalFile, no object overhead).
    # Layout: 16s Label, 8s Uom, 8s Unit, 4s Bed, 5i year/mo/d/h/mi, 1d sec
    y, mo, d, hr, mi = struct.unpack_from("iiiii", header_bytes,
                                          offset=16 + 8 + 8 + 4)
    sec = struct.unpack_from("d", header_bytes,
                             offset=16 + 8 + 8 + 4 + 5 * 4)[0]
    try:
        sec_int = int(sec)
        sec_frac = sec - sec_int
        dt = datetime(y, mo, d, hr, mi, sec_int, tzinfo=timezone.utc)
        start_ms = int(dt.timestamp() * 1000) + int(round(sec_frac * 1000))
    except Exception:
        return None

    # Memmap the samples: 4 float64 per sample. We only need columns 0 (value)
    # and 1 (offset_sec); memmap lets us slice without loading low/high.
    mm = np.memmap(str(path), dtype=np.float64, mode="r",
                   offset=vconst.VITALHEADER_SIZE, shape=(n_samp, 4))
    values = np.ascontiguousarray(mm[:, 0])
    offsets = np.ascontiguousarray(mm[:, 1])
    del mm
    return start_ms, values, offsets


def list_vital_files(raw_dir: Path, wynton_folder: str, patient_id_ge: str,
                     bed_subdir: str, wave_cycle_uid: str) -> list[Path]:
    bed_dir = raw_dir / wynton_folder / f"DE{patient_id_ge}" / bed_subdir
    if not bed_dir.is_dir():
        return []
    suffix = f"_{wave_cycle_uid}"
    return sorted(p for p in bed_dir.iterdir()
                  if p.name.endswith(".vital") and suffix in p.name)


def extract_suffix(filename: str, wave_cycle_uid: str) -> str | None:
    # Filename: DE{pid}_{ts14}_{wave_cycle_uid}_{suffix}.vital
    key = f"_{wave_cycle_uid}_"
    if key not in filename:
        return None
    tail = filename.rsplit(key, 1)[1]
    if not tail.endswith(".vital"):
        return None
    return tail[:-len(".vital")]


def process_entity(entity_id: str, entity_row: dict, raw_dir: str,
                   output_dir: str, suffix_map: dict[str, int]) -> dict:
    out_dir = Path(output_dir) / entity_id
    status = {"entity_id": entity_id, "status": "pending",
              "n_vital_files": 0, "n_matched_files": 0, "n_events": 0}
    try:
        meta_path = out_dir / "meta.json"
        if not meta_path.exists():
            status["status"] = "no_meta"
            return status
        meta = json.loads(meta_path.read_text())
        ep_start = int(meta["episode_start_ms"])
        ep_end = int(meta["episode_end_ms"])

        files = list_vital_files(Path(raw_dir), entity_row["wynton_folder"],
                                 entity_row["patient_id_ge"],
                                 entity_row["bed_subdir"],
                                 entity_row["wave_cycle_uid"])
        status["n_vital_files"] = len(files)

        per_var_count: dict[int, int] = {}
        per_suffix_count: dict[str, int] = {}
        chunks_time: list[np.ndarray] = []
        chunks_var: list[np.ndarray] = []
        chunks_val: list[np.ndarray] = []

        for fp in files:
            suffix = extract_suffix(fp.name, entity_row["wave_cycle_uid"])
            if suffix is None or suffix not in suffix_map:
                continue
            var_id = suffix_map[suffix]
            try:
                read = read_vital_file(fp)
            except Exception as e:
                status[f"err_{fp.name[-40:]}"] = str(e)[:120]
                continue
            if read is None:
                continue
            start_ms, values, offsets = read

            time_ms = (start_ms + (offsets * 1000.0)).astype(np.int64)
            # Window clip + validity filter
            mask = ((time_ms >= ep_start) & (time_ms < ep_end)
                    & np.isfinite(values)
                    & (values > VALUE_MIN) & (values < VALUE_MAX))
            if not mask.any():
                continue
            t = time_ms[mask]
            v = values[mask].astype(np.float32)
            chunks_time.append(t)
            chunks_val.append(v)
            chunks_var.append(np.full(t.shape[0], var_id, dtype=np.uint16))
            per_var_count[var_id] = per_var_count.get(var_id, 0) + int(t.shape[0])
            per_suffix_count[suffix] = (per_suffix_count.get(suffix, 0)
                                        + int(t.shape[0]))
            status["n_matched_files"] += 1

        if not chunks_time:
            # Write empty structured array so downstream can rely on file presence
            events = np.empty(0, dtype=EVENT_DTYPE)
            np.save(out_dir / "vitals_events.npy", events)
            meta["vitals"] = {"n_events": 0, "n_vital_files": len(files),
                              "per_var_count": {}, "per_suffix_count": {}}
            meta_path.write_text(json.dumps(meta, indent=2, default=str))
            status["status"] = "ok_empty"
            return status

        time_all = np.concatenate(chunks_time)
        var_all = np.concatenate(chunks_var)
        val_all = np.concatenate(chunks_val)
        order = np.argsort(time_all, kind="stable")
        events = np.empty(time_all.shape[0], dtype=EVENT_DTYPE)
        events["time_ms"] = time_all[order]
        events["var_id"] = var_all[order]
        events["value"] = val_all[order]

        np.save(out_dir / "vitals_events.npy", events)

        meta["vitals"] = {
            "n_events": int(events.shape[0]),
            "n_vital_files": len(files),
            "n_matched_files": status["n_matched_files"],
            "per_var_count": {str(k): v for k, v in sorted(per_var_count.items())},
            "per_suffix_count": dict(sorted(per_suffix_count.items())),
        }
        meta_path.write_text(json.dumps(meta, indent=2, default=str))

        status["status"] = "ok"
        status["n_events"] = int(events.shape[0])
        status["per_var_count"] = json.dumps({str(k): v for k, v in
                                              sorted(per_var_count.items())})
        return status
    except Exception as e:
        status["status"] = "error"
        status["error"] = f"{type(e).__name__}: {e}"
        status["traceback"] = traceback.format_exc()[-500:]
        return status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--workers", type=int, default=12,
                    help=f"max {MAX_WORKERS} (shared cluster cap)")
    ap.add_argument("--entities", default="",
                    help="comma-separated entity_ids; overrides limit")
    ap.add_argument("--no-resume", action="store_true",
                    help="do not skip entities with existing vitals_events.npy")
    ap.add_argument("--batch-size", type=int, default=200,
                    help="recycle pool every N entities")
    args = ap.parse_args()

    if args.workers > MAX_WORKERS:
        print(f"clamping workers {args.workers} -> {MAX_WORKERS}")
        args.workers = MAX_WORKERS

    cfg = yaml.safe_load(Path(args.config).read_text())["ucsf"]
    raw_dir = cfg["raw_waveform_dir"]
    output_dir = cfg["output_dir"]
    intermediate_dir = Path(cfg["intermediate_dir"])
    parquet = intermediate_dir / "valid_wave_window.parquet"

    suffix_map = build_suffix_map(VAR_REGISTRY_PATH)
    print(f"raw_waveform_dir = {raw_dir}")
    print(f"output_dir       = {output_dir}")
    print(f"parquet          = {parquet}")
    print(f"workers          = {args.workers}")
    print(f"suffix_map       = {len(suffix_map)} entries")

    df = pl.read_parquet(parquet).unique("entity_id", keep="first")
    rows_by_id = {r["entity_id"]: r for r in df.to_dicts()}

    # Universe: entities with Stage B meta.json on disk.
    universe = sorted(p.name for p in Path(output_dir).iterdir()
                      if p.is_dir() and (p / "meta.json").exists())
    print(f"Stage B meta.json universe: {len(universe)} entities")

    if args.entities:
        ids = [s.strip() for s in args.entities.split(",") if s.strip()]
        universe = [e for e in universe if e in ids]
    elif args.limit:
        universe = universe[:args.limit]

    statuses: list[dict] = []
    if not args.no_resume:
        remaining = []
        for eid in universe:
            if (Path(output_dir) / eid / "vitals_events.npy").exists():
                statuses.append({"entity_id": eid, "status": "already_done",
                                 "n_vital_files": 0, "n_matched_files": 0,
                                 "n_events": 0})
            else:
                remaining.append(eid)
        print(f"resume: skipping {len(statuses)} entities with existing vitals_events.npy")
    else:
        remaining = list(universe)

    print(f"processing {len(remaining)} entities workers={args.workers} "
          f"batch={args.batch_size}")

    out_status = intermediate_dir / "stage_c_status.parquet"

    def _flush_status():
        try:
            pl.DataFrame(statuses).write_parquet(out_status)
        except Exception as e:
            print(f"  (warning: could not write status parquet: {e})")

    t0 = time.time()
    total = len(remaining)
    done = 0

    if args.workers <= 1:
        for i, eid in enumerate(remaining):
            row = rows_by_id.get(eid)
            if row is None:
                s = {"entity_id": eid, "status": "no_row_in_parquet",
                     "n_vital_files": 0, "n_matched_files": 0, "n_events": 0}
            else:
                s = process_entity(eid, row, raw_dir, output_dir, suffix_map)
            statuses.append(s)
            done += 1
            if done % 25 == 0 or done == total:
                print(f"  [{done}/{total}] elapsed={time.time()-t0:.1f}s "
                      f"last={s['status']}", flush=True)
                _flush_status()
    else:
        batch_size = max(1, args.batch_size)
        for batch_start in range(0, total, batch_size):
            batch = remaining[batch_start:batch_start + batch_size]
            try:
                with ProcessPoolExecutor(max_workers=args.workers) as ex:
                    futs = {}
                    for eid in batch:
                        row = rows_by_id.get(eid)
                        if row is None:
                            statuses.append({"entity_id": eid,
                                             "status": "no_row_in_parquet",
                                             "n_vital_files": 0,
                                             "n_matched_files": 0,
                                             "n_events": 0})
                            done += 1
                            continue
                        fut = ex.submit(process_entity, eid, row, raw_dir,
                                        output_dir, suffix_map)
                        futs[fut] = eid
                    for fut in as_completed(futs):
                        eid = futs[fut]
                        try:
                            s = fut.result()
                        except BrokenProcessPool:
                            s = {"entity_id": eid, "status": "worker_killed",
                                 "n_vital_files": 0, "n_matched_files": 0,
                                 "n_events": 0,
                                 "error": "BrokenProcessPool"}
                        except Exception as e:
                            s = {"entity_id": eid, "status": "error",
                                 "n_vital_files": 0, "n_matched_files": 0,
                                 "n_events": 0,
                                 "error": f"{type(e).__name__}: {str(e)[:200]}"}
                        statuses.append(s)
                        done += 1
                        if done % 50 == 0 or done == total:
                            print(f"  [{done}/{total}] "
                                  f"elapsed={time.time()-t0:.1f}s "
                                  f"last={s['status']}", flush=True)
            except BrokenProcessPool as e:
                finished_ids = {s["entity_id"] for s in statuses}
                for eid in batch:
                    if eid not in finished_ids:
                        statuses.append({"entity_id": eid,
                                         "status": "worker_killed",
                                         "n_vital_files": 0,
                                         "n_matched_files": 0,
                                         "n_events": 0,
                                         "error": "BrokenProcessPool (batch)"})
                        done += 1
                print(f"  BATCH {batch_start}: pool broken ({e})", flush=True)
            _flush_status()

    elapsed = time.time() - t0

    by_status = {}
    for s in statuses:
        by_status[s["status"]] = by_status.get(s["status"], 0) + 1
    n_events_total = sum(s.get("n_events", 0) for s in statuses)
    summary = {
        "stage": "c_vital",
        "ran_at_unix": int(time.time()),
        "elapsed_sec": round(elapsed, 1),
        "n_entities_universe": len(universe),
        "n_entities_processed": total,
        "by_status": by_status,
        "n_events_total": n_events_total,
        "workers": args.workers,
        "output_dir": output_dir,
    }
    out_summary = intermediate_dir / "stage_c_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2))
    pl.DataFrame(statuses).write_parquet(out_status)

    print(f"wrote {out_summary}")
    print(f"wrote {out_status}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
