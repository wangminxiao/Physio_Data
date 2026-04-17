"""
Stage D (labs) - UCSF EHR labs extraction.

Two-phase pipeline:

  Phase 1 (parallel over lab shards):
    For each `.txt` shard in {raw_ehr_dir}/Filtered_Lab_New/:
      - read with csv_repair (quote-aware, latin-1)
      - filter to rows whose Patient_ID is in our cohort's patient list
      - map Lab_Common_Name (upper-cased, comma-stripped) -> var_id via
        var_registry `ucsf_lab_common_names`
      - parse Lab_Collection_Date/Time into EHR-shifted time_ms
      - drop rows where Lab_Value fails to cast to float
    Writes per-shard parquet under {intermediate_dir}/stage_d_labs_shards/
    then concatenates into {intermediate_dir}/stage_d_labs_raw.parquet.

  Phase 2 (sequential over entities):
    For each entity with a Stage B `meta.json`:
      - filter combined DataFrame by patient_id (covers all candidate
        encounters, since multi-encounter entities share patient_id)
      - shift EHR time to GE/wave time:
            time_ms_ge = time_ms_ehr - (offset_ge_days - offset_days) * 86_400_000
      - clip to [episode_start_ms, episode_end_ms]
      - sort by time_ms, write {output_dir}/{entity_id}/labs_events.npy
        as structured (time_ms:i8, var_id:u2, value:f4)
      - update meta.json with `labs` section

Resume:
  - Per-shard parquet exists  -> skip that shard in phase 1
  - Combined parquet exists   -> skip phase 1 concat (unless --redo-phase-1)
  - labs_events.npy exists    -> skip entity in phase 2 (unless --no-resume)

Shared-cluster cap: 24 workers.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np
import polars as pl
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "workzone" / "ucsf"))
from readers.csv_repair import read_dirty_csv  # noqa: E402

CONFIG_PATH = REPO_ROOT / "workzone" / "configs" / "server_paths.yaml"
VAR_REGISTRY_PATH = REPO_ROOT / "indices" / "var_registry.json"

MAX_WORKERS = 24
MS_PER_DAY = 86_400_000

EVENT_DTYPE = np.dtype([
    ("time_ms", np.int64),
    ("var_id", np.uint16),
    ("value", np.float32),
])

LAB_COLS_NEEDED = [
    "Lab_Collection_Date",
    "Lab_Collection_Time",
    "Lab_Value",
    "Lab_Common_Name",
    "Patient_ID",
]

EMPTY_SCHEMA = [
    ("patient_id", pl.Int64),
    ("time_ms_ehr", pl.Int64),
    ("var_id", pl.UInt16),
    ("value", pl.Float32),
]


def build_name_to_varid(registry_path: Path) -> dict[str, int]:
    reg = json.loads(registry_path.read_text())
    entries = reg["variables"] if isinstance(reg, dict) else reg
    out: dict[str, int] = {}
    for entry in entries:
        names = entry.get("ucsf_lab_common_names")
        if not names:
            continue
        for n in names:
            out[n.strip().upper()] = int(entry["id"])
    return out


def process_lab_shard(shard_path_str: str, patient_ids: list[int],
                      name_to_varid: dict[str, int],
                      shards_out_dir_str: str) -> dict:
    """Read one shard, filter + parse, write per-shard parquet.

    Returns only a tiny status dict to keep worker->driver traffic cheap.
    """
    shard_path = Path(shard_path_str)
    out_parquet = Path(shards_out_dir_str) / (shard_path.stem + ".parquet")
    result = {"shard": shard_path.name, "n_raw": 0, "n_filtered": 0,
              "n_mapped": 0, "n_dated": 0, "n_valued": 0,
              "out_parquet": str(out_parquet)}
    try:
        df = read_dirty_csv(str(shard_path), quote_aware=True)
        result["n_raw"] = df.height
        if df.height == 0:
            pl.DataFrame(schema=EMPTY_SCHEMA).write_parquet(out_parquet)
            return result

        df = df.with_columns(
            pl.col("Patient_ID").cast(pl.Int64, strict=False)
        ).filter(pl.col("Patient_ID").is_in(patient_ids))
        result["n_filtered"] = df.height
        if df.height == 0:
            pl.DataFrame(schema=EMPTY_SCHEMA).write_parquet(out_parquet)
            return result

        df = df.select(LAB_COLS_NEEDED)
        df = df.with_columns(
            pl.col("Lab_Common_Name").cast(pl.Utf8)
            .str.strip_chars()
            .str.to_uppercase()
            .alias("name_u")
        )
        df = df.filter(pl.col("name_u").is_in(list(name_to_varid.keys())))
        if df.height == 0:
            pl.DataFrame(schema=EMPTY_SCHEMA).write_parquet(out_parquet)
            return result
        df = df.with_columns(
            pl.col("name_u").replace_strict(name_to_varid,
                                            return_dtype=pl.UInt16).alias("var_id")
        )
        result["n_mapped"] = df.height

        dt_str = pl.concat_str([
            pl.col("Lab_Collection_Date").cast(pl.Utf8).str.replace_all("-", "/"),
            pl.lit(" "),
            pl.when(pl.col("Lab_Collection_Time").cast(pl.Utf8).fill_null("") == "")
              .then(pl.lit("00:00:00.000"))
              .otherwise(pl.col("Lab_Collection_Time").cast(pl.Utf8)),
        ])
        df = df.with_columns(
            dt_str.str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M:%S%.f",
                                strict=False).alias("dt")
        ).filter(pl.col("dt").is_not_null())
        result["n_dated"] = df.height

        df = df.with_columns(
            pl.col("Lab_Value").cast(pl.Utf8)
            .str.strip_chars()
            .str.strip_chars("%")
            .cast(pl.Float64, strict=False).alias("value_f")
        ).filter(
            pl.col("value_f").is_not_null() & pl.col("value_f").is_finite()
        )
        result["n_valued"] = df.height
        if df.height == 0:
            pl.DataFrame(schema=EMPTY_SCHEMA).write_parquet(out_parquet)
            return result

        df = df.with_columns(
            pl.col("dt").dt.timestamp("ms").alias("time_ms_ehr"),
            pl.col("value_f").cast(pl.Float32).alias("value"),
        ).select(["Patient_ID", "time_ms_ehr", "var_id", "value"]).rename(
            {"Patient_ID": "patient_id"}
        )
        df.write_parquet(out_parquet)
        return result
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        result["traceback"] = traceback.format_exc()[-400:]
        try:
            pl.DataFrame(schema=EMPTY_SCHEMA).write_parquet(out_parquet)
        except Exception:
            pass
        return result


def run_phase_1(shard_paths: list[Path], patient_ids: list[int],
                name_to_varid: dict[str, int], workers: int,
                shards_out_dir: Path, combined_parquet: Path,
                status_parquet: Path, redo: bool) -> int:
    shards_out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # resume at shard level
    todo = []
    for p in shard_paths:
        out_p = shards_out_dir / (p.stem + ".parquet")
        if out_p.exists() and not redo:
            continue
        todo.append(p)
    print(f"Phase 1: {len(todo)}/{len(shard_paths)} shards to process "
          f"(resume: skipping {len(shard_paths) - len(todo)})", flush=True)

    statuses: list[dict] = []
    if todo:
        # Use spawn to avoid polars thread-pool deadlock via fork.
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futs = {}
            for p in todo:
                fut = ex.submit(process_lab_shard, str(p), patient_ids,
                                name_to_varid, str(shards_out_dir))
                futs[fut] = p
            done = 0
            for fut in as_completed(futs):
                p = futs[fut]
                try:
                    r = fut.result()
                except BrokenProcessPool:
                    r = {"shard": p.name, "error": "BrokenProcessPool"}
                except Exception as e:
                    r = {"shard": p.name,
                         "error": f"{type(e).__name__}: {e}"}
                statuses.append(r)
                done += 1
                if done % 50 == 0 or done == len(todo):
                    kept = sum(s.get("n_valued", 0) for s in statuses)
                    print(f"  [{done}/{len(todo)}] "
                          f"elapsed={time.time()-t0:.1f}s "
                          f"rows_kept_so_far={kept:,}", flush=True)

    if statuses:
        pl.DataFrame(statuses).write_parquet(status_parquet)

    # Concatenate per-shard parquets into combined
    print(f"Phase 1: concatenating per-shard parquets...", flush=True)
    t_cc = time.time()
    all_parts = sorted(shards_out_dir.glob("*.parquet"))
    if not all_parts:
        pl.DataFrame(schema=EMPTY_SCHEMA).write_parquet(combined_parquet)
        return 0
    combined = pl.concat([pl.read_parquet(p) for p in all_parts],
                         how="vertical_relaxed")
    combined.write_parquet(combined_parquet)
    print(f"  wrote {combined_parquet} rows={combined.height:,} "
          f"({time.time()-t_cc:.1f}s)", flush=True)
    return combined.height


def run_phase_2(combined_parquet: Path, entities_df: pl.DataFrame,
                output_dir: Path, resume: bool,
                status_parquet: Path) -> list[dict]:
    print(f"Phase 2: reading {combined_parquet}", flush=True)
    events_all = pl.read_parquet(combined_parquet)
    print(f"  combined: {events_all.height:,} rows, "
          f"unique patients={events_all['patient_id'].n_unique():,}",
          flush=True)

    t0 = time.time()
    events_by_pat: dict[int, dict] = {}
    if events_all.height > 0:
        grouped = events_all.partition_by("patient_id", as_dict=True)
        for key, g in grouped.items():
            pid = key[0] if isinstance(key, tuple) else key
            events_by_pat[int(pid)] = {
                "time_ms_ehr": g["time_ms_ehr"].to_numpy(),
                "var_id": g["var_id"].to_numpy(),
                "value": g["value"].to_numpy(),
            }
    print(f"  grouped into {len(events_by_pat)} patients "
          f"({time.time()-t0:.1f}s)", flush=True)

    statuses: list[dict] = []
    total = entities_df.height
    done = 0
    t0 = time.time()

    for row in entities_df.iter_rows(named=True):
        eid = row["entity_id"]
        out_dir = output_dir / eid
        meta_path = out_dir / "meta.json"
        npy_path = out_dir / "labs_events.npy"
        status = {"entity_id": eid, "status": "pending",
                  "n_events": 0, "n_matched_patient": 0}
        try:
            if not meta_path.exists():
                status["status"] = "no_meta"
                statuses.append(status); done += 1; continue
            if resume and npy_path.exists():
                status["status"] = "already_done"
                statuses.append(status); done += 1; continue

            meta = json.loads(meta_path.read_text())
            # Clip to admission window (broader than wave window) so Stage E
            # has baseline/recent/future context outside the waveform.
            adm_start = int(row["admission_start_ms"])
            adm_end = int(row["admission_end_ms"])

            pat_id = int(row["patient_id"])
            shift_ms = (int(row["offset_ge_days"]) - int(row["offset_days"])) * MS_PER_DAY

            bucket = events_by_pat.get(pat_id)
            if bucket is None:
                events = np.empty(0, dtype=EVENT_DTYPE)
                np.save(npy_path, events)
                meta["labs"] = {"n_events": 0, "per_var_count": {},
                                "shift_ms": shift_ms,
                                "window": "admission",
                                "admission_start_ms": adm_start,
                                "admission_end_ms": adm_end}
                meta_path.write_text(json.dumps(meta, indent=2, default=str))
                status["status"] = "ok_empty"
                statuses.append(status); done += 1; continue

            status["n_matched_patient"] = int(bucket["time_ms_ehr"].shape[0])
            t_ge = bucket["time_ms_ehr"] - shift_ms
            mask = (t_ge >= adm_start) & (t_ge < adm_end)
            if not mask.any():
                events = np.empty(0, dtype=EVENT_DTYPE)
                np.save(npy_path, events)
                meta["labs"] = {"n_events": 0,
                                "n_matched_patient": status["n_matched_patient"],
                                "per_var_count": {}, "shift_ms": shift_ms,
                                "window": "admission",
                                "admission_start_ms": adm_start,
                                "admission_end_ms": adm_end}
                meta_path.write_text(json.dumps(meta, indent=2, default=str))
                status["status"] = "ok_empty"
                statuses.append(status); done += 1; continue

            t_sel = t_ge[mask]
            v_sel = bucket["var_id"][mask]
            val_sel = bucket["value"][mask]
            order = np.argsort(t_sel, kind="stable")
            events = np.empty(t_sel.shape[0], dtype=EVENT_DTYPE)
            events["time_ms"] = t_sel[order]
            events["var_id"] = v_sel[order]
            events["value"] = val_sel[order]

            np.save(npy_path, events)

            per_var_count: dict[int, int] = {}
            for vid in np.unique(events["var_id"]):
                per_var_count[int(vid)] = int((events["var_id"] == vid).sum())

            meta["labs"] = {
                "n_events": int(events.shape[0]),
                "n_matched_patient": status["n_matched_patient"],
                "per_var_count": {str(k): v for k, v in sorted(per_var_count.items())},
                "shift_ms": shift_ms,
                "window": "admission",
                "admission_start_ms": adm_start,
                "admission_end_ms": adm_end,
            }
            meta_path.write_text(json.dumps(meta, indent=2, default=str))

            status["status"] = "ok"
            status["n_events"] = int(events.shape[0])
            status["per_var_count"] = json.dumps(
                {str(k): v for k, v in sorted(per_var_count.items())}
            )
        except Exception as e:
            status["status"] = "error"
            status["error"] = f"{type(e).__name__}: {e}"
            status["traceback"] = traceback.format_exc()[-400:]

        statuses.append(status)
        done += 1
        if done % 500 == 0 or done == total:
            print(f"  [{done}/{total}] elapsed={time.time()-t0:.1f}s "
                  f"last={status['status']}", flush=True)

    pl.DataFrame(statuses).write_parquet(status_parquet)
    return statuses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--workers", type=int, default=12,
                    help=f"phase-1 shard workers (max {MAX_WORKERS})")
    ap.add_argument("--limit-shards", type=int, default=0,
                    help="phase 1: process first N shards only (debug)")
    ap.add_argument("--entities", default="",
                    help="phase 2: comma-separated entity_ids (debug)")
    ap.add_argument("--phase", choices=["1", "2", "both"], default="both")
    ap.add_argument("--no-resume", action="store_true",
                    help="phase 2: do not skip entities with existing labs_events.npy")
    ap.add_argument("--redo-phase-1", action="store_true",
                    help="reprocess all shards even if per-shard parquet exists")
    args = ap.parse_args()

    if args.workers > MAX_WORKERS:
        print(f"clamping workers {args.workers} -> {MAX_WORKERS}")
        args.workers = MAX_WORKERS

    cfg = yaml.safe_load(Path(args.config).read_text())["ucsf"]
    raw_ehr_dir = Path(cfg["raw_ehr_dir"])
    output_dir = Path(cfg["output_dir"])
    intermediate_dir = Path(cfg["intermediate_dir"])
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    parquet = intermediate_dir / "valid_wave_window.parquet"
    shards_out_dir = intermediate_dir / "stage_d_labs_shards"
    combined_parquet = intermediate_dir / "stage_d_labs_raw.parquet"
    p1_status = intermediate_dir / "stage_d_labs_phase1_status.parquet"
    p2_status = intermediate_dir / "stage_d_labs_phase2_status.parquet"

    name_to_varid = build_name_to_varid(VAR_REGISTRY_PATH)
    print(f"raw_ehr_dir      = {raw_ehr_dir}")
    print(f"output_dir       = {output_dir}")
    print(f"parquet          = {parquet}")
    print(f"shards_out_dir   = {shards_out_dir}")
    print(f"combined_parquet = {combined_parquet}")
    print(f"workers          = {args.workers}")
    print(f"name_to_varid    = {len(name_to_varid)} entries "
          f"-> var_ids {sorted(set(name_to_varid.values()))}", flush=True)

    entities_df = (
        pl.read_parquet(parquet)
        .unique("entity_id", keep="first")
        .select([
            "entity_id", "patient_id", "patient_id_ge",
            "episode_start_ms", "episode_end_ms",
            "admission_start_ms", "admission_end_ms",
            "offset_days", "offset_ge_days",
        ])
    )
    print(f"entities (unique): {entities_df.height}", flush=True)

    if args.phase in ("1", "both"):
        shard_paths = sorted((raw_ehr_dir / "Filtered_Lab_New").glob("*.txt"))
        if args.limit_shards:
            shard_paths = shard_paths[:args.limit_shards]
        print(f"shards: {len(shard_paths)}", flush=True)
        patient_ids = entities_df["patient_id"].unique().to_list()
        print(f"unique cohort patients: {len(patient_ids)}", flush=True)
        t_p1 = time.time()
        n_rows = run_phase_1(shard_paths, patient_ids, name_to_varid,
                             args.workers, shards_out_dir, combined_parquet,
                             p1_status, args.redo_phase_1)
        print(f"Phase 1 complete: {n_rows:,} rows in "
              f"{time.time()-t_p1:.1f}s", flush=True)

    if args.phase in ("2", "both"):
        if not combined_parquet.exists():
            print(f"ERROR: combined parquet missing {combined_parquet}. "
                  f"Run phase 1 first.", flush=True)
            sys.exit(1)
        e_df = entities_df
        if args.entities:
            ids = [s.strip() for s in args.entities.split(",") if s.strip()]
            e_df = e_df.filter(pl.col("entity_id").is_in(ids))
            print(f"Phase 2: filtered to {e_df.height} entities via --entities",
                  flush=True)
        t_p2 = time.time()
        statuses = run_phase_2(combined_parquet, e_df, output_dir,
                               resume=not args.no_resume, status_parquet=p2_status)
        elapsed = time.time() - t_p2

        by_status: dict[str, int] = {}
        for s in statuses:
            by_status[s["status"]] = by_status.get(s["status"], 0) + 1
        n_events_total = sum(s.get("n_events", 0) for s in statuses)
        summary = {
            "stage": "d_labs",
            "ran_at_unix": int(time.time()),
            "elapsed_sec_phase2": round(elapsed, 1),
            "n_entities": e_df.height,
            "by_status": by_status,
            "n_events_total": n_events_total,
            "workers": args.workers,
            "output_dir": str(output_dir),
        }
        out_summary = intermediate_dir / "stage_d_labs_summary.json"
        out_summary.write_text(json.dumps(summary, indent=2))
        print(f"wrote {out_summary}")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
