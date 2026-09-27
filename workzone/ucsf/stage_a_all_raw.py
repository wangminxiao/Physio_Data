"""
Stage A (all-raw) — enumerate EVERY UCSF wave cycle straight from the waveform tree.

Unlike stage_a_wave_windows.py (CA cohort CSV + offset xlsx), this stage needs no
EHR linkage: the waveform window of a wave cycle is defined by its own `.adibin`
headers (start time + samples/240 Hz), which share the GE-shifted clock with the
`.vital` numerics (see datasets/ucsf/explore/RAW_FORMAT.md).

Reads   {raw_waveform_dir}/{YYYY-MM}-deid/DE{pid}/{bed_subdir}/*.adibin|*.vital
        {raw_waveform_dir}/{YYYY-MM}-deid/DE{pid}/MRN-Mapping.csv   (optional extras)
Writes  {intermediate_dir}/stage_a_shards/{folder}.parquet   (one shard per cohort folder; resume unit)
        {intermediate_dir}/valid_wave_window.parquet          (all shards concatenated)
        {intermediate_dir}/stage_a_summary.json

One row per entity = {Patient_ID_GE}_{WaveCycleUID}. Columns consumed by Stage B:
entity_id, patient_id_ge, wave_cycle_uid, wynton_folder, bed_subdir,
episode_start_ms, episode_end_ms, adibin_files (paths relative to raw_waveform_dir).
Stage C consumes vital_files (relative paths) and the zero-time flags.

Only headers are read (68 B + 96 B/channel for .adibin, 56 B for .vital); no sample data.
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import struct
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "workzone" / "configs" / "server_paths.yaml"

MAX_WORKERS = 22  # half of xhu40-n01 (44 cores) — shared lab node
CFWB_FMT = "<4si d iiiii dd iiii"; CFWB_SIZE = struct.calcsize(CFWB_FMT)   # 68
CH_FMT = "<32s32s4d"; CH_SIZE = struct.calcsize(CH_FMT)                    # 96
VH_FMT = "<16s8s8s4s iiiii d"; VH_SIZE = struct.calcsize(VH_FMT)          # 56
SRC_RATE = 240

ABP_SUFFIX_PREFIXES = ("AR1", "AR2", "AR3", "FE1", "FE3")

SCHEMA = {
    "entity_id": pl.Utf8, "patient_id_ge": pl.Utf8, "wave_cycle_uid": pl.Utf8,
    "wynton_folder": pl.Utf8, "bed_subdir": pl.Utf8, "bed_subdirs": pl.Utf8,
    "episode_start_ms": pl.Int64, "episode_end_ms": pl.Int64, "episode_duration_sec": pl.Int64,
    "adibin_total_sec": pl.Float64, "adibin_coverage_frac": pl.Float64,
    "n_adibin": pl.Int32, "n_adibin_empty": pl.Int32, "n_adibin_dup_dropped": pl.Int32, "n_adibin_ii": pl.Int32,
    "n_adibin_spo2": pl.Int32, "n_adibin_both": pl.Int32,
    "adibin_channels": pl.Utf8, "adibin_files": pl.List(pl.Utf8),
    "n_vital": pl.Int32, "vital_suffixes": pl.Utf8, "vital_zero_time_suffixes": pl.Utf8,
    "vital_files": pl.List(pl.Utf8),
    "has_hr": pl.Boolean, "has_spo2pct": pl.Boolean, "has_resp": pl.Boolean,
    "has_abp": pl.Boolean, "has_nbp": pl.Boolean,
    "unit_bed": pl.Utf8, "mrn_wave_start_ms": pl.Int64, "mrn_wave_stop_ms": pl.Int64,
    "bed_in_ms": pl.Int64, "bed_out_ms": pl.Int64,
}


def _cstr(b: bytes) -> str:
    return b.split(b"\0", 1)[0].decode("latin-1")


def _to_ms(y, mo, d, h, mi, s) -> int | None:
    try:
        si = int(s)
        return int(datetime(y, mo, d, h, mi, si, tzinfo=timezone.utc).timestamp() * 1000) \
            + int(round((s - si) * 1000))
    except Exception:
        return None


def _parse_ts(s: str) -> int | None:
    s = (s or "").strip()
    for f in ("%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M"):
        try:
            return int(datetime.strptime(s, f).replace(tzinfo=timezone.utc).timestamp() * 1000)
        except ValueError:
            pass
    return None


def read_adibin_header(path: str) -> dict | None:
    with open(path, "rb") as f:
        hb = f.read(CFWB_SIZE)
        if len(hb) < CFWB_SIZE:
            return None
        magic, ver, spt, y, mo, d, h, mi, sec, trig, nch, nsamp, tch, fmt = struct.unpack(CFWB_FMT, hb)
        if magic != b"CFWB" or nch < 0 or nch > 64:
            return None
        titles = []
        for _ in range(nch):
            cb = f.read(CH_SIZE)
            if len(cb) < CH_SIZE:
                return None
            titles.append(_cstr(struct.unpack(CH_FMT, cb)[0]))
    start_ms = _to_ms(y, mo, d, h, mi, sec)
    if start_ms is None or spt <= 0:
        return None
    return {"start_ms": start_ms, "dur_ms": int(round(nsamp * spt * 1000)), "nsamp": nsamp,
            "fmt": fmt, "titles": titles, "fs": 1.0 / spt}


def read_vital_header(path: str) -> dict | None:
    with open(path, "rb") as f:
        hb = f.read(VH_SIZE)
    if len(hb) < VH_SIZE:
        return None
    lab, uom, unit, bed, y, mo, d, h, mi, sec = struct.unpack(VH_FMT, hb)
    n = (os.path.getsize(path) - VH_SIZE) // 32
    return {"label": _cstr(lab), "uom": _cstr(uom), "zero_time": (y == 0),
            "start_ms": None if y == 0 else _to_ms(y, mo, d, h, mi, sec), "n": int(n)}


def parse_mrn_mapping(path: str) -> dict[str, dict]:
    """WaveCycleUID -> {unit_bed, bed_in_ms, bed_out_ms, wave_start_ms, wave_stop_ms} (earliest/latest)."""
    out: dict[str, dict] = {}
    if not os.path.isfile(path):
        return out
    try:
        with open(path, newline="", encoding="latin-1") as fh:
            r = csv.reader(fh)
            hdr = [c.strip() for c in next(r, [])]
            if "WaveCycleUID" not in hdr:
                return out
            idx = {c: hdr.index(c) for c in hdr}
            def col(row, name):
                i = idx.get(name)
                return row[i].strip() if i is not None and len(row) > i else ""
            for row in r:
                uid = col(row, "WaveCycleUID")
                if not uid:
                    continue
                rec = out.setdefault(uid, {"unit_bed": col(row, "UnitBed") or None,
                                           "bed_in_ms": None, "bed_out_ms": None,
                                           "wave_start_ms": None, "wave_stop_ms": None})
                for key, cname, fn in (("bed_in_ms", "BedTransfer_In", min), ("bed_out_ms", "BedTransfer_Out", max),
                                       ("wave_start_ms", "WaveStartTime", min), ("wave_stop_ms", "WaveStopTime", max)):
                    v = _parse_ts(col(row, cname))
                    if v is not None:
                        rec[key] = v if rec[key] is None else fn(rec[key], v)
    except Exception:
        return out
    return out


def _uid_from_adibin(name: str) -> str:
    return name.rsplit("_", 1)[-1][:-len(".adibin")]


def _uid_suffix_from_vital(name: str) -> tuple[str, str]:
    stem = name[:-len(".vital")]
    head, suffix = stem.rsplit("_", 1)
    uid = head.rsplit("_", 1)[-1]
    return uid, suffix


def scan_patient_dir(task: tuple[str, str, str]) -> list[dict]:
    """(raw_dir, folder, de_name) -> entity rows for that patient directory."""
    raw_dir, folder, de_name = task
    pid = de_name[2:] if de_name.startswith("DE") else de_name
    dp = os.path.join(raw_dir, folder, de_name)
    mrn = parse_mrn_mapping(os.path.join(dp, "MRN-Mapping.csv"))
    groups: dict[str, dict] = {}
    try:
        subs = [s for s in os.scandir(dp) if s.is_dir()]
    except OSError:
        return []
    for s in subs:
        try:
            names = os.listdir(s.path)
        except OSError:
            continue
        rel_sub = f"{folder}/{de_name}/{s.name}"
        for n in names:
            if n.endswith(".adibin"):
                uid = _uid_from_adibin(n)
                g = groups.setdefault(uid, {"adibin": [], "vital": [], "beds": {}})
                g["beds"][s.name] = g["beds"].get(s.name, 0) + 1
                try:
                    h = read_adibin_header(os.path.join(s.path, n))
                except OSError:
                    h = None
                g["adibin"].append((f"{rel_sub}/{n}", h))
            elif n.endswith(".vital"):
                try:
                    uid, suffix = _uid_suffix_from_vital(n)
                except ValueError:
                    continue
                g = groups.setdefault(uid, {"adibin": [], "vital": [], "beds": {}})
                try:
                    vh = read_vital_header(os.path.join(s.path, n))
                except OSError:
                    vh = None
                g["vital"].append((f"{rel_sub}/{n}", suffix, vh))
    rows = []
    for uid, g in groups.items():
        ad_ok = [(p, h) for p, h in g["adibin"] if h and h["nsamp"] > 0 and h["fmt"] == 3]
        n_empty = sum(1 for p, h in g["adibin"] if not h or h["nsamp"] <= 0)
        ad_ok.sort(key=lambda x: (x[1]["start_ms"], x[0]))
        # Identical exports can sit in two bed sub-directories of the same patient (same start, samples,
        # channels). Keep the first path only; otherwise the window is covered twice (coverage ~2.0).
        seen = set(); dedup = []
        for p_, h in ad_ok:
            key = (h["start_ms"], h["nsamp"], len(h["titles"]))
            if key in seen:
                continue
            seen.add(key); dedup.append((p_, h))
        n_dup = len(ad_ok) - len(dedup); ad_ok = dedup
        if ad_ok:
            ep_start = ad_ok[0][1]["start_ms"]
            ep_end = max(h["start_ms"] + h["dur_ms"] for _, h in ad_ok)
            total_ms = sum(h["dur_ms"] for _, h in ad_ok)
            dur_ms = ep_end - ep_start
        else:
            ep_start = ep_end = total_ms = dur_ms = 0
        titles_union = sorted({t for _, h in ad_ok for t in h["titles"]})
        suffixes = sorted({suf for _, suf, _ in g["vital"]})
        zero_suf = sorted({suf for _, suf, vh in g["vital"] if vh and vh["zero_time"]})
        beds = sorted(g["beds"].items(), key=lambda kv: -kv[1])
        m = mrn.get(uid, {})
        rows.append({
            "entity_id": f"{pid}_{uid}", "patient_id_ge": pid, "wave_cycle_uid": uid,
            "wynton_folder": folder,
            "bed_subdir": beds[0][0] if beds else (subs[0].name if subs else ""),
            "bed_subdirs": ";".join(b for b, _ in beds),
            "episode_start_ms": int(ep_start), "episode_end_ms": int(ep_end),
            "episode_duration_sec": int(dur_ms // 1000),
            "adibin_total_sec": total_ms / 1000.0,
            "adibin_coverage_frac": (total_ms / dur_ms) if dur_ms > 0 else 0.0,
            "n_adibin": len(g["adibin"]), "n_adibin_empty": n_empty, "n_adibin_dup_dropped": n_dup,
            "n_adibin_ii": sum(1 for _, h in ad_ok if "II" in h["titles"]),
            "n_adibin_spo2": sum(1 for _, h in ad_ok if "SPO2" in h["titles"]),
            "n_adibin_both": sum(1 for _, h in ad_ok if "II" in h["titles"] and "SPO2" in h["titles"]),
            "adibin_channels": ";".join(titles_union),
            "adibin_files": [p for p, _ in ad_ok],
            "n_vital": len(g["vital"]), "vital_suffixes": ";".join(suffixes),
            "vital_zero_time_suffixes": ";".join(zero_suf),
            "vital_files": sorted(p for p, _, _ in g["vital"]),
            "has_hr": "HR" in suffixes, "has_spo2pct": "SPO2-%" in suffixes, "has_resp": "RESP" in suffixes,
            "has_abp": any(s.startswith(ABP_SUFFIX_PREFIXES) and s.endswith("-S") for s in suffixes),
            "has_nbp": "NBP-S" in suffixes,
            "unit_bed": m.get("unit_bed"), "mrn_wave_start_ms": m.get("wave_start_ms"),
            "mrn_wave_stop_ms": m.get("wave_stop_ms"), "bed_in_ms": m.get("bed_in_ms"), "bed_out_ms": m.get("bed_out_ms"),
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--dataset", default="ucsf_all")
    ap.add_argument("--folders", default="", help="comma-separated cohort folders (default: all *-deid)")
    ap.add_argument("--limit-de", type=int, default=0, help="per folder, first N DE dirs only (smoke test)")
    ap.add_argument("--de-list-file", default="", help="TSV with 'folder<TAB>DE...' rows: scan only those patient dirs (targeted smoke)")
    ap.add_argument("--workers", type=int, default=8, help=f"max {MAX_WORKERS}")
    ap.add_argument("--no-resume", action="store_true", help="rebuild shards that already exist")
    ap.add_argument("--concat-only", action="store_true", help="skip scanning; rebuild valid_wave_window.parquet from ALL existing shards")
    args = ap.parse_args()
    if args.workers > MAX_WORKERS:
        print(f"clamping workers {args.workers} -> {MAX_WORKERS}")
        args.workers = MAX_WORKERS

    cfg = yaml.safe_load(Path(args.config).read_text())[args.dataset]
    raw_dir = cfg["raw_waveform_dir"]
    inter = Path(cfg["intermediate_dir"]); shards = inter / "stage_a_shards"
    shards.mkdir(parents=True, exist_ok=True)

    de_filter: dict[str, set[str]] = {}
    if args.de_list_file:
        for line in Path(args.de_list_file).read_text().splitlines():
            parts = line.split("\t")
            if len(parts) >= 2 and parts[0].endswith("-deid"):
                de_filter.setdefault(parts[0], set()).add(parts[1])
    folders = sorted(d for d in os.listdir(raw_dir) if d.endswith("-deid"))
    if de_filter:
        folders = [f for f in folders if f in de_filter]
    if args.folders:
        want = {f.strip() for f in args.folders.split(",") if f.strip()}
        folders = [f for f in folders if f in want]
    print(f"raw_dir={raw_dir}  folders={len(folders)}  workers={args.workers}  limit_de={args.limit_de}")

    t0 = time.time(); ctx = mp.get_context("spawn"); per_folder = {}
    scan_folders = [] if args.concat_only else folders
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
        for folder in scan_folders:
            shard = shards / f"{folder}.parquet"
            if shard.exists() and not args.no_resume and not args.limit_de and not de_filter:
                per_folder[folder] = ("resumed", pl.read_parquet(shard).height)
                continue
            des = sorted(d for d in os.listdir(os.path.join(raw_dir, folder)) if d.startswith("DE"))
            if de_filter:
                des = [d for d in des if d in de_filter[folder]]
            if args.limit_de:
                des = des[:args.limit_de]
            rows = []
            futs = [ex.submit(scan_patient_dir, (raw_dir, folder, de)) for de in des]
            for fut in as_completed(futs):
                try:
                    rows.extend(fut.result())
                except Exception as e:  # keep going; a bad patient dir must not kill the folder
                    print(f"  [{folder}] worker error: {type(e).__name__}: {str(e)[:120]}", flush=True)
            df = pl.DataFrame(rows, schema=SCHEMA) if rows else pl.DataFrame(schema=SCHEMA)
            df = df.sort(["patient_id_ge", "episode_start_ms"])
            df.write_parquet(shard)
            per_folder[folder] = ("built", df.height)
            print(f"  [{folder}] DE dirs={len(des)} entities={df.height} elapsed={time.time()-t0:.0f}s", flush=True)

    # The parquet is ALWAYS rebuilt from every shard on disk, whatever subset was (re)scanned in this run
    # (a --folders rerun must never shrink the table). Smoke runs with --limit-de keep only their own folders.
    shard_files = sorted(shards.glob("*.parquet")) if not (args.limit_de or de_filter) else [shards / f"{f}.parquet" for f in folders]
    all_df = pl.concat([pl.read_parquet(f) for f in shard_files], how="diagonal_relaxed") if shard_files else pl.DataFrame(schema=SCHEMA)
    per_folder["_concat_shards"] = ("concat", len(shard_files))
    # entity ids are unique per (pid, uid); if a UID somehow appears in two cohort folders, keep the longer one
    all_df = all_df.sort(["entity_id", "episode_duration_sec"], descending=[False, True]).unique(subset=["entity_id"], keep="first")
    out_parquet = inter / "valid_wave_window.parquet"
    all_df.write_parquet(out_parquet)

    with_wave = all_df.filter(pl.col("n_adibin_both") > 0)
    core = with_wave.filter(pl.col("has_hr") & pl.col("has_spo2pct") & pl.col("has_resp"))
    summary = {
        "stage": "a_all_raw", "dataset": args.dataset, "ran_at_unix": int(time.time()),
        "elapsed_sec": round(time.time() - t0, 1), "folders": len(folders), "n_shards_concatenated": len(shard_files), "workers": args.workers,
        "limit_de": args.limit_de,
        "n_entities": all_df.height, "n_patients": all_df["patient_id_ge"].n_unique(),
        "n_with_ii_and_spo2": with_wave.height,
        "n_core_wave_plus_hr_spo2_resp": core.height,
        "n_core_plus_abp": core.filter(pl.col("has_abp")).height,
        "n_core_plus_nbp": core.filter(pl.col("has_nbp")).height,
        "n_core_plus_abp_or_nbp": core.filter(pl.col("has_abp") | pl.col("has_nbp")).height,
        "wave_hours_total": round(float(with_wave["episode_duration_sec"].sum()) / 3600, 1),
        "wave_hours_ge_5min": round(float(with_wave.filter(pl.col("episode_duration_sec") >= 300)["episode_duration_sec"].sum()) / 3600, 1),
        "episode_duration_sec_quantiles": {q: float(with_wave["episode_duration_sec"].quantile(q)) for q in (0.1, 0.5, 0.9, 0.99)} if with_wave.height else {},
        "n_adibin_files_total": int(all_df["n_adibin"].sum()), "n_vital_files_total": int(all_df["n_vital"].sum()),
        "entities_with_any_zero_time_vital": all_df.filter(pl.col("vital_zero_time_suffixes") != "").height,
        "per_folder": per_folder, "output_parquet": str(out_parquet),
    }
    (inter / "stage_a_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_folder"}, indent=2))


if __name__ == "__main__":
    main()
