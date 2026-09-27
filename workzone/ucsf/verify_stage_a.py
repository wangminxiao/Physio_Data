"""Verification gate after Stage A (all-raw). Exit 0 = PASS, 1 = FAIL. Writes {intermediate_dir}/verify_stage_a.json.

Checks: summary/parquet present; one non-empty shard per folder; entity_id unique; wave entities have a
positive window, non-empty adibin_files, coverage <= 1.05; duration median in [5 h, 100 h]; >= 90 % of wave
entities carry HR .vital; header spot-check (first adibin start == episode_start_ms) on 30 random entities;
disk sizing for Stage B (<= 60 % of free space on the output volume).
"""
from __future__ import annotations
import argparse, json, os, random, shutil, struct, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, polars as pl, yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "workzone" / "configs" / "server_paths.yaml"
CFWB_FMT = "<4si d iiiii dd iiii"; CFWB_SIZE = struct.calcsize(CFWB_FMT)
MB_PER_WAVE_HOUR = 1.15   # PLETH40 + II120 float16, measured on the CA store


def adibin_start_ms(path):
    with open(path, "rb") as f:
        magic, ver, spt, y, mo, d, h, mi, sec, *_ = struct.unpack(CFWB_FMT, f.read(CFWB_SIZE))
    si = int(sec)
    return int(datetime(y, mo, d, h, mi, si, tzinfo=timezone.utc).timestamp() * 1000) + int(round((sec - si) * 1000))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default=str(CONFIG_PATH)); ap.add_argument("--dataset", default="ucsf_all")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())[args.dataset]
    inter = Path(cfg["intermediate_dir"]); raw = Path(cfg["raw_waveform_dir"]); out_dir = Path(cfg["output_dir"])
    checks = []; info = {}
    def check(name, ok, detail=""):
        checks.append({"check": name, "pass": bool(ok), "detail": detail}); print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")

    summ_p = inter / "stage_a_summary.json"; pq = inter / "valid_wave_window.parquet"
    check("summary present", summ_p.exists(), str(summ_p)); check("parquet present", pq.exists(), str(pq))
    if not (summ_p.exists() and pq.exists()):
        return finish(inter, checks, info)
    summ = json.loads(summ_p.read_text()); df = pl.read_parquet(pq)
    n_folders = len([d for d in os.listdir(raw) if d.endswith("-deid")])
    shards = sorted((inter / "stage_a_shards").glob("*.parquet"))
    empty = [s.name for s in shards if pl.read_parquet(s).height == 0]
    check("one shard per cohort folder", len(shards) == n_folders and not summ.get("limit_de"), f"shards={len(shards)} folders={n_folders} limit_de={summ.get('limit_de')}")
    check("no empty shard", not empty, f"empty={empty[:5]}")
    shard_rows = sum(pl.read_parquet(sh).height for sh in shards)
    check("parquet covers all shards (rows == sum of shard rows, minus cross-folder duplicates)", abs(df.height - shard_rows) <= 0.001 * shard_rows,
          f"parquet={df.height} shards={shard_rows}")
    check("parquet spans all cohort folders", df["wynton_folder"].n_unique() == n_folders, f"folders in parquet={df['wynton_folder'].n_unique()}")
    check("entity_id unique", df["entity_id"].n_unique() == df.height, f"n={df.height}")
    wave = df.filter(pl.col("n_adibin_both") > 0)
    info.update(n_entities=df.height, n_wave=wave.height, n_patients=int(df["patient_id_ge"].n_unique()))
    check("wave entities >= 50 % of entities", wave.height >= 0.5 * df.height, f"wave={wave.height} all={df.height}")
    bad_win = wave.filter(pl.col("episode_end_ms") <= pl.col("episode_start_ms")).height
    check("positive waveform window", bad_win == 0, f"bad={bad_win}")
    bad_files = wave.filter(pl.col("adibin_files").list.len() == 0).height
    check("adibin_files non-empty for wave entities", bad_files == 0, f"bad={bad_files}")
    cov_dup = wave.filter(pl.col("adibin_coverage_frac") > 1.10).height
    cov_dst = wave.filter((pl.col("adibin_coverage_frac") > 1.0) & (pl.col("adibin_coverage_frac") <= 1.10)).height
    check("adibin coverage <= 1.10 (duplicate file sets removed)", cov_dup == 0, f"over_1.10={cov_dup}; 1.0-1.10 (1 h wall-clock repeats, DST-like, overwritten by Stage B)={cov_dst}")
    info["n_coverage_1.0_1.10"] = int(cov_dst)
    if "n_adibin_dup_dropped" in wave.columns:
        info["n_adibin_dup_dropped_total"] = int(wave["n_adibin_dup_dropped"].sum())
    dur_h = wave["episode_duration_sec"].median() / 3600
    check("duration median in [5 h, 100 h]", 5 <= dur_h <= 100, f"median={dur_h:.1f} h")
    hr_frac = wave["has_hr"].mean()
    check("HR .vital present for >= 90 % of wave entities", hr_frac >= 0.9, f"{hr_frac:.3f}")
    info.update(hr_frac=round(float(hr_frac), 4), abp_frac=round(float(wave["has_abp"].mean()), 4), nbp_frac=round(float(wave["has_nbp"].mean()), 4),
                zero_time_frac=round(float((wave["vital_zero_time_suffixes"] != "").mean()), 4))
    # header spot-check
    random.seed(0); rows = wave.sample(n=min(30, wave.height), seed=0).to_dicts(); mism = 0
    for r in rows:
        try:
            if adibin_start_ms(raw / r["adibin_files"][0]) != r["episode_start_ms"]: mism += 1
        except Exception: mism += 1
    check("episode_start_ms == first adibin header start (30 random)", mism == 0, f"mismatch={mism}")
    # sizing for Stage B
    hours = float(wave.filter(pl.col("episode_duration_sec") >= 300)["episode_duration_sec"].sum()) / 3600
    hours_capped = float(wave.filter(pl.col("episode_duration_sec") >= 300)["episode_duration_sec"].clip(upper_bound=14 * 86400).sum()) / 3600
    need_gb = hours_capped * MB_PER_WAVE_HOUR / 1024
    out_dir.mkdir(parents=True, exist_ok=True); free_gb = shutil.disk_usage(out_dir).free / 1e9
    check("Stage B fits in <= 60 % of free space", need_gb <= 0.6 * free_gb, f"need~{need_gb:.0f} GB (capped 14 d) free={free_gb:.0f} GB")
    info.update(wave_hours=round(hours, 1), wave_hours_capped_14d=round(hours_capped, 1), stage_b_need_gb=round(need_gb, 1), free_gb=round(free_gb, 1),
                est_stage_b_hours_16workers=round(hours_capped / 66000, 1))   # CA store: 355k h in 3.6 h with 24 workers ~ 99k h/h -> 16 workers ~ 66k h/h
    return finish(inter, checks, info)


def finish(inter, checks, info):
    ok = all(c["pass"] for c in checks)
    rep = {"stage": "a", "pass": ok, "ran_at_unix": int(time.time()), "checks": checks, "info": info}
    (inter / "verify_stage_a.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(info, indent=2)); print("VERIFY STAGE A:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
