"""Verification gate after Stage B (all-raw). Exit 0 = PASS, 1 = FAIL. Writes {intermediate_dir}/verify_stage_b.json.

Checks: summary present; error+worker_killed <= 1 % and ok >= 90 % of processed entities; on 300 random ok
entities: files present, float16 C-contiguous, shapes (n_seg,1200)/(n_seg,3600), time_ms int64 with 30 s steps,
n_seg == floor(min(duration,14 d)/30), no inf, NaN fraction median < 0.3, II |p99| in (50, 5000) uV, PLETH
p50 in (50, 4095) and non-constant. Reports disk usage.
"""
from __future__ import annotations
import argparse, json, os, random, sys, time
from pathlib import Path
import numpy as np, polars as pl, yaml
sys.path.insert(0, str(Path(__file__).resolve().parent))
from clock import ge_wall_to_grid_ms  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "workzone" / "configs" / "server_paths.yaml"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default=str(CONFIG_PATH)); ap.add_argument("--dataset", default="ucsf_all"); ap.add_argument("--sample", type=int, default=300)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())[args.dataset]
    inter = Path(cfg["intermediate_dir"]); out_dir = Path(cfg["output_dir"])
    checks = []; info = {}
    def check(name, ok, detail=""):
        checks.append({"check": name, "pass": bool(ok), "detail": detail}); print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    summ_p = inter / "stage_b_summary.json"; st_p = inter / "stage_b_status.parquet"
    check("summary present", summ_p.exists(), str(summ_p)); check("status parquet present", st_p.exists(), str(st_p))
    if not (summ_p.exists() and st_p.exists()):
        return finish(inter, checks, info)
    summ = json.loads(summ_p.read_text()); st = pl.read_parquet(st_p); by = summ["by_status"]; info["by_status"] = by
    processed = sum(v for k, v in by.items() if k != "already_done"); ok_n = by.get("ok", 0)
    bad = by.get("error", 0) + by.get("worker_killed", 0)
    check("errors + worker_killed <= 1 %", bad <= 0.01 * max(processed, 1), f"bad={bad} processed={processed}")
    check("ok >= 90 % of processed", ok_n >= 0.9 * max(processed, 1), f"ok={ok_n} processed={processed}")
    a = pl.read_parquet(inter / "valid_wave_window.parquet").select(["entity_id", "episode_duration_sec", "episode_start_ms", "episode_end_ms"]).unique("entity_id", keep="first")
    ok_ids = st.filter(pl.col("status") == "ok")["entity_id"].to_list()
    random.seed(1); sample = random.sample(ok_ids, min(args.sample, len(ok_ids)))
    dur = dict(zip(a["entity_id"].to_list(), a["episode_duration_sec"].to_list()))
    bounds = {r['entity_id']: (int(r['episode_start_ms']), int(r['episode_end_ms'])) for r in a.select(['entity_id', 'episode_start_ms', 'episode_end_ms']).to_dicts()}
    fails = {}; nan_p = []; nan_i = []; ii_p99 = []; pl_p50 = []; n_inf = 0; n_seg_tot = 0
    for eid in sample:
        d = out_dir / eid; errs = []
        try:
            m = json.loads((d / "meta.json").read_text()); n_seg = m["n_seg"]; n_seg_tot += n_seg
            t = np.load(d / "time_ms.npy"); pl_ = np.load(d / "PLETH40.npy", mmap_mode="r"); ii = np.load(d / "II120.npy", mmap_mode="r")
            if t.dtype != np.int64 or len(t) != n_seg or (len(t) > 1 and not np.all(np.diff(t) == 30000)): errs.append("time_ms")
            if pl_.dtype != np.float16 or pl_.shape != (n_seg, 1200) or not pl_.flags["C_CONTIGUOUS"]: errs.append("PLETH40 shape/dtype")
            if ii.dtype != np.float16 or ii.shape != (n_seg, 3600) or not ii.flags["C_CONTIGUOUS"]: errs.append("II120 shape/dtype")
            exp = min(dur.get(eid, 0), 14 * 86400) // 30
            if m.get("time_base") == "utc_continuous" and eid in bounds:   # Stage B v2: duration in real elapsed time
                s0, e0 = bounds[eid]; exp = min(int((ge_wall_to_grid_ms(e0, s0) - s0) // 1000), 14 * 86400) // 30
            if n_seg != exp: errs.append(f"n_seg {n_seg} != expected {exp}")
            idx = np.linspace(0, n_seg - 1, min(n_seg, 200)).astype(int)
            P = np.asarray(pl_[idx], dtype=np.float32); I = np.asarray(ii[idx], dtype=np.float32)
            n_inf += int(np.isinf(P).sum() + np.isinf(I).sum())
            nan_p.append(float(np.isnan(P).mean())); nan_i.append(float(np.isnan(I).mean()))
            if np.isfinite(I).any(): ii_p99.append(float(np.nanpercentile(np.abs(I[np.isfinite(I)]), 99)))
            if np.isfinite(P).any():
                pv = P[np.isfinite(P)]; pl_p50.append(float(np.median(pv)))
                if pv.std() == 0: errs.append("PLETH constant")
        except Exception as e:
            errs.append(f"{type(e).__name__}: {str(e)[:80]}")
        if errs: fails[eid] = errs
    check(f"sampled entities structurally valid ({len(sample)})", len(fails) == 0, f"failing={len(fails)} e.g. {list(fails.values())[:3]}")
    check("no inf values in sample", n_inf == 0, f"inf={n_inf}")
    check("PLETH NaN fraction median < 0.3", np.median(nan_p) < 0.3 if nan_p else False, f"median={np.median(nan_p) if nan_p else None:.3f}")
    check("II NaN fraction median < 0.3", np.median(nan_i) < 0.3 if nan_i else False, f"median={np.median(nan_i) if nan_i else None:.3f}")
    check("II |p99| median in (50, 5000) uV", 50 < np.median(ii_p99) < 5000 if ii_p99 else False, f"median={np.median(ii_p99) if ii_p99 else None:.0f}")
    check("PLETH p50 median in (50, 4095)", 50 < np.median(pl_p50) < 4095 if pl_p50 else False, f"median={np.median(pl_p50) if pl_p50 else None:.0f}")
    n_dirs = sum(1 for p in out_dir.iterdir() if p.is_dir() and (p / "meta.json").exists())
    check("entity dirs with meta >= ok count", n_dirs >= ok_n, f"dirs={n_dirs} ok={ok_n}")
    try:
        used_gb = sum(f.stat().st_size for f in out_dir.rglob("*.npy")) / 1e9
    except Exception: used_gb = None
    info.update(n_ok=ok_n, n_processed=processed, n_segments_total=summ.get("n_segments_total"), sample=len(sample), store_gb=None if used_gb is None else round(used_gb, 1),
                nan_pleth_median=round(float(np.median(nan_p)), 4) if nan_p else None, nan_ii_median=round(float(np.median(nan_i)), 4) if nan_i else None)
    return finish(inter, checks, info)


def finish(inter, checks, info):
    ok = all(c["pass"] for c in checks)
    (inter / "verify_stage_b.json").write_text(json.dumps({"stage": "b", "pass": ok, "ran_at_unix": int(time.time()), "checks": checks, "info": info}, indent=2))
    print(json.dumps(info, indent=2)); print("VERIFY STAGE B:", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
