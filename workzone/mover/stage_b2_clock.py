"""
Stage B2 — MOVER (SIS `mover` and EPIC `mover_epic`) per-case waveform clock correction
(datasets/CLOCK_FIX_PLAN_MIMIC_MOVER.md §2).

Finding: the XML waveform timestamps (`…Z`) were exported with a fixed UTC offset (no DST): ~70 % of devices −7 h
year-round, ~30 % −8 h. The EHR side (Pacific → UTC with DST) is right, so in winter the waveform sits 60 min early
for −7 devices and in summer 60 min late for −8 devices (≈40 % of cases). The device is unknown, so the correction
is measured per case from charted HR (vitals_events var 100, 1–2 min cadence) against the waveform's own rates:

    lag_ppg = argmin_L MAE( charted HR(t), PPG pulse rate over [t+L−5, t+L+5 min] ),  L ∈ −75…+75 min, 5-min steps
    lag_ecg = same against the ECG-derived HR (II120) when usable
    decision:  both refs within 10 min of each other                     → shift = −round60(lag)   (two_refs)
               one ref, MAE(best) < 0.75·MAE(0), best within ±60 ± 10     → shift = −round60(best)  (one_ref)
               best within ±10 min                                        → shift = 0               (aligned)
               otherwise                                                  → shift = 0, clock_unverified
    round60: |L| ≤ 10 → 0 ; 45 ≤ |L| ≤ 75 → ±60 ; else "other" (→ unverified)
    time_ms += shift · 60 000   unless the shift would move the waveform out of its OR / anesthesia window (attribution_risk)

Writes  workzone/outputs/{dataset}/clock_shift.parquet (+ .json summary) and updates time_ms.npy + meta.json
(time_base="utc_ms", clock_shift_min, clock_shift_method, clock_shift_confidence, clock_mae_before/after,
clock_fix_version=1). Idempotent: entities whose meta already has clock_fix_version are skipped unless --no-resume.
Stage E (assemble) must be re-run afterwards (seg_idx / partitions).

  python workzone/mover/stage_b2_clock.py --dataset mover --workers 16 [--limit N] [--entities a,b] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from scipy.signal import butter, filtfilt, find_peaks

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
CONFIG_PATH = REPO_ROOT / "workzone" / "configs" / "server_paths.yaml"
MAX_WORKERS = 22
LAGS = list(range(-75, 76, 5))
BP_PPG = butter(3, [0.5 / 20, 3.0 / 20], btype="band")
BP_ECG = butter(3, [5 / 60, 30 / 60], btype="band")
WINDOW_KEYS = (("or_start_ms", "or_end_ms"), ("anes_start_ms", "anes_end_ms"), ("window_start_ms", "window_end_ms"),
               ("admission_start_ms", "admission_end_ms"))


def rate_seg(x: np.ndarray, fs: float, bp, absval: bool) -> float:
    fin = np.isfinite(x)
    if fin.mean() < 0.9 or np.nanstd(x) < 1e-3:
        return np.nan
    x = np.where(fin, x, np.nanmean(x)); y = filtfilt(bp[0], bp[1], x); y = np.abs(y) if absval else y
    pk, _ = find_peaks(y, distance=int(0.3 * fs), height=np.percentile(y, 98 if absval else 90) * (0.4 if absval else 0.3))
    return pk.size * 2 if 8 <= pk.size <= 120 else np.nan


def lag_scan(t_seg, rate, tc, vc, min_pts=30):
    res = {}
    for L in LAGS:
        errs = []
        for t_ms, v in zip(tc, vc):
            c = t_ms + L * 60000; a, b = np.searchsorted(t_seg, c - 300000), np.searchsorted(t_seg, c + 300000); w = rate[a:b]; w = w[np.isfinite(w)]
            if w.size >= 2:
                errs.append(abs(float(np.median(w)) - v))
        if len(errs) >= min_pts:
            res[L] = float(np.mean(errs))
    if len(res) < 15:
        return None
    L = min(res, key=res.get)
    return {"best": L, "mae_best": res[L], "mae0": res.get(0, float("nan")), "n_lags": len(res)}


def round60(L):
    if L is None:
        return None
    if abs(L) <= 10:
        return 0
    if 45 <= abs(L) <= 75:
        return 60 if L > 0 else -60
    return "other"


ECG_MAX_MAE = 6.0    # OR ECG (electrocautery, lead noise) yields garbage rates often: trust the ECG reference only when tight
PPG_MAX_MAE = 8.0


def decide(rp, re_):
    """-> (shift_min, method, confidence, detail).
    A reference is usable when its best MAE is below an absolute bar (ECG 6 bpm, PPG 8 bpm) and 'clear' when shifting
    helps (MAE_best < 0.75·MAE_0) or it is already tight at 0 (|best| <= 10 min, MAE_0 <= 5). Two clear references that
    disagree -> unverified (no override); one clear reference decides with medium confidence."""
    def band_of(r, bar):
        if r is None or not np.isfinite(r["mae_best"]) or r["mae_best"] > bar:
            return None
        improves = np.isfinite(r["mae0"]) and r["mae_best"] < 0.75 * r["mae0"]
        tight0 = abs(r["best"]) <= 10 and np.isfinite(r["mae0"]) and r["mae0"] <= 6.0
        return {"lag": r["best"], "band": round60(r["best"]), "clear": bool(improves or tight0)}
    p, e = band_of(rp, PPG_MAX_MAE), band_of(re_, ECG_MAX_MAE)
    refs = [(n, r) for n, r in (("ppg", p), ("ecg", e)) if r is not None]
    if not refs:
        lags = " ".join(f"{n} {r['best']}({r['mae_best']:.1f})" for n, r in (("ppg", rp), ("ecg", re_)) if r)
        return 0, "none", "unverified", lags or "no reference"
    clear = [(n, r) for n, r in refs if r["clear"] and r["band"] != "other"]
    detail = " ".join(f"{n} {r['lag']}" for n, r in refs)
    if len(clear) == 2:
        if clear[0][1]["band"] == clear[1][1]["band"]:
            b = clear[0][1]["band"]; return (-b if b else 0), "two_refs", "high", detail
        return 0, "conflict", "unverified", detail
    if len(clear) == 1:
        n, r = clear[0]; b = r["band"]; other = [x for x in refs if x[0] != n]
        conf = "medium" if not other else ("medium" if abs(other[0][1]["lag"] - r["lag"]) <= 10 or other[0][1]["band"] == "other" else "low")
        if conf == "low":
            return 0, f"one_ref_{n}_disputed", "unverified", detail
        return (-b if b else 0), f"one_ref_{n}", conf, detail
    return 0, "unclear", "unverified", detail


def analyze_entity(task: dict) -> dict:
    root, eid, dry, seg_limit = task["root"], task["eid"], task["dry_run"], task["seg_limit"]
    d = Path(root) / eid; out = {"entity_id": eid, "status": "pending"}
    try:
        meta = json.loads((d / "meta.json").read_text())
        if meta.get("clock_fix_version") and not task["no_resume"]:
            out.update(status="already", shift_min=meta.get("clock_shift_min")); return out
        tm = np.load(d / "time_ms.npy"); n = len(tm)
        if n == 0:
            out["status"] = "empty"; return out
        ve = np.load(d / "vitals_events.npy") if (d / "vitals_events.npy").exists() else None
        hr = ve[ve["var_id"] == 100] if ve is not None and ve.size else None
        out["n_seg"] = int(n); out["n_hr"] = int(hr.size) if hr is not None else 0
        idx = np.arange(0, min(n, seg_limit)); t_seg = tm[idx]
        rp = re_ = None
        if hr is not None and hr.size >= 60:
            tc = hr["time_ms"].astype(np.int64); vc = hr["value"].astype(np.float64); ok = (vc > 25) & (vc < 220); tc, vc = tc[ok], vc[ok]
            pl_ = np.load(d / "PLETH40.npy", mmap_mode="r")
            pr = np.array([rate_seg(np.asarray(pl_[i], dtype=np.float32), 40.0, BP_PPG, False) for i in idx])
            out["ppg_rate_frac"] = round(float(np.isfinite(pr).mean()), 3)
            if np.isfinite(pr).sum() >= 60:
                rp = lag_scan(t_seg, pr, tc, vc)
            if (d / "II120.npy").exists():
                ii = np.load(d / "II120.npy", mmap_mode="r")
                er = np.array([rate_seg(np.asarray(ii[i], dtype=np.float32), 120.0, BP_ECG, True) for i in idx])
                out["ecg_rate_frac"] = round(float(np.isfinite(er).mean()), 3)
                if np.isfinite(er).sum() >= 60:
                    re_ = lag_scan(t_seg, er, tc, vc)
        else:
            out["status"] = "no_vitals"
        shift, method, conf, detail = decide(rp, re_)
        out.update(lag_ppg=None if rp is None else rp["best"], mae_ppg_best=None if rp is None else round(rp["mae_best"], 2), mae_ppg0=None if rp is None else round(rp["mae0"], 2),
                   lag_ecg=None if re_ is None else re_["best"], mae_ecg_best=None if re_ is None else round(re_["mae_best"], 2), mae_ecg0=None if re_ is None else round(re_["mae0"], 2),
                   shift_min=int(shift), method=method, confidence=conf, detail=detail)
        # attribution check against the OR / anesthesia window recorded in meta
        win = next(((meta.get(a), meta.get(b)) for a, b in WINDOW_KEYS if meta.get(a) and meta.get(b)), None)
        if win:
            ws, we = int(win[0]) - 2 * 3600000, int(win[1]) + 2 * 3600000
            before = ws <= int(tm[0]) and int(tm[-1]) <= we; after = ws <= int(tm[0]) + shift * 60000 and int(tm[-1]) + shift * 60000 <= we
            out["in_window_before"], out["in_window_after"] = bool(before), bool(after)
            out["window_start_minus_wave_start_min_before"] = round((int(win[0]) - int(tm[0])) / 60000, 1)
            if before and not after and shift:
                out.update(shift_min=0, method=method + "+attribution_risk", confidence="unverified"); shift = 0
        out["status"] = "ok" if out.get("status") == "pending" else out["status"]
        if dry or shift == 0 and out["status"] != "ok":
            pass
        if not dry:
            if shift:
                np.save(d / "time_ms.npy", (tm + shift * 60000).astype(np.int64))
                for k in ("wave_start_ms", "wave_end_ms"):
                    if k in meta:
                        meta[k] = int(meta[k]) + shift * 60000
            meta.update(time_base="utc_ms", clock_shift_min=int(shift), clock_shift_method=method, clock_shift_confidence=conf,
                        clock_shift_detail=detail, clock_fix_version=1,
                        clock_mae_before=out.get("mae_ecg0") if re_ is not None else out.get("mae_ppg0"),
                        clock_mae_after=out.get("mae_ecg_best") if re_ is not None else out.get("mae_ppg_best"))
            (d / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
        return out
    except Exception as ex:  # noqa: BLE001
        out.update(status="error", error=f"{type(ex).__name__}: {str(ex)[:160]}"); return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG_PATH)); ap.add_argument("--dataset", default="mover", help="mover | mover_epic")
    ap.add_argument("--root", default=None, help="override output_dir (scratch)")
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--entities", default="")
    ap.add_argument("--seg-limit", type=int, default=2880, help="segments analysed per entity (24 h)")
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--no-resume", action="store_true"); ap.add_argument("--batch-size", type=int, default=200)
    args = ap.parse_args()
    args.workers = min(args.workers, MAX_WORKERS)
    cfg = yaml.safe_load(Path(args.config).read_text())[args.dataset]
    root = Path(args.root or cfg["output_dir"]); inter = REPO_ROOT / "workzone" / "outputs" / args.dataset; inter.mkdir(parents=True, exist_ok=True)
    ids = [s.strip() for s in args.entities.split(",") if s.strip()] or sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "meta.json").exists() and (p / "time_ms.npy").exists())
    if args.limit:
        ids = ids[:args.limit]
    print(f"dataset={args.dataset} root={root} entities={len(ids)} workers={args.workers} dry_run={args.dry_run}", flush=True)
    tasks = [dict(root=str(root), eid=e, dry_run=args.dry_run, seg_limit=args.seg_limit, no_resume=args.no_resume) for e in ids]
    rows = []; t0 = time.time(); done = 0; ctx = mp.get_context("spawn")
    for b0 in range(0, len(tasks), args.batch_size):
        batch = tasks[b0:b0 + args.batch_size]
        try:
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
                futs = {ex.submit(analyze_entity, t): t["eid"] for t in batch}
                for fut in as_completed(futs):
                    try: r = fut.result()
                    except BrokenProcessPool: r = {"entity_id": futs[fut], "status": "worker_killed"}
                    except Exception as e: r = {"entity_id": futs[fut], "status": "error", "error": str(e)[:160]}
                    rows.append(r); done += 1
                    if done % 200 == 0 or done == len(tasks):
                        print(f"  [{done}/{len(tasks)}] {time.time()-t0:.0f}s", flush=True)
        except BrokenProcessPool as e:
            fin = {r["entity_id"] for r in rows}; rows.extend({"entity_id": t["eid"], "status": "worker_killed"} for t in batch if t["eid"] not in fin); print("pool broken:", e)
    df = pl.DataFrame(rows, infer_schema_length=None)
    tag = ("_dryrun" if args.dry_run else "") + (f"_{Path(args.root).name}" if args.root else "")   # scratch runs never overwrite the production report
    out_p = inter / f"clock_shift{tag}.parquet"; df.write_parquet(out_p)
    by = df.group_by("status").len().to_dicts() if df.height else []
    ok = df.filter(pl.col("status") == "ok") if "status" in df.columns else df
    summ = {"dataset": args.dataset, "root": str(root), "dry_run": args.dry_run, "n": len(rows), "by_status": {r["status"]: r["len"] for r in by},
            "shift_counts": ok.group_by("shift_min").len().sort("shift_min").to_dicts() if ok.height and "shift_min" in ok.columns else [],
            "confidence_counts": ok.group_by("confidence").len().to_dicts() if ok.height and "confidence" in ok.columns else [],
            "method_counts": ok.group_by("method").len().sort("len", descending=True).head(12).to_dicts() if ok.height and "method" in ok.columns else [],
            "elapsed_sec": round(time.time() - t0, 1), "ran_at": time.strftime("%Y-%m-%d %H:%M")}
    (inter / f"clock_shift_summary{tag}.json").write_text(json.dumps(summ, indent=2, default=str))
    print(json.dumps(summ, indent=2, default=str)); print("wrote", out_p)


if __name__ == "__main__":
    main()
