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

v4 (2026-09-27, --season-binary / --only-unverified): the device hypothesis fixes the SIGN of any shift by season —
standard time -> the waveform can only be 60 min early (+60), DST -> only 60 min late (−60); 99.2 % (SIS) / 98.7 % (EPIC)
of the v3 decisions obey this. Cases left unverified by the free scan are re-tested as a binary hypothesis H0 "aligned"
(lags −10…+10) vs H1 "the one season-allowed hour" (lags ±50…70): a reference votes when its preferred hypothesis has
MAE <= 10 (PPG) / 8 (ECG) bpm and a margin (MAE < 0.8 x other, or Pearson r >= 0.5 with r gain >= 0.15, criteria not
contradicting). Two votes -> high, one -> medium, none/conflict -> unverified (kept, excluded from tasks/ via
`exclude_meta`). --validate-decided replays the test on the already-decided cases from their pre-fix clock and fails
(exit 1) below --min-agreement.

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
from datetime import datetime
from zoneinfo import ZoneInfo
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
LA = ZoneInfo("America/Los_Angeles")
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

# ---- v4 season-constrained binary test
H0_LAGS = (-10, -5, 0, 5, 10)
SEASON_PPG_MAX_MAE = 10.0
SEASON_ECG_MAX_MAE = 8.0
SEASON_MAE_RATIO = 0.8
SEASON_CORR_GAIN = 0.15
SEASON_CORR_MIN = 0.5
SEASON_MIN_PTS = 15
SEASON_MIN_HR = 20


def is_dst_ms(ms: int) -> bool:
    return bool(datetime.fromtimestamp(int(ms) / 1000, tz=LA).dst())


def metrics_at_lags(t_seg, rate, tc, vc, lags, min_pts=SEASON_MIN_PTS):
    """lowest-MAE lag among `lags` -> {lag, mae, corr, n} or None (charted HR at t vs median rate over [t+L-5, t+L+5 min])."""
    best = None
    for L in lags:
        pairs = []
        for t_ms, v in zip(tc, vc):
            c = t_ms + L * 60000; a, b = np.searchsorted(t_seg, c - 300000), np.searchsorted(t_seg, c + 300000); w = rate[a:b]; w = w[np.isfinite(w)]
            if w.size >= 2:
                pairs.append((float(np.median(w)), float(v)))
        if len(pairs) < min_pts:
            continue
        p = np.asarray(pairs); mae = float(np.mean(np.abs(p[:, 0] - p[:, 1])))
        corr = float(np.corrcoef(p[:, 0], p[:, 1])[0, 1]) if p[:, 1].std() > 2.0 and p[:, 0].std() > 1e-6 else float("nan")
        if best is None or mae < best["mae"]:
            best = {"lag": int(L), "mae": mae, "corr": corr, "n": len(pairs)}
    return best


def decide_season(refs: dict, allowed_shift: int):
    """-> (shift_min, method, confidence, detail); refs = {"ppg": (h0, h1), "ecg": (h0, h1)} from metrics_at_lags."""
    votes = []; detail = []
    for name, bar in (("ppg", SEASON_PPG_MAX_MAE), ("ecg", SEASON_ECG_MAX_MAE)):
        h0, h1 = refs.get(name, (None, None))
        if h0 is None or h1 is None:
            continue
        detail.append(f"{name} mae0={h0['mae']:.1f}/r0={h0['corr']:.2f} mae60={h1['mae']:.1f}/r60={h1['corr']:.2f} n={h0['n']}/{h1['n']}")
        mae_pref = 0 if h0["mae"] < SEASON_MAE_RATIO * h1["mae"] else 1 if h1["mae"] < SEASON_MAE_RATIO * h0["mae"] else None
        r0, r1 = h0["corr"], h1["corr"]; corr_pref = None
        if np.isfinite(r0) and r0 >= SEASON_CORR_MIN and (not np.isfinite(r1) or r0 - r1 >= SEASON_CORR_GAIN):
            corr_pref = 0
        elif np.isfinite(r1) and r1 >= SEASON_CORR_MIN and (not np.isfinite(r0) or r1 - r0 >= SEASON_CORR_GAIN):
            corr_pref = 1
        if mae_pref is not None and corr_pref is not None and mae_pref != corr_pref:
            continue                                     # the two criteria contradict -> this reference abstains
        pref = mae_pref if mae_pref is not None else corr_pref
        if pref is None or (h0 if pref == 0 else h1)["mae"] > bar:
            continue
        votes.append((name, 0 if pref == 0 else allowed_shift))
    d = " ".join(detail) or "no reference"
    if not votes:
        return 0, "season_binary_undecided", "unverified", d
    if len({s for _, s in votes}) > 1:
        return 0, "season_binary_conflict", "unverified", d
    return votes[0][1], "season_binary_" + "+".join(n for n, _ in votes), ("high" if len(votes) == 2 else "medium"), d


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
    season, validate = bool(task.get("season_binary")), bool(task.get("validate"))
    d = Path(root) / eid; out = {"entity_id": eid, "status": "pending"}
    try:
        meta = json.loads((d / "meta.json").read_text())
        if meta.get("clock_fix_version") and not task["no_resume"] and not season:
            out.update(status="already", shift_min=meta.get("clock_shift_min")); return out
        tm = np.load(d / "time_ms.npy"); n = len(tm)
        if n == 0:
            out["status"] = "empty"; return out
        undo = int(meta.get("clock_shift_min") or 0) if validate else 0      # validation replays the test on the pre-fix clock
        tm_base = tm - undo * 60000
        if validate:
            out["expected_shift_min"] = int(meta.get("clock_shift_min") or 0); out["expected_confidence"] = meta.get("clock_shift_confidence")
        ve = np.load(d / "vitals_events.npy") if (d / "vitals_events.npy").exists() else None
        hr = ve[ve["var_id"] == 100] if ve is not None and ve.size else None
        out["n_seg"] = int(n); out["n_hr"] = int(hr.size) if hr is not None else 0
        idx = np.arange(0, min(n, seg_limit)); t_seg = tm_base[idx]
        rp = re_ = None; pr = er = None; tc = vc = None
        min_hr = SEASON_MIN_HR if season else 60
        min_rates = SEASON_MIN_PTS if season else 60
        if hr is not None and hr.size >= min_hr:
            tc = hr["time_ms"].astype(np.int64); vc = hr["value"].astype(np.float64); ok = (vc > 25) & (vc < 220); tc, vc = tc[ok], vc[ok]
            pl_ = np.load(d / "PLETH40.npy", mmap_mode="r")
            pr = np.array([rate_seg(np.asarray(pl_[i], dtype=np.float32), 40.0, BP_PPG, False) for i in idx])
            out["ppg_rate_frac"] = round(float(np.isfinite(pr).mean()), 3)
            if np.isfinite(pr).sum() < min_rates:
                pr = None
            elif not season:
                rp = lag_scan(t_seg, pr, tc, vc)
            if (d / "II120.npy").exists():
                ii = np.load(d / "II120.npy", mmap_mode="r")
                er = np.array([rate_seg(np.asarray(ii[i], dtype=np.float32), 120.0, BP_ECG, True) for i in idx])
                out["ecg_rate_frac"] = round(float(np.isfinite(er).mean()), 3)
                if np.isfinite(er).sum() < min_rates:
                    er = None
                elif not season:
                    re_ = lag_scan(t_seg, er, tc, vc)
        else:
            out["status"] = "no_vitals"
        dst = None; allowed = None
        if season:
            dst = is_dst_ms(int(tm_base[0])); allowed = -60 if dst else 60
            h1_lags = tuple(range(-70, -45, 5)) if allowed == 60 else tuple(range(50, 75, 5))
            refs = {}
            for name, rate in (("ppg", pr), ("ecg", er)):
                if rate is not None:
                    refs[name] = (metrics_at_lags(t_seg, rate, tc, vc, H0_LAGS), metrics_at_lags(t_seg, rate, tc, vc, h1_lags))
            shift, method, conf, detail = decide_season(refs, allowed)
            out.update(dst=bool(dst), allowed_shift_min=int(allowed), shift_min=int(shift), method=method, confidence=conf, detail=detail)
            for name in ("ppg", "ecg"):
                h0, h1 = refs.get(name, (None, None))
                out[f"mae_{name}0"] = None if h0 is None else round(h0["mae"], 2); out[f"mae_{name}_h1"] = None if h1 is None else round(h1["mae"], 2)
                out[f"corr_{name}0"] = None if h0 is None or not np.isfinite(h0["corr"]) else round(h0["corr"], 3)
                out[f"corr_{name}_h1"] = None if h1 is None or not np.isfinite(h1["corr"]) else round(h1["corr"], 3)
            ref_used = "ecg" if "ecg" in method else ("ppg" if "ppg" in method else None)
            mae_before = out.get(f"mae_{ref_used}0") if ref_used else None
            mae_after = (out.get(f"mae_{ref_used}0") if shift == 0 else out.get(f"mae_{ref_used}_h1")) if ref_used else None
        else:
            shift, method, conf, detail = decide(rp, re_)
            out.update(lag_ppg=None if rp is None else rp["best"], mae_ppg_best=None if rp is None else round(rp["mae_best"], 2), mae_ppg0=None if rp is None else round(rp["mae0"], 2),
                       lag_ecg=None if re_ is None else re_["best"], mae_ecg_best=None if re_ is None else round(re_["mae_best"], 2), mae_ecg0=None if re_ is None else round(re_["mae0"], 2),
                       shift_min=int(shift), method=method, confidence=conf, detail=detail)
            mae_before = out.get("mae_ecg0") if re_ is not None else out.get("mae_ppg0")
            mae_after = out.get("mae_ecg_best") if re_ is not None else out.get("mae_ppg_best")
        # attribution check against the OR / anesthesia window recorded in meta (on the clock the shift is applied to)
        win = next(((meta.get(a), meta.get(b)) for a, b in WINDOW_KEYS if meta.get(a) and meta.get(b)), None)
        if win:
            ws, we = int(win[0]) - 2 * 3600000, int(win[1]) + 2 * 3600000
            before = ws <= int(tm_base[0]) and int(tm_base[-1]) <= we; after = ws <= int(tm_base[0]) + shift * 60000 and int(tm_base[-1]) + shift * 60000 <= we
            out["in_window_before"], out["in_window_after"] = bool(before), bool(after)
            out["window_start_minus_wave_start_min_before"] = round((int(win[0]) - int(tm_base[0])) / 60000, 1)
            if before and not after and shift:
                out.update(shift_min=0, method=method + "+attribution_risk", confidence="unverified"); shift = 0
        out["status"] = "ok" if out.get("status") == "pending" else out["status"]
        if validate:
            out["agree"] = (int(shift) == out["expected_shift_min"]) if out["confidence"] != "unverified" else None
            return out
        if not dry:
            if shift:
                np.save(d / "time_ms.npy", (tm + shift * 60000).astype(np.int64))
                for k in ("wave_start_ms", "wave_end_ms"):
                    if k in meta:
                        meta[k] = int(meta[k]) + shift * 60000
            meta.update(time_base="utc_ms", clock_shift_min=int(shift), clock_shift_method=method, clock_shift_confidence=conf,
                        clock_shift_detail=detail, clock_fix_version=2 if season else 1, clock_mae_before=mae_before, clock_mae_after=mae_after)
            if season:
                meta["clock_season_prior"] = f"{'dst' if dst else 'standard_time'}:allowed_shift_min={allowed}"
                meta["clock_shift_v3_method"] = meta.get("clock_shift_v3_method") or task.get("prev_method")
            (d / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
        return out
    except Exception as ex:  # noqa: BLE001
        out.update(status="error", error=f"{type(ex).__name__}: {str(ex)[:160]}"); return out


def select_by_confidence(root: Path, ids: list[str], want) -> tuple[list[str], dict[str, str]]:
    keep, prev = [], {}
    for e in ids:
        try:
            m = json.loads((root / e / "meta.json").read_text())
        except Exception:  # noqa: BLE001
            continue
        if want(m.get("clock_shift_confidence")):
            keep.append(e); prev[e] = m.get("clock_shift_method")
    return keep, prev


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG_PATH)); ap.add_argument("--dataset", default="mover", help="mover | mover_epic")
    ap.add_argument("--root", default=None, help="override output_dir (scratch)")
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--entities", default="")
    ap.add_argument("--seg-limit", type=int, default=2880, help="segments analysed per entity (24 h)")
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--no-resume", action="store_true"); ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--season-binary", action="store_true", help="v4 season-constrained binary test instead of the free lag scan")
    ap.add_argument("--only-unverified", action="store_true", help="re-analyse only meta.clock_shift_confidence == 'unverified' entities (implies --season-binary --no-resume); merges into clock_shift.parquet")
    ap.add_argument("--validate-decided", action="store_true", help="dry-run the season test on already-decided entities from their pre-fix clock; exit 1 below --min-agreement / --min-coverage")
    ap.add_argument("--min-agreement", type=float, default=0.98); ap.add_argument("--min-coverage", type=float, default=0.6)
    args = ap.parse_args()
    args.workers = min(args.workers, MAX_WORKERS)
    if args.only_unverified or args.validate_decided:
        args.season_binary = True; args.no_resume = True
    if args.validate_decided:
        args.dry_run = True
    cfg = yaml.safe_load(Path(args.config).read_text())[args.dataset]
    root = Path(args.root or cfg["output_dir"]); inter = REPO_ROOT / "workzone" / "outputs" / args.dataset; inter.mkdir(parents=True, exist_ok=True)
    ids = [s.strip() for s in args.entities.split(",") if s.strip()] or sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "meta.json").exists() and (p / "time_ms.npy").exists())
    prev_method: dict[str, str] = {}
    if args.only_unverified:
        ids, prev_method = select_by_confidence(root, ids, lambda c: c == "unverified")
    elif args.validate_decided:
        ids, prev_method = select_by_confidence(root, ids, lambda c: c in ("high", "medium"))
    if args.limit:
        ids = ids[:args.limit]
    mode = "validate_season" if args.validate_decided else "season_unverified" if args.only_unverified else "season" if args.season_binary else "free_scan"
    print(f"dataset={args.dataset} root={root} entities={len(ids)} workers={args.workers} dry_run={args.dry_run} mode={mode}", flush=True)
    tasks = [dict(root=str(root), eid=e, dry_run=args.dry_run, seg_limit=args.seg_limit, no_resume=args.no_resume, season_binary=args.season_binary,
                  validate=args.validate_decided, prev_method=prev_method.get(e)) for e in ids]
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
    tag = ("_dryrun" if args.dry_run and not args.validate_decided else "") + (f"_{Path(args.root).name}" if args.root else "")   # scratch runs never overwrite the production report
    tag += "_validate_season" if args.validate_decided else "_season" if args.only_unverified else ""
    out_p = inter / f"clock_shift{tag}.parquet"; df.write_parquet(out_p)
    by = df.group_by("status").len().to_dicts() if df.height else []
    ok = df.filter(pl.col("status") == "ok") if "status" in df.columns else df
    summ = {"dataset": args.dataset, "root": str(root), "mode": mode, "dry_run": args.dry_run, "n": len(rows), "by_status": {r["status"]: r["len"] for r in by},
            "shift_counts": ok.group_by("shift_min").len().sort("shift_min").to_dicts() if ok.height and "shift_min" in ok.columns else [],
            "confidence_counts": ok.group_by("confidence").len().to_dicts() if ok.height and "confidence" in ok.columns else [],
            "method_counts": ok.group_by("method").len().sort("len", descending=True).head(12).to_dicts() if ok.height and "method" in ok.columns else [],
            "elapsed_sec": round(time.time() - t0, 1), "ran_at": time.strftime("%Y-%m-%d %H:%M")}
    rc = 0
    if args.validate_decided and ok.height:
        dec = ok.filter(pl.col("confidence") != "unverified")
        agree = dec.filter(pl.col("agree") == True).height  # noqa: E712
        cov = dec.height / max(1, ok.height); agr = agree / max(1, dec.height)
        tab = dec.group_by(["expected_shift_min", "shift_min"]).len().sort(["expected_shift_min", "shift_min"]).to_dicts()
        by_conf = {c: {"n": int(dec.filter(pl.col("expected_confidence") == c).height),
                       "agree": round(dec.filter((pl.col("expected_confidence") == c) & (pl.col("agree") == True)).height / max(1, dec.filter(pl.col("expected_confidence") == c).height), 4)}  # noqa: E712
                   for c in ("high", "medium")}
        summ["validation"] = {"n_decided_v3": ok.height, "n_decided_v4": dec.height, "coverage": round(cov, 4), "agreement": round(agr, 4),
                              "by_expected_confidence": by_conf, "expected_vs_v4": tab,
                              "pass": bool(agr >= args.min_agreement and cov >= args.min_coverage), "min_agreement": args.min_agreement, "min_coverage": args.min_coverage}
        rc = 0 if summ["validation"]["pass"] else 1
    if args.only_unverified and not args.dry_run and not args.root and rows:
        prod = inter / "clock_shift.parquet"
        if prod.exists():
            old = pl.read_parquet(prod); v3 = inter / "clock_shift_v3.parquet"
            if not v3.exists():
                old.write_parquet(v3)
            redone = set(df["entity_id"].to_list())
            merged = pl.concat([old.filter(~pl.col("entity_id").is_in(list(redone))), df], how="diagonal")
            merged.write_parquet(prod); summ["merged_into"] = str(prod); summ["merged_rows_replaced"] = len(redone)
            mok = merged.filter(pl.col("status") == "ok")
            summ["store_after_merge"] = {"confidence_counts": mok.group_by("confidence").len().to_dicts(), "shift_counts": mok.group_by("shift_min").len().sort("shift_min").to_dicts()}
    (inter / f"clock_shift_summary{tag}.json").write_text(json.dumps(summ, indent=2, default=str))
    print(json.dumps(summ, indent=2, default=str)); print("wrote", out_p)
    if rc:
        print("VALIDATION FAILED", file=sys.stderr)
    sys.exit(rc)


if __name__ == "__main__":
    main()
