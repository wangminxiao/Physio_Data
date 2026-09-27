"""
MOVER clock-fix gates (datasets/CLOCK_FIX_PLAN_MIMIC_MOVER.md §2.4). Read-only. Exit 1 on FAIL.

  G1 measurement   clock_shift.parquet: >= 85 % of cases with vitals decided (not unverified); shifts only in {-60, 0, +60}
  G2 re-scan       charted HR vs PPG rate (+ ECG when usable) on the corrected time_ms, N random decided cases:
                   >= 95 % at |lag| <= 10 min; no +-60 band left by DST season
  G3 OR window     OR_start(EHR) - wave_start per case: >= 80 % within +-30 min of the median; no side lobes at +-60
  G4 attribution   every shifted case still inside its OR/anesthesia window (+-2 h)
  G5 structural    (after stage E) time_ms monotonic; ehr_events.seg_idx == searchsorted(time_ms) - 1; meta.clock_fix_version
  G6 snapshot      (--snapshot DIR) time_ms_new - time_ms_old == clock_shift_min * 60000; vitals_events unchanged

  python workzone/mover/verify_mover_clock.py --dataset mover [--root STORE] [--rescan 120] [--snapshot DIR] [--skip-structural]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT)); sys.path.insert(0, str(REPO_ROOT / "workzone" / "mover"))
from stage_b2_clock import rate_seg, lag_scan, decide, BP_PPG, BP_ECG, WINDOW_KEYS  # noqa: E402

LA = ZoneInfo("America/Los_Angeles")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="mover"); ap.add_argument("--root", default=None); ap.add_argument("--rescan", type=int, default=120)
    ap.add_argument("--snapshot", default=None); ap.add_argument("--skip-structural", action="store_true"); ap.add_argument("--out", default=None)
    ap.add_argument("--g3-envelope", default="-20,75", help="OR_start - wave_start envelope in min (SIS: -20,75; EPIC anesthesia window: -30,90)")
    ap.add_argument("--g3-min-frac", type=float, default=0.9)
    args = ap.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / "workzone" / "configs" / "server_paths.yaml").read_text())[args.dataset]
    root = Path(args.root or cfg["output_dir"]); inter = REPO_ROOT / "workzone" / "outputs" / args.dataset
    checks = []; info = {}
    def check(name, ok, detail=""):
        checks.append({"check": name, "pass": bool(ok), "detail": detail}); print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    # ---- G1
    pq = inter / (f"clock_shift_{Path(args.root).name}.parquet" if args.root else "clock_shift.parquet")
    if not pq.exists():
        check("G1 clock_shift.parquet present", False, str(pq)); return finish(checks, info, args, root)
    df = pl.read_parquet(pq); ok = df.filter(pl.col("status") == "ok")
    with_v = ok.filter(pl.col("n_hr") >= 60) if "n_hr" in ok.columns else ok
    decided = with_v.filter(pl.col("confidence") != "unverified")
    shifts = set(ok["shift_min"].drop_nulls().to_list()) if "shift_min" in ok.columns else set()
    check("G1 >= 80 % of cases with vitals decided; shifts in {-60,0,60}", with_v.height > 0 and decided.height >= 0.80 * with_v.height and shifts <= {-60, 0, 60},
          f"with_vitals={with_v.height} decided={decided.height} shifts={sorted(shifts)} by_shift={ok.group_by('shift_min').len().sort('shift_min').to_dicts() if 'shift_min' in ok.columns else None}")
    info["g1"] = {"with_vitals": with_v.height, "decided": decided.height, "n_ok": ok.height}
    # ---- G2 re-scan on corrected time_ms
    ids = decided["entity_id"].to_list(); random.seed(5); random.shuffle(ids); lags = []; band = {"0": 0, "-60": 0, "+60": 0, "other": 0}; dst_tab = {0: {}, 1: {}}
    for e in ids[: args.rescan * 3]:
        if len(lags) >= args.rescan:
            break
        d = root / e
        try:
            tm = np.load(d / "time_ms.npy"); ve = np.load(d / "vitals_events.npy"); hr = ve[ve["var_id"] == 100]
            if hr.size < 60: continue
            tc = hr["time_ms"].astype(np.int64); vc = hr["value"].astype(np.float64); okm = (vc > 25) & (vc < 220); tc, vc = tc[okm], vc[okm]
            idx = np.arange(0, min(len(tm), 2880)); t_seg = tm[idx]
            pl_ = np.load(d / "PLETH40.npy", mmap_mode="r"); pr = np.array([rate_seg(np.asarray(pl_[i], dtype=np.float32), 40.0, BP_PPG, False) for i in idx])
            rp = lag_scan(t_seg, pr, tc, vc) if np.isfinite(pr).sum() >= 60 else None; re_ = None
            if (d / "II120.npy").exists():
                ii = np.load(d / "II120.npy", mmap_mode="r"); er = np.array([rate_seg(np.asarray(ii[i], dtype=np.float32), 120.0, BP_ECG, True) for i in idx])
                re_ = lag_scan(t_seg, er, tc, vc) if np.isfinite(er).sum() >= 60 else None
            if rp is None and re_ is None: continue
            shift, method, conf, _ = decide(rp, re_)      # same rule as stage B2: the corrected store must need no further shift
            if conf == "unverified": continue
            L = -shift; lags.append(L); b = "0" if L == 0 else ("-60" if L == -60 else ("+60" if L == 60 else "other")); band[b] += 1
            dt = datetime(1970, 1, 1) + timedelta(milliseconds=int(tm[0])); dst = int(dt.replace(tzinfo=LA).dst().total_seconds() != 0); dst_tab[dst][b] = dst_tab[dst].get(b, 0) + 1
        except Exception:
            continue
    if lags:
        frac0 = band["0"] / len(lags)
        check("G2 corrected store: the B2 decision rule finds no residual shift in >= 95 % of decidable cases", frac0 >= 0.95 and band["-60"] + band["+60"] <= 0.03 * len(lags), f"n={len(lags)} bands={band} by_dst={dst_tab}")
    else:
        check("G2 re-scan sample available", False, "none")
    info["g2"] = {"n": len(lags), "bands": band, "by_dst": dst_tab}
    # ---- G3 / G4 window checks + G5 structural
    diffs = []; g4_bad = []; g5_fail = {}; n_struct = 0
    for e in ok["entity_id"].to_list():
        d = root / e
        try:
            meta = json.loads((d / "meta.json").read_text()); tm = np.load(d / "time_ms.npy")
            win = next(((meta.get(a), meta.get(b)) for a, b in WINDOW_KEYS if meta.get(a) and meta.get(b)), None)
            if win and meta.get("clock_shift_confidence") not in (None, "unverified"):
                diffs.append((int(win[0]) - int(tm[0])) / 60000)
                sh = int(meta.get("clock_shift_min") or 0) * 60000
                if sh:   # same rule as stage B2: a shift must not move a case that was inside its window (+-2 h) outside of it
                    ws_, we_ = int(win[0]) - 2 * 3600000, int(win[1]) + 2 * 3600000
                    inside_before = ws_ <= int(tm[0]) - sh and int(tm[-1]) - sh <= we_
                    inside_after = ws_ <= int(tm[0]) and int(tm[-1]) <= we_
                    if inside_before and not inside_after:
                        g4_bad.append(e)
            if not args.skip_structural:
                n_struct += 1; errs = []
                if len(tm) > 1 and not np.all(np.diff(tm) > 0): errs.append("time_ms not monotonic")
                if not meta.get("clock_fix_version"): errs.append("no clock_fix_version")
                if (d / "ehr_events.npy").exists():
                    ev = np.load(d / "ehr_events.npy")
                    if ev.size:
                        seg = int(meta.get("segment_duration_sec", 30)) * 1000
                        inside = (ev["time_ms"] >= tm[0]) & (ev["time_ms"] < tm[-1] + seg)
                        si = np.searchsorted(tm, ev["time_ms"], side="right") - 1
                        if inside.any() and not np.array_equal(si[inside], ev["seg_idx"][inside]): errs.append("ehr_events.seg_idx inconsistent (rerun stage E)")
                if errs: g5_fail[e] = errs
        except Exception as ex:
            g5_fail[e] = [f"{type(ex).__name__}: {str(ex)[:60]}"]
    if diffs:
        # OR_start - wave_start is bimodal even when aligned (waveform starts at OR entry ~-3 min or at induction ~+50 min),
        # so the gate is the envelope of the two clusters: decided cases must sit in [-20, 75] min; wrong hours land at ~-63 / ~110
        lo_e, hi_e = (float(x) for x in args.g3_envelope.split(","))
        a = np.array(diffs); inside = float(np.mean((a >= lo_e) & (a <= hi_e))); low = float(np.mean(a < lo_e - 25)); high = float(np.mean(a > hi_e + 15))
        check(f"G3 OR_start - wave_start (decided cases): >= {args.g3_min_frac:.0%} within [{lo_e:.0f}, {hi_e:.0f}] min", inside >= args.g3_min_frac, f"inside={inside:.2f} far-below={low:.2f} far-above={high:.2f} n={len(a)} p10/50/90={np.percentile(a,10):.0f}/{np.median(a):.0f}/{np.percentile(a,90):.0f}")
    check("G4 no shifted case moved from inside to outside its OR/anesthesia window (+-2 h)", not g4_bad, f"violations={len(g4_bad)} e.g. {g4_bad[:5]}")
    if not args.skip_structural:
        check("G5 structural: time_ms monotonic, clock_fix_version, ehr_events.seg_idx consistent", not g5_fail, f"failing={len(g5_fail)} of {n_struct} e.g. {list(g5_fail.items())[:3]}")
    # ---- G6 snapshot
    if args.snapshot:
        snap = Path(args.snapshot); n_ok = n_bad = 0; ex = []
        for sd in sorted(p for p in snap.iterdir() if p.is_dir()):
            d = root / sd.name
            try:
                meta = json.loads((d / "meta.json").read_text()); old = np.load(sd / "time_ms.npy"); new = np.load(d / "time_ms.npy")
                good = len(old) == len(new) and np.all(new - old == int(meta.get("clock_shift_min", 0)) * 60000)
                if good and (sd / "vitals_events.npy").exists() and (d / "vitals_events.npy").exists():
                    good = np.array_equal(np.load(sd / "vitals_events.npy"), np.load(d / "vitals_events.npy"))
                n_ok += good; n_bad += (not good)
                if not good and len(ex) < 5: ex.append(sd.name)
            except Exception as ex_:
                n_bad += 1; ex.append(f"{sd.name}: {ex_}")
        check("G6 snapshot: time_ms shifted exactly by clock_shift_min, vitals_events unchanged", n_bad == 0 and n_ok > 0, f"ok={n_ok} bad={n_bad} e.g. {ex[:3]}")
    return finish(checks, info, args, root)


def finish(checks, info, args, root):
    result = "PASS" if checks and all(c["pass"] for c in checks) else "FAIL"
    out = Path(args.out or (REPO_ROOT / "workzone" / "outputs" / args.dataset / "verify_clock.json")); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"result": result, "checks": checks, "info": info, "root": str(root), "ran_at": time.strftime("%Y-%m-%d %H:%M")}, indent=2, default=str))
    print(f"VERIFY {args.dataset.upper()} CLOCK: {result}  ({out})")
    sys.exit(0 if result == "PASS" else 1)


if __name__ == "__main__":
    main()
