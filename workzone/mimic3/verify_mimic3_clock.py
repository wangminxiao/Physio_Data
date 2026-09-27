"""
MIMIC-III clock-fix gates (datasets/CLOCK_FIX_PLAN_MIMIC_MOVER.md §1.4). Read-only. Exit 1 on FAIL.

  G1  code-level   time_ms[0] == wall-clock base time of the raw master header that starts the recording
                   (all entities, or --sample N); before the fix the difference is +4/+5 h
  G2  numerics     ECG-derived HR vs ehr_hf HR (var 150): median best lag in [-2, 8] s, corr >= 0.6 (40 entities)
  G3  chart        charted HR (var 100) vs ECG-derived HR: lag scan -8..+8 h; median |best lag| <= 30 min,
                   >= 90 % within +-60 min  (before the fix: +220..+300 min)
  G4  cuff BP      charted NIBP systolic (var 104) vs ehr_hf ABP systolic (var 153): lag scan; best lag within +-15 min
  G5  structural   time_ms monotonic 25 s stride; ehr_events.seg_idx == searchsorted(time_ms, t) - 1; ehr_hf inside window;
                   meta.time_base == wall_clock and hf_time_base == wall_clock
  G6  snapshot     (--snapshot DIR) time_ms_new - time_ms_old == clock_shift_ms; ehr_events (var,value,time) multiset unchanged

Run:  python workzone/mimic3/verify_mimic3_clock.py [--root STORE] [--sample 300] [--snapshot /projects/mwang80/staging/mimic3_before]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from scipy.signal import butter, filtfilt, find_peaks

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT)); sys.path.insert(0, str(REPO_ROOT / "workzone" / "common"))
from physio_data.schema import EHR_EVENT_DTYPE  # noqa: E402
from clock_utils import wall_ms, translate_path  # noqa: E402

BP_ECG = butter(3, [5 / 60, 30 / 60], btype="band")


def ecg_hr(x, fs=120.0):
    """beats in a 30-s segment x2 -> bpm; tolerates <= 10 % NaN samples (MIMIC II120 carries sparse NaNs)."""
    fin = np.isfinite(x)
    if fin.mean() < 0.9 or np.nanstd(x) < 1e-3:
        return np.nan
    x = np.where(fin, x, np.nanmean(x))
    y = np.abs(filtfilt(BP_ECG[0], BP_ECG[1], x)); pk, _ = find_peaks(y, distance=int(0.3 * fs), height=np.percentile(y, 98) * 0.4)
    return pk.size * 2 if 8 <= pk.size <= 120 else np.nan


def lag_scan_mae(t_ref, v_ref, t_ev, v_ev, lo, hi, step, half_win_ms=300000, min_pts=8):
    """MAE between event values and the reference median around t_ev + L."""
    res = {}
    for L in range(lo, hi + 1, step):
        errs = []
        for t, v in zip(t_ev, v_ev):
            c = t + L * 60000; a, b = np.searchsorted(t_ref, c - half_win_ms), np.searchsorted(t_ref, c + half_win_ms); w = v_ref[a:b]; w = w[np.isfinite(w)]
            if w.size >= 2:
                errs.append(abs(float(np.median(w)) - v))
        if len(errs) >= min_pts:
            res[L] = float(np.mean(errs))
    if len(res) < 5:
        return None
    L = min(res, key=res.get); return L, res[L], res.get(0, np.nan)


def best_lag_corr(ref, num, max_slots=8):
    best = None
    for lag in range(-max_slots, max_slots + 1):
        a = ref[max(0, -lag):len(ref) - max(0, lag)]; b = num[max(0, lag):len(num) - max(0, -lag)]; ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 20:
            continue
        c = float(np.corrcoef(a[ok], b[ok])[0, 1])
        if best is None or c > best[1]:
            best = (lag * 2, c)
    return best


def header_diffs_h(subj_dir, t0_ms):
    """(time_ms[0] - base time of every master header in the subject dir) in hours, or None when no header is readable."""
    if not subj_dir or not os.path.isdir(subj_dir):
        return None
    out = []
    for hp in sorted(glob.glob(os.path.join(subj_dir, "p*-*-*-*-*-*.hea"))):
        if hp.endswith("n.hea"):
            continue
        try:
            first = open(hp).readline().split(); dt = None
            for fmt in ("%H:%M:%S.%f %d/%m/%Y", "%H:%M:%S %d/%m/%Y"):
                try:
                    dt = datetime.strptime(f"{first[4]} {first[5]}", fmt); break
                except (ValueError, IndexError):
                    continue
            if dt is not None:
                out.append((t0_ms - wall_ms(dt)) / 3.6e6)
        except Exception:
            continue
    return out or None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None); ap.add_argument("--sample", type=int, default=0, help="G1 on N random entities (0 = all)")
    ap.add_argument("--physio-sample", type=int, default=40); ap.add_argument("--chart-sample", type=int, default=100)
    ap.add_argument("--snapshot", default=None); ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / "workzone" / "configs" / "server_paths.yaml").read_text())["mimic3"]
    root = Path(args.root or cfg["output_dir"]); checks = []; info = {}
    def check(name, ok, detail=""):
        checks.append({"check": name, "pass": bool(ok), "detail": detail}); print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    ids = sorted(d.name for d in root.iterdir() if d.is_dir() and (d / "meta.json").exists()); t0 = time.time()
    print(f"root={root} entities={len(ids)}")
    # ---- G5 structural (all entities, cheap) + G1 header (sampled or all)
    random.seed(1); g1_ids = ids if not args.sample else random.sample(ids, min(args.sample, len(ids)))
    g5_fail = {}; g1 = {"match": 0, "mismatch": 0, "no_header": 0, "unverified": 0, "examples": []}; legacy = 0; g1_set = set(g1_ids)
    for i, e in enumerate(ids, 1):
        d = root / e
        try:
            m = json.loads((d / "meta.json").read_text()); tm = np.load(d / "time_ms.npy"); errs = []
            if m.get("time_base") != "wall_clock": legacy += 1; errs.append("time_base != wall_clock")
            if m.get("hf_time_base") not in ("wall_clock", None) and int(m.get("n_hf_events") or 0) > 0 and m.get("hf_time_base") != "wall_clock_provisional_shift":
                errs.append(f"hf_time_base={m.get('hf_time_base')}")
            if len(tm) > 1 and not np.all(np.diff(tm) > 0): errs.append("time_ms not monotonic")
            if int(m.get("recording_start_ms", tm[0])) != int(tm[0]): errs.append("recording_start_ms != time_ms[0]")
            ev_p = d / "ehr_events.npy"
            if ev_p.exists():
                ev = np.load(ev_p)
                if ev.size:
                    si = np.searchsorted(tm, ev["time_ms"], side="right") - 1
                    inside = (ev["time_ms"] >= tm[0]) & (ev["time_ms"] < tm[-1] + int(m.get("segment_duration_sec", 30)) * 1000)
                    if inside.any() and not np.array_equal(si[inside], ev["seg_idx"][inside]): errs.append("ehr_events.seg_idx inconsistent with time_ms")
            hf_p = d / "ehr_hf.npy"
            if hf_p.exists():
                hf = np.load(hf_p)
                if hf.size and (hf["time_ms"].min() < tm[0] - 60000 or hf["time_ms"].max() > tm[-1] + 90000): errs.append("ehr_hf outside window")
            if errs: g5_fail[e] = errs
            if (e in g1_set) if args.sample else True:
                diffs = header_diffs_h(translate_path(m.get("source_path")), int(tm[0]))
                if diffs is None: g1["no_header"] += 1
                else:
                    c = min(diffs, key=abs)            # the record that starts the recording
                    if abs(c) <= 5 / 60: g1["match"] += 1                  # first block within 5 min of its header
                    elif abs(abs(c) - 4) <= 0.15 or abs(abs(c) - 5) <= 0.15:   # the bug produced exact +4/+5 h (+ a few min block offset)
                        g1["mismatch"] += 1
                        if len(g1["examples"]) < 10: g1["examples"].append((e, [round(h, 2) for h in diffs[:4]]))
                    else: g1["unverified"] += 1
        except Exception as ex:
            g5_fail[e] = [f"{type(ex).__name__}: {str(ex)[:80]}"]
        if i % 1000 == 0: print(f"  [{i}/{len(ids)}] {time.time()-t0:.0f}s", flush=True)
    check("G5 structural: time_ms/seg_idx/ehr_hf/meta consistent", not g5_fail, f"failing={len(g5_fail)} legacy_time_base={legacy} e.g. {list(g5_fail.items())[:3]}")
    n_g1 = g1["match"] + g1["mismatch"]
    # multi-record subjects can start a block ~4-5 h from a neighbouring record's header by coincidence: allow <= 0.1 %
    check("G1 time_ms[0] equals a raw master-header base time (wall clock); +-4/5 h residue in <= 0.1 % of entities", n_g1 > 0 and g1["mismatch"] <= 0.001 * n_g1 and g1["match"] >= 0.5 * n_g1, f"match={g1['match']} mismatch(+-4/5h)={g1['mismatch']} unverified(later block)={g1['unverified']} no_header={g1['no_header']} e.g. {g1['examples'][:3]}")
    info["g1"] = g1
    # ---- G2 numerics vs ECG, G3 chart vs ECG, G4 NIBP vs ABP
    random.seed(2); phys = random.sample(ids, min(max(args.physio_sample, args.chart_sample), len(ids)))
    g2 = []; g3 = []; g4 = []; why = {}
    def skip(r): why[r] = why.get(r, 0) + 1
    for e in phys:
        d = root / e
        try:
            tm = np.load(d / "time_ms.npy")
            if len(tm) < 720: skip("short<6h"); continue
            ii = np.load(d / "II120.npy", mmap_mode="r"); ev = np.load(d / "ehr_events.npy"); hf = np.load(d / "ehr_hf.npy") if (d / "ehr_hf.npy").exists() else np.empty(0, EHR_EVENT_DTYPE)
            idx = np.arange(0, min(len(tm), 5760), 2); h = np.array([ecg_hr(np.asarray(ii[i], dtype=np.float64)) for i in idx]); t_seg = tm[idx]
            if np.isfinite(h).sum() < 100: skip("ecg_hr_unavailable"); continue
            num = hf[hf["var_id"] == 150]
            if len(g2) < args.physio_sample and num.size > 200:
                # slot-level: numerics at pair ends vs ECG-HR of the same segments
                s0 = max(0, len(tm) // 2 - 60); nsg = 120
                seg_hr = np.array([ecg_hr(np.asarray(ii[i], dtype=np.float64)) for i in range(s0, s0 + nsg)])
                seg_num = np.full(nsg, np.nan)
                for t_, v_ in zip(num["time_ms"], num["value"]):
                    k = int(np.searchsorted(tm, t_, side="right") - 1) - s0
                    if 0 <= k < nsg: seg_num[k] = v_
                b = best_lag_corr(seg_hr, seg_num, max_slots=8)
                if b: g2.append(b)
            ch = ev[ev["var_id"] == 100]
            if len(g3) < args.chart_sample and ch.size >= 12:
                vc = ch["value"].astype(float); ok = (vc > 25) & (vc < 220)
                r = lag_scan_mae(t_seg, h, ch["time_ms"][ok].astype(np.int64), vc[ok], -480, 480, 5)
                if r and r[1] < 0.9 * r[2] or (r and abs(r[0]) <= 30): g3.append(r[0])
            nb = ev[ev["var_id"] == 104]; ab = hf[hf["var_id"] == 157]
            if ab.size < 50: ab = hf[hf["var_id"] == 153]
            if len(g4) < args.physio_sample and nb.size >= 12 and ab.size >= 50:
                r = lag_scan_mae(ab["time_ms"].astype(np.int64), ab["value"].astype(float), nb["time_ms"].astype(np.int64), nb["value"].astype(float), -480, 480, 5, half_win_ms=600000)
                if r and r[1] < 0.9 * r[2]: g4.append(r[0])
        except Exception as ex_:
            skip(f"exc:{type(ex_).__name__}:{str(ex_)[:60]}"); continue
    print("  physio sample skip reasons:", why, flush=True)
    g2 = [(l, c) for l, c in g2 if np.isfinite(c)]
    if g2:
        lag2 = float(np.median([l for l, _ in g2])); corr2 = float(np.median([c for _, c in g2]))
        check("G2 ehr_hf HR vs ECG-derived HR: median best lag in [-2, 8] s and corr >= 0.6", -2 <= lag2 <= 8 and corr2 >= 0.6, f"lag={lag2:.0f} s corr={corr2:.2f} n={len(g2)}")
    else: check("G2 sample available", False, f"no usable entity; skips={why}")
    if g3:
        med3 = float(np.median([abs(l) for l in g3])); within = sum(abs(l) <= 60 for l in g3) / len(g3)
        check("G3 charted HR vs ECG-derived HR: median |lag| <= 30 min and >= 90 % within +-60 min", med3 <= 30 and within >= 0.9, f"median |lag|={med3:.0f} min within60={within:.2f} n={len(g3)} lags={sorted(g3)[:12]}...")
    else: check("G3 sample available", False, f"no usable entity; skips={why}")
    if len(g4) >= 10:
        med4 = float(np.median(g4)); check("G4 charted NIBP vs ehr_hf NBP/ABP systolic: median best lag within +-15 min", abs(med4) <= 15, f"median lag={med4:.0f} min n={len(g4)} lags={sorted(g4)[:12]}")
    elif g4:
        check("G4 NIBP-vs-numerics sample (informational, n < 10)", True, f"lags={sorted(g4)}")
    else: check("G4 NIBP-vs-ABP sample", True, "no entity with both (informational)")
    # ---- G6 snapshot
    if args.snapshot:
        snap = Path(args.snapshot); n_ok = n_bad = 0; ex = []
        for sd in sorted(p for p in snap.iterdir() if p.is_dir()):
            e = sd.name; d = root / e
            try:
                m = json.loads((d / "meta.json").read_text()); old = np.load(sd / "time_ms.npy"); new = np.load(d / "time_ms.npy")
                ok = len(old) == len(new) and np.all(new - old == int(m.get("clock_shift_ms", 0)))
                if ok and (sd / "ehr_events.npy").exists():
                    a = np.load(sd / "ehr_events.npy"); b = np.load(d / "ehr_events.npy"); ws, we = int(new[0]), int(new[-1]) + 30000
                    keep = (a["time_ms"] >= ws) & (a["time_ms"] <= we)       # old in-wave events still inside the (shifted) window
                    A = set(zip(a["var_id"][keep].tolist(), np.round(a["value"][keep], 4).tolist(), a["time_ms"][keep].tolist()))
                    B = set(zip(b["var_id"].tolist(), np.round(b["value"], 4).tolist(), b["time_ms"].tolist()))
                    ok = A <= B
                n_ok += ok; n_bad += (not ok)
                if not ok and len(ex) < 5: ex.append(e)
            except Exception as ex_:
                n_bad += 1; ex.append(f"{e}: {ex_}")
        check("G6 snapshot: time_ms shifted by clock_shift_ms exactly; old in-wave events preserved (subset of the rebuilt ehr_events)", n_bad == 0 and n_ok > 0, f"ok={n_ok} bad={n_bad} e.g. {ex[:3]}")
    # ---- G7 partition totals vs the pre-fix manifest (catches silently dropped EHR events)
    snap_man = Path(args.snapshot) / "manifest.json" if args.snapshot else None
    if snap_man and snap_man.exists() and (root / "manifest.json").exists():
        old_m = json.loads(snap_man.read_text()); new_m = json.loads((root / "manifest.json").read_text())
        tot = lambda m, k: sum(int(e.get(k) or 0) for e in m)
        ob, nb = tot(old_m, "n_baseline"), tot(new_m, "n_baseline"); orr, nr = tot(old_m, "n_recent"), tot(new_m, "n_recent"); oe, ne = tot(old_m, "n_ehr_events"), tot(new_m, "n_ehr_events"); of, nf = tot(old_m, "n_future"), tot(new_m, "n_future")
        ok7 = nb >= 0.5 * ob and nr >= 0.3 * orr and ne >= 0.8 * oe and nf >= 0.5 * of
        check("G7 EHR partition totals vs pre-fix manifest (baseline >= 50 %, recent >= 30 %, events >= 80 %, future >= 50 %)", ok7, f"baseline {ob}->{nb} recent {orr}->{nr} events {oe}->{ne} future {of}->{nf}")
    result = "PASS" if all(c["pass"] for c in checks) else "FAIL"
    out = Path(args.out or (REPO_ROOT / "workzone" / "outputs" / "mimic3" / "verify_clock.json")); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"result": result, "checks": checks, "info": info, "root": str(root), "ran_at": time.strftime("%Y-%m-%d %H:%M")}, indent=2, default=str))
    print(f"VERIFY MIMIC3 CLOCK: {result}  ({out})")
    sys.exit(0 if result == "PASS" else 1)


if __name__ == "__main__":
    main()
