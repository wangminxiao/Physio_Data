"""Prototype: physiological time-zero (t0') for cardiac-arrest events from PPG + ECG.

For each unique event (patient, EventTime E) in the ValidWaveTime CSV:
  window = [E-120 min, E+45 min] of the cycle that contains E (or ends within 30 min before E).
  10-s blocks:
    PPG amplitude  = p95-p5 of baseline-removed PLETH40 (signal present: >=80 % finite, std>0)
    ECG            = R peaks on band-passed II120 -> HR (median RR), QRS present, signal present
  20-s windows: CPR artefact = dominant 1.5-2.3 Hz oscillation carrying >=40 % of 0.5-5 Hz power in ECG or PPG.
  Marker A (pulse loss): amplitude < 20 % of the patient's baseline (median of [E-120,E-60] min) for >=60 s with signal present.
  Marker B (ECG collapse): for >=30 s signal present and (HR<30 or HR>180 or no QRS).
  Marker C (CPR): >=2 consecutive CPR windows.
  t0' = earliest of A/B that is followed within 15 min by C or that persists >=5 min; else C-1 min; else none.
Outputs: summary JSON (no identifiers: events are numbered) + one PNG per event (relative minutes to E).
"""
from __future__ import annotations
import argparse, json, collections, os, sys
from datetime import datetime, timezone
import numpy as np, polars as pl
from scipy.signal import butter, filtfilt, find_peaks, welch

ROOT = "/mnt/localdata100tb/physio_data/ucsf_all"
CSV = "/mnt/localdata/storage/mxwang/data/ucsf_EHR/bedanalysis_waveformExtraction/Output_new/ValidWaveTime_allEnc_eventtime.csv"
FS_P, FS_E = 40, 120; SEG = 30; BLK = 10
PRE_MIN, POST_MIN = 180, 150; SEARCH_LO, SEARCH_HI = -120, 120   # minutes relative to E (covers +-60 min clock error)

def bandpass(x, fs, lo, hi, order=3):
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band"); return filtfilt(b, a, x)

def load_window(m, e_ms):
    d = f"{ROOT}/{m['entity_id']}"; t = np.load(f"{d}/time_ms.npy"); n = len(t)
    s0 = int((e_ms - PRE_MIN * 60000 - t[0]) // 30000); s1 = int((e_ms + POST_MIN * 60000 - t[0]) // 30000) + 1
    a, b = max(0, s0), min(n, s1)
    if b - a < 20: return None
    pl_ = np.asarray(np.load(f"{d}/PLETH40.npy", mmap_mode="r")[a:b], dtype=np.float32).reshape(-1)
    ii = np.asarray(np.load(f"{d}/II120.npy", mmap_mode="r")[a:b], dtype=np.float32).reshape(-1)
    t_start = int(t[0] + a * 30000)
    return t_start, pl_, ii

def blocks_ppg(x, t_start, e_ms):
    n = x.size // (BLK * FS_P); amp = np.full(n, np.nan); present = np.zeros(n, bool)
    for i in range(n):
        seg = x[i * BLK * FS_P:(i + 1) * BLK * FS_P].astype(np.float64); fin = np.isfinite(seg)
        if fin.mean() < 0.8: continue
        seg = np.where(fin, seg, np.nanmedian(seg))
        if seg.std() < 0.5: continue                                       # flat line = probe off, not pulse loss
        hp = seg - np.convolve(seg, np.ones(FS_P) / FS_P, mode="same")     # remove ~1 s baseline
        amp[i] = np.percentile(hp, 95) - np.percentile(hp, 5); present[i] = True
    tmin = (t_start + (np.arange(n) + 0.5) * BLK * 1000 - e_ms) / 60000.0
    return tmin, amp, present

def blocks_ecg(x, t_start, e_ms):
    n = x.size // (BLK * FS_E); hr = np.full(n, np.nan); qrs = np.zeros(n, bool); present = np.zeros(n, bool)
    fin_all = np.isfinite(x); xf = np.where(fin_all, x, np.nanmedian(x) if fin_all.any() else 0).astype(np.float64)
    if fin_all.mean() > 0.05 and xf.std() > 0:
        y = bandpass(xf, FS_E, 5, 25); e = np.convolve(y ** 2, np.ones(int(0.12 * FS_E)) / int(0.12 * FS_E), mode="same")
        pk, _ = find_peaks(e, distance=int(0.25 * FS_E)); h = e[pk]
        thr = 0.25 * np.percentile(h, 90) if h.size else 0; pk = pk[h >= thr]
    else: pk = np.array([], int)
    for i in range(n):
        lo, hi = i * BLK * FS_E, (i + 1) * BLK * FS_E; fin = fin_all[lo:hi]
        if fin.mean() < 0.8: continue
        present[i] = xf[lo:hi].std() > 5.0
        p = pk[(pk >= lo) & (pk < hi)]
        qrs[i] = p.size >= 2
        if p.size >= 3:
            rr = np.diff(p) / FS_E; hr[i] = 60.0 / np.median(rr)
    tmin = (t_start + (np.arange(n) + 0.5) * BLK * 1000 - e_ms) / 60000.0
    return tmin, hr, qrs, present

def cpr_windows(x, fs, t_start, e_ms, qrs_ok_blocks, tblk, base_lf=None):
    """CPR artefact: 20-s windows where ECG R-peak detection fails (no QRS / HR<30 in the overlapping 10-s
    blocks) AND a strong regular 1.5-2.3 Hz oscillation dominates 0.5-5 Hz with amplitude >= 3x the patient's
    baseline low-frequency amplitude. Returns (t_min, flag, lf_amp)."""
    W = 20 * fs; n = x.size // W; flag = np.zeros(n, bool); lf = np.full(n, np.nan)
    for i in range(n):
        seg = x[i * W:(i + 1) * W].astype(np.float64); fin = np.isfinite(seg)
        if fin.mean() < 0.8: continue
        seg = np.where(fin, seg, np.nanmedian(seg))
        if seg.std() == 0: continue
        y = bandpass(seg, fs, 0.5, 5.0); lf[i] = np.percentile(y, 95) - np.percentile(y, 5)
        f, P = welch(y, fs=fs, nperseg=min(seg.size, 8 * fs)); band = (f >= 0.5) & (f <= 5.0); tot = P[band].sum() + 1e-12
        j = np.argmax(P * band); f0 = f[j]; near = (f >= f0 - 0.15) & (f <= f0 + 0.15)
        tw = (t_start + (i + 0.5) * 20 * 1000 - e_ms) / 60000.0
        kb = np.flatnonzero(np.abs(tblk - tw) <= (BLK / 60.0))          # overlapping 10-s blocks
        qrs_fail = kb.size > 0 and (~qrs_ok_blocks[kb]).all()
        flag[i] = qrs_fail and (1.5 <= f0 <= 2.3) and (P[near].sum() / tot >= 0.4)
    if base_lf is not None and np.isfinite(base_lf) and base_lf > 0: flag &= (lf >= 3 * base_lf)
    tmin = (t_start + (np.arange(n) + 0.5) * 20 * 1000 - e_ms) / 60000.0
    return tmin, flag, lf

def gap_onsets(pp, tp, pe, te, min_minutes=2):
    """start times (min) of periods where BOTH PPG and ECG are absent for >= min_minutes."""
    n = min(pp.size, pe.size); absent = (~pp[:n]) & (~pe[:n]); t = tp[:n]; out = []; i = 0
    need = int(min_minutes * 60 / BLK)
    while i < n:
        if absent[i]:
            j = i
            while j < n and absent[j]: j += 1
            if j - i >= need: out.append(float(t[i]))
            i = j
        else: i += 1
    return out

def first_run(cond, tmin, need, lo, hi):
    """first block index in [lo,hi] min where cond holds for `need` consecutive blocks."""
    idx = np.flatnonzero((tmin >= lo) & (tmin <= hi))
    for i in idx:
        if i + need <= cond.size and cond[i:i + need].all(): return i
    return None

def detect(m, e_ms):
    w = load_window(m, e_ms)
    if w is None: return None
    t_start, pl_, ii = w
    tp, amp, pp = blocks_ppg(pl_, t_start, e_ms); te, hr, qrs, pe = blocks_ecg(ii, t_start, e_ms)
    qrs_ok = pe & qrs & np.isfinite(hr) & (hr >= 30)
    # baselines from [-180,-60] min (fallback: everything before -30)
    base_sel = (tp >= -180) & (tp <= -60) & pp & np.isfinite(amp)
    if base_sel.sum() < 30: base_sel = (tp <= -30) & pp & np.isfinite(amp)
    base = float(np.nanmedian(amp[base_sel])) if base_sel.any() else np.nan
    tc_e, ce, lf_e = cpr_windows(ii, FS_E, t_start, e_ms, qrs_ok, te)
    bsel = (tc_e >= -180) & (tc_e <= -60) & np.isfinite(lf_e); base_lf = float(np.nanmedian(lf_e[bsel])) if bsel.sum() >= 30 else (float(np.nanmedian(lf_e[np.isfinite(lf_e)])) if np.isfinite(lf_e).any() else np.nan)
    tc_e, ce, lf_e = cpr_windows(ii, FS_E, t_start, e_ms, qrs_ok, te, base_lf)
    condB = pe & ((~qrs) | (hr < 30) | (hr > 180))
    condA_raw = pp & np.isfinite(amp) & (amp < 0.2 * base) if np.isfinite(base) and base > 0 else np.zeros_like(pp)
    # A must be corroborated by ECG trouble (B) or CPR within +-2 min, otherwise it is a probe problem
    ecg_bad_t = np.concatenate([te[condB], tc_e[ce]]) if (condB.any() or ce.any()) else np.array([])
    condA = condA_raw & np.array([bool(ecg_bad_t.size) and (np.abs(ecg_bad_t - t).min() <= 2.0) for t in tp])
    iA = first_run(condA, tp, 6, SEARCH_LO, SEARCH_HI); iB = first_run(condB, te, 3, SEARCH_LO, SEARCH_HI); iC = first_run(ce, tc_e, 2, SEARCH_LO, SEARCH_HI)
    gaps = [g for g in gap_onsets(pp, tp, pe, te) if SEARCH_LO <= g <= SEARCH_HI]
    cyc_end = (m["wave_end_ms"] + 30000 - e_ms) / 60000.0
    tA = float(tp[iA]) if iA is not None else None; tB = float(te[iB]) if iB is not None else None; tC = float(tc_e[iC]) if iC is not None else None
    tD = gaps[0] if gaps else (cyc_end if SEARCH_LO <= cyc_end <= SEARCH_HI else None)
    def persists(cond, t, i, mins=5):
        j = np.flatnonzero((t >= t[i]) & (t <= t[i] + mins)); return j.size and cond[j].mean() >= 0.8
    cands = [c for c in [(tA, "A", condA, tp, iA), (tB, "B", condB, te, iB)] if c[0] is not None]; cands.sort(key=lambda c: c[0])
    t0 = None; method = None; quality = "none"
    for tcand, name, cond, t, i in cands:
        if tC is not None and 0 <= tC - tcand <= 15: t0, method, quality = tcand, name + "+C", "good"; break
        if tD is not None and 0 <= tD - tcand <= 15: t0, method, quality = tcand, name + "+gap", "good"; break
        if persists(cond, t, i): t0, method, quality = tcand, name + "+persist", "fair"; break
    if t0 is None and tC is not None: t0, method, quality = tC - 1.0, "C-1min", "fair"
    if t0 is None and tD is not None:
        # instability in the 10 min before the gap: HR change > 30 bpm or PPG amp < 50 % baseline
        sel = (te >= tD - 10) & (te < tD) & np.isfinite(hr); selp = (tp >= tD - 10) & (tp < tD) & np.isfinite(amp)
        unstable = (sel.sum() >= 6 and (np.nanmax(hr[sel]) - np.nanmin(hr[sel]) > 30)) or (selp.sum() >= 6 and np.isfinite(base) and np.nanmin(amp[selp]) < 0.5 * base)
        t0, method, quality = tD, "gap" + ("+unstable" if unstable else "-only"), ("fair" if unstable else "weak")
    return dict(entity=m["entity_id"], unit=m.get("unit"), base_amp=None if not np.isfinite(base) else round(base, 1),
                tA=tA, tB=tB, tC=tC, tD=tD, cycle_end=round(cyc_end, 1), t0=t0, method=method, quality=quality,
                ppg_present_frac=round(float(pp[(tp >= -60) & (tp <= 0)].mean()) if ((tp >= -60) & (tp <= 0)).any() else 0, 2),
                ecg_present_frac=round(float(pe[(te >= -60) & (te <= 0)].mean()) if ((te >= -60) & (te <= 0)).any() else 0, 2),
                _tracks=(tp, amp, pp, te, hr, qrs, pe, tc_e, ce, t_start, pl_, ii))

def plot_event(k, r, e_ms, out_dir):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    tp, amp, pp, te, hr, qrs, pe, tc, cc, t_start, pl_, ii = r["_tracks"]
    fig, ax = plt.subplots(4, 1, figsize=(12, 11)); base = r["base_amp"] or np.nan
    ax[0].plot(tp, 100 * amp / base if base else amp, lw=1); ax[0].set_ylabel("PPG amp (% baseline)"); ax[0].set_ylim(0, 300); ax[0].set_xlim(-150, 150)
    ax[0].fill_between(tp, 0, 300, where=~pp, color="grey", alpha=0.2, step="mid", label="PPG absent"); ax[0].axhline(20, color="r", ls=":", lw=0.8)
    ax[1].plot(te, hr, lw=1); ax[1].set_ylabel("ECG HR (bpm)"); ax[1].set_ylim(0, 220); ax[1].set_xlim(-150, 150)
    ax[1].fill_between(te, 0, 220, where=~pe, color="grey", alpha=0.2, step="mid"); ax[1].fill_between(te, 0, 220, where=pe & ~qrs, color="orange", alpha=0.25, step="mid", label="no QRS")
    ax[1].fill_between(tc, 0, 220, where=cc, color="purple", alpha=0.15, step="mid", label="CPR artefact")
    for a in ax[:2]:
        a.axvline(0, color="k", lw=1.2, label="EventTime"); 
        for tt, nm, col in ((r["tA"], "A", "red"), (r["tB"], "B", "blue"), (r["tC"], "C", "purple"), (r["tD"], "D gap", "grey")):
            if tt is not None: a.axvline(tt, color=col, ls="--", lw=1, label=nm)
        if r["t0"] is not None: a.axvline(r["t0"], color="green", lw=2, alpha=0.7, label="t0'")
        a.legend(loc="upper left", fontsize=7, ncol=6); a.grid(alpha=0.3)
    # raw strips: 10 s at t0' (or E) and 10 s five minutes earlier
    tref = r["t0"] if r["t0"] is not None else 0.0
    for row, off, lab in ((2, -5.0, "5 min before t0'"), (3, 0.0, "at t0'")):
        c = e_ms + (tref + off) * 60000; i0 = int((c - t_start) / 1000 * FS_P); j0 = int((c - t_start) / 1000 * FS_E)
        sp = pl_[max(0, i0):max(0, i0) + 10 * FS_P]; se = ii[max(0, j0):max(0, j0) + 10 * FS_E]
        if sp.size: ax[row].plot(np.arange(sp.size) / FS_P, sp, lw=0.8, color="tab:red", label="PPG")
        a2 = ax[row].twinx()
        if se.size: a2.plot(np.arange(se.size) / FS_E, se, lw=0.6, color="tab:blue", label="ECG (uV)")
        ax[row].set_title(f"raw 10 s {lab} ({tref + off:+.1f} min vs EventTime)", fontsize=9); ax[row].set_xlabel("s")
    fig.suptitle(f"event #{k:03d}  unit={r['unit']}  t0'={None if r['t0'] is None else round(r['t0'],1)} min  method={r['method']}  quality={r['quality']}  A={None if r['tA'] is None else round(r['tA'],1)} B={None if r['tB'] is None else round(r['tB'],1)} C={None if r['tC'] is None else round(r['tC'],1)} D={None if r['tD'] is None else round(r['tD'],1)} cycle_end={r['cycle_end']}", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{out_dir}/event_{k:03d}.png", dpi=90); plt.close(fig)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--plots", type=int, default=200)
    ap.add_argument("--events-json", default="", help="override {patient_id_ge: event_ms} (e.g. Code Blue times re-derived on the monitor clock)")
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
    df = pl.read_csv(CSV, infer_schema_length=0); df = df.rename({c: c.strip() for c in df.columns}); events = {}
    for p, e in zip(df["Patient_ID_GE"].to_list(), df["EventTime"].fill_null("").to_list()):
        p = p.strip()[2:] if p.strip().startswith("DE") else p.strip(); e = e.strip()
        if e and e != "-1" and p not in events: events[p] = int(datetime.strptime(e, "%Y-%m-%dT%H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp() * 1000)
    if args.events_json:
        events = {k: int(v) for k, v in json.load(open(args.events_json)).items()}; print(f"events overridden from {args.events_json}: {len(events)}")
    man = json.load(open(f"{ROOT}/manifest.json")); by_pat = collections.defaultdict(list)
    for m in man: by_pat[str(m["patient_id_ge"])].append(m)
    out = []; k = 0
    for pid, e_ms in sorted(events.items()):
        ms = [m for m in by_pat.get(pid, []) if m["wave_start_ms"] - 120 * 60000 <= e_ms <= m["wave_end_ms"] + 120 * 60000]
        if not ms: continue
        m = max(ms, key=lambda q: q["wave_end_ms"] - q["wave_start_ms"]); k += 1
        r = detect(m, e_ms)
        if r is None: continue
        if k <= args.plots: plot_event(k, r, e_ms, args.out)
        rr = {kk: v for kk, v in r.items() if kk != "_tracks"}; rr["k"] = k; out.append(rr)
        if args.limit and k >= args.limit: break
    json.dump(out, open(f"{args.out}/summary.json", "w"), indent=1)
    q = collections.Counter(r["quality"] for r in out); meth = collections.Counter(r["method"] for r in out)
    t0s = np.array([r["t0"] for r in out if r["t0"] is not None])
    print(f"events with a covering cycle: {len(out)} | quality: {dict(q)} | method: {dict(meth)}")
    if t0s.size:
        print(f"t0' - EventTime (min): n={t0s.size} p10/p25/p50/p75/p90 = {np.percentile(t0s,10):.1f}/{np.percentile(t0s,25):.1f}/{np.median(t0s):.1f}/{np.percentile(t0s,75):.1f}/{np.percentile(t0s,90):.1f}")
        bins = {"<-70": int((t0s < -70).sum()), "-70..-50": int(((t0s >= -70) & (t0s <= -50)).sum()), "-50..-10": int(((t0s > -50) & (t0s < -10)).sum()), "-10..+10": int(((t0s >= -10) & (t0s <= 10)).sum()), "10..50": int(((t0s > 10) & (t0s < 50)).sum()), "50..70": int(((t0s >= 50) & (t0s <= 70)).sum()), ">70": int((t0s > 70).sum())}
        print("binned:", bins)
    ce_ = np.array([r["cycle_end"] for r in out]); print(f"cycle end - EventTime (min): p10/p50/p90 = {np.percentile(ce_,10):.0f}/{np.median(ce_):.0f}/{np.percentile(ce_,90):.0f}; within [-120,120]: {int(((ce_>=-120)&(ce_<=120)).sum())}")
    for nm in ("tA", "tB", "tC", "tD"):
        v = np.array([r[nm] for r in out if r[nm] is not None]); print(f"{nm}: found in {v.size} events; median {np.median(v) if v.size else float('nan'):.1f} min")

if __name__ == "__main__":
    main()
