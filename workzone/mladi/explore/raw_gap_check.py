#!/usr/bin/env python3
"""What are the short NaN runs in the store's PLETH40? Compare with the CHOA 'small gaps' (xml2bin DataHandler v1:
NaN holes of ~16-32 ms, 30-75 per hour, inserted when the integer-second block stamps run ahead of the sample
clock). Per entity, over up to --hours of raw H5 (Pleth, II, ART):
  time axis   dt == nominal? small gaps (1.5 periods < dt <= 2 s): count/h, length (ms), interval between them (s);
              dt <= 0 (repeats / backward steps); large gaps (> 2 s)
  values      NaN-value runs and invalid-code runs (|v| > 1e3): count/h, run length (ms), interval (s)
  store       every interior NaN run of PLETH40 up to 1 s: which raw cause sits under it (time gap / NaN value /
              invalid code / none = only spread by resampling)
Prints per entity, then pooled medians.
    python workzone/mladi/explore/raw_gap_check.py --n 30 --hours 6
"""
import argparse, collections, json, os, random, sys
import h5py
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import cfg  # noqa: E402

C = cfg()
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=30); ap.add_argument("--hours", type=float, default=6.0)
a = ap.parse_args()
ents = sorted(d for d in os.listdir(C["output_dir"]) if os.path.exists(os.path.join(C["output_dir"], d, "PLETH40.npy")))
random.seed(11); random.shuffle(ents)


def runs(mask):
    i = np.flatnonzero(mask)
    if i.size == 0:
        return np.zeros((0, 2), int)
    cut = np.r_[0, np.flatnonzero(np.diff(i) > 1) + 1, i.size]
    return np.array([(i[cut[k]], i[cut[k + 1] - 1] + 1) for k in range(cut.size - 1)])


def chan(f, key, t0, t1):
    if f"data/waveforms/{key}" not in f:
        return None
    ds = f[f"data/waveforms/{key}"]
    per = float(json.loads(ds.attrs[".meta"])["dwc_meta"]["samplePeriod"]) / 1000.0 if ".meta" in ds.attrs else None
    tt = ds["time"] if False else None
    n = ds.shape[0]
    # binary search the time range (time column is sorted)
    def find(x):
        lo, hi = 0, n
        while lo < hi:
            m = (lo + hi) // 2
            if ds[m]["time"] < x: lo = m + 1
            else: hi = m
        return lo
    j0, j1 = find(t0), find(t1)
    if j1 - j0 < 100:
        return None
    x = ds[j0:j1]
    return per, x["time"].astype(np.float64), x["value"].astype(np.float64)


def describe(per, t, v, hours):
    dt = np.diff(t)
    out = {"period_ms": round(per * 1000, 3), "dt_exact_frac": float(np.mean(np.abs(dt - per) < 1e-6))}
    small = (dt > 1.5 * per) & (dt <= 2.0)
    out["small_gaps_per_h"] = round(small.sum() / hours, 1)
    if small.any():
        out["small_gap_ms_p50"] = round(float(np.median(dt[small] - per)) * 1000, 1)
        st = t[:-1][small]
        out["small_gap_interval_s_p50"] = round(float(np.median(np.diff(st))), 1) if st.size > 2 else None
    out["dt_nonpos"] = int((dt <= 0).sum()); out["dt_off_nominal_small"] = int(((np.abs(dt - per) >= 1e-6) & (dt > 0) & (dt <= 1.5 * per)).sum())
    out["large_gaps"] = int((dt > 2.0).sum())
    for name, m in (("nan", ~np.isfinite(v)), ("invalid", np.isfinite(v) & (np.abs(v) > 1e3))):
        r = runs(m)
        out[f"{name}_runs_per_h"] = round(len(r) / hours, 1)
        if len(r):
            out[f"{name}_run_ms_p50"] = round(float(np.median(r[:, 1] - r[:, 0])) * per * 1000, 1)
            out[f"{name}_run_ms_p90"] = round(float(np.percentile(r[:, 1] - r[:, 0], 90)) * per * 1000, 1)
            st = t[r[:, 0]]
            out[f"{name}_interval_s_p50"] = round(float(np.median(np.diff(st))), 1) if st.size > 2 else None
    return out


pool = collections.defaultdict(list); cause = collections.Counter(); n_done = 0
for e in ents:
    if n_done >= a.n:
        break
    od = os.path.join(C["output_dir"], e); m = json.load(open(os.path.join(od, "meta.json")))
    seg = json.load(open(os.path.join(C["pretrain_wav_dir"], e + "__meta.json")))["seg_list"]
    st = np.array(sorted(s[2] for s in seg), float)
    t0 = st[0]; t1 = min(st[-1] + 30, t0 + a.hours * 3600)
    hours = (t1 - t0) / 3600
    rec = {"e": e[:14], "hours": round(hours, 2)}
    with h5py.File(os.path.join(C["raw_h5_dir"], e + ".h5"), "r") as f:
        R = {}
        for key in ("Pleth", "II", "ART"):
            x = chan(f, key, t0, t1)
            if x is None:
                continue
            R[key] = x
            d = describe(*x, hours)
            rec[key] = d
            for k, v in d.items():
                if isinstance(v, (int, float)) and v is not None:
                    pool[f"{key}.{k}"].append(v)
    # store NaN runs (interior, <= 1 s) over the same rows -> raw cause
    tms = np.load(os.path.join(od, "time_ms.npy")); P = np.load(os.path.join(od, "PLETH40.npy"), mmap_mode="r")
    rows = np.flatnonzero((tms / 1000.0 >= t0 - 1) & (tms / 1000.0 < t1))
    if "Pleth" in R and rows.size:
        per, tt, vv = R["Pleth"]
        # contiguous row stretches
        cut = np.r_[0, np.flatnonzero(np.diff(tms[rows]) != 30000) + 1, rows.size]
        for k in range(cut.size - 1):
            rr = rows[cut[k]:cut[k + 1]]
            x = np.asarray(P[rr[0]:rr[-1] + 1], np.float32).reshape(-1)
            g0 = tms[rr[0]] / 1000.0
            for s0, s1 in runs(np.isnan(x)):
                if s0 == 0 or s1 == x.size or (s1 - s0) > 40:
                    continue
                a_, b_ = g0 + s0 / 40.0, g0 + s1 / 40.0
                j0, j1 = np.searchsorted(tt, a_ - 0.05), np.searchsorted(tt, b_ + 0.05)
                seg_t, seg_v = tt[j0:j1], vv[j0:j1]
                if seg_t.size < 2:
                    cause["no raw samples"] += 1; continue
                if np.any(~np.isfinite(seg_v)):
                    cause["raw NaN value"] += 1
                elif np.any(np.abs(seg_v) > 1e3):
                    cause["raw invalid code"] += 1
                elif np.max(np.diff(seg_t)) > 2.5 * per:
                    cause["raw time gap"] += 1
                else:
                    cause["nothing in raw (resampling spread)"] += 1
    n_done += 1
    print(json.dumps(rec), flush=True)
print("== pooled medians over", n_done, "entities")
for k in sorted(pool):
    v = [x for x in pool[k] if x is not None]
    if v:
        print(f"  {k:36s} median {np.median(v):10.3f}   p90 {np.percentile(v, 90):10.3f}   (n {len(v)})")
print("== store PLETH40 interior NaN runs <= 1 s, raw cause:", dict(cause))
