#!/usr/bin/env python3
"""Are the ~264 ms forward steps in the raw H5 time axis lost data, or a timestamp re-sync over continuous
samples (the CHOA DataHandler v1 'small gaps' were the latter)? Around each step in II (500 Hz): R peaks found in
SAMPLE index (Physio_Data pleth_timing_lib.ecg_beats on the samples as stored), then the R-R interval that spans
the step measured two ways against the median of the 6 neighbouring intervals:
  by sample count  (n_samples x 2 ms)   ~0 if no sample is missing, ~-256 ms if 256 ms of signal is gone
  by timestamp     (t[j] - t[i])        ~+256 ms if the samples are continuous and only the stamp jumped
Same for the Pleth foot intervals (125 Hz). Prints per step and pooled.
    python workzone/mladi/explore/gap_content_check.py --n 15
"""
import argparse, json, os, random, sys
import h5py
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from common import cfg  # noqa: E402
import pleth_timing_lib as L  # noqa: E402

C = cfg()
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=15); ap.add_argument("--max-steps", type=int, default=8)
a = ap.parse_args()
ents = sorted(d for d in os.listdir(C["output_dir"]) if os.path.exists(os.path.join(C["output_dir"], d, "PLETH40.npy")))
random.seed(5); random.shuffle(ents)
res = {"II": {"samp": [], "time": []}, "Pleth": {"samp": [], "time": []}}
n_ent = 0
for e in ents:
    if n_ent >= a.n:
        break
    with h5py.File(os.path.join(C["raw_h5_dir"], e + ".h5"), "r") as f:
        if "data/waveforms/II" not in f or "data/waveforms/Pleth" not in f:
            continue
        dsE = f["data/waveforms/II"]; perE = float(json.loads(dsE.attrs[".meta"])["dwc_meta"]["samplePeriod"]) / 1000
        n = dsE.shape[0]
        tE = dsE[: min(n, int(6 * 3600 / perE))]["time"].astype(np.float64)
        dt = np.diff(tE)
        steps = np.flatnonzero((dt > 0.2) & (dt < 0.4))[: a.max_steps]
        if steps.size == 0:
            continue
        n_ent += 1
        for k in steps:
            tg = tE[k]
            for key, fs, det in (("II", 1 / perE, L.ecg_beats), ("Pleth", 125.0, L.ppg_beats)):
                ds = f[f"data/waveforms/{key}"]
                # samples within +-15 s of the step (by timestamp), detected in sample index
                t = ds[max(0, int((tg - 15 - ds[0]["time"]) * fs) - 2000): int((tg + 15 - ds[0]["time"]) * fs) + 2000]
                tt, vv = t["time"].astype(np.float64), t["value"].astype(np.float64)
                m = (tt >= tg - 15) & (tt <= tg + 15)
                tt, vv = tt[m], vv[m]
                if tt.size < fs * 20:
                    continue
                pk = det(vv, fs)                      # seconds as sample index / fs
                if pk is None or pk.size < 10:
                    continue
                idx = pk * fs
                ts = np.interp(idx, np.arange(tt.size), tt)  # timestamps of the beats
                j = np.searchsorted(ts, tg + 1e-6)              # first beat after the step
                if j < 4 or j + 3 > ts.size:
                    continue
                rr_s = np.diff(idx) / fs; rr_t = np.diff(ts)
                ref = np.median(np.r_[rr_s[j - 4:j - 1], rr_s[j:j + 3]])
                res[key]["samp"].append((rr_s[j - 1] - ref) * 1000); res[key]["time"].append((rr_t[j - 1] - ref) * 1000)
                print(json.dumps({"e": e[:14], "chan": key, "step_ms": round(float(dt[k] - perE) * 1000, 1),
                                  "rr_by_samples_minus_ref_ms": round(float((rr_s[j - 1] - ref) * 1000), 1),
                                  "rr_by_time_minus_ref_ms": round(float((rr_t[j - 1] - ref) * 1000), 1)}), flush=True)
for key, v in res.items():
    if v["samp"]:
        print(f"== {key}: n {len(v['samp'])} | RR across the step minus neighbours: by sample count median "
              f"{np.median(v['samp']):.1f} ms (IQR {np.percentile(v['samp'], 25):.1f}..{np.percentile(v['samp'], 75):.1f}) | "
              f"by timestamp median {np.median(v['time']):.1f} ms (IQR {np.percentile(v['time'], 25):.1f}..{np.percentile(v['time'], 75):.1f})")
