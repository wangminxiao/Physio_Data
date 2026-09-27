# For entities whose charted-NBP lag disagrees with the prediction: is the whole vitals_hf/nbp time base shifted vs the ECG?
import numpy as np, pandas as pd, json, collections
from scipy.signal import butter, filtfilt, find_peaks
ROOT = "/mnt/localdata100tb/physio_data/ucsf_all"
df = pd.read_csv("/projects/mwang80/staging/fs_nbp_probe_2014_11_2015_03_2015_11_2016_03.csv")
bad = df[(df.margin >= 3) & (df.win != df.p1)]; good = df[(df.margin >= 3) & (df.win == df.p1)].sample(12, random_state=0)
b, a = butter(3, [5 / 60, 30 / 60], btype="band")
def ecg_hr(ecg):   # per 30-s segment HR from II120 (n_seg, 3600)
    out = np.full(ecg.shape[0], np.nan, np.float32)
    for i in range(ecg.shape[0]):
        x = np.asarray(ecg[i], dtype=np.float32)
        if not np.isfinite(x).all() or np.nanstd(x) < 20: continue
        y = filtfilt(b, a, x); pk, _ = find_peaks(np.abs(y), distance=36, height=np.percentile(np.abs(y), 98) * 0.4)
        if 8 <= pk.size <= 120: out[i] = pk.size * 2
    return out
def best_lag(h_ecg, h_hf, step_segs=10, max_segs=360):
    res = {}
    for L in range(-max_segs, max_segs + 1, step_segs):   # lag in segments (30 s); positive = vitals_hf later than ECG
        if L >= 0: a_, b_ = h_ecg[:len(h_ecg) - L], h_hf[L:]
        else: a_, b_ = h_ecg[-L:], h_hf[:len(h_hf) + L]
        m = np.isfinite(a_) & np.isfinite(b_)
        if m.sum() > 200: res[L] = float(np.corrcoef(a_[m], b_[m])[0, 1])
    if not res: return None, None, None
    L = max(res, key=res.get); return L * 0.5, res[L], res.get(0)
for label, sub in (("DISAGREE", bad), ("agree-control", good)):
    print(f"===== {label} n={len(sub)}")
    for r in sub.itertuples(index=False):
        d = f"{ROOT}/{r.entity}"; meta = json.load(open(f"{d}/meta.json")); vh = meta.get("vitals_hf", {}); files = vh.get("files", {})
        cls = {k: v.get("class") for k, v in files.items() if any(s in k for s in ("HR", "NBP"))}
        ecg = np.load(f"{d}/II120.npy", mmap_mode="r"); hf = np.load(f"{d}/vitals_hf.npy", mmap_mode="r"); n = min(ecg.shape[0], 2880)   # first 24 h
        h_ecg = ecg_hr(ecg[:n]); h_hf = np.nanmean(np.asarray(hf[:n, :, 0], dtype=np.float32), axis=1)
        lag_min, corr, corr0 = best_lag(h_ecg, h_hf)
        print(f"{r.entity} {r.unit} p1={r.p1} win={r.win} hits(-60/0/+60)={r.hit_m60}/{r.hit_0}/{r.hit_p60} n_charted={r.n_charted} | vital classes {cls} | ECG-vs-HR_hf best lag {lag_min} min corr {corr if corr is None else round(corr,2)} (corr@0 {corr0 if corr0 is None else round(corr0,2)}) | wave_start {meta.get('episode_start_ms')} n_seg {ecg.shape[0]}")
