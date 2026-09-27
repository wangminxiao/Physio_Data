import json, sys, os, random, collections, numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from datetime import datetime, timedelta
sys.path.insert(0, "/mnt/localdata/storage/mxwang/Physio_Data"); from physio_data.schema import EHR_EVENT_DTYPE  # noqa
ms2dt = lambda ms: datetime(1970, 1, 1) + timedelta(milliseconds=int(ms))
bpp = butter(3, [0.5 / 20, 3.0 / 20], btype="band"); bpe = butter(3, [5 / 60, 30 / 60], btype="band")
def rate_seg(x, fs, bp, absval=False):
    if np.isfinite(x).mean() < 0.9 or np.nanstd(x) < 1e-3: return np.nan
    x = np.where(np.isfinite(x), x, np.nanmean(x)); y = filtfilt(bp[0], bp[1], x); y = np.abs(y) if absval else y
    pk, _ = find_peaks(y, distance=int(0.3 * fs), height=np.percentile(y, 90) * 0.3 if not absval else np.percentile(y, 98) * 0.4); return pk.size * 2 if 8 <= pk.size <= 120 else np.nan
def lag_scan(t_seg, rate, tc, vc, lo, hi, step=5, min_pts=30):
    res = {}
    for L in range(lo, hi + 1, step):
        errs = []
        for t_ms, v in zip(tc, vc):
            c = t_ms + L * 60000; a, b = np.searchsorted(t_seg, c - 300000), np.searchsorted(t_seg, c + 300000); w = rate[a:b]; w = w[np.isfinite(w)]
            if w.size >= 2: errs.append(abs(float(np.median(w)) - v))
        if len(errs) >= min_pts: res[L] = float(np.mean(errs))
    return res
def best(res):
    if len(res) < 20: return None
    L = min(res, key=res.get); return L, res[L], res.get(0, np.nan)
print("===== MC_MED robust (>=6 h wave, >=120 HR events, PPG-rate AND ECG-HR references, visit window containment)", flush=True)
root = "/mnt/localdata100tb/physio_data/mcmed"; man = json.load(open(f"{root}/manifest.json")); random.seed(13); random.shuffle(man); out = []; cont = collections.Counter()
for m in man:
    if len(out) >= 40: break
    d = f"{root}/{m['entity_id']}"
    try:
        tm = np.load(f"{d}/time_ms.npy"); ev = np.load(f"{d}/ehr_events.npy"); hr = ev[ev["var_id"] == 100]; meta = json.load(open(f"{d}/meta.json"))
        if len(tm) < 720 or hr.size < 120: continue
        arr, dep = meta.get("arrival_ms"), meta.get("departure_ms")
        if arr and dep: cont["inside" if arr - 3600000 <= tm[0] and tm[-1] <= dep + 3600000 else "OUTSIDE"] += 1
        n = min(len(tm), 2880); idx = np.arange(0, n); t_seg = tm[idx]
        pl_ = np.load(f"{d}/PLETH40.npy", mmap_mode="r"); pr = np.array([rate_seg(np.asarray(pl_[i], dtype=np.float32), 40.0, bpp) for i in idx])
        ii = np.load(f"{d}/II120.npy", mmap_mode="r"); er = np.array([rate_seg(np.asarray(ii[i], dtype=np.float32), 120.0, bpe, absval=True) for i in idx[::2]]); t_e = tm[idx[::2]]
        tc = hr["time_ms"].astype(np.int64); vc = hr["value"].astype(np.float64); ok = (vc > 25) & (vc < 220); tc, vc = tc[ok], vc[ok]
        bp_ = best(lag_scan(t_seg, pr, tc, vc, -360, 360)) if np.isfinite(pr).sum() >= 120 else None
        be_ = best(lag_scan(t_e, er, tc, vc, -360, 360)) if np.isfinite(er).sum() >= 60 else None
        out.append((m["entity_id"], bp_, be_, int(tc.size), len(tm), round(float(np.isfinite(pr).mean()), 2), round(float(np.isfinite(er).mean()), 2), (tm[0] - arr) / 3.6e6 if arr else None))
    except Exception as e: continue
print("visit-window containment:", dict(cont))
agree0 = agree_shift = disagree = 0
for e, bp_, be_, nhr, nseg, fp, fe, dt_arr in out:
    def band(b): return None if b is None else ("0" if abs(b[0]) <= 10 else str(b[0]))
    bb, eb = band(bp_), band(be_)
    if bb == "0" and eb == "0": agree0 += 1
    elif bb is not None and eb is not None and bb == eb and bb != "0": agree_shift += 1
    else: disagree += 1
    print(f"  {e}: PPG {bp_ and (bp_[0], round(bp_[1],1), round(bp_[2],1))} | ECG {be_ and (be_[0], round(be_[1],1), round(be_[2],1))} | n_hr {nhr} n_seg {nseg} ppg_ok {fp} ecg_ok {fe} | wave_start - arrival {dt_arr and round(dt_arr,1)} h")
print(f"entities {len(out)}: both refs at 0 -> {agree0}; both refs agree on a non-zero lag -> {agree_shift}; disagree/one missing -> {disagree}")
print("\n===== MIMIC: numerics ehr_hf (var 150 HR) vs ECG-HR (expect 0) and vs charted HR (expect +4/5 h)", flush=True)
root = "/mnt/localdata100tb/physio_data/mimic3"; man = json.load(open(f"{root}/manifest.json")); random.seed(17); random.shuffle(man); res_hf = []; res_chart = []
for m in man:
    if len(res_hf) >= 12: break
    eid = os.path.basename(m["dir"].rstrip("/")); d = f"{root}/{eid}"
    try:
        tm = np.load(f"{d}/time_ms.npy"); hf = np.load(f"{d}/ehr_hf.npy"); ev = np.load(f"{d}/ehr_events.npy")
        if len(tm) < 2000: continue
        num = hf[hf["var_id"] == 150]; ch = ev[ev["var_id"] == 100]
        if num.size < 200 or ch.size < 12: continue
        ii = np.load(f"{d}/II120.npy", mmap_mode="r"); idx = np.arange(0, min(len(tm), 2880), 2); er = np.array([rate_seg(np.asarray(ii[i], dtype=np.float32), 120.0, bpe, absval=True) for i in idx]); t_e = tm[idx]
        if np.isfinite(er).sum() < 200: continue
        tn = num["time_ms"].astype(np.int64); vn = num["value"].astype(np.float64); okn = (vn > 25) & (vn < 220)
        r1 = best(lag_scan(t_e, er, tn[okn][::10], vn[okn][::10], -120, 120, 5, 20))
        # numerics vs charted: for each charted point, numerics median in +-5 min at lag L
        tc = ch["time_ms"].astype(np.int64); vc = ch["value"].astype(np.float64); okc = (vc > 25) & (vc < 220)
        r2 = best(lag_scan(tn[okn], vn[okn], tc[okc], vc[okc], -480, 480, 5, 8))
        if r1: res_hf.append(r1[0])
        if r2: res_chart.append(r2[0])
    except Exception: continue
print("ehr_hf numerics vs ECG-HR best lag (min):", res_hf, "| numerics vs charted HR best lag (min):", res_chart)
