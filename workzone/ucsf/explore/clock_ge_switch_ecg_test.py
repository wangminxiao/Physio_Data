# In cycles that straddle a GE-calendar DST switch: is the .vital-derived HR (vitals_hf) aligned with the adibin ECG before AND after the switch?
import numpy as np, json, collections
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from scipy.signal import butter, filtfilt, find_peaks
LA = ZoneInfo("America/Los_Angeles"); ROOT = "/mnt/localdata100tb/physio_data/ucsf_all"
man = json.load(open(f"{ROOT}/manifest.json")); ms2dt = lambda ms: datetime(1970, 1, 1) + timedelta(milliseconds=int(ms))
def ds(t): return int(t.replace(tzinfo=LA).dst().total_seconds() != 0)
strad = [q for q in man if ds(ms2dt(q["wave_start_ms"])) != ds(ms2dt(q["wave_end_ms"]))]
print(f"entities straddling a GE-calendar DST switch: {len(strad)} / {len(man)}  (units: {collections.Counter(q.get('unit') for q in strad).most_common(6)})")
b, a = butter(3, [5 / 60, 30 / 60], btype="band")
def ecg_hr(ecg, i0, i1):
    out = np.full(i1 - i0, np.nan, np.float32)
    for k, i in enumerate(range(i0, i1)):
        x = np.asarray(ecg[i], dtype=np.float32)
        if not np.isfinite(x).all() or np.nanstd(x) < 20: continue
        y = filtfilt(b, a, x); pk, _ = find_peaks(np.abs(y), distance=36, height=np.percentile(np.abs(y), 98) * 0.4)
        if 8 <= pk.size <= 120: out[k] = pk.size * 2
    return out
def best_lag(h_ecg, h_hf, max_segs=300, step=5):
    res = {}
    for L in range(-max_segs, max_segs + 1, step):
        if L >= 0: a_, b_ = h_ecg[:len(h_ecg) - L], h_hf[L:]
        else: a_, b_ = h_ecg[-L:], h_hf[:len(h_hf) + L]
        m = np.isfinite(a_) & np.isfinite(b_)
        if m.sum() > 120: res[L] = float(np.corrcoef(a_[m], b_[m])[0, 1])
    if not res: return None
    L = max(res, key=res.get); return (L * 0.5, round(res[L], 2), round(res.get(0, np.nan), 2))
rng = np.random.default_rng(0); pick = list(rng.choice(len(strad), size=min(14, len(strad)), replace=False))
want = {"172519704483858_25379", "249219710629842_13833", "669941987999192_40996", "726929272216222_14884", "445584457776_5920", "48708793864063_35930"}
sel = [q for q in strad if q["entity_id"] in want] + [strad[i] for i in pick if strad[i]["entity_id"] not in want]
for q in sel[:18]:
    ent = q["entity_id"]; tm = np.load(f"{ROOT}/{ent}/time_ms.npy"); ecg = np.load(f"{ROOT}/{ent}/II120.npy", mmap_mode="r"); hf = np.load(f"{ROOT}/{ent}/vitals_hf.npy", mmap_mode="r")
    s = ms2dt(tm[0]); t = s.replace(minute=0, second=0, microsecond=0); sw = None
    while t < ms2dt(tm[-1]):
        if ds(t) != ds(t + timedelta(hours=1)): sw = t + timedelta(hours=1); break
        t += timedelta(hours=1)
    k_sw = int(np.searchsorted(tm, int((sw - datetime(1970, 1, 1)).total_seconds() * 1000))); n = len(tm)
    i0, i1 = max(0, k_sw - 2880), max(0, k_sw - 120); j0, j1 = min(n, k_sw + 120), min(n, k_sw + 2880)
    meta = json.load(open(f"{ROOT}/{ent}/meta.json")); cls = {k: v.get("class") for k, v in meta.get("vitals_hf", {}).get("files", {}).items() if k.startswith("HR")}
    before = best_lag(ecg_hr(ecg, i0, i1), np.nanmean(np.asarray(hf[i0:i1, :, 0], dtype=np.float32), axis=1)) if i1 - i0 > 240 else None
    after = best_lag(ecg_hr(ecg, j0, j1), np.nanmean(np.asarray(hf[j0:j1, :, 0], dtype=np.float32), axis=1)) if j1 - j0 > 240 else None
    print(f"{ent} {q.get('unit')} {q.get('wynton_folder')} HR class {cls} GE {s:%Y-%m-%d %H:%M} .. {ms2dt(tm[-1]):%m-%d %H:%M} switch {sw:%m-%d %H:%M} | ECG-vs-vitals_hf lag (min, corr, corr@0): before {before} after {after}")
