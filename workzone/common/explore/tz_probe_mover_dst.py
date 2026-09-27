import json, sys, os, random, collections, numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
sys.path.insert(0, "/mnt/localdata/storage/mxwang/Physio_Data"); from physio_data.schema import EHR_EVENT_DTYPE  # noqa
LA = ZoneInfo("America/Los_Angeles"); ms2dt = lambda ms: datetime(1970, 1, 1) + timedelta(milliseconds=int(ms))
bp = butter(3, [0.5 / 20, 3.0 / 20], btype="band")
def rate_seg(x, fs=40.0):
    if np.isfinite(x).mean() < 0.9 or np.nanstd(x) < 1e-3: return np.nan
    x = np.where(np.isfinite(x), x, np.nanmean(x)); y = filtfilt(bp[0], bp[1], x); pk, _ = find_peaks(y, distance=int(0.3 * fs), height=np.percentile(y, 90) * 0.3); return pk.size * 2 if 8 <= pk.size <= 120 else np.nan
def lag_scan(t_seg, rate, tc, vc, lo=-180, hi=180, step=5):
    res = {}
    for L in range(lo, hi + 1, step):
        errs = []
        for t_ms, v in zip(tc, vc):
            c = t_ms + L * 60000; a, b = np.searchsorted(t_seg, c - 300000), np.searchsorted(t_seg, c + 300000); w = rate[a:b]; w = w[np.isfinite(w)]
            if w.size >= 2: errs.append(abs(float(np.median(w)) - v))
        if len(errs) >= 8: res[L] = float(np.mean(errs))
    return res
root = "/mnt/localdata100tb/physio_data/mover"; man = json.load(open(f"{root}/manifest.json")); random.seed(21); random.shuffle(man); out = []
for m in man:
    if len(out) >= 120: break
    d = f"{root}/{m['entity_id']}"
    try:
        tm = np.load(f"{d}/time_ms.npy"); ev = np.load(f"{d}/ehr_events.npy"); hr = ev[ev["var_id"] == 100]
        if len(tm) < 360 or hr.size < 60: continue
        pl_ = np.load(f"{d}/PLETH40.npy", mmap_mode="r"); idx = np.arange(0, min(len(tm), 2880)); pr = np.array([rate_seg(np.asarray(pl_[i], dtype=np.float32)) for i in idx]); t_seg = tm[idx]
        if np.isfinite(pr).sum() < 60: continue
        tc = hr["time_ms"].astype(np.int64); vc = hr["value"].astype(np.float64); ok = (vc > 25) & (vc < 220); tc, vc = tc[ok], vc[ok]
        res = lag_scan(t_seg, pr, tc, vc)
        if len(res) < 20: continue
        L = min(res, key=res.get); r0 = res.get(0, np.nan); clear = res[L] < 0.75 * r0
        band = "0" if abs(L) <= 10 else ("-60" if -75 <= L <= -45 else ("+60" if 45 <= L <= 75 else "other"))
        dt = ms2dt(tm[0]); meta = json.load(open(f"{d}/meta.json"))
        out.append(dict(e=m["entity_id"][:10], band=band, clear=clear, L=L, mae=round(res[L], 1), mae0=round(r0, 1), dst=int(dt.replace(tzinfo=LA).dst().total_seconds() != 0), ym=dt.strftime("%Y-%m"), or_start=meta.get("or_start_ms"), n_xml=meta.get("n_xml_files_parsed")))
    except Exception: continue
print("entities:", len(out))
tab = collections.defaultdict(collections.Counter)
for o in out: tab[o["ym"]][o["band"] + ("*" if o["clear"] else "")] += 1
for ym in sorted(tab): print(ym, "dst" if int(ym[5:]) in (4,5,6,7,8,9,10) else "std", dict(tab[ym]))
print("\nby DST:", {k: dict(v) for k, v in {d_: collections.Counter(o["band"] + ("*" if o["clear"] else "") for o in out if o["dst"] == d_) for d_ in (0, 1)}.items()})
# waveform start vs OR start (EHR) for the +-60 groups
for band in ("0", "-60", "+60"):
    v = [(o["or_start"] - int(np.load(f"{root}/{[m for m in man if m['entity_id'].startswith(o['e'])][0]['entity_id']}/time_ms.npy")[0])) / 60000 for o in out if o["band"] == band and o["or_start"]]
    if v: print(f"band {band}: OR_start(EHR) - wave_start (min) median {np.median(v):.0f}, p25 {np.percentile(v,25):.0f}, p75 {np.percentile(v,75):.0f}, n={len(v)}")
