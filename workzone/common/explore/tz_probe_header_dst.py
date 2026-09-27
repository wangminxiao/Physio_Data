import json, sys, os, random, collections, glob, numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
sys.path.insert(0, "/mnt/localdata/storage/mxwang/Physio_Data"); from physio_data.schema import EHR_EVENT_DTYPE  # noqa
import wfdb
NY, LA = ZoneInfo("America/New_York"), ZoneInfo("America/Los_Angeles")
ms2dt = lambda ms: datetime(1970, 1, 1) + timedelta(milliseconds=int(ms))
bp_ppg = butter(3, [0.5 / 20, 3.0 / 20], btype="band")
def rate_seg(x, fs=40.0):
    if np.isfinite(x).mean() < 0.9 or np.nanstd(x) < 1e-3: return np.nan
    x = np.where(np.isfinite(x), x, np.nanmean(x)); y = filtfilt(bp_ppg[0], bp_ppg[1], x)
    pk, _ = find_peaks(y, distance=int(0.3 * fs), height=np.percentile(y, 90) * 0.3); return pk.size * 2 if 8 <= pk.size <= 120 else np.nan
def lag_scan(t_seg, rate, tc, vc, lo=-720, hi=720, step=5):
    res = {}
    for L in range(lo, hi + 1, step):
        errs = []
        for t_ms, v in zip(tc, vc):
            c = t_ms + L * 60000; a, b = np.searchsorted(t_seg, c - 300000), np.searchsorted(t_seg, c + 300000); w = rate[a:b]; w = w[np.isfinite(w)]
            if w.size >= 2: errs.append(abs(float(np.median(w)) - v))
        if len(errs) >= 8: res[L] = float(np.mean(errs))
    return res
# ---------- 1. MIMIC: code-level offset per entity (time_ms[0] vs raw master header base time), 40 entities
print("===== MIMIC header check", flush=True)
root = "/mnt/localdata100tb/physio_data/mimic3"; man = json.load(open(f"{root}/manifest.json")); random.seed(3); random.shuffle(man); offs = collections.Counter(); n_chk = 0; rows = []
for m in man:
    if n_chk >= 40: break
    eid = os.path.basename(m["dir"].rstrip("/")); d = f"{root}/{eid}"
    try:
        meta = json.load(open(f"{d}/meta.json")); tm0 = int(np.load(f"{d}/time_ms.npy")[0]); sp = (meta.get("source_path") or "").replace("/labs/hulab/", "/mnt/localdata/storage/")
        heas = sorted(glob.glob(os.path.join(sp, "p*-*-*-*-*-*.hea"))); best = None
        for hp in heas:
            if hp.endswith("n.hea"): continue
            bd = wfdb.rdheader(hp[:-4]).base_datetime
            if bd is None: continue
            wall = int((bd - datetime(1970, 1, 1)).total_seconds() * 1000); dh = (tm0 - wall) / 3.6e6
            if best is None or abs(dh) < abs(best[0]): best = (dh, bd)
        if best is None: continue
        dh, bd = best; n_chk += 1
        exp = -bd.replace(tzinfo=NY).utcoffset().total_seconds() / 3600   # +4 (EDT) / +5 (EST) at the surrogate date
        offs[(round(dh, 2), exp)] += 1; rows.append((eid, round(dh, 2), exp, bd.month))
    except Exception as e: continue
print("(time_ms[0] - header wall clock in h, expected local UTC offset at surrogate date):", dict(offs))
print("match expected:", sum(1 for r in rows if abs(r[1] - r[2]) < 0.02), "of", len(rows))
# ---------- 2. MOVER: lag vs Pacific DST state, 60 entities
print("\n===== MOVER lag vs DST", flush=True)
root = "/mnt/localdata100tb/physio_data/mover"; man = json.load(open(f"{root}/manifest.json")); random.seed(5); random.shuffle(man); out = []
for m in man:
    if len(out) >= 60: break
    d = f"{root}/{m['entity_id']}"
    try:
        tm = np.load(f"{d}/time_ms.npy"); ev = np.load(f"{d}/ehr_events.npy")
        if len(tm) < 360: continue
        hr = ev[ev["var_id"] == 100]
        if hr.size < 60: continue
        pl_ = np.load(f"{d}/PLETH40.npy", mmap_mode="r"); idx = np.arange(0, min(len(tm), 2880), 1); pr = np.array([rate_seg(np.asarray(pl_[i], dtype=np.float32)) for i in idx]); t_seg = tm[idx]
        if np.isfinite(pr).sum() < 60: continue
        tc = hr["time_ms"].astype(np.int64); vc = hr["value"].astype(np.float64); ok = (vc > 25) & (vc < 220); tc, vc = tc[ok], vc[ok]
        res = lag_scan(t_seg, pr, tc, vc, -180, 180, 5)
        if len(res) < 20: continue
        L = min(res, key=res.get); r0 = res.get(0, np.nan); clear = res[L] < 0.75 * r0
        dt = ms2dt(tm[0]); dst = int(dt.replace(tzinfo=LA).dst().total_seconds() != 0)
        out.append((m["entity_id"][:12], L, round(res[L], 1), round(r0, 1), clear, dst, dt.strftime("%Y-%m")))
    except Exception: continue
tab = collections.defaultdict(collections.Counter)
for e, L, mb, m0, clear, dst, ym in out:
    band = "0" if abs(L) <= 10 else ("-60" if -75 <= L <= -45 else ("+60" if 45 <= L <= 75 else "other"))
    tab[dst][band + ("*" if clear else "")] += 1
print("entities:", len(out), "| DST=1 (PDT):", dict(tab[1]), "| DST=0 (PST):", dict(tab[0]), "  (* = clear MAE improvement)")
print("MAE at lag 0 median:", np.median([o[3] for o in out]), "| at best:", np.median([o[2] for o in out]))
# ---------- 3. MC_MED: shorter stays allowed, >=60 HR events, 40 entities
print("\n===== MC_MED extended", flush=True)
root = "/mnt/localdata100tb/physio_data/mcmed"; man = json.load(open(f"{root}/manifest.json")); random.seed(9); random.shuffle(man); out = []
for m in man:
    if len(out) >= 40: break
    d = f"{root}/{m['entity_id']}"
    try:
        tm = np.load(f"{d}/time_ms.npy"); ev = np.load(f"{d}/ehr_events.npy")
        if len(tm) < 240: continue
        hr = ev[ev["var_id"] == 100]
        if hr.size < 60: continue
        pl_ = np.load(f"{d}/PLETH40.npy", mmap_mode="r"); idx = np.arange(0, min(len(tm), 2880), 1); pr = np.array([rate_seg(np.asarray(pl_[i], dtype=np.float32)) for i in idx]); t_seg = tm[idx]
        if np.isfinite(pr).sum() < 60: continue
        tc = hr["time_ms"].astype(np.int64); vc = hr["value"].astype(np.float64); ok = (vc > 25) & (vc < 220); tc, vc = tc[ok], vc[ok]
        res = lag_scan(t_seg, pr, tc, vc, -360, 360, 5)
        if len(res) < 20: continue
        L = min(res, key=res.get); r0 = res.get(0, np.nan); clear = res[L] < 0.75 * r0
        out.append((m["entity_id"], L, round(res[L], 1), round(r0, 1), clear, int(tc.size)))
    except Exception: continue
c = collections.Counter(("0" if abs(o[1]) <= 10 else ("+-60" if 45 <= abs(o[1]) <= 75 else "other")) + ("*" if o[4] else "") for o in out)
print("entities:", len(out), "| lag bands:", dict(c), "| MAE at 0 median:", np.median([o[3] for o in out]) if out else None, "| best:", np.median([o[2] for o in out]) if out else None)
print("non-zero clear cases:", [(o[0], o[1], o[2], o[3], o[5]) for o in out if o[4] and abs(o[1]) > 10])
