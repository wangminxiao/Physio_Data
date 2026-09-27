# Read-only: do charted HR events (ehr_events var 100) align with the ECG-derived HR of the waveform in each store?
# Lag scan -8 h .. +8 h (5-min steps); a +4/+5 h peak = the naive-local .timestamp() vs pandas-UTC mismatch.
import json, sys, os, random, collections, numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from datetime import datetime, timedelta
sys.path.insert(0, "/mnt/localdata/storage/mxwang/Physio_Data"); from physio_data.schema import EHR_EVENT_DTYPE  # noqa
STORES = {"mimic3": "/mnt/localdata100tb/physio_data/mimic3", "mover": "/mnt/localdata100tb/physio_data/mover", "mcmed": "/mnt/localdata100tb/physio_data/mcmed"}
N_ENT = int(sys.argv[1]) if len(sys.argv) > 1 else 20
b_, a_ = butter(3, [5 / 60, 30 / 60], btype="band")
def ecg_hr_seg(x, fs):
    if not np.isfinite(x).all() or np.nanstd(x) < 20: return np.nan
    y = filtfilt(b_, a_, x); pk, _ = find_peaks(np.abs(y), distance=int(0.3 * fs), height=np.percentile(np.abs(y), 98) * 0.4)
    return pk.size * 2 if 8 <= pk.size <= 120 else np.nan
ms2dt = lambda ms: datetime(1970, 1, 1) + timedelta(milliseconds=int(ms))
for name, root in STORES.items():
    print(f"\n===== {name}: {root}", flush=True)
    try: man = json.load(open(f"{root}/manifest.json"))
    except Exception as e: print("no manifest:", e); continue
    man = man if isinstance(man, list) else man.get("entities", list(man.values()))
    print("manifest entries:", len(man), "| keys:", sorted(man[0].keys())[:30])
    for m in man:
        if "entity_id" not in m: m["entity_id"] = os.path.basename(str(m.get("dir") or m.get("path") or "").rstrip("/"))
    ents = [m for m in man if int(m.get("n_seg") or m.get("n_segments") or m.get("n_windows") or 0) >= 2000] or list(man)
    random.seed(7); random.shuffle(ents); ent0 = ents[0]
    meta = json.load(open(f"{root}/{ent0['entity_id']}/meta.json")); print("meta keys:", sorted(meta.keys())[:40]); print("entity files:", sorted(os.listdir(f"{root}/{ent0['entity_id']}")))
    for k in ("record", "record_name", "wav_start", "wav_start_ms", "base_time", "episode_start_ms", "source_record", "start_time", "wave_start_ms"):
        if k in meta: print(f"  meta[{k}] = {meta[k]}")
    tm0 = int(np.load(f"{root}/{ent0['entity_id']}/time_ms.npy")[0]); print(f"  time_ms[0] = {tm0} -> as UTC {ms2dt(tm0)}")
    import re, glob
    blob = json.dumps(meta); recs = sorted(set(re.findall(r"p\d{6}-\d{4}-\d{2}-\d{2}-\d{2}-\d{2}", blob)))
    if recs and name == "mimic3":
        import wfdb
        RAW = "/mnt/localdata/storage/MIMIC_waveform_matched_subset/physionet.org/files/mimic3wdb-matched/1.0"
        for r in recs[:3]:
            hits = glob.glob(f"{RAW}/p0*/{r[:7]}/{r}.hea")
            if hits:
                h = wfdb.rdheader(hits[0][:-4]); bd = h.base_datetime
                wall = int((bd - datetime(1970, 1, 1)).total_seconds() * 1000) if bd else None
                print(f"  raw header {r}: base_datetime={bd} wall_ms={wall} | time_ms[0]-wall = {(tm0 - wall)/3.6e6 if wall else None} h")
            else: print("  raw header not found for", r)
    lags = []; n_try = 0
    for m in ents:
        if len(lags) >= N_ENT or n_try > 6 * N_ENT: break
        n_try += 1; d = f"{root}/{m['entity_id']}"
        try:
            ev = np.load(f"{d}/ehr_events.npy"); tm = np.load(f"{d}/time_ms.npy")
            if len(tm) < 2000: continue
            hr = ev[ev["var_id"] == 100]
            if hr.size < 12: continue
            ecg_path = f"{d}/II120.npy"
            if not os.path.exists(ecg_path): continue
            ecg = np.load(ecg_path, mmap_mode="r"); fs = ecg.shape[1] / 30.0
            # ECG-HR on every 2nd segment over the first 48 h
            n = min(len(tm), 5760); idx = np.arange(0, n, 2); h = np.array([ecg_hr_seg(np.asarray(ecg[i], dtype=np.float32), fs) for i in idx]); t_seg = tm[idx]
            if np.isfinite(h).sum() < 200: continue
            tc = hr["time_ms"].astype(np.int64); vc = hr["value"].astype(np.float64); m_ok = (vc > 20) & (vc < 220); tc, vc = tc[m_ok], vc[m_ok]
            best = None; res = {}
            for L in range(-480, 481, 5):
                errs = []
                for t_ms, v in zip(tc, vc):
                    c = t_ms + L * 60000; a, b = np.searchsorted(t_seg, c - 300000), np.searchsorted(t_seg, c + 300000); w = h[a:b]; w = w[np.isfinite(w)]
                    if w.size >= 3: errs.append(abs(float(np.median(w)) - v))
                if len(errs) >= 8: res[L] = float(np.mean(errs))
            if len(res) < 20: continue
            L = min(res, key=res.get); base = res.get(0, np.nan)
            lags.append((L, round(res[L], 1), round(base, 1), len(tc)))
        except Exception as e:
            continue
    if not lags: print("no usable entity"); continue
    Ls = [l for l, _, _, _ in lags]; c = collections.Counter(int(round(l / 30.0)) * 30 for l in Ls)
    print(f"entities with a lag estimate: {len(lags)} | best lag (min) histogram (30-min bins): {dict(sorted(c.items()))}")
    print(f"median best lag = {np.median(Ls):.0f} min | MAE at best vs at 0: {np.median([b for _, b, _, _ in lags]):.1f} vs {np.median([z for _, _, z, _ in lags]):.1f} bpm")
    print("examples (lag, mae_best, mae_at_0, n_charted):", lags[:8])
