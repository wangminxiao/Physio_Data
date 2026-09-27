import json, sys, os, random, collections, glob, re, numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from datetime import datetime, timedelta
sys.path.insert(0, "/mnt/localdata/storage/mxwang/Physio_Data"); from physio_data.schema import EHR_EVENT_DTYPE  # noqa
STORES = {"mimic3": "/mnt/localdata100tb/physio_data/mimic3", "mover": "/mnt/localdata100tb/physio_data/mover", "mcmed": "/mnt/localdata100tb/physio_data/mcmed"}
N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
bp_ppg = butter(3, [0.5 / 20, 3.0 / 20], btype="band"); bp_ecg = butter(3, [5 / 60, 30 / 60], btype="band")
ms2dt = lambda ms: datetime(1970, 1, 1) + timedelta(milliseconds=int(ms))
def rate_seg(x, fs, bp, dist_s=0.3, absval=False):
    if np.isfinite(x).mean() < 0.9 or np.nanstd(x) < 1e-3: return np.nan
    x = np.where(np.isfinite(x), x, np.nanmean(x)); y = filtfilt(bp[0], bp[1], x); y = np.abs(y) if absval else y
    pk, _ = find_peaks(y, distance=int(dist_s * fs), height=np.percentile(y, 90) * 0.3); return pk.size * 2 if 8 <= pk.size <= 120 else np.nan
def lag_scan(t_seg, rate, tc, vc, lo=-720, hi=720, step=5):
    res = {}
    for L in range(lo, hi + 1, step):
        errs = []
        for t_ms, v in zip(tc, vc):
            c = t_ms + L * 60000; a, b = np.searchsorted(t_seg, c - 300000), np.searchsorted(t_seg, c + 300000); w = rate[a:b]; w = w[np.isfinite(w)]
            if w.size >= 2: errs.append(abs(float(np.median(w)) - v))
        if len(errs) >= 8: res[L] = float(np.mean(errs))
    return res
for name, root in STORES.items():
    print(f"\n===== {name}", flush=True)
    man = json.load(open(f"{root}/manifest.json"))
    for m in man:
        if "entity_id" not in m: m["entity_id"] = os.path.basename(str(m.get("dir")).rstrip("/"))
    random.seed(11); random.shuffle(man); reasons = collections.Counter(); out = []
    for m in man[:400]:
        if len(out) >= N: break
        d = f"{root}/{m['entity_id']}"
        try:
            tm = np.load(f"{d}/time_ms.npy"); ev = np.load(f"{d}/ehr_events.npy")
        except Exception as e: reasons["load_fail"] += 1; continue
        if len(tm) < 720: reasons["short(<6h)"] += 1; continue
        vc_all = collections.Counter(ev["var_id"].tolist())
        hr = ev[(ev["var_id"] == 100) | (ev["var_id"] == 156)]
        if hr.size < 10: reasons[f"few_hr_events(top vars {vc_all.most_common(4)})"] += 1; continue
        pl_ = np.load(f"{d}/PLETH40.npy", mmap_mode="r"); n = min(len(tm), 5760); idx = np.arange(0, n, 2)
        pr = np.array([rate_seg(np.asarray(pl_[i], dtype=np.float32), 40.0, bp_ppg) for i in idx]); t_seg = tm[idx]
        if np.isfinite(pr).sum() < 100: reasons["ppg_rate_unavailable"] += 1; continue
        tc = hr["time_ms"].astype(np.int64); vc = hr["value"].astype(np.float64); ok = (vc > 25) & (vc < 220); tc, vc = tc[ok], vc[ok]
        res = lag_scan(t_seg, pr, tc, vc)
        if len(res) < 20: reasons["no_overlap_in_scan"] += 1; continue
        L = min(res, key=res.get); out.append((m["entity_id"][:24], L, round(res[L], 1), round(res.get(0, float("nan")), 1), int(tc.size), round(float(np.isfinite(pr).mean()), 2)))
        if len(out) <= 3:
            print("first entity:", m["entity_id"], "| n_seg", len(tm), "| event var counts:", vc_all.most_common(6), "| charted HR events:", int(hr.size), "| time_ms[0]:", ms2dt(tm[0]), "| first HR event:", ms2dt(tc.min()) if tc.size else None)
            if name == "mimic3":
                meta = json.load(open(f"{d}/meta.json")); sp = (meta.get("source_path") or "").replace("/labs/hulab/", "/mnt/localdata/storage/"); print("  source_path:", sp, "| recording_start_ms:", meta.get("recording_start_ms"), ms2dt(meta["recording_start_ms"]) if meta.get("recording_start_ms") else "")
                try:
                    import wfdb
                    heas = sorted(glob.glob(os.path.join(sp, "p*-*-*-*-*-*.hea"))) if sp and os.path.isdir(sp) else ([sp + ".hea"] if sp and os.path.exists(sp + ".hea") else [])
                    for hp in heas[:4]:
                        h = wfdb.rdheader(hp[:-4]); bd = h.base_datetime; wall = int((bd - datetime(1970, 1, 1)).total_seconds() * 1000) if bd else None
                        print(f"  raw master header {os.path.basename(hp)}: base_datetime={bd} | time_ms[0] - wall = {(int(tm[0]) - wall) / 3.6e6 if wall else None:.2f} h")
                except Exception as e: print("  header check failed:", e)
    print("skip reasons:", dict(reasons))
    if out:
        Ls = [o[1] for o in out]; print(f"entities: {len(out)} | best lag (min) per entity: {Ls} | median {np.median(Ls):.0f} | MAE best vs at 0 (bpm): {np.median([o[2] for o in out]):.1f} vs {np.median([o[3] for o in out]):.1f}")
        clear = [o for o in out if o[2] < 0.75 * o[3] and o[4] >= 10]
        print(f"CLEAR improvements (MAE_best < 0.75*MAE_0): {len(clear)} of {len(out)} | their lags: {sorted(o[1] for o in clear)}")
        print("rows (id, lag, mae_best, mae0, n_hr, ppg_rate_frac):", out)
