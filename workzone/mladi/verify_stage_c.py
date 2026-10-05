#!/usr/bin/env python3
"""Gate C for MLADI (non-zero exit on failure; verify_stage_c.json in the Stage A dir).

  completion  meta.vitals_hf present on >= 99 % of Stage B entities
  sample      300 entities: vitals_hf float32 [n_seg, 30, 11], abp_src uint8 [n_seg, 30], nbp_events
              EHR_EVENT_DTYPE sorted, var_id in {157, 158, 159}, seg_idx in [0, n_seg), values in range
  coverage    HR_hf slot coverage median >= 0.8
  physiology  40 entities with II: heart rate from II120 R-peaks vs HR_hf, 1-min means over 2 h,
              median per-entity correlation >= 0.6
"""
import argparse, glob, json, os, random, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import cfg  # noqa: E402


def ecg_hr_per_min(x, fs=120):
    from scipy.signal import butter, filtfilt, find_peaks
    x = np.nan_to_num(x.astype(np.float64))
    b, a = butter(2, [5 / (fs / 2), 20 / (fs / 2)], "band")
    e = filtfilt(b, a, x) ** 2
    pk, _ = find_peaks(e, distance=int(0.27 * fs), height=0.2 * np.percentile(e, 99))
    n = x.size // (60 * fs)
    return np.array([np.sum((pk >= k * 60 * fs) & (pk < (k + 1) * 60 * fs)) for k in range(n)], float)


def main():
    C = cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=C["output_dir"]); ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--stage-a", default=os.path.join(C["intermediate_dir"], "stage_a"))
    a = ap.parse_args()
    ents = [d for d in os.listdir(a.out) if os.path.exists(os.path.join(a.out, d, "meta.json"))]
    done = [d for d in ents if "vitals_hf" in json.load(open(os.path.join(a.out, d, "meta.json")))]
    res, fails = {"n_entities": len(ents), "n_done": len(done)}, []
    if len(done) < 0.99 * len(ents): fails.append(f"vitals_hf on {len(done)} of {len(ents)}")
    random.seed(0); S = random.sample(done, min(a.n, len(done)))
    probs, cov = [], []
    for e in S:
        d = os.path.join(a.out, e)
        try:
            V = np.load(os.path.join(d, "vitals_hf.npy"), mmap_mode="r"); s = np.load(os.path.join(d, "vitals_hf_abp_src.npy"), mmap_mode="r")
            ev = np.load(os.path.join(d, "nbp_events.npy")); n = np.load(os.path.join(d, "time_ms.npy"), mmap_mode="r").size
            ok = V.dtype == np.float32 and V.shape == (n, 30, 11) and s.dtype == np.uint8 and s.shape == (n, 30)
            if ev.size:
                ok &= bool(np.all(np.diff(ev["time_ms"]) >= 0) and set(np.unique(ev["var_id"]).tolist()) <= {157, 158, 159}
                           and ev["seg_idx"].min() >= 0 and ev["seg_idx"].max() < n and np.all((ev["value"] > 10) & (ev["value"] < 320)))
            if not ok: probs.append(e[:8]); continue
            cov.append(float(np.isfinite(V[:, :, 0]).mean()))
        except Exception as ex:
            probs.append(f"{e[:8]} {type(ex).__name__}")
    res.update(sample=len(S), n_problems=len(probs), hr_cov_median=float(np.median(cov)) if cov else None)
    if len(probs) > 0.01 * max(1, len(S)): fails.append(f"{len(probs)} problem entities")
    if cov and np.median(cov) < 0.8: fails.append(f"HR_hf coverage median {np.median(cov):.2f} < 0.8")
    rr = []
    for e in S[:120]:
        if len(rr) >= 40:
            break
        d = os.path.join(a.out, e)
        E = np.load(os.path.join(d, "II120.npy"), mmap_mode="r"); V = np.load(os.path.join(d, "vitals_hf.npy"), mmap_mode="r")
        n = E.shape[0]
        if n < 300:
            continue
        i0 = n // 2; i1 = i0 + 240                     # 2 h of consecutive rows (if contiguous)
        t = np.load(os.path.join(d, "time_ms.npy"))
        if not np.all(np.diff(t[i0:i1]) == 30000):
            continue
        x = np.asarray(E[i0:i1], np.float32).reshape(-1)
        if not np.isfinite(x).mean() > 0.95:
            continue
        h_ecg = ecg_hr_per_min(x)
        h = np.asarray(V[i0:i1, :, 0]).reshape(-1)
        h_mon = np.array([np.nanmean(h[k * 60:(k + 1) * 60]) if np.isfinite(h[k * 60:(k + 1) * 60]).any() else np.nan for k in range(h_ecg.size)])
        m = np.isfinite(h_mon) & (h_ecg > 20)
        if m.sum() > 30 and np.std(h_mon[m]) > 0.5 and np.std(h_ecg[m]) > 0.5:
            rr.append(float(np.corrcoef(h_ecg[m], h_mon[m])[0, 1]))
    res.update(ecg_vs_hr_hf_corr_median=float(np.median(rr)) if rr else None, n_corr=len(rr))
    if rr and np.median(rr) < 0.6: fails.append(f"ECG HR vs HR_hf corr median {np.median(rr):.2f} < 0.6")
    res["fails"] = fails
    json.dump(res, open(os.path.join(a.stage_a, "verify_stage_c.json"), "w"), indent=1)
    print(json.dumps(res, indent=1)); print("GATE C:", "FAIL" if fails else "PASS", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
