#!/usr/bin/env python3
"""Gate C4 for MLADI (non-zero exit on failure; verify_stage_c4.json in the Stage A dir). On a sample of entities,
three 10-min chunks from runs with a fitted sawtooth, R -> foot delays measured on PLETH40 and on PLETH40_aligned:

  completion   meta.pleth_aligned on >= 99 % of entities with Stage C3 output
  level        median (raw level - aligned level) within 5 ms of D_device
  sawtooth     residual ramp: delays folded on C4's refined resets (time since the last reset, per-cycle demeaned)
               -> within-cycle slope; median on the aligned trace < 5 ms/min (raw ~29). A free refit
               (fit_sawtooth) on the aligned delays is reported only: with real beat noise it finds a few spurious
               drops in 30 min, so "ok" alone does not mean a sawtooth is left; its plausible-pattern share
               (period 50-90 s, ramp 15-45 ms/min, >= 30 resets/h) is reported too
  spread       median 6-s spread (10-min detrended) aligned / raw < 0.9
  morphology   beat templates (foot - 0.2 s .. + 0.8 s, mean over the chunk) aligned vs raw: median r >= 0.99
  nan          median added NaN fraction < 1 %
"""
import argparse, json, os, random, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pleth_timing_lib as L  # noqa: E402
from common import cfg  # noqa: E402


def delays(E, P, tms, c0, c1, prior):
    r = L.ecg_beats(np.asarray(E[c0:c1], np.float32).reshape(-1), 120)
    f = L.ppg_beats(np.asarray(P[c0:c1], np.float32).reshape(-1), 40)
    if r is None or f is None or r.size < 50 or f.size < 50:
        return None
    est = L.estimate_offset(r, f, prior=prior)
    if est is None:
        return None
    idx, _ = L.pair_nearest(r, f, est[0], tol=0.12); ok = idx >= 0
    return tms[c0] / 1000.0 + r[ok], f[idx[ok]] - r[ok], f


def folded_slope(t, d, resets):
    """ms/min: slope of d against time since the last reset, each cycle demeaned (cycles of 20-90 s only)."""
    resets = np.sort(np.asarray(resets, float))
    k = np.searchsorted(resets, t, side="right") - 1
    ok = (k >= 0) & (k < resets.size - 1)
    if ok.sum() < 50:
        return None
    k, tt, dd = k[ok], t[ok], d[ok]
    clen = resets[k + 1] - resets[k]
    m = (clen > 20) & (clen < 90)
    k, tt, dd = k[m], tt[m], dd[m]
    if k.size < 50:
        return None
    ph = tt - resets[k]
    x = np.empty_like(ph); y = np.empty_like(dd)
    for c in np.unique(k):
        i = k == c
        x[i] = ph[i] - ph[i].mean(); y[i] = dd[i] - dd[i].mean()
    return float(np.sum(x * y) / max(1e-12, np.sum(x * x))) * 60000.0


def plausible(fit, hours):
    return bool(fit["ok"] and 50 <= fit["period"] <= 90 and 15 <= fit["ramp"] * 60000 <= 45 and len(fit["resets"]) / max(hours, 1e-9) >= 30)


def spread6(t, d):
    _, m = L._bin_medians(t, d, 6.0); _, m10 = L._bin_medians(t, d, 600.0)
    j = np.minimum((np.arange(m.size) * 6.0 / 600.0).astype(int), m10.size - 1)
    return float(np.nanstd(m - m10[j]))


def template(P, c0, c1, feet):
    x = np.asarray(P[c0:c1], np.float32).reshape(-1).astype(np.float64)
    k = (np.round(feet * 40)).astype(int)
    k = k[(k - 8 >= 0) & (k + 32 < x.size)]
    if k.size < 20:
        return None
    W = x[k[:, None] + np.arange(-8, 32)[None]]
    W = W[np.isfinite(W).all(1)]
    return W.mean(0) if W.shape[0] >= 20 else None


def main():
    C = cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=C["output_dir"]); ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--stage-a", default=os.path.join(C["intermediate_dir"], "stage_a"))
    a = ap.parse_args()
    ents = [d for d in os.listdir(a.out) if os.path.exists(os.path.join(a.out, d, "pleth_timing_runs.npy"))]
    done = [d for d in ents if "pleth_aligned" in json.load(open(os.path.join(a.out, d, "meta.json")))]
    res, fails = {"n_entities": len(ents), "n_done": len(done)}, []
    if len(done) < 0.99 * len(ents): fails.append(f"pleth_aligned on {len(done)} of {len(ents)}")
    random.seed(0); random.shuffle(done)
    dlev, saw_raw, saw_al, ratio, tr, nan_add, Ddev, n_used = [], [], [], [], [], [], None, 0
    slope_raw, slope_al, pl_raw, pl_al = [], [], [], []
    for e in done:
        if n_used >= a.n:
            break
        d = os.path.join(a.out, e); m = json.load(open(os.path.join(d, "meta.json")))
        rt = np.load(os.path.join(d, "pleth_timing_runs.npy")); runs = [r for r in rt if r["ok"] and r["seg1"] - r["seg0"] >= 60]
        if not runs:
            continue
        Ddev = m["pleth_aligned"]["D_device_ms"] / 1000.0
        tms = np.load(os.path.join(d, "time_ms.npy"))
        E = np.load(os.path.join(d, "II120.npy"), mmap_mode="r"); Pr = np.load(os.path.join(d, "PLETH40.npy"), mmap_mode="r")
        Pa = np.load(os.path.join(d, "PLETH40_aligned.npy"), mmap_mode="r")
        r = max(runs, key=lambda x: x["seg1"] - x["seg0"]); a0, b0 = int(r["seg0"]), int(r["seg1"])
        prior = (m["pleth_timing"].get("initial_offset_ms") or 1200.0) / 1000.0
        TR, DR, TA, DA = [], [], [], []
        for c0 in np.linspace(a0, b0 - 20, 3).astype(int):
            x = delays(E, Pr, tms, c0, c0 + 20, prior); y = delays(E, Pa, tms, c0, c0 + 20, prior - Ddev)
            if x is None or y is None:
                continue
            TR.append(x[0]); DR.append(x[1]); TA.append(y[0]); DA.append(y[1])
            t1, t2 = template(Pr, c0, c0 + 20, x[2]), template(Pa, c0, c0 + 20, y[2])
            if t1 is not None and t2 is not None:
                tr.append(float(np.corrcoef(t1, t2)[0, 1]))
        if not TR:
            continue
        TR, DR, TA, DA = map(np.concatenate, (TR, DR, TA, DA))
        dlev.append((np.median(DR) - np.median(DA)) * 1000.0 - Ddev * 1000.0)
        fr, fa = L.fit_sawtooth(TR, DR), L.fit_sawtooth(TA, DA)
        saw_raw.append(fr["ok"]); saw_al.append(fa["ok"])
        hrs = (len(TR) and (TR.max() - TR.min()) / 3600.0) or 0.0
        pl_raw.append(plausible(fr, min(hrs, 0.5))); pl_al.append(plausible(fa, min(hrs, 0.5)))
        rres = np.load(os.path.join(d, "pleth_aligned_resets.npy"))[r["reset0"]: r["reset0"] + r["n_resets"]] / 1000.0
        sr_, sa_ = folded_slope(TR, DR, rres), folded_slope(TA, DA, rres)
        if sr_ is not None and sa_ is not None:
            slope_raw.append(sr_); slope_al.append(sa_)
        ratio.append(spread6(TA, DA) / max(1e-9, spread6(TR, DR)))
        nan_add.append(m["pleth_aligned"]["nan_frac_aligned"] - m["pleth_aligned"]["nan_frac_raw"])
        n_used += 1
    res.update(n_checked=n_used, D_device_ms=None if Ddev is None else round(Ddev * 1000, 2),
               level_minus_Ddev_ms=dict(median=float(np.median(dlev)), p5=float(np.percentile(dlev, 5)), p95=float(np.percentile(dlev, 95))) if dlev else None,
               sawtooth_refit_ok_raw=float(np.mean(saw_raw)) if saw_raw else None, sawtooth_refit_ok_aligned=float(np.mean(saw_al)) if saw_al else None,
               refit_plausible_raw=float(np.mean(pl_raw)) if pl_raw else None, refit_plausible_aligned=float(np.mean(pl_al)) if pl_al else None,
               folded_slope_ms_per_min_raw=dict(median=float(np.median(slope_raw)), p90=float(np.percentile(slope_raw, 90))) if slope_raw else None,
               folded_slope_ms_per_min_aligned=dict(median=float(np.median(slope_al)), p10=float(np.percentile(slope_al, 10)),
                                                    p90=float(np.percentile(slope_al, 90))) if slope_al else None,
               spread6_ratio_median=float(np.median(ratio)) if ratio else None, template_r_median=float(np.median(tr)) if tr else None,
               nan_added_median=float(np.median(nan_add)) if nan_add else None)
    if dlev and abs(np.median(dlev)) > 5: fails.append(f"level shift off D_device by {np.median(dlev):.1f} ms")
    if slope_al and abs(np.median(slope_al)) >= 5: fails.append(f"residual within-cycle ramp {np.median(slope_al):.1f} ms/min on the aligned trace")
    if not slope_al: fails.append("no folded-slope measurement")
    if ratio and np.median(ratio) >= 0.9: fails.append(f"6-s spread ratio {np.median(ratio):.2f}")
    if tr and np.median(tr) < 0.99: fails.append(f"template r {np.median(tr):.3f}")
    if nan_add and np.median(nan_add) >= 0.01: fails.append(f"added NaN {np.median(nan_add):.3f}")
    res["fails"] = fails
    os.makedirs(a.stage_a, exist_ok=True)
    json.dump(res, open(os.path.join(a.stage_a, "verify_stage_c4.json"), "w"), indent=1)
    print(json.dumps(res, indent=1)); print("GATE C4:", "FAIL" if fails else "PASS", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
