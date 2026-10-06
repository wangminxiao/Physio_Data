#!/usr/bin/env python3
"""Duplicated-copy check: is part of an entity's recording present twice, shifted by a zone offset?

For each entity: (1) NBP twins = share of monitor NBP readings (157, 158) that reappear with the same values
D minutes later (+-1.5 min), D in {60, 240, 300}; (2) waveform twins = share of 300 sampled segments i
for which a segment j with time_ms[j] = time_ms[i] + D (+-15 s) exists and PLETH40[j] equals PLETH40[i]
(max |diff| < 1e-3 on finite samples, >= 90 % finite) -- also the best lag within +-2 s via correlation,
in case the copy is resampled; (3) block layout (start/end hours from the first segment) to see whether
the copy is a separate block.

    python workzone/mladi/explore/dup_check.py --only conflict --res 240,300   # entities from ehr_clock
    python workzone/mladi/explore/dup_check.py --sample 300 --min-year 2020     # prevalence
"""
import argparse, json, os, random, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import cfg  # noqa: E402

C = cfg()
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=C["output_dir"]); ap.add_argument("--only", default="")
ap.add_argument("--res", default=""); ap.add_argument("--sample", type=int, default=0)
ap.add_argument("--min-year", type=int, default=0); ap.add_argument("--max", type=int, default=120)
a = ap.parse_args()
res_set = {int(x) for x in a.res.replace(":", ",").split(",") if x}
ents = sorted(d for d in os.listdir(a.out) if os.path.exists(os.path.join(a.out, d, "meta.json")))
random.seed(3); random.shuffle(ents)
D_LIST = (60, 240, 300)
n_done = 0; tw_any = 0
for e in ents:
    if n_done >= (a.sample or a.max):
        break
    d = os.path.join(a.out, e); m = json.load(open(os.path.join(d, "meta.json")))
    ck = m.get("ehr_clock") or {}
    if a.only and ck.get("confidence") != a.only:
        continue
    if res_set and (ck.get("residual_min") is None or abs(ck["residual_min"]) not in res_set):
        continue
    if a.min_year and (m.get("origin_year") or 0) < a.min_year:
        continue
    t = np.load(os.path.join(d, "time_ms.npy")); n = t.size
    nb = np.load(os.path.join(d, "nbp_events.npy"))
    s = nb[nb["var_id"] == 157]
    out = {"e": e, "oy": m.get("origin_year"), "res": ck.get("residual_min"), "n_seg": int(n), "n_blocks": m.get("n_blocks"),
           "n_runs": m.get("n_runs"), "n_nbp": int(s.size)}
    for D in D_LIST:
        if s.size:
            j = np.searchsorted(s["time_ms"], s["time_ms"] + D * 60000)
            hit = 0
            for k, jj in enumerate(j):
                c = s[max(0, jj - 3): jj + 3]
                hit += bool(np.any((np.abs(c["time_ms"] - s["time_ms"][k] - D * 60000) <= 90000) & (c["value"] == s["value"][k])))
            out[f"nbp_twin_{D}"] = round(hit / s.size, 3)
    P = np.load(os.path.join(d, "PLETH40.npy"), mmap_mode="r")
    idx = np.sort(np.random.default_rng(0).choice(n, size=min(300, n), replace=False))
    for D in D_LIST:
        tw = cand = 0
        for i in idx:
            jj = np.searchsorted(t, t[i] + D * 60000 - 15000)
            if jj < n and abs(t[jj] - t[i] - D * 60000) <= 15000:
                x = np.asarray(P[i], np.float32); y = np.asarray(P[jj], np.float32)
                f = np.isfinite(x) & np.isfinite(y)
                if f.mean() < 0.9:
                    continue
                cand += 1
                if np.max(np.abs(x[f] - y[f])) < 1e-3:
                    tw += 1
                else:
                    xs, ys = x[f] - x[f].mean(), y[f] - y[f].mean()
                    if xs.std() > 0 and ys.std() > 0:
                        cc = max(np.corrcoef(np.roll(xs, L), ys)[0, 1] for L in range(-80, 81, 2))
                        tw += cc > 0.99
        out[f"wav_twin_{D}"] = f"{tw}/{cand}"
    # block layout from time gaps (> 30 s)
    gaps = np.flatnonzero(np.diff(t) > 30000)
    starts = np.r_[0, gaps + 1]; ends = np.r_[gaps, n - 1]
    out["blocks_h"] = [(round((t[a_] - t[0]) / 3.6e6, 2), round((t[b_] - t[0]) / 3.6e6 + 1 / 120, 2)) for a_, b_ in zip(starts, ends)][:8]
    n_done += 1
    if any(str(v).split("/")[0] not in ("0", "0.0") for k, v in out.items() if k.startswith("wav_twin")):
        tw_any += 1
    print(json.dumps(out), flush=True)
print(f"== {n_done} entities, {tw_any} with any waveform twin")
