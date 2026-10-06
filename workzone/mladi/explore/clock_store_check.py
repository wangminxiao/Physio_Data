#!/usr/bin/env python3
"""Clock check on the WRITTEN store: charted SBP (ehr_events var 104) vs monitor NBP (nbp_events var 157), both
on the entity grid. Per entity: every charted SBP against every NBP with |dv| <= 0.5 within +-6 h, dt binned to
1 min -> mode, share of charted SBP explained within 1.5 min of the mode, share within 1.5 min of 0.
Grouped by clock_confidence, dst_crossing_runs, runs_continued_after_dst. Prints per entity as it goes.

    python workzone/mladi/explore/clock_store_check.py [--n 400]
"""
import argparse, collections, json, os, random, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import cfg  # noqa: E402

C = cfg()
ap = argparse.ArgumentParser(); ap.add_argument("--out", default=C["output_dir"]); ap.add_argument("--n", type=int, default=400)
a = ap.parse_args()
ents = [d for d in os.listdir(a.out) if os.path.exists(os.path.join(a.out, d, "ehr_events.npy"))]
random.seed(1); random.shuffle(ents)
rows = []
for e in ents:
    if len(rows) >= a.n:
        break
    d = os.path.join(a.out, e); m = json.load(open(os.path.join(d, "meta.json")))
    ev = np.load(os.path.join(d, "ehr_events.npy")); nb = np.load(os.path.join(d, "nbp_events.npy"))
    s = ev[ev["var_id"] == 104]; k = nb[nb["var_id"] == 157]
    if s.size < 5 or k.size < 5:
        continue
    dts = []
    for t, v in zip(s["time_ms"], s["value"]):
        z = (t - k["time_ms"][np.abs(k["value"] - v) <= 0.5]) / 60000.0
        dts.append(z[np.abs(z) <= 360])
    allz = np.concatenate(dts) if dts else np.empty(0)
    if allz.size == 0:
        mode, f_mode = None, 0.0
    else:
        mode = collections.Counter(np.round(allz).astype(int).tolist()).most_common(1)[0][0]
        f_mode = float(np.mean([np.any(np.abs(z - mode) <= 1.5) for z in dts]))
    f0 = float(np.mean([np.any(np.abs(z) <= 1.5) for z in dts]))
    f1 = float(np.mean([np.any(np.abs(z) <= 1.0) for z in dts]))
    r = {"e": e[:12], "cc": m.get("clock_confidence"), "res_A": (m.get("clock_check") or {}).get("residual_min"),
         "shift": m.get("ehr_extra_shift_min"), "dst": bool(m.get("dst_crossing_runs")), "cont": m.get("runs_continued_after_dst", 0),
         "n": int(s.size), "mode": mode, "f_mode": round(f_mode, 2), "f0": round(f0, 2), "f1": round(f1, 2), "rule": m.get("clock_rule")}
    rows.append(r); print(json.dumps(r), flush=True)
print("== summary by clock_confidence")
for cc in sorted({r["cc"] for r in rows}):
    R = [r for r in rows if r["cc"] == cc]
    print(cc, "n", len(R), "| mode counts", collections.Counter(r["mode"] for r in R).most_common(6),
          "| f0 median", np.median([r["f0"] for r in R]), "| f_mode median", np.median([r["f_mode"] for r in R]),
          "| f1 median", np.median([r["f1"] for r in R]))
R = [r for r in rows if r["cc"] == "verified"]
print("verified with f0 < 0.6:", [(r["e"], r["mode"], r["f0"], r["f_mode"], r["dst"], r["n"]) for r in R if r["f0"] < 0.6][:25])
