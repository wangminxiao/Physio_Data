#!/usr/bin/env python3
"""EHR-vs-monitor clock offset from non-NBP vitals: charted Pulse (100) vs monitor HR_hf (150) / PULSE_hf (156),
charted SpO2 (101) vs SpO2_hf (151), on the written store (ehr_events vs ehr_hf, both on the grid). For each
charted value, every monitor tick within +-6 h with an equal value (|dv| <= 0.5) votes for its lag (1-min bins);
the vote histogram's peak = offset (charted - monitor, min). Shares reported at lags 0, +-240, +-300.
Independent of NBP, so it arbitrates entities whose NBP stream holds twin copies (dup_check.py).

    python workzone/mladi/explore/ehr_vital_offset.py --from-dup-log <log> [--normal 30]
"""
import argparse, collections, json, os, random, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import cfg  # noqa: E402

C = cfg()
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=C["output_dir"]); ap.add_argument("--from-dup-log", default="")
ap.add_argument("--normal", type=int, default=30); ap.add_argument("--max", type=int, default=40)
a = ap.parse_args()
twin = []
if a.from_dup_log:
    for ln in open(a.from_dup_log):
        if ln.startswith("{"):
            r = json.loads(ln)
            if max(r.get("nbp_twin_240", 0), r.get("nbp_twin_300", 0)) > 0.2:
                twin.append(r["e"])
twin = twin[: a.max]
ents = sorted(d for d in os.listdir(a.out) if os.path.exists(os.path.join(a.out, d, "ehr_hf.npy")))
random.seed(5); random.shuffle(ents)
normal = []
for e in ents:
    if len(normal) >= a.normal:
        break
    m = json.load(open(os.path.join(a.out, e, "meta.json")))
    if (m.get("ehr_clock") or {}).get("confidence") == "verified" and (m.get("origin_year") or 0) >= 2020:
        normal.append(e)

PAIRS = ((100, (150, 156)), (101, (151,)))


def offsets(e):
    d = os.path.join(a.out, e)
    ev = np.load(os.path.join(d, "ehr_events.npy")); hf = np.load(os.path.join(d, "ehr_hf.npy"))
    votes = collections.Counter(); n_ch = 0
    for cv, mvs in PAIRS:
        c = ev[ev["var_id"] == cv]; mon = hf[np.isin(hf["var_id"], mvs)]
        if c.size == 0 or mon.size == 0:
            continue
        o = np.argsort(mon["time_ms"]); mt, mv = mon["time_ms"][o], mon["value"][o]
        for t, v in zip(c["time_ms"], c["value"]):
            lo, hi = np.searchsorted(mt, t - 361 * 60000), np.searchsorted(mt, t + 361 * 60000)
            z = (t - mt[lo:hi][np.abs(mv[lo:hi] - v) <= 0.5]) / 60000.0
            for L in set(np.round(z).astype(int).tolist()):
                votes[L] += 1
            n_ch += 1
    if not n_ch:
        return None
    def sh(L):
        return round(sum(votes.get(L + k, 0) for k in (-1, 0, 1)) / n_ch, 3)
    peak = max(votes, key=votes.get) if votes else None
    # background = median share over lags 30..200 min
    bg = float(np.median([sh(L) for L in range(30, 200, 7)]))
    return {"n_charted": n_ch, "peak": peak, "share_peak": sh(peak) if peak is not None else None, "bg": round(bg, 3),
            **{f"s{L}": sh(L) for L in (0, 240, -240, 300, -300)}}


for grp, L in (("twin", twin), ("normal", normal)):
    for e in L:
        m = json.load(open(os.path.join(a.out, e, "meta.json")))
        r = offsets(e)
        print(json.dumps({"grp": grp, "e": e[:16], "res_nbp": (m.get("ehr_clock") or {}).get("residual_min"), **(r or {})}), flush=True)
