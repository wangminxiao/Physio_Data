#!/usr/bin/env python3
"""NBP twin copies in the raw H5: which copy is the real measurement?

For entities whose NBP readings reappear D = 240 / 300 min apart (dup_check.py): read data/numerics NBP.NBPs /
NBPd / NBPm / Pulse and HR.HR (raw seconds); collapse held rows to readings (value change); pair readings
whose (s, d, m) are equal at lag D (+-90 s). For each pair: the cuff pulse vs the median HR.HR within +-60 s of
each copy -> the copy with the smaller |pulse - HR| is taken as real. Also: raw row layout around each copy
(rows within +-10 s, spacing), monotonicity of the NBP time column, and which copy is earlier.

    python workzone/mladi/explore/nbp_twins_raw.py --entities a,b,c   |  --from-dup-log <log> --max 15
"""
import argparse, json, os, re, sys
import h5py
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import cfg  # noqa: E402

C = cfg()
ap = argparse.ArgumentParser()
ap.add_argument("--entities", default=""); ap.add_argument("--from-dup-log", default=""); ap.add_argument("--max", type=int, default=15)
a = ap.parse_args()
ents = [e for e in re.split(r"[,:]", a.entities) if e]
if a.from_dup_log:
    for ln in open(a.from_dup_log):
        if ln.startswith("{"):
            r = json.loads(ln)
            if max(r.get("nbp_twin_240", 0), r.get("nbp_twin_300", 0)) > 0.2:
                ents.append(r["e"])
ents = ents[: a.max]


def readings(f, key):
    if f"data/numerics/{key}" not in f:
        return None, None, None
    x = f[f"data/numerics/{key}"][:]
    t, v = x["time"].astype(float), x["value"].astype(float)
    mono = bool(np.all(np.diff(t) >= 0))
    o = np.argsort(t, kind="stable"); t, v = t[o], v[o]
    ok = np.isfinite(v)
    t, v = t[ok], v[ok]
    keep = np.r_[True, np.diff(v) != 0] if v.size else np.zeros(0, bool)
    return t[keep], v[keep], {"rows": int(x.size), "monotonic": mono, "readings": int(keep.sum())}


tot = {"pairs": 0, "earlier_real": 0, "later_real": 0, "undecided": 0}
for e in ents:
    with h5py.File(os.path.join(C["raw_h5_dir"], e + ".h5"), "r") as f:
        ts, vs, info = readings(f, "NBP.NBPs")
        if ts is None:
            continue
        td, vd, _ = readings(f, "NBP.NBPd")
        tp, vp, _ = readings(f, "NBP.Pulse") if "data/numerics/NBP.Pulse" in f else (None, None, None)
        hr = f["data/numerics/HR.HR"][:] if "data/numerics/HR.HR" in f else None
        raw_s = f["data/numerics/NBP.NBPs"][:]["time"].astype(float)
    dmap = dict(zip(np.round(td, 0), vd)) if td is not None else {}
    pmap = (lambda t0: vp[np.argmin(np.abs(tp - t0))] if tp is not None and tp.size and np.min(np.abs(tp - t0)) <= 5 else np.nan)
    hr_t = hr["time"].astype(float) if hr is not None else None; hr_v = hr["value"].astype(float) if hr is not None else None

    def hr_at(t0):
        if hr_t is None:
            return np.nan
        m = np.abs(hr_t - t0) <= 60
        return float(np.nanmedian(hr_v[m])) if m.any() else np.nan

    res = {"e": e, **info, "pairs": {}}
    dec = {"earlier_real": 0, "later_real": 0, "undecided": 0}
    ex = []
    for D in (240, 300):
        n_pair = 0
        for i in range(ts.size):
            j = np.flatnonzero((np.abs(ts - ts[i] - D * 60) <= 90) & (vs == vs[i]))
            if not j.size:
                continue
            j = j[0]
            if np.isfinite(dmap.get(round(ts[i], 0), np.nan)) and dmap.get(round(ts[i], 0)) != dmap.get(round(ts[j], 0), np.nan):
                continue
            n_pair += 1
            p_i, p_j = pmap(ts[i]), pmap(ts[j]); h_i, h_j = hr_at(ts[i]), hr_at(ts[j])
            e_i, e_j = abs(p_i - h_i), abs(p_j - h_j)
            if np.isfinite(e_i) and np.isfinite(e_j) and abs(e_i - e_j) >= 3:
                dec["earlier_real" if e_i < e_j else "later_real"] += 1
            else:
                dec["undecided"] += 1
            rows_i = int(np.sum(np.abs(raw_s - ts[i]) <= 10)); rows_j = int(np.sum(np.abs(raw_s - ts[j]) <= 10))
            if len(ex) < 4:
                ex.append({"D": D, "t_early_h": round(ts[i] / 3600, 2), "sbp": vs[i], "pulse": [p_i, p_j], "hr": [round(h_i, 1), round(h_j, 1)],
                           "raw_rows_10s": [rows_i, rows_j]})
        res["pairs"][D] = n_pair
    res.update(dec); res["examples"] = ex
    for k in dec:
        tot[k] += dec[k]
    tot["pairs"] += sum(res["pairs"].values())
    print(json.dumps(res, default=float), flush=True)
print("== total", tot)
