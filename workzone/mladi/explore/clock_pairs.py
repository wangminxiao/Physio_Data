#!/usr/bin/env python3
"""MLADI: in encounters whose charted events span a DST change, where does the charted-vs-monitor
offset step? Per exact NBP match (charted SBP and DBP equal to a monitor NBP s/d within 0.5 mmHg,
nearest monitor reading), the raw difference t_ehr - t_dwc (seconds) against the time from the DST
transition (hours, on the EHR clock read as elapsed). Prints, per encounter, the median raw difference
before and after the transition and the per-pair series coarsened to 2 h, so a step shows where it is.
Also reports, per encounter, whether its Pleth clock has a 3600-s gap and where it sits relative to
the transition (raw t). Read-only.
"""
import collections, glob, json, os, random, sys
from datetime import datetime, timedelta, timezone
from multiprocessing import Pool
from zoneinfo import ZoneInfo
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clock_verify import Clock, factor, num, transitions, NY, UTC  # noqa: E402

RAW = "/ocean/projects/med250003p/shared/mladi_extract_2023_waves"


def one(p):
    import h5py
    try:
        with h5py.File(p, "r") as f:
            ck = Clock(json.loads(f.attrs[".meta"])["time_origin"])
            if ck.tz == "LMT" or "ehr/low_rate" not in f or "data/numerics/NBP.NBPs" not in f or "data/numerics/NBP.NBPd" not in f:
                return None
            d = factor(f["ehr/low_rate"])
            name, t, v = d["eventName"], d["date"].astype(float), np.array([num(x) for x in d["resultVal"]])
            sb = (name == "Systolic BP") & np.isfinite(v) & np.isfinite(t)
            db = (name == "Diastolic BP") & np.isfinite(v) & np.isfinite(t)
            if sb.sum() < 6:
                return None
            te = t[sb]; u = ck.ehr_utc(te)
            hits = [(T, k) for T, k in transitions(datetime.fromtimestamp(u.min(), UTC).year, datetime.fromtimestamp(u.max(), UTC).year)
                    if u.min() < T < u.max()]
            if not hits:
                return None
            T, kind = hits[0]
            ns, nd = f["data/numerics/NBP.NBPs"][:], f["data/numerics/NBP.NBPd"][:]
            if ns.shape != nd.shape:
                return None
            tn, vs, vd = ns["time"].astype(float), ns["value"].astype(float), nd["value"].astype(float)
            dmap = dict(zip(np.round(t[db], 0), v[db]))
            rows = []
            for x, a_ in zip(te, v[sb]):
                b_ = dmap.get(round(x, 0))
                if b_ is None:
                    continue
                m = (np.abs(vs - a_) <= 0.5) & (np.abs(vd - b_) <= 0.5) & (np.abs(x - tn) <= 4 * 3600)
                if m.any():
                    j = np.argmin(np.abs(x - tn[m] - np.median(x - tn[m])))
                    rows.append(((ck.ehr0 + x - T) / 3600, x - tn[m][j]))
            if len(rows) < 4:
                return None
            R = np.array(rows)
            gap = None
            if "data/waveforms/Pleth" in f:
                ds = f["data/waveforms/Pleth"]; n = ds.shape[0]
                # coarse scan of the raw Pleth clock for an hour-sized jump near the transition
                idx = np.linspace(0, n - 1, 4000).astype(int)
                tt = ds[idx]["time"].astype(float) if n > 4000 else ds[:]["time"].astype(float)
                dd = np.diff(tt) - np.diff(idx) / 125.0 if n > 4000 else np.diff(tt)
                k = int(np.argmax(dd))
                if dd[k] > 3000:
                    gap = {"size_s": float(dd[k]), "raw_t_h_from_transition_on_ehr_axis": float((ck.ehr0 + tt[k] - T) / 3600)}
            return {"kind": kind, "origin_tz": ck.tz, "pairs": R.tolist(), "gap": gap}
    except Exception as ex:
        return {"error": f"{type(ex).__name__}: {ex}"}


def main():
    files = sorted(glob.glob(os.path.join(RAW, "*.h5")))
    random.seed(2)
    pick = random.sample(files, min(int(os.environ.get("N_ENC", 4000)), len(files)))
    with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", 16))) as pool:
        R = [r for r in pool.map(one, pick, chunksize=4) if r and "pairs" in r]
    print(f"{len(R)} encounters with >= 4 exact NBP matches whose charted events span a DST change", flush=True)
    summ = collections.Counter()
    for r in R[:40]:
        P = np.array(r["pairs"]); pre, post = P[P[:, 0] < 0, 1], P[P[:, 0] >= 0, 1]
        coarse = collections.defaultdict(list)
        for h, dsec in P:
            coarse[int(np.floor(h / 2) * 2)].append(dsec / 60)
        line = " ".join(f"{k:+d}h:{np.median(v):+.0f}" for k, v in sorted(coarse.items()) if abs(k) <= 24)
        g = r["gap"]
        print(f"{r['kind']:6s} origin {r['origin_tz']} | before {np.median(pre) / 60 if pre.size else float('nan'):+.0f} min (n={pre.size}) "
              f"after {np.median(post) / 60 if post.size else float('nan'):+.0f} min (n={post.size}) | "
              f"Pleth 1-h gap {('at %+.1f h' % g['raw_t_h_from_transition_on_ehr_axis']) if g else 'none'} | {line}", flush=True)
    for r in R:
        P = np.array(r["pairs"]); pre, post = P[P[:, 0] < 0, 1], P[P[:, 0] >= 0, 1]
        if pre.size and post.size:
            summ[(r["kind"], r["origin_tz"], int(round(np.median(pre) / 60 / 30) * 30), int(round(np.median(post) / 60 / 30) * 30))] += 1
    print("summary (kind, origin zone, median raw offset before -> after the transition, min):", flush=True)
    for k, n in summ.most_common():
        print(f"   {k}: {n}", flush=True)


if __name__ == "__main__":
    main()
