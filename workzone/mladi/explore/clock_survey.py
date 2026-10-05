#!/usr/bin/env python3
"""MLADI: is the /ehr clock the waveform clock? Survey across encounters.

The Step 0c demo found charted Systolic BP equal to the monitor's cuff NBP to the mmHg, but 60 min
apart (charted later). Nurses validate the monitor's NBP into the flowsheet, so an exact value match
pins the offset between the two clocks without any physiology. Per encounter:

  nbp offset   every (charted SBP, charted DBP at the same time) vs every monitor NBP reading
               (s and d both within 0.5 mmHg): histogram of time differences in 1-min bins over
               +-6 h; the mode = the offset (charted - monitor), with the share of charted readings
               it explains.
  hr lag       charted Pulse vs 1-min medians of HR.HR, lag sweep -180..180 min (median |diff|).
  context      time_origin's stamped zone (EDT/EST/LMT), the DST state (America/New_York) at the
               waveform start and at the median charted event, year.

The cross-table of offset by (origin zone, DST state of the events) says whether the offset is a DST
convention (EHR seconds computed in wall-clock time from a stamped origin) and how to undo it.
Read-only; prints running tables every 100 encounters.
"""
import collections, glob, json, os, random, sys, time
from datetime import datetime, timezone
from multiprocessing import Pool
from zoneinfo import ZoneInfo
import numpy as np

RAW = "/ocean/projects/med250003p/shared/mladi_extract_2023_waves"
NY = ZoneInfo("America/New_York")
T0 = time.time()


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:6.0f}s]", *a, flush=True)


def num(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else np.nan
    except (TypeError, ValueError):
        return np.nan


def factor(ds):
    a = ds[:]
    cm = json.loads(ds.attrs[".meta"]).get("columns", {}) if ".meta" in ds.attrs else {}
    out = {}
    for c in a.dtype.names:
        lev = cm.get(c, {}).get("levels")
        out[c] = (np.array([lev[k] if 0 <= k < len(lev) else None for k in a[c].astype(int)], dtype=object)
                  if lev else a[c])
    return out


def one(p):
    import h5py
    r = {"year": os.path.basename(p)[:4]}
    try:
        with h5py.File(p, "r") as f:
            s = json.loads(f.attrs[".meta"])["time_origin"]
            stamp, tz = s.rsplit(" ", 1)
            o = datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=NY)
            o_utc = o.timestamp()
            r["origin_tz"] = tz; r["origin_year"] = stamp[:4]
            if "ehr/low_rate" not in f or "data/numerics/NBP.NBPs" not in f:
                r["skip"] = "missing"; return r
            d = factor(f["ehr/low_rate"])
            name, t, v = d["eventName"], d["date"].astype(float), np.array([num(x) for x in d["resultVal"]])
            sb = (name == "Systolic BP") & np.isfinite(v) & np.isfinite(t)
            db = (name == "Diastolic BP") & np.isfinite(v) & np.isfinite(t)
            pu = (name == "Pulse") & np.isfinite(v) & np.isfinite(t)
            ns = f["data/numerics/NBP.NBPs"][:]; nd = f["data/numerics/NBP.NBPd"][:] if "data/numerics/NBP.NBPd" in f else None
            tn, vs = ns["time"].astype(float), ns["value"].astype(float)
            vd = None
            if nd is not None and nd.shape == ns.shape:
                vd = nd["value"].astype(float)
            keep = np.r_[True, np.diff(vs) != 0] & np.isfinite(vs)
            tn, vs = tn[keep], vs[keep]; vd = vd[keep] if vd is not None else None
            # charted SBP with its DBP at the same time
            ts, cs = t[sb], v[sb]
            dmap = dict(zip(np.round(t[db], 0), v[db]))
            cd = np.array([dmap.get(round(x, 0), np.nan) for x in ts])
            diffs = []
            for x, a_, b_ in zip(ts, cs, cd):
                m = np.abs(vs - a_) <= 0.5
                if vd is not None and np.isfinite(b_):
                    m &= np.abs(vd - b_) <= 0.5
                dd = (x - tn[m]) / 60.0
                dd = dd[np.abs(dd) <= 360]
                diffs.append(dd)
            if ts.size >= 3 and any(len(z) for z in diffs):
                allz = np.concatenate(diffs)
                h = collections.Counter(np.round(allz).astype(int).tolist())
                best, cnt = h.most_common(1)[0]
                r["nbp_offset_min"] = int(best)
                r["nbp_explained"] = float(np.mean([np.any(np.abs(z - best) <= 1.5) for z in diffs]))
                r["n_charted_sbp"] = int(ts.size)
            # HR lag sweep
            if pu.sum() >= 5 and "data/numerics/HR.HR" in f:
                hr = f["data/numerics/HR.HR"][:]
                th, vh = hr["time"].astype(float), hr["value"].astype(float)
                k = np.floor(th / 60).astype(np.int64); k0 = k.min()
                ser = np.full(int(k.max() - k0) + 1, np.nan)
                o_ = np.argsort(k, kind="stable"); ks, vv = k[o_] - k0, vh[o_]
                u, i0 = np.unique(ks, return_index=True)
                for kk, a_, b_ in zip(u, i0, np.r_[i0[1:], ks.size]):
                    ser[kk] = np.nanmedian(vv[a_:b_])
                kc = np.floor(t[pu] / 60).astype(np.int64) - k0; vc = v[pu]
                best = None
                for lag in range(-180, 181, 5):
                    kk = kc - lag; ok = (kk >= 0) & (kk < ser.size)
                    y = ser[kk[ok]]; g = np.isfinite(y)
                    if g.sum() >= 5:
                        mad = float(np.median(np.abs(vc[ok][g] - y[g])))
                        if best is None or mad < best[1]:
                            best = (lag, mad)
                if best:
                    r["hr_lag_min"], r["hr_mad"] = best
            # DST state at the waveform start and at the median charted SBP
            if "data/waveforms/Pleth" in f:
                w0 = json.loads(f["data/waveforms/Pleth"].attrs[".meta"])["dwc_meta"]["minTime"]
                r["tz_wave"] = datetime.fromtimestamp(o_utc + w0, tz=NY).tzname()
            if ts.size:
                r["tz_events"] = datetime.fromtimestamp(o_utc + float(np.median(ts)), tz=NY).tzname()
                r["events_cross_dst"] = len({datetime.fromtimestamp(o_utc + x, tz=NY).tzname() for x in (ts.min(), ts.max())}) > 1
    except Exception as ex:
        r["error"] = f"{type(ex).__name__}: {ex}"
    return r


def report(R, final=False):
    ok = [r for r in R if "nbp_offset_min" in r]
    log(("FINAL " if final else "") + f"{len(R)} encounters, {len(ok)} with an NBP exact-match offset "
        f"(errors {sum('error' in r for r in R)}, skipped {sum('skip' in r for r in R)})")
    if not ok:
        return
    off = np.array([r["nbp_offset_min"] for r in ok])
    log("  NBP offset (charted - monitor, min) distribution: " + str(collections.Counter(off.tolist()).most_common(12)))
    log(f"  share of charted SBP explained at the best offset: p25/p50/p75 "
        f"{np.percentile([r['nbp_explained'] for r in ok], [25, 50, 75]).round(2).tolist()}")
    tab = collections.defaultdict(collections.Counter)
    for r in ok:
        key = (r.get("origin_tz"), r.get("tz_events"), "crossDST" if r.get("events_cross_dst") else "")
        b = int(np.round(r["nbp_offset_min"] / 30.0) * 30)
        tab[key][b] += 1
    log("  offset (rounded to 30 min) by (origin zone, zone of the charted events, crosses DST):")
    for key, c in sorted(tab.items(), key=lambda kv: -sum(kv[1].values())):
        log(f"    {str(key):34s} n={sum(c.values()):4d} | " + ", ".join(f"{k:+d}:{v}" for k, v in sorted(c.items())))
    yt = collections.defaultdict(collections.Counter)
    for r in ok:
        yt[r["origin_year"]][int(np.round(r["nbp_offset_min"] / 30.0) * 30)] += 1
    log("  by origin year: " + "; ".join(f"{y}: " + ",".join(f"{k:+d}:{v}" for k, v in sorted(c.items())) for y, c in sorted(yt.items())))
    hl = [r for r in ok if "hr_lag_min" in r]
    if hl:
        agree = np.mean([abs(r["hr_lag_min"] - r["nbp_offset_min"]) <= 15 for r in hl])
        log(f"  HR lag agrees with the NBP offset (within 15 min) in {agree:.1%} of {len(hl)}")


def main():
    files = sorted(glob.glob(os.path.join(RAW, "*.h5")))
    random.seed(0)
    n = int(os.environ.get("N_ENC", 600))
    pick = random.sample(files, min(n, len(files)))
    log(f"surveying {len(pick)} of {len(files)} encounters")
    R = []
    with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", 16))) as pool:
        for k, r in enumerate(pool.imap_unordered(one, pick, chunksize=4), 1):
            R.append(r)
            if k % 100 == 0:
                report(R)
    report(R, final=True)
    out = "/ocean/projects/med250003p/mwang11/Physio_Data/workzone/outputs/mladi/explore/clock_survey.json"
    json.dump(R, open(out, "w"), default=str)
    log("wrote " + out)


if __name__ == "__main__":
    main()
