#!/usr/bin/env python3
"""MLADI: verify the clock rule found by clock_survey.py, and what DST does to the waveform clock.

Rule under test (from the charted-vs-monitor NBP exact matches):
  DWC (waveforms, numerics): seconds of LOCAL WALL-CLOCK time since the origin's wall-clock stamp,
      so  utc = America/New_York(wall = origin_wall + t)
  EHR (/ehr tables):          true elapsed seconds since the origin instant,
      so  utc = origin_instant + t, where origin_instant = origin_wall in New York -- except for
      origins stamped LMT (year 1800), whose instant is origin_wall read as UTC.

1. offsets  recompute the charted-minus-monitor NBP offset with both sides converted by the rule,
            per (origin zone, event zone, crosses-DST) class: all should be 0.
2. dst      for encounters whose waveform span (rule-converted) contains a DST transition, read the
            raw Pleth time column around the transition: a wall-clock clock shows a +1 h gap at
            spring-forward and a -1 h step / duplicated hour at fall-back.
Read-only; N_ENC encounters (default 1500) for 1, all crossing encounters among them for 2.
"""
import collections, glob, json, os, random, time
from datetime import datetime, timedelta, timezone
from multiprocessing import Pool
from zoneinfo import ZoneInfo
import numpy as np

RAW = "/ocean/projects/med250003p/shared/mladi_extract_2023_waves"
NY = ZoneInfo("America/New_York")
UTC = timezone.utc
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
    return {c: (np.array([cm[c]["levels"][k] if 0 <= k < len(cm[c]["levels"]) else None for k in a[c].astype(int)], dtype=object)
                if cm.get(c, {}).get("levels") else a[c]) for c in a.dtype.names}


class Clock:
    """Rule conversions for one file. Wall -> UTC offsets are looked up per wall-clock hour."""

    def __init__(self, origin_str):
        stamp, self.tz = origin_str.rsplit(" ", 1)
        self.wall0 = datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S.%f")
        inst = self.wall0.replace(tzinfo=UTC) if self.tz == "LMT" else self.wall0.replace(tzinfo=NY)
        self.ehr0 = inst.timestamp()
        self._cache = {}

    def ehr_utc(self, t):
        return self.ehr0 + np.asarray(t, float)

    def dwc_utc(self, t):
        t = np.atleast_1d(np.asarray(t, float))
        wall_epoch = (self.wall0 - datetime(1970, 1, 1)).total_seconds() + t
        hrs = np.floor(wall_epoch / 3600).astype(np.int64)          # wall-clock hours, not hours since origin
        off = np.empty(t.size)
        for h in np.unique(hrs):
            if h not in self._cache:
                w = datetime(1970, 1, 1) + timedelta(hours=int(h), minutes=30)
                self._cache[h] = w.replace(tzinfo=NY).utcoffset().total_seconds()
            off[hrs == h] = self._cache[h]
        return wall_epoch - off


def offsets(p):
    import h5py
    r = {}
    try:
        with h5py.File(p, "r") as f:
            ck = Clock(json.loads(f.attrs[".meta"])["time_origin"])
            r["origin_tz"] = ck.tz; r["origin_year"] = ck.wall0.year
            if "ehr/low_rate" not in f or "data/numerics/NBP.NBPs" not in f:
                r["skip"] = 1; return r
            d = factor(f["ehr/low_rate"])
            name, t, v = d["eventName"], d["date"].astype(float), np.array([num(x) for x in d["resultVal"]])
            sb = (name == "Systolic BP") & np.isfinite(v) & np.isfinite(t)
            if sb.sum() < 3:
                r["skip"] = 1; return r
            ns = f["data/numerics/NBP.NBPs"][:]
            tn, vs = ns["time"].astype(float), ns["value"].astype(float)
            keep = np.r_[True, np.diff(vs) != 0] & np.isfinite(vs)
            tn, vs = tn[keep], vs[keep]
            te_raw, ve = t[sb], v[sb]
            for mode, a_, b_ in (("raw", te_raw, tn), ("rule", ck.ehr_utc(te_raw), ck.dwc_utc(tn))):
                dd = []
                for x, val in zip(a_, ve):
                    m = np.abs(vs - val) <= 0.5
                    z = (x - b_[m]) / 60.0
                    dd.append(z[np.abs(z) <= 360])
                if any(len(z) for z in dd):
                    h = collections.Counter(np.round(np.concatenate(dd)).astype(int).tolist())
                    r[f"off_{mode}"] = int(h.most_common(1)[0][0])
            ev = ck.ehr_utc(np.median(te_raw))
            r["tz_events"] = datetime.fromtimestamp(float(ev), tz=NY).tzname()
            r["cross"] = len({datetime.fromtimestamp(float(x), tz=NY).tzname() for x in ck.ehr_utc([te_raw.min(), te_raw.max()])}) > 1
    except Exception as ex:
        r["error"] = f"{type(ex).__name__}: {ex}"
    return r


def transitions(y0, y1):
    """UTC instants of New York DST changes between years y0..y1."""
    out = []
    for y in range(y0, y1 + 1):
        prev = None
        t = datetime(y, 1, 1, tzinfo=UTC)
        while t.year == y:
            off = t.astimezone(NY).utcoffset()
            if prev is not None and off != prev:
                lo, hi = t - timedelta(hours=6), t       # refine to the hour
                while (hi - lo) > timedelta(minutes=1):
                    mid = lo + (hi - lo) / 2
                    if mid.astimezone(NY).utcoffset() == prev:
                        lo = mid
                    else:
                        hi = mid
                out.append((hi.timestamp(), "spring" if off > prev else "fall"))
            prev = off; t += timedelta(hours=6)
    return out


def dst(p):
    import h5py
    r = {}
    try:
        with h5py.File(p, "r") as f:
            if "data/waveforms/Pleth" not in f:
                return r
            ck = Clock(json.loads(f.attrs[".meta"])["time_origin"])
            ds = f["data/waveforms/Pleth"]; n = ds.shape[0]
            t0, t1 = float(ds[0]["time"]), float(ds[n - 1]["time"])
            u0, u1 = ck.dwc_utc([t0, t1])
            hits = [(T, kind) for T, kind in transitions(datetime.fromtimestamp(u0, UTC).year, datetime.fromtimestamp(u1, UTC).year)
                    if u0 < T < u1]
            if not hits:
                return r
            T, kind = hits[0]
            # raw wall-clock t of the transition instant
            wall_T = (datetime.fromtimestamp(T, tz=NY).replace(tzinfo=None) - ck.wall0).total_seconds()
            lo, hi = 0, n
            target = wall_T - 2 * 3600
            while lo < hi:
                mid = (lo + hi) // 2
                if ds[mid]["time"] < target:
                    lo = mid + 1
                else:
                    hi = mid
            seg = ds[lo:min(n, lo + int(4.5 * 3600 * 125))]["time"].astype(float)
            dt = np.diff(seg)
            r.update(kind=kind, n=int(seg.size), max_gap_s=float(dt.max()) if dt.size else None,
                     min_dt_s=float(dt.min()) if dt.size else None, n_back=int((dt < 0).sum()),
                     n_dup=int(seg.size - np.unique(seg).size), span_s=float(seg[-1] - seg[0]) if seg.size else None)
    except Exception as ex:
        r["error"] = f"{type(ex).__name__}: {ex}"
    return r


def main():
    files = sorted(glob.glob(os.path.join(RAW, "*.h5")))
    random.seed(1)
    pick = random.sample(files, min(int(os.environ.get("N_ENC", 1500)), len(files)))
    cpu = int(os.environ.get("SLURM_CPUS_PER_TASK", 16))
    with Pool(cpu) as pool:
        R = pool.map(offsets, pick, chunksize=4)
    ok = [r for r in R if "off_rule" in r]
    log(f"offsets: {len(ok)} encounters with exact NBP matches (errors {sum('error' in r for r in R)})")
    tab = collections.defaultdict(lambda: [collections.Counter(), collections.Counter()])
    for r in ok:
        key = (r["origin_tz"], r["tz_events"], "cross" if r["cross"] else "", "1990" if r["origin_year"] == 1990 else "")
        tab[key][0][int(np.round(r.get("off_raw", 0) / 30) * 30)] += 1
        tab[key][1][int(np.round(r["off_rule"] / 30) * 30)] += 1
    log("  class (origin zone, event zone, crosses DST, 1990 origin): raw offset -> offset after the rule (rounded to 30 min)")
    for key, (a, b) in sorted(tab.items(), key=lambda kv: -sum(kv[1][1].values())):
        log(f"    {str(key):40s} n={sum(b.values()):4d} | raw {dict(sorted(a.items()))} -> rule {dict(sorted(b.items()))}")
    exact0 = np.mean([abs(r["off_rule"]) <= 2 for r in ok])
    log(f"  after the rule: |offset| <= 2 min in {exact0:.1%} of encounters")
    with Pool(cpu) as pool:
        D = [d for d in pool.map(dst, pick, chunksize=4) if d]
    log(f"dst: {len(D)} sampled encounters whose waveform spans a DST change")
    for kind in ("spring", "fall"):
        g = [d for d in D if d.get("kind") == kind]
        if g:
            log(f"  {kind}: n={len(g)} | max gap s p50/max {np.median([d['max_gap_s'] for d in g]):.1f}/{max(d['max_gap_s'] for d in g):.1f} | "
                f"min dt s p50 {np.median([d['min_dt_s'] for d in g]):.4f} | files with backward steps {sum(d['n_back'] > 0 for d in g)} | "
                f"with duplicated times {sum(d['n_dup'] > 0 for d in g)} | raw span of the 4.5-h read p50 {np.median([d['span_s'] for d in g]) / 3600:.2f} h")
    json.dump({"offsets": R, "dst": D}, open("/ocean/projects/med250003p/mwang11/Physio_Data/workzone/outputs/mladi/explore/clock_verify.json", "w"), default=str)
    log("done")


if __name__ == "__main__":
    main()
