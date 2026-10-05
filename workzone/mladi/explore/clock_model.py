#!/usr/bin/env python3
"""MLADI: test model M for the two raw clocks, and find which instant sets the EHR origin's DST state.

Model M (from clock_survey / clock_verify / clock_pairs):
  raw_dwc = wall(event) - W            W = the stamped time_origin wall time
  raw_ehr = wall(event) - W_ehr        W_ehr = the origin INSTANT rendered in the DST state of some
                                       instant B (state at origin -> W_ehr = W; else W +- 1 h)
  so raw_ehr - raw_dwc = W - W_ehr in {0, -60, +60} min, constant within an encounter.
  (LMT / year-1800 origins: raw_ehr is UTC-elapsed from the origin read as UTC -- handled apart.)

For every encounter with exact charted-vs-monitor NBP matches: the observed offset, and the offset M
predicts when B = each candidate: waveform start / end, first / last charted SBP, last EHR row of any
table, regDate, dischDate, file creation (h5 attr if any). Accuracy per candidate. Then, with the best
B, the residual offset after converting both sides by M (both wall-based): should be 0 everywhere.
"""
import collections, glob, json, os, random, sys
from datetime import datetime, timedelta, timezone
from multiprocessing import Pool
from zoneinfo import ZoneInfo
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clock_verify import factor, num, NY, UTC  # noqa: E402

RAW = "/ocean/projects/med250003p/shared/mladi_extract_2023_waves"


def parse(origin):
    stamp, tz = origin.rsplit(" ", 1)
    return datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S.%f"), tz


def state(wall_naive):
    """DST state ('EDT'/'EST') of a New York wall-clock time."""
    return wall_naive.replace(tzinfo=NY).tzname()


def one(p):
    import h5py
    try:
        with h5py.File(p, "r") as f:
            W, tz = parse(json.loads(f.attrs[".meta"])["time_origin"])
            if tz == "LMT" or "ehr/low_rate" not in f or "data/numerics/NBP.NBPs" not in f:
                return None
            d = factor(f["ehr/low_rate"])
            name, t, v = d["eventName"], d["date"].astype(float), np.array([num(x) for x in d["resultVal"]])
            sb = (name == "Systolic BP") & np.isfinite(v) & np.isfinite(t)
            if sb.sum() < 3:
                return None
            ns = f["data/numerics/NBP.NBPs"][:]
            tn, vs = ns["time"].astype(float), ns["value"].astype(float)
            keep = np.r_[True, np.diff(vs) != 0] & np.isfinite(vs); tn, vs = tn[keep], vs[keep]
            dd = []
            for x, val in zip(t[sb], v[sb]):
                z = (x - tn[np.abs(vs - val) <= 0.5]) / 60.0
                dd.append(z[np.abs(z) <= 360])
            if not any(len(z) for z in dd):
                return None
            obs = collections.Counter(np.round(np.concatenate(dd)).astype(int).tolist()).most_common(1)[0][0]
            # candidate instants (all on the raw clocks: walls = W + raw)
            cand = {"origin": W}
            if "data/waveforms/Pleth" in f:
                m = json.loads(f["data/waveforms/Pleth"].attrs[".meta"])["dwc_meta"]
                cand["wave_start"] = W + timedelta(seconds=float(m["minTime"]))
                cand["wave_end"] = W + timedelta(seconds=float(m["maxTime"]))
            cand["first_sbp"] = W + timedelta(seconds=float(t[sb].min()))
            cand["last_sbp"] = W + timedelta(seconds=float(t[sb].max()))
            last_any = -np.inf
            for k in f["ehr"].keys():
                ds = f["ehr"][k]
                if getattr(ds, "dtype", None) is not None and ds.dtype.names:
                    for c in ("time", "date", "endDate", "dischDate"):
                        if c in ds.dtype.names:
                            a = ds[c][:].astype(float); a = a[np.isfinite(a)]
                            if a.size:
                                last_any = max(last_any, a.max())
            if np.isfinite(last_any):
                cand["last_ehr_any"] = W + timedelta(seconds=float(last_any))
            if "ehr/demographic" in f:
                dm = f["ehr/demographic"][:]
                for c in ("regDate", "dischDate"):
                    if c in dm.dtype.names and np.isfinite(dm[c][0]):
                        cand[c] = W + timedelta(seconds=float(dm[c][0]))
            s0 = state(W)
            pred = {}
            for k, wall in cand.items():
                sB = state(wall)
                pred[k] = 0 if sB == s0 else (-60 if s0 == "EST" else 60)
            return {"obs": int(obs), "pred": pred, "origin_state": s0}
    except Exception as ex:
        return {"error": f"{type(ex).__name__}: {ex}"}


def main():
    files = sorted(glob.glob(os.path.join(RAW, "*.h5")))
    random.seed(3)
    pick = random.sample(files, min(int(os.environ.get("N_ENC", 6000)), len(files)))
    with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", 16))) as pool:
        R = [r for r in pool.map(one, pick, chunksize=4) if r]
    E = [r for r in R if "error" in r]; R = [r for r in R if "obs" in r]
    print(f"{len(R)} encounters with exact NBP matches (errors {len(E)})", flush=True)
    obs = np.array([r["obs"] for r in R])
    near = lambda a, b: abs(a - b) <= 2
    print("observed offset (min): " + str(collections.Counter(int(np.round(o / 30) * 30) for o in obs).most_common(8)), flush=True)
    keys = sorted({k for r in R for k in r["pred"]})
    for k in keys:
        rr = [r for r in R if k in r["pred"]]
        acc = np.mean([near(r["obs"], r["pred"][k]) for r in rr])
        nz = [r for r in rr if not near(r["obs"], 0)]
        acc_nz = np.mean([near(r["obs"], r["pred"][k]) for r in nz]) if nz else float("nan")
        print(f"  B = {k:13s}: predicts the observed offset in {acc:.1%} of {len(rr)} | among the {len(nz)} non-zero ones {acc_nz:.1%}", flush=True)
    # residual misses for the best candidate
    best = max(keys, key=lambda k: np.mean([near(r["obs"], r["pred"][k]) for r in R if k in r["pred"]]))
    miss = collections.Counter((int(np.round(r["obs"] / 30) * 30), r["pred"].get(best)) for r in R if not near(r["obs"], r["pred"].get(best, 999)))
    print(f"best B = {best}; misses (observed rounded, predicted): {miss.most_common(10)}", flush=True)


if __name__ == "__main__":
    main()
