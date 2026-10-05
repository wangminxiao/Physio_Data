"""Why does clock_pairs find no DST change where clock_verify saw zone changes? Same sample as
clock_verify (seed 1, 1500 files): for files whose first and last charted SBP differ in zone, print
the span (UTC dates) and the transitions found."""
import glob, json, os, random, sys
from datetime import datetime
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clock_verify import Clock, factor, num, transitions, NY, UTC
import h5py
files = sorted(glob.glob("/ocean/projects/med250003p/shared/mladi_extract_2023_waves/*.h5"))
random.seed(1); pick = random.sample(files, 1500)
n = 0
for p in pick:
    with h5py.File(p, "r") as f:
        ck = Clock(json.loads(f.attrs[".meta"])["time_origin"])
        if "ehr/low_rate" not in f:
            continue
        d = factor(f["ehr/low_rate"])
        name, t, v = d["eventName"], d["date"].astype(float), np.array([num(x) for x in d["resultVal"]])
        sb = (name == "Systolic BP") & np.isfinite(v) & np.isfinite(t)
        if sb.sum() < 3:
            continue
        te = t[sb]; u = ck.ehr_utc([te.min(), te.max()])
        z = [datetime.fromtimestamp(float(x), tz=NY).tzname() for x in u]
        if z[0] != z[1]:
            a, b = (datetime.fromtimestamp(float(x), UTC) for x in u)
            tr = transitions(a.year, b.year)
            inside = [datetime.fromtimestamp(T, UTC).strftime("%Y-%m-%d") for T, k in tr if u[0] < T < u[1]]
            print(f"{ck.tz} {z} span {a:%Y-%m-%d}..{b:%Y-%m-%d} ({(u[1]-u[0])/86400:.1f} d) | transitions inside {inside} | all {[datetime.fromtimestamp(T, UTC).strftime('%m-%d') for T, k in tr][:4]}", flush=True)
            n += 1
            if n >= 15:
                break
