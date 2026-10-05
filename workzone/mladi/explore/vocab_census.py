#!/usr/bin/env python3
"""MLADI Step 0b (cont.): decode the audata factor columns of every file and count, per clinical
term, how many encounters carry it -- the input for mapping /ehr onto the variable registry.

audata: each table's `.meta` attribute lists `levels` per factor column (per file); a code is an
index into them (negative = missing). Time columns are seconds from the root `.meta` time_origin.
Counts per term are encounter counts (and total rows), clinical vocabulary only. Also: the
distribution of time_origin (year only) and of the Pleth start offset, to explain the ~1 % of files
whose waveform clock starts near 6.9e9 s. Read-only; streams partial tables every 4000 files.
"""
import collections, glob, json, os, sys, time
from multiprocessing import Pool
import numpy as np

RAW = "/ocean/projects/med250003p/shared/mladi_extract_2023_waves"
OUT = "/ocean/projects/med250003p/mwang11/Physio_Data/workzone/outputs/mladi/explore"
COLS = {"lab_results": ("eventDisp", "orderedAs", "resultUnit"), "low_rate": ("eventName", "resultUnit"),
        "medications": ("catalogDisp", "route", "doseUnit"), "infusions_and_outputs": ("name",),
        "diagnostic_codes": ("codeType", "type"), "demographic": ("sex", "race", "encntrType", "dischDisp", "ethnicity"),
        "patient": ("clinicalUnit",), "location": ("unit",), "csce": ("eventTag",)}
T0 = time.time()


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:6.0f}s]", *a, flush=True)


def one(p):
    import h5py
    r = {"terms": collections.Counter(), "rows": collections.Counter()}
    try:
        with h5py.File(p, "r") as f:
            m = json.loads(f.attrs[".meta"]) if ".meta" in f.attrs else {}
            r["origin_year"] = str(m.get("time_origin", "?"))[:4]
            r["origin_tz"] = str(m.get("time_origin", "?"))[-3:]
            if "data/waveforms/Pleth" in f:
                pm = json.loads(f["data/waveforms/Pleth"].attrs[".meta"])["dwc_meta"]
                r["pleth_t0"] = pm.get("minTime")
            for t, cols in COLS.items():
                k = f"ehr/{t}"
                if k not in f:
                    continue
                d = f[k]; a = d[:]
                cm = json.loads(d.attrs[".meta"]).get("columns", {}) if ".meta" in d.attrs else {}
                for c in cols:
                    if c not in (a.dtype.names or ()):
                        continue
                    lev = cm.get(c, {}).get("levels")
                    if not lev:
                        continue
                    codes = a[c].astype(int)
                    for code, n in collections.Counter(codes.tolist()).items():
                        name = lev[code] if 0 <= code < len(lev) else "<NA>"
                        r["terms"][(t, c, name)] += 1
                        r["rows"][(t, c, name)] += n
    except Exception as ex:
        r["error"] = f"{type(ex).__name__}: {ex}"
    return r


def report(R, final=False):
    T = collections.Counter(); N = collections.Counter()
    for r in R:
        T.update(r["terms"]); N.update(r["rows"])
    log(("FINAL " if final else "") + f"{len(R)} files")
    by = collections.defaultdict(list)
    for (t, c, name), n in T.items():
        by[(t, c)].append((n, N[(t, c, name)], name))
    out = {}
    for (t, c), lst in sorted(by.items()):
        lst.sort(reverse=True)
        top = 60 if (t, c) in (("lab_results", "eventDisp"), ("low_rate", "eventName"), ("medications", "catalogDisp")) else 15
        log(f"  {t}.{c}: {len(lst)} distinct | " + "; ".join(f"{name[:40]} [{n} enc, {rows} rows]" for n, rows, name in lst[:top]))
        out[f"{t}.{c}"] = [[name, n, rows] for n, rows, name in lst]
    return out


def main():
    files = sorted(glob.glob(os.path.join(RAW, "*.h5")))
    log(f"{len(files)} files")
    for fn in ("config.json", "conv.py"):
        p = os.path.join(RAW, fn)
        if os.path.exists(p):
            txt = open(p, errors="replace").read()
            log(f"---- {fn} (first 2500 chars of {len(txt)})\n" + txt[:2500])
    R = []
    with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", 16))) as pool:
        for k, r in enumerate(pool.imap_unordered(one, files, chunksize=8), 1):
            R.append(r)
            if k % 4000 == 0:
                report(R)
    vocab = report(R, final=True)
    log("errors: " + str(collections.Counter(r["error"].split(":")[0] for r in R if "error" in r).most_common(5)))
    log("time_origin year: " + str(sorted(collections.Counter(r.get("origin_year") for r in R).items())))
    log("time_origin tz: " + str(collections.Counter(r.get("origin_tz") for r in R).most_common(5)))
    t0 = np.array([r["pleth_t0"] for r in R if r.get("pleth_t0") is not None], float)
    big = t0 > 1e8
    log(f"Pleth start > 1e8 s in {big.sum()} of {t0.size} files; their origin years: "
        + str(collections.Counter(r.get("origin_year") for r in R if (r.get("pleth_t0") or 0) > 1e8).most_common(5)))
    os.makedirs(OUT, exist_ok=True)
    json.dump({"vocab": vocab}, open(os.path.join(OUT, "vocab_census.json"), "w"))
    log(f"wrote {OUT}/vocab_census.json")


if __name__ == "__main__":
    main()
