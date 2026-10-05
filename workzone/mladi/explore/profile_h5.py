#!/usr/bin/env python3
"""MLADI Step 0b: structural exploration of the raw DWC HDF5 files (read-only).

Two passes, each printing as it goes (a timeout still leaves findings in the log):

  census   every file, in parallel: waveform / numerics / ehr keys present and their sizes, the
           waveform span (dwc_meta minTime/maxTime), whether the earlier 30-s mmap set
           (pretrain_wav_v2 __meta.json) and the e1 split know the encounter, patient linkage.
  detail   a year-stratified sample: every dataset with shape/dtype/attrs, numerics cadence and
           value ranges, each EHR table's columns, and for each time-like column its TYPE and FORMAT
           PATTERN (digits -> D, letters -> A) plus, for numeric times, the range relative to the
           waveform span -- never a raw date. Top names in the vocab columns (lab names, med names,
           charted-vital names) are counted: they are clinical terms, not identifiers.

Writes <intermediate_dir>/explore/dataset_profile.json (aggregate counts and formats only).

    python workzone/mladi/explore/profile_h5.py --n-detail 60 --workers 16
"""
from __future__ import annotations
import argparse, collections, glob, json, os, random, re, sys, time
from multiprocessing import Pool
import numpy as np

T0 = time.time()
CFG = dict(raw_h5_dir="/ocean/projects/med250003p/shared/mladi_extract_2023_waves",
           pretrain_wav_dir="/ocean/projects/med250003p/shared/pretrain_wav_v2",
           e1_split="/ocean/projects/med250003p/shared/data_cache/e1_mladi/mladi_encounter_index.json",
           intermediate_dir="/ocean/projects/med250003p/mwang11/Physio_Data/workzone/outputs/mladi")
VOCAB = {"lab_results": ("eventTag", "orderedAs", "resultUnit"), "medications": ("catalogDisp", "route", "doseUnit"),
         "infusions_and_outputs": ("name", "unit"), "low_rate": ("eventName", "resultUnit"),
         "diagnostic_codes": ("codeType", "type", "source"), "patient": ("category", "clinicalUnit"),
         "demographic": ("encntrType", "dischDisp", "sex"), "csce": ("eventName",), "location": ("unit",)}
TIME_HINT = re.compile(r"time|date", re.I)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:6.0f}s]", *a, flush=True)


def _s(v):
    return v.decode(errors="replace") if isinstance(v, (bytes, np.bytes_)) else v


def pattern(v) -> str:
    """Format of a value with its content removed: digits -> D, letters -> A."""
    s = str(_s(v))
    return re.sub(r"\d", "D", re.sub(r"[A-Za-z]", "A", s))[:40]


def meta_of(ds):
    try:
        raw = ds.attrs.get(".meta")
        raw = _s(raw) if raw is not None else None
        return json.loads(raw).get("dwc_meta", {}) if raw else {}
    except Exception:
        return {}


def census(path):
    import h5py
    base = os.path.basename(path)[:-3]
    out = {"base": base, "size_gb": os.path.getsize(path) / 1e9}
    try:
        with h5py.File(path, "r") as f:
            wf = f.get("data/waveforms"); nm = f.get("data/numerics"); ehr = f.get("ehr")
            out["top"] = sorted(f.keys())
            out["data_keys"] = sorted(f["data"].keys()) if "data" in f else []
            if wf is not None:
                W = {}
                for k in wf.keys():
                    m = meta_of(wf[k])
                    W[k] = {"rows": int(wf[k].shape[0]), "sp_ms": m.get("samplePeriod"),
                            "t0": m.get("minTime"), "t1": m.get("maxTime")}
                out["waves"] = W
            if nm is not None:
                out["numerics"] = {k: int(nm[k].shape[0]) for k in nm.keys()}
            if ehr is not None:
                out["ehr"] = {k: (int(ehr[k].shape[0]) if hasattr(ehr[k], "shape") else -1) for k in ehr.keys()}
    except Exception as ex:
        out["error"] = f"{type(ex).__name__}: {ex}"
    return out


def detail(path):
    """Everything about one file, as formats and ranges."""
    import h5py
    base = os.path.basename(path)[:-3]
    out = {"base": base, "datasets": {}, "numerics": {}, "ehr": {}}
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                out["datasets"][name] = {"shape": list(obj.shape), "dtype": str(obj.dtype)[:200],
                                         "attrs": sorted(obj.attrs.keys())}
        f.visititems(visit)
        wf = f.get("data/waveforms")
        span = None
        if wf is not None and "Pleth" in wf:
            m = meta_of(wf["Pleth"]); span = (m.get("minTime"), m.get("maxTime"))
        out["wave_span"] = span
        nm = f.get("data/numerics")
        if nm is not None:
            for k in nm.keys():
                ds = nm[k]
                try:
                    a = ds[:]
                    names = a.dtype.names or ()
                    t = a["time"].astype(float) if "time" in names else None
                    v = a["value"].astype(float) if "value" in names else None
                    rec = {"rows": int(a.shape[0]), "fields": list(names), "meta": {kk: meta_of(ds).get(kk) for kk in ("label", "unitLabel", "samplePeriod")}}
                    if t is not None and t.size > 2:
                        dt = np.diff(t)
                        rec.update(dt_p50=float(np.median(dt)), dt_p90=float(np.percentile(dt, 90)),
                                   t_rel_wave=[float(t[0] - span[0]), float(t[-1] - span[1])] if span and span[0] else None)
                    if v is not None and v.size:
                        fv = v[np.isfinite(v)]
                        rec["value_p1_p50_p99"] = np.percentile(fv, [1, 50, 99]).round(3).tolist() if fv.size else None
                        rec["nonfinite_frac"] = float(1 - fv.size / v.size)
                    out["numerics"][k] = rec
                except Exception as ex:
                    out["numerics"][k] = {"error": str(ex)}
        ehr = f.get("ehr")
        if ehr is not None:
            for k in ehr.keys():
                ds = ehr[k]
                if not hasattr(ds, "dtype") or ds.dtype.names is None:
                    out["ehr"][k] = {"kind": type(ds).__name__}; continue
                a = ds[:]
                rec = {"rows": int(a.shape[0]), "columns": {}}
                for c in a.dtype.names:
                    col = a[c]
                    info = {"dtype": str(col.dtype)}
                    if TIME_HINT.search(c) and col.size:
                        if np.issubdtype(col.dtype, np.number):
                            fv = col[np.isfinite(col.astype(float))].astype(float)
                            if fv.size:
                                info["num_min"], info["num_max"] = float(fv.min()), float(fv.max())
                                if span and span[0]:
                                    info["rel_to_wave_start_s"] = [float(fv.min() - span[0]), float(fv.max() - span[0])]
                        else:
                            info["patterns"] = dict(collections.Counter(pattern(x) for x in col[:200]).most_common(3))
                    rec["columns"][c] = info
                out["ehr"][k] = rec
                out["ehr"][k]["vocab"] = {}
                for c in VOCAB.get(k, ()):
                    if c in a.dtype.names:
                        out["ehr"][k]["vocab"][c] = collections.Counter(str(_s(x)).strip() for x in a[c]).most_common(40)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-detail", type=int, default=60); ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="census only the first N files (testing)")
    for k in CFG:
        ap.add_argument("--" + k.replace("_", "-"), default=None)
    a = ap.parse_args()
    for k in CFG:
        if getattr(a, k):
            CFG[k] = getattr(a, k)
    out_dir = os.path.join(CFG["intermediate_dir"], "explore"); os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(CFG["raw_h5_dir"], "*.h5")))[: a.limit or None]
    log(f"{len(files)} h5 files, {sum(os.path.getsize(p) for p in files) / 1e12:.2f} TB")
    have_meta = {os.path.basename(p)[: -len("__meta.json")] for p in glob.glob(os.path.join(CFG["pretrain_wav_dir"], "*__meta.json"))}
    e1 = json.load(open(CFG["e1_split"]))["encounters"]
    log(f"pretrain_wav_v2 __meta.json for {len(have_meta)} encounters; e1 index {len(e1)} "
        f"({collections.Counter(v.get('split') for v in e1.values())})")

    # ---- census
    C = []
    with Pool(a.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(census, files, chunksize=8), 1):
            C.append(r)
            if k % 2000 == 0 or k == len(files):
                log(f"census {k}/{len(files)}")
    ok = [r for r in C if "error" not in r]
    log(f"errors {len(C) - len(ok)}: {collections.Counter(r['error'].split(':')[0] for r in C if 'error' in r).most_common(5)}")
    S = {"n_files": len(C), "n_ok": len(ok)}
    S["top_keys"] = collections.Counter(tuple(r["top"]) for r in ok).most_common(5)
    S["data_keys"] = collections.Counter(tuple(r["data_keys"]) for r in ok).most_common(5)
    S["waveforms"] = collections.Counter(k for r in ok for k in r.get("waves", {})).most_common(40)
    S["numerics"] = collections.Counter(k for r in ok for k in r.get("numerics", {})).most_common(80)
    S["ehr_tables"] = collections.Counter(k for r in ok for k in r.get("ehr", {})).most_common(20)
    S["ehr_rows_median"] = {t: float(np.median([r["ehr"][t] for r in ok if t in r.get("ehr", {})])) for t, _ in S["ehr_tables"]}
    for k in ("waveforms", "numerics", "ehr_tables"):
        log(f"{k}: " + ", ".join(f"{n} {c}" for n, c in S[k]))
    log("ehr rows (median per file): " + json.dumps(S["ehr_rows_median"]))
    durs = [(r["waves"]["Pleth"]["t1"] - r["waves"]["Pleth"]["t0"]) / 3600 for r in ok
            if "Pleth" in r.get("waves", {}) and r["waves"]["Pleth"]["t0"] is not None]
    S["pleth_span_h_p5_p50_p95"] = np.percentile(durs, [5, 50, 95]).round(1).tolist() if durs else None
    t0s = [r["waves"]["Pleth"]["t0"] for r in ok if "Pleth" in r.get("waves", {}) and r["waves"]["Pleth"]["t0"] is not None]
    S["pleth_t0_s_p1_p50_p99"] = np.percentile(t0s, [1, 50, 99]).round(0).tolist() if t0s else None
    log(f"Pleth span h p5/p50/p95 {S['pleth_span_h_p5_p50_p95']} | Pleth start (waveform clock, s) p1/p50/p99 {S['pleth_t0_s_p1_p50_p99']}")
    pat = collections.Counter(b.rsplit("_", 1)[-1] for b in (r["base"] for r in ok))
    S["n_patients"] = len(pat); S["encounters_per_patient_p50_p99_max"] = [float(np.median(list(pat.values()))), float(np.percentile(list(pat.values()), 99)), max(pat.values())]
    S["with_pretrain_meta"] = sum(r["base"] in have_meta for r in ok)
    S["e1_split"] = dict(collections.Counter(e1.get(r["base"], {}).get("split", "absent") for r in ok))
    # does the e1 split keep each patient on one side?
    psplit = collections.defaultdict(set)
    for b, v in e1.items():
        if v.get("split") in ("train", "val", "test"):
            psplit[b.rsplit("_", 1)[-1]].add(v["split"])
    S["e1_patients_on_two_sides"] = sum(len(s) > 1 for s in psplit.values())
    log(f"patients {S['n_patients']} (encounters/patient p50/p99/max {S['encounters_per_patient_p50_p99_max']}) | "
        f"with pretrain_wav_v2 meta {S['with_pretrain_meta']} | e1 split {S['e1_split']} | e1 patients on two sides {S['e1_patients_on_two_sides']}")
    json.dump({"census": S}, open(os.path.join(out_dir, "dataset_profile.json"), "w"), indent=1, default=str)

    # ---- detail on a year-stratified sample
    by_year = collections.defaultdict(list)
    for r in ok:
        by_year[r["base"][:4]].append(r["base"])
    rng = random.Random(0); pick = []
    per = max(1, a.n_detail // max(1, len(by_year)))
    for y in sorted(by_year):
        pick += rng.sample(by_year[y], min(per, len(by_year[y])))
    log(f"detail on {len(pick)} files across years {sorted(by_year)}")
    D = []
    with Pool(min(a.workers, 8)) as pool:
        for k, r in enumerate(pool.imap_unordered(detail, [os.path.join(CFG["raw_h5_dir"], b + ".h5") for b in pick]), 1):
            D.append(r)
            if k == 1:
                log("first file, datasets: " + json.dumps({n: (v["shape"], v["dtype"][:60]) for n, v in list(r["datasets"].items())[:60]}))
    # numerics summary
    NM = collections.defaultdict(list)
    for r in D:
        for k, v in r["numerics"].items():
            NM[k].append(v)
    log("numerics in the sample (files, rows p50, dt p50 s, value p1/p50/p99 of the first file, unit):")
    for k, v in sorted(NM.items(), key=lambda kv: -len(kv[1])):
        g = [x for x in v if "rows" in x]
        if g:
            log(f"   {k:28s} {len(g):3d} files | rows {np.median([x['rows'] for x in g]):9.0f} | dt {np.median([x.get('dt_p50', np.nan) for x in g]):7.2f} s"
                f" | {g[0].get('value_p1_p50_p99')} {g[0].get('meta', {}).get('unitLabel')} | start-wave_start p50 "
                f"{np.median([x['t_rel_wave'][0] for x in g if x.get('t_rel_wave')]) if any(x.get('t_rel_wave') for x in g) else None}")
    # ehr summary
    E = collections.defaultdict(list)
    for r in D:
        for t, v in r["ehr"].items():
            E[t].append(v)
    for t, v in sorted(E.items()):
        cols = v[0].get("columns", {})
        log(f"ehr/{t}: in {len(v)} files, rows p50 {np.median([x.get('rows', 0) for x in v]):.0f}")
        for c, info in cols.items():
            if "patterns" in info or "num_min" in info:
                pats = collections.Counter(p for x in v for p, n in (x["columns"].get(c, {}).get("patterns") or {}).items())
                rel = [x["columns"][c]["rel_to_wave_start_s"] for x in v if x["columns"].get(c, {}).get("rel_to_wave_start_s")]
                log(f"     time col {c:12s} dtype {info['dtype']:10s} | patterns {dict(pats.most_common(3))}"
                    + (f" | numeric, (min,max) - wave start, p50 over files: {np.median([q[0] for q in rel]):.0f} / {np.median([q[1] for q in rel]):.0f} s" if rel else ""))
        vc = collections.Counter()
        for x in v:
            for c, lst in x.get("vocab", {}).items():
                for name, n in lst:
                    vc[(c, name)] += n
        if vc:
            log(f"     top values: " + "; ".join(f"{c}={name[:30]}:{n}" for (c, name), n in vc.most_common(25)))
    json.dump({"census": S, "detail_sample": [r["base"][:4] for r in D], "numerics": {k: v[:3] for k, v in NM.items()},
               "ehr": {t: {"columns": v[0].get("columns"), "files": len(v)} for t, v in E.items()}},
              open(os.path.join(out_dir, "dataset_profile.json"), "w"), indent=1, default=str)
    log(f"wrote {out_dir}/dataset_profile.json")


if __name__ == "__main__":
    main()
