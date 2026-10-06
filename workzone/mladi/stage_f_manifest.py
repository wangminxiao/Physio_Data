#!/usr/bin/env python3
"""MLADI Stage F: manifest.json, pretrain_splits.json, downstream_splits.json, demographics.csv.

Per entity (header-level checks; content was gated in B/C/D): meta.json readable; time_ms int64 strictly
increasing with n_seg rows; PLETH40 / II120 float16 [n_seg, 1200 / 3600]; the 4 EHR files, ehr_actions,
vitals_hf (+ abp_src, nbp_events) and ehr_hf present. Invalid entities are left out of the manifest and
listed in stage_f_summary.json.

Splits (API.md): the e1 split is kept (patient-level 70/15/15, seed 42). An entity outside e1 takes its
patient's e1 split if the patient has one, else a patient-hash split (seed 42, 70/15/15). No patient on two
sides. pretrain_splits.json == downstream_splits.json (UNIPHY list format [dir, patient_id, 0, n_seg, -1, 0]).

meta.json += clock_shift_confidence (= clock_confidence; the key build_estimation_task.py `exclude_meta` reads).

demographics.csv (one row per entity, decoded strings from /ehr/demographic and /ehr/location): entity_id,
patient_id, age, sex, race, ethnicity, encntrType, dischDisp, stay_days, admit_to_wave_days, first_unit,
has_ehr, has_art, e1_split, split.

    python workzone/mladi/stage_f_manifest.py [--out <root>] [--workers 16]
"""
from __future__ import annotations
import argparse, collections, csv, hashlib, json, os, sys, time
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import cfg, factor, num, EHR_EVENT_DTYPE  # noqa: E402

T0 = time.time()
FILES = ("time_ms.npy", "PLETH40.npy", "II120.npy", "ehr_baseline.npy", "ehr_recent.npy", "ehr_events.npy", "ehr_future.npy",
         "ehr_actions.npy", "vitals_hf.npy", "vitals_hf_abp_src.npy", "nbp_events.npy", "ehr_hf.npy")
_OUT, _RAW = None, None


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:6.0f}s]", *a, flush=True)


def _init(out, raw):
    global _OUT, _RAW
    _OUT, _RAW = out, raw


def first(d, c):
    v = d.get(c)
    return None if v is None or len(v) == 0 else v[0]


def one(eid):
    import h5py
    od = os.path.join(_OUT, eid); errs = []
    try:
        meta = json.load(open(os.path.join(od, "meta.json")))
    except Exception as ex:
        return {"entity_id": eid, "errors": [f"meta.json: {ex}"]}
    n = int(meta.get("n_seg") or 0)
    for fn in FILES:
        if not os.path.exists(os.path.join(od, fn)):
            errs.append(f"missing {fn}")
    if not errs:
        t = np.load(os.path.join(od, "time_ms.npy"))
        if t.dtype != np.int64 or t.size != n or not np.all(np.diff(t) > 0):
            errs.append("time_ms")
        for fn, L in (("PLETH40.npy", 1200), ("II120.npy", 3600)):
            x = np.load(os.path.join(od, fn), mmap_mode="r")
            if x.dtype != np.float16 or x.shape != (n, L):
                errs.append(f"{fn} {x.dtype} {x.shape}")
        for fn in FILES[3:8] + ("nbp_events.npy", "ehr_hf.npy"):
            if np.load(os.path.join(od, fn), mmap_mode="r").dtype != EHR_EVENT_DTYPE:
                errs.append(f"{fn} dtype")
    vh = meta.get("vitals_hf", {})
    has_art = bool(sum((vh.get("abp_slots_per_line") or {}).values()))
    entry = {"entity_id": eid, "patient_id": meta.get("patient_id"), "e1_split": meta.get("e1_split"), "n_seg": n,
             "source_dataset": "mladi", "time_base": "utc_continuous", "has_ehr": bool(meta.get("has_ehr")), "has_art": has_art,
             "clock_confidence": meta.get("clock_confidence"), "clock_risk": meta.get("clock_risk"),
             "dst_crossing_runs": meta.get("dst_crossing_runs"), "nbp_twins": bool((meta.get("nbp_twins") or {}).get("flag")),
             "n_baseline": meta.get("ehr", {}).get("n_baseline"), "n_recent": meta.get("ehr", {}).get("n_recent"),
             "n_events": meta.get("ehr", {}).get("n_events"), "n_future": meta.get("ehr", {}).get("n_future"),
             "n_actions": meta.get("ehr_actions", {}).get("n"), "n_ehr_hf": meta.get("ehr_hf", {}).get("n_events"),
             "first_time_ms": None}
    if not errs:
        entry["first_time_ms"] = int(np.load(os.path.join(od, "time_ms.npy"), mmap_mode="r")[0])
    if meta.get("clock_shift_confidence") != meta.get("clock_confidence"):
        meta["clock_shift_confidence"] = meta.get("clock_confidence")
        json.dump(meta, open(os.path.join(od, "meta.json.tmp"), "w"), indent=1, default=str)
        os.replace(os.path.join(od, "meta.json.tmp"), os.path.join(od, "meta.json"))
    demo = {"entity_id": eid, "patient_id": meta.get("patient_id"), "has_ehr": entry["has_ehr"], "has_art": has_art,
            "e1_split": meta.get("e1_split")}
    try:
        with h5py.File(os.path.join(_RAW, eid + ".h5"), "r") as f:
            ehr = f.get("ehr")
            if ehr is not None and "demographic" in ehr and ehr["demographic"].shape[0]:
                d = factor(ehr["demographic"])
                for c in ("age", "sex", "race", "ethnicity", "encntrType", "dischDisp"):
                    v = first(d, c)
                    demo[c] = (float(num(v)) if c == "age" else (None if v is None else str(v)))
                # episode bounds on the entity grid were written by Stage D (meta.ehr)
                e0, e1 = meta.get("ehr", {}).get("episode_start_ms"), meta.get("ehr", {}).get("episode_end_ms")
                if e0 is not None and e1 is not None:
                    demo["stay_days"] = round((e1 - e0) / 86400000.0, 3)
                if e0 is not None and entry["first_time_ms"] is not None:
                    demo["admit_to_wave_days"] = round((entry["first_time_ms"] - e0) / 86400000.0, 3)
            if ehr is not None and "location" in ehr and ehr["location"].shape[0]:
                d = factor(ehr["location"])
                for c in ("clinicalUnit", "unit"):
                    if c in d:
                        tcol = "time" if "time" in d else None
                        order = np.argsort(np.asarray(d[tcol], float)) if tcol else np.arange(len(d[c]))
                        demo["first_unit"] = str(d[c][order[0]]); break
    except Exception as ex:
        errs.append(f"demographics: {type(ex).__name__}: {ex}")
    return {"entity_id": eid, "entry": entry, "demo": demo, "errors": errs}


def hash_split(pid: str) -> str:
    h = int(hashlib.sha256(f"42:{pid}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "train" if h < 0.70 else ("val" if h < 0.85 else "test")


def main():
    C = cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=C["output_dir"]); ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--raw-h5-dir", default=C["raw_h5_dir"]); ap.add_argument("--intermediate", default=C["intermediate_dir"])
    a = ap.parse_args()
    ents = sorted(d for d in os.listdir(a.out) if os.path.exists(os.path.join(a.out, d, "meta.json")))
    log(f"{len(ents)} entity dirs")
    res = []
    with Pool(a.workers, initializer=_init, initargs=(a.out, a.raw_h5_dir)) as pool:
        for i, r in enumerate(pool.imap_unordered(one, ents, chunksize=8), 1):
            res.append(r)
            if i % 2000 == 0 or i == len(ents):
                log(f"{i}/{len(ents)} | invalid {sum(bool([e for e in x['errors'] if not e.startswith('demographics')]) for x in res)}")
    res.sort(key=lambda r: r["entity_id"])
    bad = {r["entity_id"]: r["errors"] for r in res if [e for e in r["errors"] if not e.startswith("demographics")]}
    valid = [r for r in res if r["entity_id"] not in bad]
    # splits: e1 first, then the patient's e1 split, then patient hash
    pat_e1 = collections.defaultdict(set)
    for r in valid:
        s = r["entry"]["e1_split"]
        if s in ("train", "val", "test"):
            pat_e1[r["entry"]["patient_id"]].add(s)
    conflict = [p for p, s in pat_e1.items() if len(s) > 1]
    if conflict:
        raise SystemExit(f"FATAL: {len(conflict)} patients on two e1 sides, e.g. {conflict[:3]}")
    src = collections.Counter()
    for r in valid:
        e = r["entry"]; s = e["e1_split"]
        if s in ("train", "val", "test"):
            e["split"], how = s, "e1"
        elif pat_e1.get(e["patient_id"]):
            e["split"], how = next(iter(pat_e1[e["patient_id"]])), "patient_e1"
        else:
            e["split"], how = hash_split(str(e["patient_id"])), "patient_hash"
        e["split_source"] = how; src[how] += 1; r["demo"]["split"] = e["split"]
    sides = collections.defaultdict(set)
    for r in valid:
        sides[r["entry"]["patient_id"]].add(r["entry"]["split"])
    assert not [p for p, s in sides.items() if len(s) > 1], "patient on two sides"
    man = [r["entry"] for r in valid]
    json.dump(man, open(os.path.join(a.out, "manifest.json"), "w"), indent=1, default=str)
    sp = {k: [e["entity_id"] for e in man if e["split"] == k] for k in ("train", "val", "test")}
    json.dump({"seed": 42, "ratios": [0.7, 0.15, 0.15], "group_by": "patient_id",
               "source": "e1 split (Physio_HNET data_cache/e1_mladi), then the patient's e1 split, then patient hash",
               "split_source_counts": dict(src), **{f"n_{k}": len(v) for k, v in sp.items()}, **sp},
              open(os.path.join(a.out, "pretrain_splits.json"), "w"), indent=1)
    by = {e["entity_id"]: e for e in man}
    lst = lambda ids: [[os.path.join(a.out, i), str(by[i]["patient_id"]), 0, int(by[i]["n_seg"]), -1, 0] for i in ids]
    json.dump({f"{k}_control_list": lst(v) for k, v in sp.items()}, open(os.path.join(a.out, "downstream_splits.json"), "w"), indent=1)
    cols = ["entity_id", "patient_id", "age", "sex", "race", "ethnicity", "encntrType", "dischDisp", "stay_days",
            "admit_to_wave_days", "first_unit", "has_ehr", "has_art", "e1_split", "split"]
    with open(os.path.join(a.out, "demographics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        for r in valid:
            w.writerow(r["demo"])
    summ = {"n_dirs": len(ents), "n_valid": len(valid), "n_invalid": len(bad), "invalid": dict(list(bad.items())[:50]),
            "split_counts": {k: len(v) for k, v in sp.items()}, "split_source_counts": dict(src),
            "n_segments": int(sum(e["n_seg"] for e in man)), "has_ehr": int(sum(e["has_ehr"] for e in man)),
            "has_art": int(sum(e["has_art"] for e in man)),
            "demographics_errors": int(sum(any(x.startswith("demographics") for x in r["errors"]) for r in valid))}
    os.makedirs(a.intermediate, exist_ok=True)
    json.dump(summ, open(os.path.join(a.intermediate, "stage_f_summary.json"), "w"), indent=1)
    log(json.dumps({k: v for k, v in summ.items() if k != "invalid"}))
    fails = []
    if len(valid) < 0.99 * len(ents): fails.append(f"valid {len(valid)} of {len(ents)}")
    moved = [e["entity_id"] for e in man if e["e1_split"] in ("train", "val", "test") and e["split"] != e["e1_split"]]
    if moved: fails.append(f"{len(moved)} entities moved off their e1 split")
    print("GATE F:", "FAIL " + "; ".join(fails) if fails else "PASS", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
