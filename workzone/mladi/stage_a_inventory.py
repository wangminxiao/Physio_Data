#!/usr/bin/env python3
"""MLADI Stage A: per-entity inventory, clock classification and clock check (read-only on raw data).

One row per raw HDF5 (= encounter = entity), written to <intermediate>/stage_a/inventory.jsonl:

  identity   entity_id (= base), patient_id (last field), e1_split
  waveform   Pleth / II presence, sample periods, spans; ECG filter edges; the pretrain_wav_v2 grid
             (__meta.json seg_list): rows, blocks, first / last segment
  numerics   which data/numerics keys exist among those the store uses, and their row counts
  ehr        tables present, row counts, has_ehr; lab coverage of the waveform span
  clock      origin year / label, clock_rule, discharge seconds, ehr_origin_shift_min (the rule of
             workzone/mladi/clock.py), and the CHECK: charted SBP/DBP vs monitor NBP exact matches after
             conversion -> residual offset (min), n matches, clock_confidence:
               verified   residual within +-2 min on >= 3 matched readings
               corrected  residual +-60 (+-2) on >= 3 -> ehr_extra_shift_min applied by later stages
               conflict   any other residual on >= 3
               inferred   fewer than 3 matches (rule only)
  included   has Pleth and >= 1 grid row (the canonical store's entity set)

Resumable: entities already in inventory.jsonl are skipped. Prints a running summary every 1000.

    python workzone/mladi/stage_a_inventory.py --limit 5 --workers 4
    python workzone/mladi/stage_a_inventory.py --workers 32
"""
from __future__ import annotations
import argparse, collections, glob, json, os, sys, time
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clock  # noqa: E402
from common import cfg, factor, nums, nbp_offset  # noqa: E402

T0 = time.time()
USED_NUMERICS = ["HR.HR", "SpO₂.SpO₂", "RR.RR", "SpO₂.Pulse", "Perf.Perf", "PVC.PVC", "CVP.CVPm",
                 "ART.Systolic", "ART.Diastolic", "ART.Mean", "ART.Pulse", "ABP.ABPs", "ABP.ABPd", "ABP.ABPm",
                 "ABP.Pulse", "NBP.NBPs", "NBP.NBPd", "NBP.NBPm"]
EHR_TABLES = ["lab_results", "low_rate", "medications", "infusions_and_outputs", "diagnostic_codes",
              "demographic", "patient", "location", "csce", "culture_sensitivity"]
DAY_MS = 86_400_000


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:6.0f}s]", *a, flush=True)


def dwc_meta(ds):
    try:
        return json.loads(ds.attrs[".meta"])["dwc_meta"]
    except Exception:
        return {}


_E1, _WAV = {}, None


def _init(e1, wav_dir):
    global _E1, _WAV
    _E1, _WAV = e1, wav_dir


def one(path):
    e1, wav_dir = _E1, _WAV
    import h5py
    base = os.path.basename(path)[:-3]
    r = {"entity_id": base, "patient_id": base.rsplit("_", 1)[-1], "e1_split": e1.get(base, "absent")}
    try:
        meta_p = os.path.join(wav_dir, base + "__meta.json")
        seg = []
        if os.path.exists(meta_p):
            seg = json.load(open(meta_p)).get("seg_list", [])
        st = np.sort(np.array([s[2] for s in seg], float)) if seg else np.zeros(0)
        r.update(grid_rows=int(st.size), grid_blocks=int(len({s[0] for s in seg})) if seg else 0,
                 grid_first_s=float(st[0]) if st.size else None, grid_last_s=float(st[-1]) if st.size else None)
        with h5py.File(path, "r") as f:
            W, label = clock.parse_origin(json.loads(f.attrs[".meta"])["time_origin"])
            r.update(origin_year=W.year, origin_label=label)
            wf = f.get("data/waveforms")
            for ch in ("Pleth", "II"):
                if wf is not None and ch in wf:
                    m = dwc_meta(wf[ch])
                    r[f"has_{ch.lower()}"] = True
                    r[f"{ch.lower()}_sp_ms"] = m.get("samplePeriod")
                    r[f"{ch.lower()}_span_s"] = [m.get("minTime"), m.get("maxTime")]
                    if ch == "II":
                        r["ecg_filter"] = [m.get("lowEdgeFrequency"), m.get("highEdgeFrequency")]
                else:
                    r[f"has_{ch.lower()}"] = False
            nm = f.get("data/numerics")
            r["numerics"] = {k: int(nm[k].shape[0]) for k in USED_NUMERICS if nm is not None and k in nm}
            ehr = f.get("ehr")
            r["ehr_rows"] = {t: int(ehr[t].shape[0]) for t in EHR_TABLES if ehr is not None and t in ehr}
            r["has_ehr"] = any(r["ehr_rows"].get(t, 0) > 0 for t in ("lab_results", "low_rate", "medications"))
            disch = None
            if ehr is not None and "demographic" in ehr and "dischDate" in ehr["demographic"].dtype.names:
                x = float(ehr["demographic"][0]["dischDate"]); disch = x if np.isfinite(x) else None
            utc_rule = label == "LMT" or W.year in clock.UTC_ORIGIN_YEARS
            r["clock_rule"] = "dwc_wall/ehr_utc_origin" if utc_rule else "dwc_wall/ehr_wall_disch"
            r["disch_s"] = disch
            r["ehr_origin_shift_min"] = (None if utc_rule else
                                         (clock.ehr_origin_wall(W, label, disch) - W).total_seconds() / 60)
            r["included"] = bool(r["has_pleth"] and r["grid_rows"] > 0)
            if not r["included"]:
                return r
            grid = clock.Grid(W, st[0])
            g0, g1 = grid.dwc([st[0], st[-1] + 30], W)
            r["grid_span_h"] = float((g1 - g0) / 3.6e6)
            # ---- clock check: charted SBP/DBP vs monitor NBP, both on the grid
            r["clock_check"] = {"n_matches": 0, "n_charted": 0, "residual_min": None}
            if ehr is not None and "low_rate" in ehr and nm is not None and "NBP.NBPs" in nm:
                d = factor(ehr["low_rate"])
                name, t, v = d["eventName"], d["date"].astype(float), nums(d["resultVal"])
                sb = (name == "Systolic BP") & np.isfinite(v) & np.isfinite(t)
                db = (name == "Diastolic BP") & np.isfinite(v) & np.isfinite(t)
                if sb.any():
                    dmap = dict(zip(np.round(t[db], 0), v[db]))
                    cd = np.array([dmap.get(round(x, 0), np.nan) for x in t[sb]])
                    a = nm["NBP.NBPs"][:]; ts_, vs_ = a["time"].astype(float), a["value"].astype(float)
                    keep = np.r_[True, np.diff(vs_) != 0] & np.isfinite(vs_)
                    ts_, vs_ = ts_[keep], vs_[keep]
                    vd_ = np.full(vs_.size, np.nan)
                    if "NBP.NBPd" in nm:
                        b = nm["NBP.NBPd"][:]
                        dm = dict(zip(np.round(b["time"].astype(float), 1), b["value"].astype(float)))
                        vd_ = np.array([dm.get(x, np.nan) for x in np.round(ts_, 1)])
                    off, n_ok, n_ch = nbp_offset(grid.ehr(t[sb], W, label, disch), v[sb], cd,
                                                 grid.dwc(ts_, W), vs_, vd_)
                    r["clock_check"] = {"n_matches": n_ok, "n_charted": n_ch, "residual_min": off}
            cc = r["clock_check"]
            if cc["n_matches"] >= 3 and cc["residual_min"] is not None:
                res = cc["residual_min"]
                if abs(res) <= 2:
                    r["clock_confidence"], r["ehr_extra_shift_min"] = "verified", 0
                elif abs(abs(res) - 60) <= 2:
                    r["clock_confidence"], r["ehr_extra_shift_min"] = "corrected", -int(np.sign(res)) * 60
                else:
                    r["clock_confidence"], r["ehr_extra_shift_min"] = "conflict", 0
            else:
                r["clock_confidence"], r["ehr_extra_shift_min"] = "inferred", 0
            # ---- lab coverage of the waveform span
            if ehr is not None and "lab_results" in ehr and ehr["lab_results"].shape[0]:
                lt = ehr["lab_results"]["time"][:].astype(float); lt = lt[np.isfinite(lt)]
                if lt.size:
                    lg = grid.ehr(lt, W, label, disch)
                    r["lab_within_24h_frac"] = float(np.mean((lg >= g0 - DAY_MS) & (lg <= g1 + DAY_MS)))
                    r["lab_within_7d"] = bool(np.any((lg >= g0 - 7 * DAY_MS) & (lg <= g1 + 7 * DAY_MS)))
    except Exception as ex:
        r["error"] = f"{type(ex).__name__}: {ex}"
    return r


def summary(R, final=False):
    inc = [r for r in R if r.get("included")]
    cc = collections.Counter(r.get("clock_confidence") for r in inc)
    log(("FINAL " if final else "") + f"{len(R)} files | included {len(inc)} | errors {sum('error' in r for r in R)} | "
        f"has_ehr {sum(r.get('has_ehr', False) for r in inc)} | has_ii {sum(r.get('has_ii', False) for r in inc)} | "
        f"ART numerics {sum('ART.Systolic' in r.get('numerics', {}) for r in inc)} | clock {dict(cc)} | "
        f"rules {dict(collections.Counter(r.get('clock_rule') for r in inc))} | "
        f"e1 {dict(collections.Counter(r.get('e1_split') for r in inc))}")
    return {"n_files": len(R), "n_included": len(inc), "clock_confidence": dict(cc),
            "errors": sum("error" in r for r in R)}


def main():
    C = cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(C["intermediate_dir"], "stage_a"))
    for k in ("raw_h5_dir", "pretrain_wav_dir", "e1_split"):
        ap.add_argument("--" + k.replace("_", "-"), default=C[k])
    a = ap.parse_args()
    C.update(raw_h5_dir=a.raw_h5_dir, pretrain_wav_dir=a.pretrain_wav_dir, e1_split=a.e1_split)
    os.makedirs(a.out, exist_ok=True)
    inv = os.path.join(a.out, "inventory.jsonl")
    done = set()
    if os.path.exists(inv):
        for line in open(inv):
            try:
                done.add(json.loads(line)["entity_id"])
            except Exception:
                pass
    files = sorted(glob.glob(os.path.join(C["raw_h5_dir"], "*.h5")))[: a.limit or None]
    todo = [p for p in files if os.path.basename(p)[:-3] not in done]
    e1 = {b: v.get("split", "unassigned") for b, v in json.load(open(C["e1_split"]))["encounters"].items()}
    log(f"{len(files)} files, {len(done)} already in {inv}, {len(todo)} to do, {a.workers} workers")
    R = []
    with open(inv, "a") as fo, Pool(a.workers, initializer=_init, initargs=(e1, C["pretrain_wav_dir"])) as pool:
        for k, r in enumerate(pool.imap_unordered(one, todo, chunksize=4), 1):
            fo.write(json.dumps(r, default=str) + "\n"); R.append(r)
            if k % 1000 == 0:
                fo.flush(); summary(R)
    allR = [json.loads(line) for line in open(inv)]
    s = summary(allR, final=True)
    json.dump(s, open(os.path.join(a.out, "inventory_summary.json"), "w"), indent=1)
    log(f"wrote {inv} and inventory_summary.json")


if __name__ == "__main__":
    main()
