#!/usr/bin/env python3
"""MLADI Stage D2: actions -> ehr_actions.npy per entity (var 200-220; MIMIC stage3b_actions_v2 / MOVER /
MC-MED convention: a sidecar with the EHR event dtype, never written into ehr_events.npy).

Sources (times through the same EHR clock as Stage D, incl. its measured shift meta.ehr_clock.shift_min):
  medications        catalogDisp (orderedAs as fallback) -> ehr_map.drug_to_var; non-systemic routes
                     dropped; value = ehr_map.action_value (exact unit conversions only, else NaN =
                     "given, magnitude unknown"; vasopressors are NaN: MLADI charts amounts, the registry
                     variables are rates). Each vasopressor administration also adds var 200 = NaN.
  low_rate           FiO2 (203, % -> fraction), PEEP (204), ventilator rows (205 = 1)
  infusions_outputs  Urine Output (206, mL); Blood Products/Colloids with a red-cell detail (214, NaN)
Kept: events inside the waveform span ([time_ms[0], time_ms[-1] + 30 s]); seg_idx = the segment at or
before. meta.json += ehr_actions {version, n, per_var, n_outside_window, value_semantics}.
"""
from __future__ import annotations
import argparse, collections, json, os, sys, time, zlib
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import clock  # noqa: E402
import ehr_map as M  # noqa: E402
from common import cfg, factor, num, EHR_EVENT_DTYPE  # noqa: E402

T0 = time.time()
VERSION = "mladi-d2-2"
VASO = set(range(207, 214))
_WAV, _RAW = None, None


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:6.0f}s]", *a, flush=True)


def _init(wav, raw):
    global _WAV, _RAW
    _WAV, _RAW = wav, raw


def one(od):
    import h5py
    eid = os.path.basename(od); mp = os.path.join(od, "meta.json")
    try:
        meta = json.load(open(mp))
        if meta.get("ehr_actions", {}).get("version") == VERSION:
            return {"entity_id": eid, "skipped": True}
        tms = np.load(os.path.join(od, "time_ms.npy"))
        seg = json.load(open(os.path.join(_WAV, eid + "__meta.json")))["seg_list"]
        st0 = float(min(s[2] for s in seg))
        raw = []                                   # (t_s, var, value)
        with h5py.File(os.path.join(_RAW, eid + ".h5"), "r") as f:
            W, label = clock.parse_origin(json.loads(f.attrs[".meta"])["time_origin"])
            grid = clock.Grid(W, st0)
            disch = meta.get("disch_s"); disch = float(disch) if disch not in (None, "None") else None
            if "ehr_clock" not in meta:
                raise RuntimeError("run Stage D first (meta.ehr_clock holds the EHR shift)")
            shift_ms = int(round(float(meta["ehr_clock"]["shift_min"]) * 60000))
            ehr = f.get("ehr")
            if ehr is not None and "medications" in ehr and ehr["medications"].shape[0]:
                d = factor(ehr["medications"]); n = len(d["time"])
                g = lambda c: d.get(c, np.full(n, None, object))
                for cat, ordd, t, route, dose, du, vol, vu in zip(g("catalogDisp"), g("orderedAs"), d["time"], g("route"),
                                                                  g("dose"), g("doseUnit"), g("volumeDose"), g("volumeDoseUnit")):
                    if route is not None and M.NONSYSTEMIC_ROUTES.search(str(route)):
                        continue
                    vid = M.drug_to_var(str(cat)) if cat is not None else None
                    if vid is None and ordd is not None:
                        vid = M.drug_to_var(str(ordd))
                    tt = num(t)
                    if vid is None or not np.isfinite(tt):
                        continue
                    raw.append((tt, vid, M.action_value(vid, dose, du, vol, vu)))
                    if vid in VASO:
                        raw.append((tt, 200, float("nan")))
            if ehr is not None and "low_rate" in ehr and ehr["low_rate"].shape[0]:
                d = factor(ehr["low_rate"])
                for nm, t, v in zip(d["eventName"], d["date"], d["resultVal"]):
                    tt = num(t)
                    if not np.isfinite(tt):
                        continue
                    if nm in M.FIO2_NAMES:
                        x = num(v)
                        if np.isfinite(x):
                            x = x / 100.0 if x > 1.0 else x
                            if 0.21 <= x <= 1.0:
                                raw.append((tt, 203, x))
                    elif nm in M.PEEP_NAMES:
                        x = num(v)
                        if np.isfinite(x) and 0 <= x <= 40:
                            raw.append((tt, 204, x))
                    elif nm in M.VENT_NAMES:
                        raw.append((tt, 205, 1.0))
            if ehr is not None and "infusions_and_outputs" in ehr and ehr["infusions_and_outputs"].shape[0]:
                d = factor(ehr["infusions_and_outputs"]); n = len(d["time"])
                det = d.get("detail", np.full(n, None, object)); un = d.get("unit", np.full(n, None, object))
                for nm, t, vol, dt_, u in zip(d["name"], d["time"], d["volume"], det, un):
                    tt = num(t)
                    if not np.isfinite(tt):
                        continue
                    if nm == "Urine Output":
                        x = num(vol)
                        if np.isfinite(x) and 0 <= x <= 2500 and (u is None or str(u).strip().lower() in ("ml", "<na>", "none")):
                            raw.append((tt, 206, x))
                    elif nm == "Blood Products/Colloids" and dt_ is not None and M.drug_to_var(str(dt_)) == 214:
                        raw.append((tt, 214, float("nan")))
        ev = []
        if raw:
            ts = np.array([r[0] for r in raw], float)
            tg = grid.ehr(ts, W, label, disch) + shift_ms
            lo, hi = int(tms[0]), int(tms[-1]) + 30_000
            si = np.searchsorted(tms, tg, side="right") - 1
            inw = (tg >= lo) & (tg <= hi) & (si >= 0)
            for (t_, vid, val), g_, s_, k in zip(raw, tg, si, inw):
                if k:
                    ev.append((int(g_), int(s_), vid, val))
            n_out = int((~inw).sum())
        else:
            n_out = 0
        arr = np.array(ev, dtype=EHR_EVENT_DTYPE) if ev else np.empty(0, dtype=EHR_EVENT_DTYPE)
        arr.sort(order=["time_ms", "var_id"])
        np.save(os.path.join(od, "ehr_actions.npy"), arr)
        meta["ehr_actions"] = {"version": VERSION, "file": "ehr_actions.npy", "n": int(arr.size),
                               "per_var": {int(u): int((arr["var_id"] == u).sum()) for u in np.unique(arr["var_id"])},
                               "n_outside_window": n_out,
                               "value_semantics": "native dose where the unit converts exactly (insulin units, KCl / bicarbonate mEq, "
                                                  "calcium g, dextrose g, bolus mL, FiO2 fraction, PEEP cmH2O, urine mL, vent 1); "
                                                  "NaN = given, magnitude unknown (vasopressors 207-213 and aggregate 200: MLADI charts "
                                                  "amounts, not rates)"}
        json.dump(meta, open(mp + ".tmp", "w"), indent=1, default=str); os.replace(mp + ".tmp", mp)
        return {"entity_id": eid, "n": int(arr.size)}
    except Exception as ex:
        return {"entity_id": eid, "error": f"{type(ex).__name__}: {ex}"}


def main():
    C = cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=C["output_dir"]); ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--shard", default="0/1")
    ap.add_argument("--raw-h5-dir", default=C["raw_h5_dir"]); ap.add_argument("--pretrain-wav-dir", default=C["pretrain_wav_dir"])
    a = ap.parse_args()
    k, n_sh = (int(x) for x in a.shard.split("/"))
    dirs = sorted(os.path.join(a.out, d) for d in os.listdir(a.out)
                  if zlib.crc32(d.encode()) % n_sh == k and os.path.exists(os.path.join(a.out, d, "time_ms.npy")))[: a.limit or None]
    log(f"{len(dirs)} entities (shard {a.shard})")
    err = done = tot = 0; pv = collections.Counter()
    with Pool(a.workers, initializer=_init, initargs=(a.pretrain_wav_dir, a.raw_h5_dir)) as pool:
        for i, r in enumerate(pool.imap_unordered(one, dirs, chunksize=4), 1):
            if "error" in r:
                err += 1; log(f"ERROR {r['entity_id'][:8]}..: {r['error']}")
            elif not r.get("skipped"):
                done += 1; tot += r["n"]
            if i % 1000 == 0 or i == len(dirs):
                log(f"{i}/{len(dirs)} | written {done} | action events {tot:,} | errors {err}")
    log("done")


if __name__ == "__main__":
    main()
