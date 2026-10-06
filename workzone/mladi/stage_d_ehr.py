#!/usr/bin/env python3
"""MLADI Stage D: EHR events (labs, charted vitals) -> ehr_baseline / recent / events / future per entity.

Times: clock.py EHR rule (origin rendered in the DST state at discharge; LMT origins as UTC) plus Stage A's
measured correction (`ehr_extra_shift_min`), mapped onto the entity grid (`Grid.from_utc`), then split by
physio_data.ehr_trajectory.split_events with the admission (regDate) and discharge (dischDate) as episode
bounds. Values numeric only; labs unit-checked (ehr_map.unit_ok), temperatures to deg C, registry
physio_min / physio_max; exact duplicates removed. Entities without EHR get four empty files.

meta.json += ehr {version, n_baseline, n_recent, n_events, n_future, per-var counts in events, dropped
counts (non-numeric / unit / range), clock_confidence, ehr_extra_shift_min applied}.

    python workzone/mladi/stage_d_ehr.py --limit 5 --workers 4 [--out <root>]
"""
from __future__ import annotations
import argparse, collections, json, os, sys, time, zlib
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import clock  # noqa: E402
import ehr_map as M  # noqa: E402
from common import cfg, factor, num, EHR_EVENT_DTYPE  # noqa: E402
from physio_data.ehr_trajectory import split_events, ALL_FNAMES  # noqa: E402

T0 = time.time()
VERSION = "mladi-d1"
_RNG, _WAV, _RAW = {}, None, None


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:6.0f}s]", *a, flush=True)


def ranges():
    p = os.path.join(os.path.dirname(os.path.dirname(HERE)), "indices", "var_registry.json")
    return {v["id"]: (v.get("physio_min"), v.get("physio_max")) for v in json.load(open(p))["variables"]}


def _init(rng, wav, raw):
    global _RNG, _WAV, _RAW
    _RNG, _WAV, _RAW = rng, wav, raw


def in_range(vid, x):
    lo, hi = _RNG.get(vid, (None, None))
    return np.isfinite(x) and (lo is None or x >= lo) and (hi is None or x <= hi)


def one(od):
    import h5py
    eid = os.path.basename(od); mp = os.path.join(od, "meta.json")
    try:
        meta = json.load(open(mp))
        if meta.get("ehr", {}).get("version") == VERSION:
            return {"entity_id": eid, "skipped": True}
        tms = np.load(os.path.join(od, "time_ms.npy"))
        seg = json.load(open(os.path.join(_WAV, eid + "__meta.json")))["seg_list"]
        st0 = float(min(s[2] for s in seg))
        drops = collections.Counter(); ev = []
        reg_ms = disch_ms = None
        with h5py.File(os.path.join(_RAW, eid + ".h5"), "r") as f:
            W, label = clock.parse_origin(json.loads(f.attrs[".meta"])["time_origin"])
            grid = clock.Grid(W, st0)
            disch = meta.get("disch_s"); disch = float(disch) if disch not in (None, "None") else None
            shift_ms = int(round(float(meta.get("ehr_extra_shift_min") or 0) * 60000))

            def to_grid(t_s):
                t = np.atleast_1d(np.asarray(t_s, float))
                return grid.ehr(t, W, label, disch) + shift_ms

            ehr = f.get("ehr")
            if ehr is not None and "demographic" in ehr:
                dm = ehr["demographic"][0]
                for c, tgt in (("regDate", "reg"), ("dischDate", "disch")):
                    if c in dm.dtype.names and np.isfinite(float(dm[c])):
                        v_ = int(to_grid(float(dm[c]))[0])
                        if tgt == "reg":
                            reg_ms = v_
                        else:
                            disch_ms = v_
            if ehr is not None and "lab_results" in ehr and ehr["lab_results"].shape[0]:
                d = factor(ehr["lab_results"]); names = d["eventDisp"]
                units = d.get("resultUnit", np.full(len(names), None, object))
                for nm, t, v, u in zip(names, d["time"], d["resultVal"], units):
                    vid = M.LAB_NAME.get(nm)
                    if vid is None:
                        continue
                    x, tt = num(v), num(t)
                    if not (np.isfinite(x) and np.isfinite(tt)):
                        drops[f"{vid}:nonnumeric"] += 1; continue
                    if not M.unit_ok(vid, u):
                        drops[f"{vid}:unit"] += 1; continue
                    if not in_range(vid, x):
                        drops[f"{vid}:range"] += 1; continue
                    ev.append((tt, vid, x))
            if ehr is not None and "low_rate" in ehr and ehr["low_rate"].shape[0]:
                d = factor(ehr["low_rate"]); names = d["eventName"]
                units = d.get("resultUnit", np.full(len(names), None, object))
                for nm, t, v, u in zip(names, d["date"], d["resultVal"], units):
                    vid = M.VITAL_NAME.get(nm)
                    if vid is None:
                        continue
                    x, tt = num(v), num(t)
                    if not (np.isfinite(x) and np.isfinite(tt)):
                        drops[f"{vid}:nonnumeric"] += 1; continue
                    if vid == 103:
                        x = M.temperature_c(nm, x, u)
                    if vid == 107 and u is not None and "h2o" in str(u).lower():
                        x = x * 0.7356                       # cmH2O -> mmHg
                    if not in_range(vid, x):
                        drops[f"{vid}:range"] += 1; continue
                    ev.append((tt, vid, x))
        if ev:
            ts = np.array([e[0] for e in ev], float)
            arr = np.zeros(len(ev), dtype=EHR_EVENT_DTYPE)
            arr["time_ms"] = to_grid(ts); arr["var_id"] = [e[1] for e in ev]; arr["value"] = [e[2] for e in ev]
            from numpy.lib import recfunctions as rfn
            key = np.unique(rfn.repack_fields(arr[["time_ms", "var_id", "value"]]), return_index=True)[1]
            arr = arr[np.sort(key)]
        else:
            arr = np.empty(0, dtype=EHR_EVENT_DTYPE)
        parts = split_events(arr, tms, episode_start_ms=reg_ms, episode_end_ms=disch_ms, wave_end_pad_ms=30_000)
        for (k, fn) in zip(("baseline", "recent", "events", "future"), ALL_FNAMES):
            np.save(os.path.join(od, fn), parts[k])
        evv = parts["events"]
        meta["ehr"] = {"version": VERSION, "n_baseline": int(parts["baseline"].size), "n_recent": int(parts["recent"].size),
                       "n_events": int(evv.size), "n_future": int(parts["future"].size),
                       "events_per_var": {int(u): int((evv["var_id"] == u).sum()) for u in np.unique(evv["var_id"])},
                       "dropped": dict(drops), "episode_start_ms": reg_ms, "episode_end_ms": disch_ms,
                       "time_rule": "clock.py EHR rule + ehr_extra_shift_min, grid-mapped",
                       "ehr_extra_shift_min_applied": meta.get("ehr_extra_shift_min") or 0,
                       "mapping": "workzone/mladi/ehr_map.py (labs 0-18, charted vitals 100-117)"}
        json.dump(meta, open(mp + ".tmp", "w"), indent=1, default=str); os.replace(mp + ".tmp", mp)
        return {"entity_id": eid, "n": int(arr.size), "n_events": int(evv.size)}
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
    err = done = 0; tot = 0
    with Pool(a.workers, initializer=_init, initargs=(ranges(), a.pretrain_wav_dir, a.raw_h5_dir)) as pool:
        for i, r in enumerate(pool.imap_unordered(one, dirs, chunksize=4), 1):
            if "error" in r:
                err += 1; log(f"ERROR {r['entity_id'][:8]}..: {r['error']}")
            elif not r.get("skipped"):
                done += 1; tot += r["n"]
            if i % 1000 == 0 or i == len(dirs):
                log(f"{i}/{len(dirs)} | written {done} | events {tot:,} | errors {err}")
    log("done")


if __name__ == "__main__":
    main()
