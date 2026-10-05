#!/usr/bin/env python3
"""MLADI Stage C: dense monitor numerics (vitals_hf.npy, vitals_hf_abp_src.npy) and cuff NBP events
(nbp_events.npy) per entity, after Stage B.

  vitals_hf.npy      float32 [n_seg, 30, n_var], 1-s slots: slot k of segment i covers
                     [time_ms[i] + k s, +1 s); readings placed on the raw DWC clock (the same wall-clock
                     seconds the segment starts are on), last reading wins; NaN = none; values outside the
                     registry physio_min / physio_max are dropped.
  vitals_hf_abp_src  uint8 [n_seg, 30]: 1 = ART line, 2 = ABP line, 0 = none. Per slot the ART line is
                     used when its systolic is there, else ABP; s / d / m / pulse never mix lines.
  nbp_events.npy     EHR_EVENT_DTYPE, var 157 / 158 / 159 (NBPs / d / m _hf): value-change points of the
                     NBP numerics, time on the entity grid (clock.py monitor rule), seg_idx = the segment
                     at or before it; only events inside the waveform span.
  meta.json          + vitals_hf {version, shape, slot_sec, slots_per_seg, var_ids, var_names, valid
                     fractions, abp line codes / slots per line, sources} and nbp_events {...}.

Resumable (meta.vitals_hf.version == VERSION -> skip).

    python workzone/mladi/stage_c_vitals_hf.py --limit 5 --workers 4 [--out <stage B root>]
"""
from __future__ import annotations
import argparse, json, os, sys, time, zlib
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clock  # noqa: E402
from common import cfg, EHR_EVENT_DTYPE  # noqa: E402

T0 = time.time()
VERSION = "mladi-c1"
SLOT_SEC, SLOTS = 1.0, 30
# (var_id, name, ART-line key, ABP-line key or None, plain key)
HF = [(150, "HR_hf", None, None, "HR.HR"), (151, "SpO2_hf", None, None, "SpO₂.SpO₂"), (152, "RR_hf", None, None, "RR.RR"),
      (153, "ABPs_hf", "ART.Systolic", "ABP.ABPs", None), (154, "ABPd_hf", "ART.Diastolic", "ABP.ABPd", None),
      (155, "ABPm_hf", "ART.Mean", "ABP.ABPm", None), (156, "PULSE_hf", None, None, "SpO₂.Pulse"),
      (113, "PR_art", "ART.Pulse", "ABP.Pulse", None), (160, "CVP_hf", None, None, "CVP.CVPm"),
      (164, "PVCrate_hf", None, None, "PVC.PVC"), (165, "PERF_hf", None, None, "Perf.Perf")]
NBP = [(157, "NBP.NBPs"), (158, "NBP.NBPd"), (159, "NBP.NBPm")]
LINE_CODE = {"ART": 1, "ABP": 2}


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:6.0f}s]", *a, flush=True)


def ranges():
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "indices", "var_registry.json")
    return {v["id"]: (v.get("physio_min"), v.get("physio_max")) for v in json.load(open(p))["variables"]}


_RNG, _WAV, _RAW = {}, None, None


def _init(rng, wav, raw):
    global _RNG, _WAV, _RAW
    _RNG, _WAV, _RAW = rng, wav, raw


def one(od):
    import h5py
    eid = os.path.basename(od)
    mp = os.path.join(od, "meta.json")
    try:
        meta = json.load(open(mp))
        if meta.get("vitals_hf", {}).get("version") == VERSION:
            return {"entity_id": eid, "skipped": True}
        if not meta.get("stage_b", {}).get("done"):
            return {"entity_id": eid, "error": "stage B not done"}
        seg = json.load(open(os.path.join(_WAV, eid + "__meta.json")))["seg_list"]
        st = np.sort(np.array([s[2] for s in seg], float))
        n = st.size
        tms = np.load(os.path.join(od, "time_ms.npy"))
        assert tms.size == n, "time_ms / seg_list length mismatch"
        V = np.full((n * SLOTS, len(HF)), np.nan, np.float32)
        src = np.zeros(n * SLOTS, np.uint8)
        used, line_slots = {}, {"ART": 0, "ABP": 0}

        def slots_of(t):
            i = np.searchsorted(st, t, side="right") - 1
            k = np.floor((t - st[np.clip(i, 0, None)]) / SLOT_SEC).astype(np.int64)
            ok = (i >= 0) & (k >= 0) & (k < SLOTS)
            return np.where(ok, i * SLOTS + k, -1)

        with h5py.File(os.path.join(_RAW, eid + ".h5"), "r") as f:
            W, label = clock.parse_origin(json.loads(f.attrs[".meta"])["time_origin"])
            nm = f.get("data/numerics")

            def load(key, vid):
                if nm is None or key not in nm:
                    return None
                a = nm[key][:]
                t, v = a["time"].astype(np.float64), a["value"].astype(np.float64)
                lo, hi = _RNG.get(vid, (None, None))
                ok = np.isfinite(t) & np.isfinite(v)
                if lo is not None:
                    ok &= v >= lo
                if hi is not None:
                    ok &= v <= hi
                return t[ok], v[ok]

            # plain variables
            for j, (vid, name, ka, kb, kp) in enumerate(HF):
                if kp is None:
                    continue
                r = load(kp, vid)
                if r is None:
                    continue
                s = slots_of(r[0]); m = s >= 0
                V[s[m], j] = r[1][m]
                used[name] = [kp]
            # arterial lines: the slot's line is decided by its systolic
            jS = [k for k, h in enumerate(HF) if h[0] == 153][0]
            for line, idx in (("ART", 2), ("ABP", 3)):
                sysk = HF[jS][idx]
                rs = load(sysk, 153)
                if rs is None:
                    continue
                s = slots_of(rs[0]); m = (s >= 0)
                free = np.zeros(n * SLOTS, bool); free[s[m]] = True
                free &= (src == 0)
                if not free.any():
                    continue
                src[free] = LINE_CODE[line]; line_slots[line] += int(free.sum())
                for j, (vid, name, ka, kb, kp) in enumerate(HF):
                    key = ka if line == "ART" else kb
                    if key is None or kp is not None:
                        continue
                    r = load(key, vid)
                    if r is None:
                        continue
                    s2 = slots_of(r[0]); m2 = (s2 >= 0)
                    m2[m2] = src[s2[m2]] == LINE_CODE[line]
                    V[s2[m2], j] = r[1][m2]
                    used.setdefault(name, []).append(key)
            # NBP events on the grid
            grid = clock.Grid(W, st[0])
            ev = []
            for vid, key in NBP:
                r = load(key, vid)
                if r is None:
                    continue
                o = np.argsort(r[0], kind="stable"); t, v = r[0][o], r[1][o]
                chg = np.r_[True, np.diff(v) != 0]
                te = grid.dwc(t[chg], W); ve = v[chg]
                si = np.searchsorted(tms, te, side="right") - 1
                inw = (si >= 0) & (te < tms[-1] + 30_000)
                for a, b, c in zip(te[inw], si[inw], ve[inw]):
                    ev.append((int(a), int(b), vid, float(c)))
        evs = np.array(ev, dtype=EHR_EVENT_DTYPE) if ev else np.empty(0, dtype=EHR_EVENT_DTYPE)
        evs.sort(order=["time_ms", "var_id"])
        np.save(os.path.join(od, "vitals_hf.npy"), np.ascontiguousarray(V.reshape(n, SLOTS, len(HF))))
        np.save(os.path.join(od, "vitals_hf_abp_src.npy"), np.ascontiguousarray(src.reshape(n, SLOTS)))
        np.save(os.path.join(od, "nbp_events.npy"), evs)
        valid = np.isfinite(V).sum(0)
        meta["vitals_hf"] = {
            "version": VERSION, "file": "vitals_hf.npy", "dtype": "float32", "time_base": "utc_continuous",
            "shape": [n, SLOTS, len(HF)], "slot_sec": SLOT_SEC, "slots_per_seg": SLOTS,
            "var_ids": [h[0] for h in HF], "var_names": [h[1] for h in HF],
            "valid_frac_per_var": {h[1]: round(float(c) / (n * SLOTS), 4) for h, c in zip(HF, valid)},
            "abp_src_file": "vitals_hf_abp_src.npy", "abp_line_codes": LINE_CODE, "abp_slots_per_line": line_slots,
            "sources": used, "source": "DWC data/numerics (1.024 s), placed on the raw wall-clock seconds of the segments"}
        meta["nbp_events"] = {"file": "nbp_events.npy", "dtype": "EHR_EVENT_DTYPE", "var_ids": [v[0] for v in NBP],
                              "n_events": int(evs.size), "rule": "value-change points of NBP.NBPs/d/m; time on the entity grid"}
        json.dump(meta, open(mp + ".tmp", "w"), indent=1, default=str); os.replace(mp + ".tmp", mp)
        return {"entity_id": eid, "hr_valid": float(valid[0]) / (n * SLOTS), "abp_slots": int(sum(line_slots.values())), "n_nbp": int(evs.size)}
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
                  if os.path.isdir(os.path.join(a.out, d)) and zlib.crc32(d.encode()) % n_sh == k
                  and os.path.exists(os.path.join(a.out, d, "meta.json")))[: a.limit or None]
    log(f"{len(dirs)} entities (shard {a.shard}) under {a.out}")
    err = done = 0; hr = []
    with Pool(a.workers, initializer=_init, initargs=(ranges(), a.pretrain_wav_dir, a.raw_h5_dir)) as pool:
        for i, r in enumerate(pool.imap_unordered(one, dirs, chunksize=4), 1):
            if "error" in r:
                err += 1; log(f"ERROR {r['entity_id'][:8]}..: {r['error']}")
            elif not r.get("skipped"):
                done += 1; hr.append(r["hr_valid"])
            if i % 500 == 0 or i == len(dirs):
                log(f"{i}/{len(dirs)} | written {done} | errors {err} | HR_hf slot coverage p50 {np.median(hr) if hr else float('nan'):.3f}")
    log("done")


if __name__ == "__main__":
    main()
