#!/usr/bin/env python3
"""MLADI Stage C3: ECG -> Pleth timing per entity (API.md known issue 6), from II120 / PLETH40 of the store.

Per contiguous run of the grid (segments 30 s apart), in 10-min chunks: R and Pleth foot detected
(pleth_timing_lib), beats paired through the chunk's offset (prior = previous chunk, start 1.2 s), then the
re-sync sawtooth fitted on the run's paired delays. Outputs (nothing else in the entity changes):

  pleth_timing.npy       [n_seg]  measured_ms f4  median R -> foot delay of the beats whose R lies in the segment
                                                  (NaN when < 3 paired beats)
                                  n_beats     u1  paired beats in the segment
                                  saw_ms      f4  sawtooth at the segment centre (0 where the run's fit failed)
                                  run         u2  index into pleth_timing_runs.npy
  pleth_timing_runs.npy  [n_runs] seg0, seg1 (i4, segments [seg0, seg1)), ok (u1), ramp_ms_per_min, period_s,
                                  jump_ms, level_ms (median measured - saw), n_beats (i4), reset0, n_resets (i4)
  pleth_resets.npy       int64 grid ms of the detected resets, runs concatenated (run r owns
                                  [reset0, reset0 + n_resets))
  meta.json              + pleth_timing {version, coverage, levels, sd before / after the sawtooth, ...}

Device part of the delay at time t (ms): D_global + saw(t), D_global in <store>/pleth_timing_summary.json
(verify_stage_c3.py); saw(t) = eval_sawtooth(t, that run's resets, ramp, period). measured - device = the
physiological part (PAT relative to the cohort level).

    python workzone/mladi/stage_c3_pleth_timing.py --limit 5 --workers 4 [--shard i/n]
"""
from __future__ import annotations
import argparse, json, os, sys, time, zlib
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pleth_timing_lib as L  # noqa: E402
from common import cfg  # noqa: E402

T0 = time.time()
VERSION = "mladi-c3-1"
FS_E, FS_P = 120, 40
CHUNK_SEG = 20                      # 10 min
PRIOR_S = 1.2
TOL_S = 0.12
D_RANGE = (0.6, 2.2)
SEG_DTYPE = np.dtype([("measured_ms", "f4"), ("n_beats", "u1"), ("saw_ms", "f4"), ("run", "u2")])
RUN_DTYPE = np.dtype([("seg0", "i4"), ("seg1", "i4"), ("ok", "u1"), ("ramp_ms_per_min", "f4"), ("period_s", "f4"),
                      ("jump_ms", "f4"), ("level_ms", "f4"), ("n_beats", "i4"), ("reset0", "i4"), ("n_resets", "i4")])


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:6.0f}s]", *a, flush=True)


def runs_of(tms):
    cut = np.flatnonzero(np.diff(tms) != 30000) + 1
    a = np.r_[0, cut]; b = np.r_[cut, tms.size]
    return list(zip(a.tolist(), b.tolist()))


def paired_delays(E, P, tms, a, b, prior):
    """(t_R s on the grid, delay s) over segments [a, b), chunked; returns arrays and the last prior."""
    T, D = [], []
    for c0 in range(a, b, CHUNK_SEG):
        c1 = min(b, c0 + CHUNK_SEG)
        if c1 - c0 < 2:
            continue
        t0 = tms[c0] / 1000.0
        r = L.ecg_beats(np.asarray(E[c0:c1], np.float32).reshape(-1), FS_E)
        f = L.ppg_beats(np.asarray(P[c0:c1], np.float32).reshape(-1), FS_P)
        if r is None or f is None or r.size < 20 or f.size < 20:
            continue
        est = L.estimate_offset(r, f, prior=prior)
        if est is None:
            continue
        off, amb, _ = est
        if not (D_RANGE[0] <= off <= D_RANGE[1]):
            continue
        idx, _ = L.pair_nearest(r, f, off, tol=TOL_S)
        ok = idx >= 0
        d = f[idx[ok]] - r[ok]
        keep = (d >= D_RANGE[0]) & (d <= D_RANGE[1])
        T.append(t0 + r[ok][keep]); D.append(d[keep])
        if amb < 0.8:
            prior = off
    if not T:
        return np.zeros(0), np.zeros(0), prior
    return np.concatenate(T), np.concatenate(D), prior


def one(od):
    eid = os.path.basename(od); mp = os.path.join(od, "meta.json")
    try:
        meta = json.load(open(mp))
        if meta.get("pleth_timing", {}).get("version") == VERSION:
            return {"entity_id": eid, "skipped": True}
        tms = np.load(os.path.join(od, "time_ms.npy")); n = tms.size
        E = np.load(os.path.join(od, "II120.npy"), mmap_mode="r"); P = np.load(os.path.join(od, "PLETH40.npy"), mmap_mode="r")
        seg = np.zeros(n, SEG_DTYPE); seg["measured_ms"] = np.nan
        runs = runs_of(tms); rt = np.zeros(len(runs), RUN_DTYPE); resets_all = []
        prior = PRIOR_S; sd_b, sd_a, n_beats_tot = [], [], 0
        for k, (a, b) in enumerate(runs):
            seg["run"][a:b] = k
            t, d, prior = paired_delays(E, P, tms, a, b, prior)
            rt[k]["seg0"], rt[k]["seg1"] = a, b
            rt[k]["reset0"] = sum(len(x) for x in resets_all)
            n_beats_tot += t.size; rt[k]["n_beats"] = t.size
            if t.size == 0:
                rt[k]["ramp_ms_per_min"] = rt[k]["period_s"] = rt[k]["jump_ms"] = rt[k]["level_ms"] = np.nan
                resets_all.append(np.zeros(0, np.int64)); continue
            fit = L.fit_sawtooth(t, d)
            centres = tms[a:b] / 1000.0 + 15.0
            saw_c = L.eval_sawtooth(centres, fit["resets"], fit["ramp"], fit["period"]) if fit["ok"] else np.zeros(b - a)
            saw_b = L.eval_sawtooth(t, fit["resets"], fit["ramp"], fit["period"]) if fit["ok"] else np.zeros(t.size)
            si = np.clip(np.searchsorted(tms[a:b] / 1000.0, t, side="right") - 1, 0, b - a - 1)
            cnt = np.bincount(si, minlength=b - a)
            seg["n_beats"][a:b] = np.minimum(cnt, 255)
            med = np.full(b - a, np.nan)
            o = np.argsort(si, kind="stable"); ss = si[o]; dd = d[o]
            cut = np.r_[0, np.flatnonzero(np.diff(ss)) + 1, ss.size]
            for i in range(cut.size - 1):
                if cut[i + 1] - cut[i] >= 3:
                    med[ss[cut[i]]] = np.median(dd[cut[i]:cut[i + 1]])
            seg["measured_ms"][a:b] = med * 1000.0
            seg["saw_ms"][a:b] = saw_c * 1000.0
            rs = (np.asarray(fit["resets"]) * 1000.0).round().astype(np.int64) if fit["ok"] else np.zeros(0, np.int64)
            resets_all.append(rs)
            rt[k]["ok"] = int(fit["ok"]); rt[k]["n_resets"] = rs.size
            rt[k]["ramp_ms_per_min"] = fit["ramp"] * 60000.0 if fit["ok"] else np.nan
            rt[k]["period_s"] = fit["period"]; rt[k]["jump_ms"] = fit["jump"] * 1000.0 if np.isfinite(fit["jump"]) else np.nan
            rt[k]["level_ms"] = float(np.median(d - saw_b)) * 1000.0
            if fit["ok"] and t.size >= 200:
                # spread of 2-s bin medians before / after removing the sawtooth, detrended by 10-min medians
                for arr, store in ((d, sd_b), (d - saw_b, sd_a)):
                    _, m = L._bin_medians(t, arr, 2.0)
                    _, m10 = L._bin_medians(t, arr, 600.0)
                    j = np.minimum((np.arange(m.size) * 2.0 / 600.0).astype(int), m10.size - 1)
                    r_ = m - m10[j]
                    store.append(float(np.nanstd(r_)) * 1000.0)
        np.save(os.path.join(od, "pleth_timing.npy"), seg)
        np.save(os.path.join(od, "pleth_timing_runs.npy"), rt)
        np.save(os.path.join(od, "pleth_resets.npy"), np.concatenate(resets_all) if resets_all else np.zeros(0, np.int64))
        okr = rt["ok"] == 1
        segs_ok = int(sum(r["seg1"] - r["seg0"] for r in rt[okr]))
        meas = seg["measured_ms"]
        meta["pleth_timing"] = {
            "version": VERSION, "files": ["pleth_timing.npy", "pleth_timing_runs.npy", "pleth_resets.npy"],
            "n_paired_beats": int(n_beats_tot), "seg_with_delay_frac": float(np.isfinite(meas).mean()) if n else 0.0,
            "seg_in_fitted_run_frac": segs_ok / n if n else 0.0, "n_runs": len(runs), "n_runs_fitted": int(okr.sum()),
            "measured_median_ms": float(np.nanmedian(meas)) if np.isfinite(meas).any() else None,
            "level_median_ms": float(np.nanmedian(meas - seg["saw_ms"])) if np.isfinite(meas).any() else None,
            "ramp_ms_per_min_median": float(np.nanmedian(rt["ramp_ms_per_min"][okr])) if okr.any() else None,
            "period_s_median": float(np.nanmedian(rt["period_s"][okr])) if okr.any() else None,
            "jump_ms_median": float(np.nanmedian(rt["jump_ms"][okr])) if okr.any() else None,
            "sd_2s_before_ms": float(np.median(sd_b)) if sd_b else None, "sd_2s_after_ms": float(np.median(sd_a)) if sd_a else None,
            "detector": "R: 5-20 Hz energy peak -> signed extreme of 0.5-40 Hz within 40 ms (parabola); foot: intersecting "
                        "tangent at the steepest upstroke (10 Hz low-pass); pairing: sharpest R-to-foot difference peak "
                        "per 10 min (prior: previous chunk, start 1.2 s), nearest within 120 ms",
            "device_part": "D_global (pleth_timing_summary.json) + saw(t)",
        }
        json.dump(meta, open(mp + ".tmp", "w"), indent=1, default=str); os.replace(mp + ".tmp", mp)
        return {"entity_id": eid, "beats": int(n_beats_tot), "cov": meta["pleth_timing"]["seg_with_delay_frac"]}
    except Exception as ex:
        import traceback
        return {"entity_id": eid, "error": f"{type(ex).__name__}: {ex} | {traceback.format_exc()[-300:]}"}


def main():
    C = cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=C["output_dir"]); ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    k, n_sh = (int(x) for x in a.shard.split("/"))
    dirs = sorted(os.path.join(a.out, d) for d in os.listdir(a.out)
                  if zlib.crc32(d.encode()) % n_sh == k and os.path.exists(os.path.join(a.out, d, "PLETH40.npy")))[: a.limit or None]
    log(f"{len(dirs)} entities (shard {a.shard})")
    err = done = 0; covs = []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(one, dirs, chunksize=1), 1):
            if "error" in r:
                err += 1; log(f"ERROR {r['entity_id'][:8]}..: {r['error']}")
            elif not r.get("skipped"):
                done += 1; covs.append(r["cov"])
            if i % 200 == 0 or i == len(dirs):
                log(f"{i}/{len(dirs)} | written {done} | seg-with-delay p50 {np.median(covs) if covs else float('nan'):.3f} | errors {err}")
    log("done")


if __name__ == "__main__":
    main()
