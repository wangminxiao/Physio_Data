#!/usr/bin/env python3
"""MLADI Stage C4: PLETH40_aligned.npy, the Pleth with the DEVICE part of its delay removed (decided with the user,
2026-10-06: the model should learn ECG<->PPG timing from the waveforms, so the input keeps physiological PAT and
loses the device parts; all MLADI monitors are Philips, so one device constant).

Device delay at stamped time s (grid ms): delta(s) = D_device + saw(s - D_device), where
  D_device = D_global (pleth_timing_summary.json, cohort median of R -> Pleth-foot levels) - R_ART_MS, R_ART_MS
             = 184 ms the median R -> radial-artery-foot delay (Physio_HNET scripts/dev/mladi_art_decomp.py, 164
             encounters): the arterial line has no device delay, so the remaining delay is ~ R -> ART plus the
             radial -> finger transit, i.e. physiology;
  saw        = the run's IntelliVue re-sync sawtooth from Stage C3 (applied only where C3's fit was plausible),
             with each reset moved from C3's 2-s bins to between the two beats where the per-beat delay drops.
Each stamped sample gets a content time c = s - delta(s), increasing (at a reset it jumps ~29 ms ahead: the
monitor dropped those samples); the aligned trace is the cubic spline through (c, value) evaluated on the grid,
per contiguous run and per finite stretch (NaN stays NaN; the last ~D_device of a run has no content -> NaN).
PLETH40 itself is not touched. pleth_aligned_resets.npy keeps the refined resets (layout of pleth_resets.npy), so
readers can map raw-PLETH40 positions (e.g. fiducial labels) onto the aligned trace with exactly this delta.

    python workzone/mladi/stage_c4_pleth_aligned.py --limit 5 --workers 4 [--shard i/n]
"""
from __future__ import annotations
import argparse, json, os, sys, time, zlib
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pleth_timing_lib as L  # noqa: E402
import stage_c3_pleth_timing as C3  # noqa: E402
from common import cfg  # noqa: E402

T0 = time.time()
VERSION = "mladi-c4-1"
R_ART_MS = 184.0
FS = 40
REFINE_HALF_S = 3.0          # look for the drop within +-3 s of C3's reset
REFINE_K = 4                 # beats averaged on each side of a candidate split
_D = None


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:6.0f}s]", *a, flush=True)


def refine_resets(t, d, resets, ramp, period):
    """Move each reset to the midpoint between the two beats where the de-ramped delay drops most (two-level
    split, REFINE_K beats a side), if that drop is 0.5-1.5x the expected ramp * period; else keep it."""
    if t.size < 2 * REFINE_K + 1 or not ramp:
        return np.asarray(resets, float), 0, len(resets)
    e = d - ramp * t
    exp_jump = ramp * period
    out, moved = [], 0
    for r in resets:
        j0, j1 = np.searchsorted(t, r - REFINE_HALF_S), np.searchsorted(t, r + REFINE_HALF_S)
        best, bj = 0.0, None
        for j in range(max(j0, REFINE_K), min(j1, t.size - REFINE_K) + 1):
            drop = np.median(e[j - REFINE_K:j]) - np.median(e[j:j + REFINE_K])
            if drop > best:
                best, bj = drop, j
        if bj is not None and 0.5 * exp_jump <= best <= 1.5 * exp_jump:
            out.append(0.5 * (t[bj - 1] + t[bj])); moved += 1
        else:
            out.append(float(r))
    return np.array(out), moved, len(resets)


def warp_run(P, tms, a, b, delta_fn):
    """Aligned Pleth for segments [a, b) of one contiguous run: content time c = s - delta(s) per sample, cubic
    spline through (c, v) per finite stretch, evaluated at the grid times. Returns float32 [b - a, 1200]."""
    from scipy.interpolate import CubicSpline
    v = np.asarray(P[a:b], np.float32).reshape(-1).astype(np.float64)
    s = (tms[a] / 1000.0) + np.arange(v.size) / FS
    c = s - delta_fn(s)
    out = np.full(v.size, np.nan)
    ok = np.isfinite(v)
    if ok.sum() < 8:
        return out.reshape(b - a, -1).astype(np.float32)
    idx = np.flatnonzero(ok)
    cut = np.r_[0, np.flatnonzero(np.diff(idx) > 1) + 1, idx.size]
    for k in range(cut.size - 1):
        seg = idx[cut[k]:cut[k + 1]]
        if seg.size < 8:
            continue
        cs = CubicSpline(c[seg], v[seg])
        lo, hi = np.searchsorted(s, c[seg[0]]), np.searchsorted(s, c[seg[-1]], side="right")
        if hi > lo:
            out[lo:hi] = cs(s[lo:hi])
    return out.reshape(b - a, -1).astype(np.float32)


def one(od):
    eid = os.path.basename(od); mp = os.path.join(od, "meta.json")
    try:
        meta = json.load(open(mp))
        if meta.get("pleth_aligned", {}).get("version") == VERSION:
            return {"entity_id": eid, "skipped": True}
        if "pleth_timing" not in meta:
            return {"entity_id": eid, "error": "no Stage C3 output"}
        tms = np.load(os.path.join(od, "time_ms.npy")); n = tms.size
        P = np.load(os.path.join(od, "PLETH40.npy"), mmap_mode="r"); E = np.load(os.path.join(od, "II120.npy"), mmap_mode="r")
        rt = np.load(os.path.join(od, "pleth_timing_runs.npy")); rs_all = np.load(os.path.join(od, "pleth_resets.npy"))
        D = _D / 1000.0
        out = np.full((n, 1200), np.nan, np.float32)
        refined_all = rs_all.astype(np.int64).copy()            # same layout as pleth_resets.npy (runs' reset0 / n_resets)
        n_ok_runs = n_res = n_moved = 0; segs_saw = 0
        prior = (meta["pleth_timing"].get("initial_offset_ms") or 1200.0) / 1000.0
        for r in rt:
            a, b = int(r["seg0"]), int(r["seg1"])
            resets = rs_all[r["reset0"]: r["reset0"] + r["n_resets"]] / 1000.0
            ramp = float(r["ramp_ms_per_min"]) / 60000.0 if np.isfinite(r["ramp_ms_per_min"]) else 0.0
            period = float(r["period_s"])
            if r["ok"] and resets.size:
                t, d, prior = C3.paired_delays(E, P, tms, a, b, prior)
                resets, moved, nr = refine_resets(t, d, resets, ramp, period)
                refined_all[r["reset0"]: r["reset0"] + r["n_resets"]] = np.round(resets * 1000.0).astype(np.int64)
                n_ok_runs += 1; n_res += nr; n_moved += moved; segs_saw += b - a
                delta_fn = (lambda s, rs=resets, ra=ramp, pe=period: D + L.eval_sawtooth(s - D, rs, ra, pe))
            else:
                delta_fn = (lambda s: np.full_like(s, D))
            out[a:b] = warp_run(P, tms, a, b, delta_fn)
        np.save(os.path.join(od, "pleth_aligned_resets.npy"), refined_all)
        tmp = os.path.join(od, "PLETH40_aligned.tmp.npy")
        np.save(tmp, out.astype(np.float16)); os.replace(tmp, os.path.join(od, "PLETH40_aligned.npy"))
        nan_raw = float(np.isnan(np.asarray(P, np.float32)).mean()) if n else 0.0
        meta["pleth_aligned"] = {
            "version": VERSION, "file": "PLETH40_aligned.npy", "resets_file": "pleth_aligned_resets.npy",
            "D_device_ms": round(_D, 2), "R_ART_ms": R_ART_MS,
            "delay_model": "delta(s) = D_device + (run ok and n_resets > 0: eval_sawtooth(s - D_device, that run's slice of "
                           "pleth_aligned_resets.npy, ramp_ms_per_min / 60000, period_s) else 0), s in grid seconds; a raw "
                           "sample stamped s sits at s - delta(s) in PLETH40_aligned",
            "segments_with_sawtooth_removed_frac": segs_saw / n if n else 0.0, "n_runs_sawtooth": n_ok_runs,
            "n_resets": int(n_res), "n_resets_refined": int(n_moved),
            "nan_frac_raw": nan_raw, "nan_frac_aligned": float(np.isnan(out).mean()) if n else 0.0,
            "method": "content time c = s - (D_device + saw(s - D_device)); cubic spline through (c, PLETH40) per run and "
                      "finite stretch, on the same grid; PLETH40 untouched",
        }
        json.dump(meta, open(mp + ".tmp", "w"), indent=1, default=str); os.replace(mp + ".tmp", mp)
        return {"entity_id": eid, "refined": n_moved, "resets": n_res}
    except Exception as ex:
        import traceback
        return {"entity_id": eid, "error": f"{type(ex).__name__}: {ex} | {traceback.format_exc()[-300:]}"}


def _init(D):
    global _D
    _D = D


def main():
    C = cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=C["output_dir"]); ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--shard", default="0/1")
    ap.add_argument("--d-global-ms", type=float, default=None, help="override pleth_timing_summary.json (tests)")
    a = ap.parse_args()
    Dg = a.d_global_ms if a.d_global_ms is not None else json.load(open(os.path.join(a.out, "pleth_timing_summary.json")))["D_global_ms"]
    Ddev = float(Dg) - R_ART_MS
    k, n_sh = (int(x) for x in a.shard.split("/"))
    dirs = sorted(os.path.join(a.out, d) for d in os.listdir(a.out)
                  if zlib.crc32(d.encode()) % n_sh == k and os.path.exists(os.path.join(a.out, d, "pleth_timing_runs.npy")))[: a.limit or None]
    log(f"{len(dirs)} entities (shard {a.shard}) | D_global {Dg:.1f} ms -> D_device {Ddev:.1f} ms")
    err = done = 0; ref = res = 0
    with Pool(a.workers, initializer=_init, initargs=(Ddev,)) as pool:
        for i, r in enumerate(pool.imap_unordered(one, dirs, chunksize=1), 1):
            if "error" in r:
                err += 1; log(f"ERROR {r['entity_id'][:8]}..: {r['error']}")
            elif not r.get("skipped"):
                done += 1; ref += r["refined"]; res += r["resets"]
            if i % 200 == 0 or i == len(dirs):
                log(f"{i}/{len(dirs)} | written {done} | resets refined {ref}/{res} | errors {err}")
    log("done")


if __name__ == "__main__":
    main()
