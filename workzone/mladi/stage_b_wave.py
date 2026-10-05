#!/usr/bin/env python3
"""MLADI Stage B: PLETH40 / II120 / time_ms / meta.json per entity, from the raw DWC HDF5.

Grid = the entity's pretrain_wav_v2 rows (`__meta.json` seg_list, sorted by start): canonical segment
i == mmap row i, 30 s, no overlap. Each run of consecutive rows (same block, 30-s steps) is processed in
pieces of <= PIECE_ROWS rows with CONTEXT_S seconds of extra signal on each side (cut afterwards), so
resample_poly's edges never touch a kept sample and a worker never holds a whole multi-day block:

    wall-clock seconds t (raw DWC clock) -> grid start + n / fs_src   (the block's own sample grid)
    np.interp from the raw (time, value) samples; samples on an invalid code (non-finite or
    |v| > 1e3) and the stretches between them stay NaN (never bridged)
    scipy.signal.resample_poly(fs_src -> 40 / 120); NO band-pass (canonical = raw)

II absent -> II120 all NaN; II at 250 Hz handled by its own samplePeriod. time_ms = the UTC-continuous
grid of workzone/mladi/clock.py (wall clock of the first segment + real elapsed ms; the monitor's DWC
wall-clock rule). meta.json carries the Stage A identity and clock fields plus what this stage measured.

Resumable: an entity whose meta.json has stage_b.done is skipped. Entities come from Stage A's
inventory (included == true).

    python workzone/mladi/stage_b_wave.py --limit 5 --workers 4 --out <scratch>
    python workzone/mladi/stage_b_wave.py --workers 32 [--shard i/n]
"""
from __future__ import annotations
import argparse, json, os, sys, time, zlib
from fractions import Fraction
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clock  # noqa: E402
from common import cfg  # noqa: E402

T0 = time.time()
CH = {"PLETH40": ("Pleth", 40, 1200), "II120": ("II", 120, 3600)}
PIECE_ROWS = 60            # 30 min per piece
CONTEXT_S = 10.0
INVALID_ABS = 1e3


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:6.0f}s]", *a, flush=True)


def _find(ds, t, lo=0):
    hi = ds.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if ds[mid]["time"] < t:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _piece(ds, fs_src, fs_dst, t_start, n_rows, L, guess=0):
    """Rows [t_start, t_start + 30 n_rows) of one channel at fs_dst, NaN where invalid / absent."""
    t0, t1 = t_start - CONTEXT_S, t_start + 30.0 * n_rows + CONTEXT_S
    j0 = _find(ds, t0 - 1.0 / fs_src, guess); j1 = _find(ds, t1 + 1.0 / fs_src, j0)
    out = np.full(n_rows * L, np.nan, np.float32)
    if j1 - j0 < 4:
        return out, j0
    a = ds[j0:j1]
    tt, vv = a["time"].astype(np.float64), a["value"].astype(np.float64)
    bad = ~np.isfinite(vv) | (np.abs(vv) > INVALID_ABS)
    n_ctx = int(round(CONTEXT_S * fs_src))
    g = t_start + (np.arange(int(round(30.0 * n_rows * fs_src)) + 2 * n_ctx) - n_ctx) / fs_src
    good = ~bad
    if good.sum() < 4:
        return out, j0
    y = np.interp(g, tt[good], vv[good])
    # NaN where the grid point is not between two good raw samples < 2 sample periods apart, or sits on a bad one
    k = np.searchsorted(tt, g)
    k0, k1 = np.clip(k - 1, 0, tt.size - 1), np.clip(k, 0, tt.size - 1)
    hole = (np.abs(tt[k1] - tt[k0]) > 2.5 / fs_src) | bad[k0] | bad[k1] | (g < tt[0]) | (g > tt[-1])
    y[hole] = np.nan
    from scipy.signal import resample_poly
    r = Fraction(fs_dst, int(round(fs_src))).limit_denominator(1000)
    z = resample_poly(np.where(hole, 0.0, y), r.numerator, r.denominator)
    if hole.any():
        m = resample_poly(hole.astype(np.float64), r.numerator, r.denominator) > 0.05
        z[m] = np.nan
    c = int(round(CONTEXT_S * fs_dst))
    z = z[c:c + n_rows * L]
    out[: z.size] = z
    return out, j0


def one(task):
    row, wav_dir, raw_dir, out_root = task
    import h5py
    eid = row["entity_id"]
    od = os.path.join(out_root, eid)
    mp = os.path.join(od, "meta.json")
    if os.path.exists(mp):
        try:
            if json.load(open(mp)).get("stage_b", {}).get("done"):
                return {"entity_id": eid, "skipped": True}
        except Exception:
            pass
    t_start = time.time()
    try:
        seg = json.load(open(os.path.join(wav_dir, eid + "__meta.json")))["seg_list"]
        order = np.argsort([s[2] for s in seg], kind="stable")
        st = np.array([seg[i][2] for i in order], float); blk = np.array([seg[i][0] for i in order])
        n = st.size
        runs, r0 = [], 0
        for i in range(1, n + 1):
            if i == n or abs(st[i] - st[i - 1] - 30.0) > 1e-3 or blk[i] != blk[i - 1]:
                runs.append((r0, i)); r0 = i
        os.makedirs(od, exist_ok=True)
        stats = {}
        with h5py.File(os.path.join(raw_dir, eid + ".h5"), "r") as f:
            W, label = clock.parse_origin(json.loads(f.attrs[".meta"])["time_origin"])
            for name, (key, fs_dst, L) in CH.items():
                X = np.lib.format.open_memmap(os.path.join(od, name + ".npy.tmp"), mode="w+", dtype=np.float16, shape=(n, L))
                if "data/waveforms/" + key not in f:
                    X[:] = np.nan; X.flush(); del X
                    os.replace(os.path.join(od, name + ".npy.tmp"), os.path.join(od, name + ".npy"))
                    stats[name] = {"present": False, "nan_frac": 1.0}
                    continue
                ds = f["data/waveforms/" + key]
                sp = float(json.loads(ds.attrs[".meta"])["dwc_meta"]["samplePeriod"])
                fs_src = 1000.0 / sp
                guess, nan_n = 0, 0
                for (i0, i1) in runs:
                    for p0 in range(i0, i1, PIECE_ROWS):
                        p1 = min(i1, p0 + PIECE_ROWS)
                        z, guess = _piece(ds, fs_src, fs_dst, st[p0], p1 - p0, L, guess)
                        nan_n += int(np.isnan(z).sum())
                        X[p0:p1] = z.reshape(p1 - p0, L).astype(np.float16)
                X.flush(); del X
                os.replace(os.path.join(od, name + ".npy.tmp"), os.path.join(od, name + ".npy"))
                stats[name] = {"present": True, "fs_src": fs_src, "nan_frac": nan_n / float(n * L)}
        grid = clock.Grid(W, st[0])
        tms = grid.dwc(st, W).astype(np.int64)
        assert tms.size == n and np.all(np.diff(tms) > 0), "time_ms not strictly increasing"
        np.save(os.path.join(od, "time_ms.npy"), tms)
        meta = {k: row.get(k) for k in ("entity_id", "patient_id", "e1_split", "origin_year", "origin_label", "clock_rule",
                                        "disch_s", "ehr_origin_shift_min", "ehr_extra_shift_min", "clock_confidence", "clock_risk",
                                        "clock_check", "has_ehr", "ecg_filter", "lab_within_24h_frac", "lab_within_7d")}
        meta.update(source_dataset="mladi", n_seg=int(n), seg_duration_sec=30.0, seg_stride_sec=30.0,
                    channels={"PLETH40": {"fs": 40, "samples_per_seg": 1200, "unit": "DWC Pleth (raw)"},
                              "II120": {"fs": 120, "samples_per_seg": 3600, "unit": "mV"}},
                    grid_source="pretrain_wav_v2 __meta.json seg_list (segment i == mmap row i)",
                    time_base="UTC-continuous grid anchored at the New York wall clock of the first segment",
                    time_rule="workzone/mladi/clock.py (monitor: wall clock from the stamped origin)",
                    n_runs=len(runs), n_blocks=int(len(set(blk.tolist()))),
                    stage_b={"done": True, "stats": stats, "seconds": round(time.time() - t_start, 1),
                             "band_pass": None})
        json.dump(meta, open(mp, "w"), indent=1, default=str)
        return {"entity_id": eid, "n_seg": int(n), "stats": stats, "seconds": time.time() - t_start}
    except Exception as ex:
        return {"entity_id": eid, "error": f"{type(ex).__name__}: {ex}"}


def main():
    C = cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--inv", default=os.path.join(C["intermediate_dir"], "stage_a", "inventory.jsonl"))
    ap.add_argument("--out", default=C["output_dir"]); ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--shard", default="0/1")
    ap.add_argument("--raw-h5-dir", default=C["raw_h5_dir"]); ap.add_argument("--pretrain-wav-dir", default=C["pretrain_wav_dir"])
    a = ap.parse_args()
    k, n_sh = (int(x) for x in a.shard.split("/"))
    rows = [json.loads(l) for l in open(a.inv)]
    rows = [r for r in rows if r.get("included") and zlib.crc32(r["entity_id"].encode()) % n_sh == k]
    rows.sort(key=lambda r: r["entity_id"])
    rows = rows[: a.limit or None]
    os.makedirs(a.out, exist_ok=True)
    log(f"{len(rows)} entities (shard {a.shard}), {sum(r.get('grid_rows', 0) for r in rows):,} segments -> {a.out}")
    done = err = segs = 0; secs = []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(one, [(r, a.pretrain_wav_dir, a.raw_h5_dir, a.out) for r in rows]), 1):
            if "error" in r:
                err += 1; log(f"ERROR {r['entity_id'][:8]}..: {r['error']}")
            elif not r.get("skipped"):
                done += 1; segs += r["n_seg"]; secs.append(r["seconds"])
            if i % 200 == 0 or i == len(rows):
                rate = segs / max(1.0, time.time() - T0)
                log(f"{i}/{len(rows)} | written {done} ({segs:,} segments, {rate:,.0f} seg/s overall) | errors {err} | "
                    f"entity seconds p50 {np.median(secs) if secs else float('nan'):.0f}")
    log("done")


if __name__ == "__main__":
    main()
