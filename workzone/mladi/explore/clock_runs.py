#!/usr/bin/env python3
"""MLADI: inside a contiguous run that spans a DST change, is raw time wall clock or elapsed?

Two candidate monitor grids for the pretrain_wav_v2 rows:
  wall     every row's raw seconds -> New York wall clock -> UTC  (current clock.Grid.dwc)
  runs     a run = consecutive rows 30 s apart (same block); its FIRST row is placed by the wall rule,
           later rows add raw elapsed seconds (DWC times are synthesized from sample counts)
For entities with exact charted-vs-monitor NBP matches on BOTH sides of a DST change that falls INSIDE
one run, the residual (EHR rule of clock.py minus monitor grid) is measured before and after the change
under each candidate. The right one gives the same residual on both sides.
Also counts, over all included entities, runs that span a DST change, and where the wall grid goes
backwards (spring, continuous raw).
"""
import collections, json, os, sys
from datetime import timedelta
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import clock  # noqa: E402
from common import cfg, factor, nums, nbp_offset  # noqa: E402

C = cfg()


def runs_of(st, blk):
    out, r0 = [], 0
    for i in range(1, st.size + 1):
        if i == st.size or abs(st[i] - st[i - 1] - 30.0) > 1e-3 or blk[i] != blk[i - 1]:
            out.append((r0, i)); r0 = i
    return out


def runs_grid(st, blk, W):
    g = clock.Grid(W, st[0])
    utc = np.empty(st.size, np.int64)
    for a, b in runs_of(st, blk):
        u0 = int(clock.dwc_to_utc_ms(st[a], W)[0])
        utc[a:b] = u0 + np.round((st[a:b] - st[a]) * 1000).astype(np.int64)
    return g.from_utc(utc)


def one(r):
    import h5py
    e = r["entity_id"]
    try:
        seg = json.load(open(os.path.join(C["pretrain_wav_dir"], e + "__meta.json")))["seg_list"]
        o = np.argsort([s[2] for s in seg], kind="stable")
        st = np.array([seg[i][2] for i in o], float); blk = np.array([seg[i][0] for i in o])
        with h5py.File(os.path.join(C["raw_h5_dir"], e + ".h5"), "r") as f:
            W, label = clock.parse_origin(json.loads(f.attrs[".meta"])["time_origin"])
            if label == "LMT":
                return None
            # runs spanning a DST change (by the wall state at their first and last row)
            span = []
            for a, b in runs_of(st, blk):
                s0 = (W + timedelta(seconds=float(st[a]))).replace(tzinfo=clock.NY).tzname()
                s1 = (W + timedelta(seconds=float(st[b - 1]))).replace(tzinfo=clock.NY).tzname()
                if s0 != s1:
                    span.append((a, b, "spring" if s0 == "EST" else "fall"))
            gw = clock.Grid(W, st[0]).dwc(st, W)
            res = {"entity_id": e[:8], "n_span": len(span), "kinds": [k for _, _, k in span],
                   "wall_backwards": int(np.sum(np.diff(gw) <= 0))}
            if not span or not r.get("has_ehr") or "ehr/low_rate" not in f or "data/numerics/NBP.NBPs" not in f:
                return res
            disch = r.get("disch_s")
            d = factor(f["ehr/low_rate"]); name, t, v = d["eventName"], d["date"].astype(float), nums(d["resultVal"])
            sb = (name == "Systolic BP") & np.isfinite(v) & np.isfinite(t)
            a_ = f["data/numerics/NBP.NBPs"][:]; tn, vn = a_["time"].astype(float), a_["value"].astype(float)
            keep = np.r_[True, np.diff(vn) != 0] & np.isfinite(vn); tn, vn = tn[keep], vn[keep]
            te = clock.Grid(W, st[0]).ehr(t[sb], W, label, disch)
            # monitor NBP times on each candidate grid: interpolate the raw->grid map of the rows
            out = {}
            for nm_, grid_rows in (("wall", gw), ("runs", runs_grid(st, blk, W))):
                gm = np.interp(tn, st, grid_rows.astype(float))       # raw -> grid, piecewise linear over rows
                inside = (tn >= st[0]) & (tn <= st[-1] + 30)
                for a, b, kind in span[:1]:
                    # the transition inside this run, on the raw clock: first row whose wall state differs
                    s0 = (W + timedelta(seconds=float(st[a]))).replace(tzinfo=clock.NY).tzname()
                    k = a + next(i for i in range(b - a) if (W + timedelta(seconds=float(st[a + i]))).replace(tzinfo=clock.NY).tzname() != s0)
                    for side, m in (("before", inside & (tn < st[k])), ("after", inside & (tn >= st[k]))):
                        off, n_ok, _ = nbp_offset(te, v[sb], np.full(sb.sum(), np.nan), gm[m].astype(np.int64), vn[m], np.full(m.sum(), np.nan))
                        out[f"{nm_}_{side}"] = (off, n_ok)
                    out["kind"] = kind
            res["check"] = out
            return res
    except Exception as ex:
        return {"entity_id": e[:8], "error": f"{type(ex).__name__}: {ex}"}


def main():
    inv = [json.loads(l) for l in open(os.path.join(C["intermediate_dir"], "stage_a", "inventory.jsonl"))]
    inv = [r for r in inv if r.get("included")]
    with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", 16))) as pool:
        R = [x for x in pool.map(one, inv, chunksize=8) if x]
    E = [x for x in R if "error" in x]; R = [x for x in R if "error" not in x]
    print(f"{len(R)} entities (errors {len(E)}: {collections.Counter(x['error'][:50] for x in E).most_common(3)})", flush=True)
    sp = [x for x in R if x["n_span"]]
    print(f"entities with a run spanning a DST change: {len(sp)} ({collections.Counter(k for x in sp for k in x['kinds'])}) | "
          f"wall grid goes backwards in {sum(x['wall_backwards'] > 0 for x in R)} entities", flush=True)
    T = collections.defaultdict(collections.Counter)
    for x in R:
        c = x.get("check")
        if not c or "kind" not in c:
            continue
        for nm_ in ("wall", "runs"):
            b, a = c.get(f"{nm_}_before"), c.get(f"{nm_}_after")
            if b and a and b[1] >= 2 and a[1] >= 2 and b[0] is not None and a[0] is not None:
                T[(c["kind"], nm_)][f"{b[0]:+d} -> {a[0]:+d}"] += 1
    for k, cnt in sorted(T.items()):
        print(f"  {k}: residual before -> after the change: {cnt.most_common(8)}", flush=True)


if __name__ == "__main__":
    main()
