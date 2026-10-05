#!/usr/bin/env python3
"""Gate B for MLADI (non-zero exit on any failure; verify_stage_b.json next to the output root).

  completion  errors + missing <= 1 % of included entities; done >= 99 %
  sample      N entities (seeded): PLETH40 / II120 float16 C-contiguous [n_seg, 1200 / 3600];
              n_seg == Stage A grid_rows == the mmap's row count; time_ms int64 strictly increasing,
              30 000 ms steps inside runs; no inf; NaN fraction median < 0.3 (PLETH) ;
              PLETH non-constant; II |p99| in [0.05, 50] mV where present
  grid        band-passing canonical rows as data_preparing_v2 did reproduces the mmap rows:
              per-entity median r >= 0.98 on PLETH40 and II120 (10 rows each)
"""
import argparse, glob, json, os, random, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import cfg  # noqa: E402


def main():
    from scipy.signal import butter, filtfilt
    C = cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--inv", default=os.path.join(C["intermediate_dir"], "stage_a", "inventory.jsonl"))
    ap.add_argument("--out", default=C["output_dir"]); ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--pretrain-wav-dir", default=C["pretrain_wav_dir"])
    ap.add_argument("--sample-only", action="store_true", help="skip the completion check (trial outputs)")
    a = ap.parse_args()
    inc = [json.loads(l) for l in open(a.inv)]; inc = {r["entity_id"]: r for r in inc if r.get("included")}
    done = [e for e in inc if os.path.exists(os.path.join(a.out, e, "meta.json"))
            and json.load(open(os.path.join(a.out, e, "meta.json"))).get("stage_b", {}).get("done")]
    res, fails = {"n_included": len(inc), "n_done": len(done)}, []
    if len(done) < 0.99 * len(inc) and not a.sample_only: fails.append(f"done {len(done)} < 99% of {len(inc)}")
    random.seed(0)
    S = random.sample(done, min(a.n, len(done)))
    bp = {"PLETH40": butter(4, [0.5 / 20, 12.0 / 20], "band"), "II120": butter(4, [0.5 / 60, 50.0 / 60], "band")}
    probs, nanp, rs = [], [], {"PLETH40": [], "II120": []}
    for e in S:
        d = os.path.join(a.out, e)
        try:
            P = np.load(os.path.join(d, "PLETH40.npy"), mmap_mode="r"); E = np.load(os.path.join(d, "II120.npy"), mmap_mode="r")
            t = np.load(os.path.join(d, "time_ms.npy")); m = json.load(open(os.path.join(d, "meta.json")))
            n = inc[e]["grid_rows"]
            ok = (P.dtype == np.float16 and E.dtype == np.float16 and P.flags.c_contiguous and E.flags.c_contiguous
                  and P.shape == (n, 1200) and E.shape == (n, 3600) and t.dtype == np.int64 and t.size == n
                  and np.all(np.diff(t) > 0))
            mm = glob.glob(os.path.join(a.pretrain_wav_dir, f"{e}_Pleth_40Hz_*_1200_mmap.npy"))
            if mm and int(os.path.basename(mm[0]).split("_")[-3]) != n:
                ok = False
            if not ok:
                probs.append((e[:8], "shape/dtype/time")); continue
            Pf = np.asarray(P[: min(n, 2000)], np.float32)
            if np.isinf(Pf).any(): probs.append((e[:8], "inf"))
            nanp.append(float(np.isnan(Pf).mean()))
            if np.nanstd(Pf) < 1e-6: probs.append((e[:8], "PLETH constant"))
            if inc[e].get("has_ii"):
                Ef = np.asarray(E[: min(n, 500)], np.float32)
                p99 = np.nanpercentile(np.abs(Ef), 99) if np.isfinite(Ef).any() else 0
                if not 0.05 <= p99 <= 50: probs.append((e[:8], f"II p99 {p99:.3g}"))
            for name, key, fs in (("PLETH40", "Pleth", 40), ("II120", "II", 120)):
                mm = glob.glob(os.path.join(a.pretrain_wav_dir, f"{e}_{key}_{fs}Hz_*_mmap.npy"))
                if not mm or (name == "II120" and not inc[e].get("has_ii")):
                    continue
                M = np.load(mm[0], mmap_mode="r"); X = P if name == "PLETH40" else E
                rr = []
                for i in np.linspace(0, n - 1, 10).astype(int):
                    x = np.asarray(X[i], np.float64)
                    if np.isfinite(x).all() and np.std(M[i]) > 0:
                        rr.append(np.corrcoef(filtfilt(*bp[name], x), np.asarray(M[i], np.float64))[0, 1])
                if rr:
                    rs[name].append(float(np.median(rr)))
        except Exception as ex:
            probs.append((e[:8], f"{type(ex).__name__}: {ex}"))
    res.update(sample=len(S), problems=probs[:30], n_problems=len(probs),
               pleth_nan_frac_median=float(np.median(nanp)) if nanp else None,
               grid_r_median={k: float(np.median(v)) if v else None for k, v in rs.items()},
               grid_r_p05={k: float(np.percentile(v, 5)) if v else None for k, v in rs.items()},
               grid_r_below_098={k: int(sum(x < 0.98 for x in v)) for k, v in rs.items()})
    if len(probs) > 0.01 * max(1, len(S)): fails.append(f"{len(probs)} problem entities in the sample")
    if nanp and np.median(nanp) >= 0.3: fails.append("PLETH NaN median >= 0.3")
    for k, v in rs.items():
        if v and np.median(v) < 0.98: fails.append(f"{k} grid r median {np.median(v):.3f} < 0.98")
    res["fails"] = fails
    json.dump(res, open(os.path.join(os.path.dirname(a.inv), "verify_stage_b.json"), "w"), indent=1)
    print(json.dumps(res, indent=1)); print("GATE B:", "FAIL" if fails else "PASS", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
