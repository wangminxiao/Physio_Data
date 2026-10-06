#!/usr/bin/env python3
"""Gate D for MLADI (non-zero exit on failure; verify_stage_d.json in the Stage A dir).

  completion  meta.ehr and meta.ehr_actions on >= 99 % of Stage B entities
  sample      400 entities: the 4 EHR files + ehr_actions.npy exist, EHR_EVENT_DTYPE, pass
              physio_data.ehr_trajectory.validate_partition; actions var_id in 200-220 with
              seg_idx in [0, n_seg); EHR values inside the registry physio range
  coverage    has_ehr entities with >= 1 in-waveform event: >= 70 %
  clock       charted SBP (104) vs monitor NBP (157) on the stored grid, per entity the share of
              charted SBP with an NBP within 60 s and 1 mmHg; median >= 0.8 for verified, reported for
              corrected / inferred
  plausible   pooled medians of the main labs and vitals inside clinical bands (catches unit or mapping
              mistakes)
"""
import argparse, collections, json, os, random, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from common import cfg, EHR_EVENT_DTYPE  # noqa: E402
from physio_data.ehr_trajectory import ALL_FNAMES, validate_partition  # noqa: E402

BANDS = {0: (3.3, 5.0), 2: (133, 145), 3: (80, 200), 4: (0.6, 4.0), 5: (0.4, 2.5), 7: (100, 350), 8: (4, 16),
         9: (7, 14), 11: (8, 40), 13: (7.25, 7.5), 14: (60, 250), 15: (30, 55), 16: (18, 30),
         100: (60, 110), 103: (36.0, 38.0), 104: (95, 150), 105: (50, 85)}


def main():
    C = cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=C["output_dir"]); ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--stage-a", default=os.path.join(C["intermediate_dir"], "stage_a"))
    a = ap.parse_args()
    reg = {v["id"]: (v.get("physio_min"), v.get("physio_max"))
           for v in json.load(open(os.path.join(os.path.dirname(os.path.dirname(HERE)), "indices", "var_registry.json")))["variables"]}
    ents = [d for d in os.listdir(a.out) if os.path.exists(os.path.join(a.out, d, "meta.json"))]
    metas = {d: json.load(open(os.path.join(a.out, d, "meta.json"))) for d in ents}
    done = [d for d in ents if "ehr" in metas[d] and "ehr_actions" in metas[d]]
    res, fails = {"n_entities": len(ents), "n_done": len(done)}, []
    if len(done) < 0.99 * len(ents): fails.append(f"ehr + actions on {len(done)} of {len(ents)}")
    has = [d for d in done if metas[d].get("has_ehr")]
    cov = float(np.mean([metas[d]["ehr"]["n_events"] > 0 for d in has])) if has else 0.0
    res.update(n_has_ehr=len(has), has_ehr_with_events=cov)
    if cov < 0.7: fails.append(f"has_ehr entities with in-waveform events {cov:.2f} < 0.7")
    tot = collections.Counter(); act = collections.Counter()
    for d in done:
        for k, v in metas[d]["ehr"]["events_per_var"].items(): tot[int(k)] += v
        for k, v in metas[d]["ehr_actions"]["per_var"].items(): act[int(k)] += v
    res.update(events_per_var=dict(sorted(tot.items())), actions_per_var=dict(sorted(act.items())),
               entities_with_vasopressor=int(sum(any(int(k) in range(207, 214) for k in metas[d]["ehr_actions"]["per_var"]) for d in done)))
    random.seed(0); S = random.sample(has, min(a.n, len(has))) + random.sample(sorted(set(done) - set(has)), min(50, len(set(done) - set(has))))
    probs, vals, clk = [], collections.defaultdict(list), collections.defaultdict(list)
    for e in S:
        d = os.path.join(a.out, e)
        try:
            n = np.load(os.path.join(d, "time_ms.npy"), mmap_mode="r").size
            errs = []
            parts = {}
            for kind, fn in zip(("baseline", "recent", "events", "future"), ALL_FNAMES):
                x = np.load(os.path.join(d, fn)); parts[kind] = x
                if x.dtype != EHR_EVENT_DTYPE: errs.append(f"{fn} dtype")
                errs += validate_partition(x, kind=kind, n_seg=n)
                if x.size:
                    lo = np.array([reg.get(int(v), (None, None))[0] if reg.get(int(v), (None, None))[0] is not None else -np.inf for v in x["var_id"]])
                    hi = np.array([reg.get(int(v), (None, None))[1] if reg.get(int(v), (None, None))[1] is not None else np.inf for v in x["var_id"]])
                    if np.any((x["value"] < lo) | (x["value"] > hi)): errs.append(f"{fn} out of range")
            A = np.load(os.path.join(d, "ehr_actions.npy"))
            if A.dtype != EHR_EVENT_DTYPE: errs.append("actions dtype")
            if A.size and (A["var_id"].min() < 200 or A["var_id"].max() > 220 or A["seg_idx"].min() < 0 or A["seg_idx"].max() >= n
                           or np.any(np.diff(A["time_ms"]) < 0)): errs.append("actions content")
            if errs: probs.append(f"{e[:8]} {errs[:2]}"); continue
            ev = parts["events"]
            for v in BANDS:
                vals[v] += ev["value"][ev["var_id"] == v].tolist()
            s = ev[ev["var_id"] == 104]; nb = np.load(os.path.join(d, "nbp_events.npy")); nb = nb[nb["var_id"] == 157]
            if s.size >= 5 and nb.size:
                j = np.searchsorted(nb["time_ms"], s["time_ms"])
                hit = []
                for t_, v_, jj in zip(s["time_ms"], s["value"], j):
                    c = nb[max(0, jj - 3): jj + 3]
                    hit.append(bool(np.any((np.abs(c["time_ms"] - t_) <= 60_000) & (np.abs(c["value"] - v_) <= 1))))
                clk[metas[e].get("clock_confidence", "?")].append(float(np.mean(hit)))
        except Exception as ex:
            probs.append(f"{e[:8]} {type(ex).__name__}: {ex}")
    res.update(sample=len(S), n_problems=len(probs), problems=probs[:10])
    if len(probs) > 0.01 * max(1, len(S)): fails.append(f"{len(probs)} problem entities")
    res["clock_nbp_match"] = {k: {"n": len(v), "median": float(np.median(v)), "p10": float(np.percentile(v, 10))} for k, v in clk.items()}
    if clk.get("verified") and np.median(clk["verified"]) < 0.8: fails.append(f"verified clock match median {np.median(clk['verified']):.2f} < 0.8")
    med = {}
    for v, (lo, hi) in BANDS.items():
        if len(vals[v]) >= 50:
            m = float(np.median(vals[v])); med[v] = round(m, 2)
            if not lo <= m <= hi: fails.append(f"var {v} pooled median {m:.2f} outside [{lo}, {hi}]")
    res["pooled_median"] = med
    res["fails"] = fails
    json.dump(res, open(os.path.join(a.stage_a, "verify_stage_d.json"), "w"), indent=1)
    print(json.dumps(res, indent=1)); print("GATE D:", "FAIL" if fails else "PASS", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
