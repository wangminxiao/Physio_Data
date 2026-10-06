#!/usr/bin/env python3
"""Gate C3 for MLADI + the store-level summary <store>/pleth_timing_summary.json (non-zero exit on failure;
verify_stage_c3.json in the Stage A dir).

  completion  meta.pleth_timing on >= 99 % of entities with PLETH40
  D_global    median over entities (>= 1000 paired beats) of their median (measured - saw): the cohort
              device constant readers add to saw(t); must lie in [1150, 1300] ms
  coverage    median share of segments with a delay >= 0.6; median share in a fitted run >= 0.7
  sawtooth    medians: ramp 25-34 ms/min, period 60-72 s, jump -35..-20 ms; the 2-s spread drops after
              removing the sawtooth (median after / before < 0.85)
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import cfg  # noqa: E402


def q(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    return {p: round(float(np.percentile(x, p)), 2) for p in (5, 25, 50, 75, 95)} if x.size else {}


def main():
    C = cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=C["output_dir"])
    ap.add_argument("--stage-a", default=os.path.join(C["intermediate_dir"], "stage_a"))
    a = ap.parse_args()
    ents = [d for d in os.listdir(a.out) if os.path.exists(os.path.join(a.out, d, "PLETH40.npy"))]
    M = []
    for e in ents:
        pt = json.load(open(os.path.join(a.out, e, "meta.json"))).get("pleth_timing")
        if pt:
            M.append(pt)
    res, fails = {"n_entities": len(ents), "n_done": len(M)}, []
    if len(M) < 0.99 * len(ents): fails.append(f"pleth_timing on {len(M)} of {len(ents)}")
    G = [m for m in M if m["n_paired_beats"] >= 1000]
    lev = [m["level_median_ms"] for m in G]
    D = float(np.median([v for v in lev if v is not None])) if G else float("nan")
    res.update(n_good=len(G), D_global_ms=round(D, 2), level_ms=q(lev),
               seg_with_delay_frac=q([m["seg_with_delay_frac"] for m in M]),
               seg_in_fitted_run_frac=q([m["seg_in_fitted_run_frac"] for m in M]),
               ramp_ms_per_min=q([m["ramp_ms_per_min_median"] for m in G]), period_s=q([m["period_s_median"] for m in G]),
               jump_ms=q([m["jump_ms_median"] for m in G]),
               sd_2s_before_ms=q([m["sd_2s_before_ms"] for m in G]), sd_2s_after_ms=q([m["sd_2s_after_ms"] for m in G]))
    if not 1150 <= D <= 1300: fails.append(f"D_global {D:.1f} ms outside [1150, 1300]")
    if res["seg_with_delay_frac"].get(50, 0) < 0.6: fails.append("median seg_with_delay_frac < 0.6")
    if res["seg_in_fitted_run_frac"].get(50, 0) < 0.7: fails.append("median seg_in_fitted_run_frac < 0.7")
    if not 25 <= res["ramp_ms_per_min"].get(50, 0) <= 34: fails.append(f"ramp median {res['ramp_ms_per_min'].get(50)}")
    if not 60 <= res["period_s"].get(50, 0) <= 72: fails.append(f"period median {res['period_s'].get(50)}")
    if not -35 <= res["jump_ms"].get(50, 0) <= -20: fails.append(f"jump median {res['jump_ms'].get(50)}")
    ratio = res["sd_2s_after_ms"].get(50, np.nan) / res["sd_2s_before_ms"].get(50, np.nan) if res["sd_2s_before_ms"] else np.nan
    res["sd_ratio_median"] = round(float(ratio), 3)
    if not ratio < 0.85: fails.append(f"2-s spread after/before {ratio:.2f} >= 0.85")
    res["fails"] = fails
    summary = {"version": "mladi-c3-1", "written": time.strftime("%Y-%m-%d"), "D_global_ms": round(D, 2),
               "use": "device part of the ECG -> Pleth delay at time t = D_global_ms + saw(t); saw(t) from the entity's "
                      "pleth_timing_runs.npy / pleth_resets.npy (workzone/mladi/pleth_timing_lib.eval_sawtooth, per run) or "
                      "pleth_timing.npy saw_ms at segment centres; measured_ms - device = physiological part",
               "foot_definition": "intersecting tangent at the steepest upstroke (~+24 ms after the true onset of a smooth upstroke)",
               **{k: res[k] for k in ("n_entities", "n_good", "level_ms", "ramp_ms_per_min", "period_s", "jump_ms",
                                      "seg_with_delay_frac", "seg_in_fitted_run_frac", "sd_2s_before_ms", "sd_2s_after_ms")}}
    if not fails:
        json.dump(summary, open(os.path.join(a.out, "pleth_timing_summary.json"), "w"), indent=1)
    os.makedirs(a.stage_a, exist_ok=True)
    json.dump(res, open(os.path.join(a.stage_a, "verify_stage_c3.json"), "w"), indent=1)
    print(json.dumps(res, indent=1)); print("GATE C3:", "FAIL" if fails else "PASS", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
