"""
Why did MOVER cases stay clock-unverified after stage B2, and does the device fixed-offset hypothesis give a season prior?

Hypothesis (CLOCK_FIX_PLAN §2): waveform `…Z` stamps come from devices with a fixed UTC offset, −7 (right in DST, 60 min
early in standard time -> needs +60) or −8 (right in standard time, 60 min late in DST -> needs −60). If true, the sign
of every decided shift is fixed by the season of the case: standard time -> {0, +60}; DST -> {0, −60}.

  python workzone/mover/explore/unverified_analysis.py --dataset mover \
      --report /mnt/localdata/storage/mxwang/Physio_Data/workzone/outputs/mover/clock_shift.parquet
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
LA = ZoneInfo("America/Los_Angeles")


def is_dst(ms: int) -> bool:
    return bool(datetime.fromtimestamp(ms / 1000, tz=LA).dst())


def q(a, ps=(5, 25, 50, 75, 95)):
    a = np.asarray([x for x in a if x is not None and np.isfinite(x)], dtype=float)
    return {str(p): round(float(np.percentile(a, p)), 2) for p in ps} if a.size else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mover"); ap.add_argument("--root", default=None); ap.add_argument("--report", default=None)
    ap.add_argument("--sample-vitals", type=int, default=300); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / "workzone" / "configs" / "server_paths.yaml").read_text())[a.dataset]
    root = Path(a.root or cfg["output_dir"])
    rep = pl.read_parquet(a.report or REPO_ROOT / "workzone" / "outputs" / a.dataset / "clock_shift.parquet")
    rows = []
    for eid in rep["entity_id"].to_list():
        mp = root / eid / "meta.json"
        if not mp.exists():
            continue
        m = json.loads(mp.read_text())
        ws = m.get("wave_start_ms")
        rows.append({"entity_id": eid, "conf": m.get("clock_shift_confidence"), "shift": m.get("clock_shift_min"),
                     "method": m.get("clock_shift_method"), "dst": is_dst(int(ws)) if ws else None, "n_seg": m.get("n_segments"),
                     "hours": m.get("total_duration_hours")})
    meta = pl.DataFrame(rows)
    df = rep.join(meta, on="entity_id", how="left")
    out = {"dataset": a.dataset, "n": df.height, "by_status": dict(Counter(df["status"].to_list())),
           "by_confidence": dict(Counter(df["conf"].to_list())),
           "shift_by_confidence": {}}
    for c in sorted({x for x in df["conf"].to_list() if x}):
        sub = df.filter(pl.col("conf") == c)
        out["shift_by_confidence"][c] = dict(Counter(sub["shift"].to_list()))
    # --- season prior on decided (non-zero) shifts
    dec = df.filter((pl.col("conf").is_in(["high", "medium"])) & (pl.col("shift") != 0))
    tab = Counter(zip(dec["dst"].to_list(), dec["shift"].to_list()))
    out["season_vs_shift_decided_nonzero"] = {f"dst={k[0]} shift={k[1]}": v for k, v in sorted(tab.items(), key=lambda kv: str(kv[0]))}
    consistent = sum(v for (d, s), v in tab.items() if (d and s == -60) or (not d and s == 60))
    out["season_prior_consistency"] = round(consistent / max(1, dec.height), 4)
    # by confidence level
    for c in ("high", "medium"):
        sub = dec.filter(pl.col("conf") == c); t = Counter(zip(sub["dst"].to_list(), sub["shift"].to_list()))
        ok = sum(v for (d, s), v in t.items() if (d and s == -60) or (not d and s == 60))
        out[f"season_prior_consistency_{c}"] = {"n": sub.height, "frac": round(ok / max(1, sub.height), 4)}
    # season of aligned decided cases (shift 0) and of everything
    out["dst_frac_all"] = round(float(np.mean([bool(x) for x in df["dst"].to_list() if x is not None])), 3)
    # --- unverified population
    unv = df.filter(pl.col("conf") == "unverified")
    out["unverified"] = {"n": unv.height, "by_method": dict(Counter(unv["method"].to_list())), "by_status": dict(Counter(unv["status"].to_list())),
                         "n_hr_quantiles": q(unv["n_hr"].to_list()) if "n_hr" in unv.columns else {},
                         "hours_quantiles": q(unv["hours"].to_list()),
                         "ppg_rate_frac_q": q(unv["ppg_rate_frac"].to_list()) if "ppg_rate_frac" in unv.columns else {},
                         "ecg_rate_frac_q": q(unv["ecg_rate_frac"].to_list()) if "ecg_rate_frac" in unv.columns else {}}
    has = unv.filter(pl.col("lag_ppg").is_not_null() | pl.col("lag_ecg").is_not_null())
    out["unverified"]["n_with_any_reference"] = has.height
    out["unverified"]["lag_ppg_hist"] = dict(Counter(has["lag_ppg"].to_list()))
    out["unverified"]["lag_ecg_hist"] = dict(Counter(has["lag_ecg"].to_list()))
    out["unverified"]["mae_ppg_best_q"] = q(has["mae_ppg_best"].to_list()); out["unverified"]["mae_ppg0_q"] = q(has["mae_ppg0"].to_list())
    out["unverified"]["mae_ecg_best_q"] = q(has["mae_ecg_best"].to_list()); out["unverified"]["mae_ecg0_q"] = q(has["mae_ecg0"].to_list())

    # candidates under a season-constrained binary test using the recorded scan: best lag in the season-allowed band or
    # at 0, with improvement (or tightness) — counted per relaxation level of the PPG MAE bar
    def band(L):
        if L is None: return None
        if abs(L) <= 10: return 0
        if 45 <= abs(L) <= 75: return 60 if L > 0 else -60
        return "other"
    def allowed(d, b):
        return b == 0 or (d is False and b == 60) or (d is True and b == -60)
    cand = {}
    for bar_p, bar_e in ((8, 6), (10, 8), (12, 10), (15, 12)):
        n_dec = 0; n_conf = 0; per_method = Counter()
        for r in has.iter_rows(named=True):
            votes = []
            for pre, bar in (("ppg", bar_p), ("ecg", bar_e)):
                L, mb, m0 = r.get(f"lag_{pre}"), r.get(f"mae_{pre}_best"), r.get(f"mae_{pre}0")
                if L is None or mb is None or not np.isfinite(mb) or mb > bar: continue
                b = band(L)
                if b == "other" or not allowed(r["dst"], b): continue
                improves = m0 is not None and np.isfinite(m0) and mb < 0.75 * m0
                tight0 = b == 0 and m0 is not None and np.isfinite(m0) and m0 <= bar
                if improves or tight0: votes.append(b)
            if votes and len(set(votes)) == 1: n_dec += 1; per_method[r["method"]] += 1
            elif len(set(votes)) > 1: n_conf += 1
        cand[f"ppg<={bar_p},ecg<={bar_e}"] = {"decidable": n_dec, "conflict": n_conf, "by_method": dict(per_method)}
    out["unverified"]["season_constrained_candidates_from_recorded_scan"] = cand
    # disallowed-band evidence: unverified refs whose best lag falls in the season-forbidden band (device hypothesis says impossible)
    forb = 0
    for r in has.iter_rows(named=True):
        for pre in ("ppg", "ecg"):
            b = band(r.get(f"lag_{pre}"))
            if b in (60, -60) and not allowed(r["dst"], b): forb += 1
    out["unverified"]["n_refs_in_forbidden_band"] = forb
    # decided cases: how many had a ref in the forbidden band (sanity)
    forb_dec = 0; n_refs_dec = 0
    for r in df.filter(pl.col("conf").is_in(["high", "medium"])).iter_rows(named=True):
        for pre in ("ppg", "ecg"):
            L = r.get(f"lag_{pre}")
            if L is None: continue
            n_refs_dec += 1; b = band(L)
            if b in (60, -60) and not allowed(r["dst"], b): forb_dec += 1
    out["decided_refs_in_forbidden_band"] = {"n_refs": n_refs_dec, "forbidden": forb_dec}
    # vitals available for unverified entities (sample)
    vc = Counter(); n_s = 0
    for eid in unv["entity_id"].to_list()[: a.sample_vitals]:
        p = root / eid / "vitals_events.npy"
        if p.exists():
            v = np.load(p)
            if v.size:
                for vid in np.unique(v["var_id"]): vc[int(vid)] += 1
        n_s += 1
    out["unverified"]["vitals_var_presence_in_sample"] = {"n_sampled": n_s, **{str(k): v for k, v in sorted(vc.items())}}
    txt = json.dumps(out, indent=1, default=str)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt)


if __name__ == "__main__":
    main()
