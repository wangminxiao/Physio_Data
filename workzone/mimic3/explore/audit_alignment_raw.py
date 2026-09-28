#!/usr/bin/env python3
"""
Independent audit of the MIMIC-III waveform <-> EHR clock alignment, from RAW data.

Nothing here reuses the pipeline's time conversions: raw WFDB headers/numerics are read with the `wfdb` package,
raw chart events straight from CHARTEVENTS.csv, and both are interpreted as the naive (surrogate) wall clock they are
written in. The processed store (after the 2026-09 fix) and the pre-fix snapshot are then compared against that raw
ground truth. Lag convention everywhere: chart_time + lag = monitor/waveform time (positive lag = chart earlier).

Evidence produced (per sampled entity, then aggregated):
  E1 base time   store time_ms[0] vs the raw master-header base time (wall clock) — before/after
  E2 numerics    raw chart HR (ITEMID 211/220045) vs raw numerics HR (WFDB n-record): best lag by MAE and by
                 exact-value matches (|dHR| <= 1 bpm, monitor sample within 90 s) — raw-vs-raw ground truth
  E3 store       chart HR events (var 100) vs the store's numerics sidecar HR (var 150) on the store's own clock —
                 before (snapshot) and after; and store ehr_hf sample times vs raw numerics times
  E4 waveform    ECG-derived HR from the store's II120 segments (time_ms clock) vs raw chart HR — before/after
  E5 cuff BP     raw chart NBP systolic (455/220179) vs raw numerics 'NBP Sys': exact matches at lag 0 vs +-4/5 h

  step 1 (once):  python audit_alignment_raw.py extract --out DIR          # chart rows for the sampled subjects
  step 2:         python audit_alignment_raw.py analyze --out DIR [--n-ecg 60] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_WAVE = "/mnt/localdata/storage/MIMIC_waveform_matched_subset/physionet.org/files/mimic3wdb-matched/1.0"
RAW_EHR = "/mnt/localdata/storage/MIMICIII-v1.4"
STORE = "/mnt/localdata100tb/physio_data/mimic3"
SNAP = "/projects/mwang80/staging/mimic3_before"
EPOCH = datetime(1970, 1, 1)
HR_ITEMS = (211, 220045)
NBP_ITEMS = (455, 220179)
COARSE = list(range(-420, 421, 5))         # minutes
FINE = list(range(-30, 31, 1))


def wall_ms(dt: datetime) -> int:
    return int((dt.replace(tzinfo=None) - EPOCH).total_seconds() * 1000)


def to_dt(ms: int) -> datetime:
    return EPOCH + timedelta(milliseconds=int(ms))


def translate(p: str) -> str:
    return p.replace("/labs/hulab/", "/mnt/localdata/storage/")


def load_entities():
    ids = [l.strip() for l in open(os.path.join(SNAP, "entities.txt")) if l.strip()]
    out = []
    for e in ids:
        mp = os.path.join(SNAP, e, "meta.json")
        if not os.path.exists(mp):
            continue
        m = json.load(open(mp))
        out.append({"entity": e, "subject_id": int(m["subject_id"]), "hadm_id": int(m["hadm_id"]) if m.get("hadm_id") is not None else None,
                    "raw_dir": translate(m["source_path"]), "start_before": int(m.get("recording_start_ms", np.load(os.path.join(SNAP, e, "time_ms.npy"))[0]))})
    return out


# ------------------------------------------------------------------ step 1: raw chart rows
def extract(args):
    import polars as pl
    ents = load_entities()
    subj = sorted({e["subject_id"] for e in ents})
    items = list(HR_ITEMS + NBP_ITEMS)
    t0 = time.time()
    lf = pl.scan_csv(os.path.join(RAW_EHR, "CHARTEVENTS.csv"), infer_schema_length=0, ignore_errors=True).select(
        ["SUBJECT_ID", "HADM_ID", "ITEMID", "CHARTTIME", "VALUENUM"])
    lf = lf.with_columns(pl.col("SUBJECT_ID").cast(pl.Int64), pl.col("ITEMID").cast(pl.Int64, strict=False))
    lf = lf.filter(pl.col("SUBJECT_ID").is_in(subj) & pl.col("ITEMID").is_in(items))
    try:
        df = lf.collect(engine="streaming")
    except TypeError:
        df = lf.collect(streaming=True)
    df = df.with_columns(pl.col("HADM_ID").cast(pl.Int64, strict=False), pl.col("VALUENUM").cast(pl.Float64, strict=False),
                         pl.col("CHARTTIME").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False))
    os.makedirs(args.out, exist_ok=True)
    df.write_parquet(os.path.join(args.out, "chartevents_sample.parquet"))
    print(f"extract: {df.height} rows for {len(subj)} subjects in {time.time()-t0:.0f}s -> {args.out}/chartevents_sample.parquet")


# ------------------------------------------------------------------ helpers
def read_master_headers(raw_dir):
    """[(name, base_datetime)] for every master record pNNNNNN-YYYY-MM-DD-hh-mm in the patient dir."""
    import wfdb
    out = []
    for f in sorted(os.listdir(raw_dir)):
        if f.startswith("p") and f.endswith(".hea") and not f.endswith("n.hea") and "_" not in f and "layout" not in f:
            h = wfdb.rdheader(os.path.join(raw_dir, f[:-4]))
            if h.base_date is None or h.base_time is None:
                continue
            out.append((f[:-4], datetime.combine(h.base_date, h.base_time)))
    return out


def read_numerics(raw_dir, name):
    """raw numerics record name+'n' -> dict(base, fs, t_ms, HR, NBPSys) with NaN for invalid samples."""
    import wfdb
    p = os.path.join(raw_dir, name + "n")
    if not os.path.exists(p + ".hea"):
        return None
    rec = wfdb.rdrecord(p)
    base = datetime.combine(rec.base_date, rec.base_time)
    n = rec.p_signal.shape[0]
    t_ms = wall_ms(base) + (np.arange(n) * (1000.0 / rec.fs)).astype(np.int64)
    names = [s.upper().replace(" ", "") for s in rec.sig_name]
    def col(cands):
        for c in cands:
            if c in names:
                v = rec.p_signal[:, names.index(c)].astype(np.float64)
                return v
        return None
    hr = col(["HR"]); nbp = col(["NBPSYS"])
    return {"name": name, "base": base, "fs": rec.fs, "t_ms": t_ms, "HR": hr, "NBP": nbp, "n": n}


def nearest_values(t_ref, v_ref, t_query, tol_ms):
    """value of the nearest finite reference sample within tol (NaN otherwise), vectorised."""
    fin = np.isfinite(v_ref); tr = t_ref[fin]; vr = v_ref[fin]
    if tr.size == 0:
        return np.full(t_query.shape, np.nan)
    idx = np.searchsorted(tr, t_query)
    lo = np.clip(idx - 1, 0, tr.size - 1); hi = np.clip(idx, 0, tr.size - 1)
    pick = np.where(np.abs(tr[hi] - t_query) < np.abs(tr[lo] - t_query), hi, lo)
    out = vr[pick].copy(); out[np.abs(tr[pick] - t_query) > tol_ms] = np.nan
    return out


def lag_profile(t_ref, v_ref, t_ev, v_ev, lags, tol_ms=90000, match_tol=1.0, min_pts=10):
    """per lag: (mae, match_frac, n) ; returns dict lag -> tuple, or None if too few points at every lag."""
    res = {}
    for L in lags:
        q = t_ev + L * 60000
        r = nearest_values(t_ref, v_ref, q, tol_ms)
        ok = np.isfinite(r) & np.isfinite(v_ev)
        if ok.sum() < min_pts:
            continue
        d = np.abs(r[ok] - v_ev[ok])
        res[L] = (float(np.mean(d)), float(np.mean(d <= match_tol)), int(ok.sum()))
    return res or None


def best_of(prof):
    if not prof:
        return None
    by_mae = min(prof, key=lambda L: prof[L][0]); by_match = max(prof, key=lambda L: prof[L][1])
    return {"lag_mae": by_mae, "mae": round(prof[by_mae][0], 2), "mae_at0": round(prof[0][0], 2) if 0 in prof else None,
            "lag_match": by_match, "match": round(prof[by_match][1], 3), "match_at0": round(prof[0][1], 3) if 0 in prof else None,
            "n": prof[by_mae][2]}


def ecg_hr_series(entity, seg_limit=2880):
    """(t_ms, bpm) per 30-s segment from the store's II120 (peak counting, independent of numerics)."""
    from scipy.signal import butter, filtfilt, find_peaks
    d = os.path.join(STORE, entity)
    ii = np.load(os.path.join(d, "II120.npy"), mmap_mode="r"); tm = np.load(os.path.join(d, "time_ms.npy"))
    n = min(len(tm), seg_limit); bp = butter(3, [5 / 60, 30 / 60], btype="band")
    out = np.full(n, np.nan)
    for i in range(n):
        x = np.asarray(ii[i], dtype=np.float32); fin = np.isfinite(x)
        if fin.mean() < 0.9 or np.nanstd(x) < 1e-3:
            continue
        x = np.where(fin, x, np.nanmean(x)); y = np.abs(filtfilt(bp[0], bp[1], x))
        pk, _ = find_peaks(y, distance=int(0.3 * 120), height=np.percentile(y, 98) * 0.4)
        if 8 <= pk.size <= 120:
            out[i] = pk.size * 2
    return tm[:n], out


def summarize(vals):
    a = np.asarray([v for v in vals if v is not None], dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "median": float(np.median(a)), "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90)),
            "frac_within_15min": float(np.mean(np.abs(a) <= 15)), "frac_within_60min": float(np.mean(np.abs(a) <= 60)),
            "frac_in_4to5h": float(np.mean((a >= 225) & (a <= 315)))}


# ------------------------------------------------------------------ step 2: analysis
def analyze(args):
    import polars as pl
    ch = pl.read_parquet(os.path.join(args.out, "chartevents_sample.parquet"))
    ents = load_entities()
    if args.limit:
        ents = ents[: args.limit]
    rows = []; examples = {}; t0 = time.time(); n_ecg_done = 0
    for k, e in enumerate(ents, 1):
        r = {"entity": e["entity"], "subject_id": e["subject_id"], "hadm_id": e["hadm_id"]}
        try:
            # --- raw chart rows of this admission (naive wall clock)
            sub = ch.filter(pl.col("SUBJECT_ID") == e["subject_id"])
            if e["hadm_id"] is not None:
                sub = sub.filter(pl.col("HADM_ID") == e["hadm_id"])
            sub = sub.filter(pl.col("VALUENUM").is_not_null() & pl.col("CHARTTIME").is_not_null())
            hr = sub.filter(pl.col("ITEMID").is_in(list(HR_ITEMS)) & (pl.col("VALUENUM") > 20) & (pl.col("VALUENUM") < 250)).sort("CHARTTIME")
            nbp = sub.filter(pl.col("ITEMID").is_in(list(NBP_ITEMS)) & (pl.col("VALUENUM") > 30) & (pl.col("VALUENUM") < 300)).sort("CHARTTIME")
            t_hr = np.array([wall_ms(x) for x in hr["CHARTTIME"].to_list()], dtype=np.int64); v_hr = hr["VALUENUM"].to_numpy().astype(float)
            t_nbp = np.array([wall_ms(x) for x in nbp["CHARTTIME"].to_list()], dtype=np.int64); v_nbp = nbp["VALUENUM"].to_numpy().astype(float)
            r["n_chart_hr"] = int(t_hr.size); r["n_chart_nbp"] = int(t_nbp.size)
            # --- store after / snapshot before
            sd = os.path.join(STORE, e["entity"]); pd_ = os.path.join(SNAP, e["entity"])
            tm_after = np.load(os.path.join(sd, "time_ms.npy")); tm_before = np.load(os.path.join(pd_, "time_ms.npy"))
            meta_after = json.load(open(os.path.join(sd, "meta.json")))
            r["store_shift_h"] = round((int(tm_after[0]) - int(tm_before[0])) / 3.6e6, 3)
            # --- raw master record that this entity's recording starts from (after-fix start == base wall time when the
            #     first stored block is the record start; otherwise pick the record with the closest base <= start)
            masters = read_master_headers(e["raw_dir"])
            r["n_master_records"] = len(masters)
            if masters:
                bases = np.array([wall_ms(b) for _, b in masters]); i_after = int(np.argmin(np.abs(bases - int(tm_after[0]))))
                name, base_dt = masters[i_after]
                r["master"] = name; r["base_wall"] = base_dt.isoformat()
                r["E1_after_minus_base_h"] = round((int(tm_after[0]) - bases[i_after]) / 3.6e6, 4)
                r["E1_before_minus_base_h"] = round((int(tm_before[0]) - bases[i_after]) / 3.6e6, 4)
                num = read_numerics(e["raw_dir"], name)
            else:
                num = None
            # --- E2 raw chart HR vs raw numerics HR (both naive wall clock)
            if num is not None and num["HR"] is not None and t_hr.size >= 10:
                r["numerics_fs"] = num["fs"]; r["numerics_n"] = num["n"]; r["numerics_base_wall"] = num["base"].isoformat()
                prof = lag_profile(num["t_ms"], num["HR"], t_hr, v_hr, COARSE)
                r["E2_raw"] = best_of(prof)
                if r["E2_raw"] and abs(r["E2_raw"]["lag_match"]) <= 30:
                    r["E2_raw_fine"] = best_of(lag_profile(num["t_ms"], num["HR"], t_hr, v_hr, FINE, tol_ms=60000))
                # raw numerics re-based the way the OLD code did (local-tz epoch): what the snapshot used
                shift_ms = int(tm_before[0]) - int(tm_after[0])
                r["E2_raw_legacy_base"] = best_of(lag_profile(num["t_ms"] + shift_ms, num["HR"], t_hr, v_hr, COARSE))
                # --- E5 cuff NBP exact matches
                if num["NBP"] is not None and t_nbp.size >= 5:
                    pn = lag_profile(num["t_ms"], num["NBP"], t_nbp, v_nbp, [0, 240, 300, -240, -300], tol_ms=300000, match_tol=2.0, min_pts=5)
                    if pn:
                        r["E5_nbp_match_by_lag"] = {str(L): round(v[1], 3) for L, v in pn.items()}
                        r["E5_nbp_n"] = pn[0][2] if 0 in pn else None
                if k <= 3 and num["HR"] is not None:
                    examples[e["entity"]] = {"t_num": num["t_ms"][::60].tolist(), "hr_num": [None if not np.isfinite(x) else float(x) for x in num["HR"][::60]],
                                             "t_chart": t_hr.tolist(), "hr_chart": v_hr.tolist(), "shift_ms": shift_ms}
            # --- E3 store: chart HR events (var 100) vs numerics sidecar HR (var 150) on the store's clock
            for tag, d in (("after", sd), ("before", pd_)):
                ev = np.load(os.path.join(d, "ehr_events.npy")); hf = np.load(os.path.join(d, "ehr_hf.npy")) if os.path.exists(os.path.join(d, "ehr_hf.npy")) else None
                ev_hr = ev[ev["var_id"] == 100]
                if hf is not None and hf.size and ev_hr.size >= 10:
                    h = hf[hf["var_id"] == 150]
                    if h.size >= 100:
                        prof = lag_profile(h["time_ms"].astype(np.int64), h["value"].astype(float), ev_hr["time_ms"].astype(np.int64), ev_hr["value"].astype(float), COARSE)
                        r[f"E3_{tag}"] = best_of(prof)
                        if num is not None and num["HR"] is not None:
                            # store hf times vs raw numerics times: align by value sequence -> first finite raw sample
                            fin = np.isfinite(num["HR"]); first_raw = int(num["t_ms"][fin][0]) if fin.any() else None
                            r[f"E3_{tag}_hf_first_minus_raw_first_h"] = None if first_raw is None else round((int(h["time_ms"].min()) - first_raw) / 3.6e6, 3)
                # the store's chart HR event times must be raw CHARTTIMEs (wall clock) — before and after
                if ev_hr.size and t_hr.size:
                    r[f"chart_times_in_raw_frac_{tag}"] = round(float(np.isin(ev_hr["time_ms"].astype(np.int64), t_hr).mean()), 3)
                # chart events themselves must be identical before/after (only the waveform side moved)
                if tag == "before":
                    ev_after = np.load(os.path.join(sd, "ehr_events.npy")); a_hr = ev_after[ev_after["var_id"] == 100]
                    common = np.intersect1d(ev_hr["time_ms"], a_hr["time_ms"])
                    r["chart_hr_times_unchanged_frac"] = round(common.size / max(1, ev_hr.size), 3)
            # --- E4 waveform (ECG) vs raw chart HR, before/after clock
            if n_ecg_done < args.n_ecg and t_hr.size >= 10 and os.path.exists(os.path.join(sd, "II120.npy")):
                tm_e, bpm = ecg_hr_series(e["entity"])
                if np.isfinite(bpm).sum() >= 60:
                    r["E4_after"] = best_of(lag_profile(tm_e, bpm, t_hr, v_hr, COARSE, tol_ms=300000, match_tol=3.0))
                    r["E4_before"] = best_of(lag_profile(tm_e - (int(tm_after[0]) - int(tm_before[0])), bpm, t_hr, v_hr, COARSE, tol_ms=300000, match_tol=3.0))
                    if num is not None and num["HR"] is not None:
                        r["E4_ecg_vs_raw_numerics"] = best_of(lag_profile(num["t_ms"], num["HR"], tm_e, bpm, [-10, -5, 0, 5, 10], tol_ms=60000, match_tol=3.0, min_pts=30))
                    n_ecg_done += 1
            r["status"] = "ok"
        except Exception as ex:  # noqa: BLE001
            r["status"] = f"error {type(ex).__name__}: {str(ex)[:120]}"
        rows.append(r)
        if k % 25 == 0 or k == len(ents):
            print(f"  [{k}/{len(ents)}] {time.time()-t0:.0f}s", flush=True)

    # ---------------- aggregate
    def col(key, sub):
        return [r[key][sub] for r in rows if r.get(key)]
    agg = {
        "n_entities": len(rows), "n_ok": sum(1 for r in rows if r.get("status") == "ok"),
        "store_shift_hours": {str(k): v for k, v in zip(*np.unique([r["store_shift_h"] for r in rows if "store_shift_h" in r], return_counts=True))},
        "E1_after_minus_base_h": {str(k): int(v) for k, v in zip(*np.unique(np.round([r["E1_after_minus_base_h"] for r in rows if "E1_after_minus_base_h" in r], 2), return_counts=True))},
        "E1_before_minus_base_h": {str(k): int(v) for k, v in zip(*np.unique(np.round([r["E1_before_minus_base_h"] for r in rows if "E1_before_minus_base_h" in r], 2), return_counts=True))},
        "E2_raw_chart_vs_raw_numerics": {"best_lag_by_match_min": summarize(col("E2_raw", "lag_match")), "best_lag_by_mae_min": summarize(col("E2_raw", "lag_mae")),
                                          "match_frac_at_best": summarize(col("E2_raw", "match")), "match_frac_at_0": summarize(col("E2_raw", "match_at0")),
                                          "fine_lag_min": summarize(col("E2_raw_fine", "lag_match"))},
        "E2_raw_chart_vs_numerics_on_LEGACY_base": {"best_lag_by_match_min": summarize(col("E2_raw_legacy_base", "lag_match")), "match_frac_at_0": summarize(col("E2_raw_legacy_base", "match_at0"))},
        "E3_store_after_chart_vs_hf": {"best_lag_by_match_min": summarize(col("E3_after", "lag_match")), "by_mae": summarize(col("E3_after", "lag_mae")), "match_frac_at_0": summarize(col("E3_after", "match_at0"))},
        "E3_store_before_chart_vs_hf": {"best_lag_by_match_min": summarize(col("E3_before", "lag_match")), "by_mae": summarize(col("E3_before", "lag_mae")), "match_frac_at_0": summarize(col("E3_before", "match_at0"))},
        "E3_hf_first_minus_raw_first_h": {"after": summarize([r.get("E3_after_hf_first_minus_raw_first_h") for r in rows]), "before": summarize([r.get("E3_before_hf_first_minus_raw_first_h") for r in rows])},
        "E4_ecg_vs_chart": {"after": summarize(col("E4_after", "lag_mae")), "before": summarize(col("E4_before", "lag_mae")),
                             "ecg_vs_raw_numerics_lag_min": summarize(col("E4_ecg_vs_raw_numerics", "lag_mae")), "n": len(col("E4_after", "lag_mae"))},
        "E5_nbp_exact_match_frac": {L: summarize([float(r["E5_nbp_match_by_lag"][L]) for r in rows if r.get("E5_nbp_match_by_lag") and L in r["E5_nbp_match_by_lag"]]) for L in ("0", "240", "300", "-240", "-300")},
        "chart_hr_times_unchanged_frac": summarize([r.get("chart_hr_times_unchanged_frac") for r in rows]),
        "chart_times_in_raw_frac": {"after": summarize([r.get("chart_times_in_raw_frac_after") for r in rows]), "before": summarize([r.get("chart_times_in_raw_frac_before") for r in rows])},
        "elapsed_sec": round(time.time() - t0, 1), "ran_at": time.strftime("%Y-%m-%d %H:%M"),
    }
    os.makedirs(args.out, exist_ok=True)
    json.dump({"aggregate": agg, "rows": rows}, open(os.path.join(args.out, "audit_alignment_raw.json"), "w"), indent=1, default=str)
    json.dump(examples, open(os.path.join(args.out, "audit_examples.json"), "w"))
    print(json.dumps(agg, indent=1, default=str))
    try:
        make_figure(rows, examples, args.out)
    except Exception as ex:  # noqa: BLE001
        print("figure failed:", ex)


def make_figure(rows, examples, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    bins = np.arange(-427.5, 430, 15)
    for ax, key, title in ((axes[0, 0], "E2_raw", "raw chart HR vs raw numerics HR (both naive wall clock)"),
                           (axes[0, 1], "E3_before", "store BEFORE fix: chart HR vs numerics sidecar"),
                           (axes[1, 0], "E3_after", "store AFTER fix: chart HR vs numerics sidecar")):
        v = [r[key]["lag_match"] for r in rows if r.get(key)]
        ax.hist(v, bins=bins, color="#4c72b0"); ax.set_title(f"{title}\nbest lag by exact-value matches, n={len(v)}", fontsize=9)
        ax.set_xlabel("lag (min): chart time + lag = monitor time"); ax.axvline(0, color="k", lw=0.8)
    ax = axes[1, 1]
    if examples:
        eid, ex = next(iter(examples.items()))
        t0 = min(ex["t_chart"]); tn = (np.array(ex["t_num"]) - t0) / 3.6e6; hn = np.array([np.nan if x is None else x for x in ex["hr_num"]])
        tc = (np.array(ex["t_chart"]) - t0) / 3.6e6
        ax.plot(tn, hn, lw=0.6, color="#888", label="raw numerics HR (raw wall clock)")
        ax.plot(tn + ex["shift_ms"] / 3.6e6, hn, lw=0.6, color="#dd8452", label="numerics on the pre-fix (legacy) clock")
        ax.scatter(tc, ex["hr_chart"], s=12, color="#c44e52", zorder=3, label="raw charted HR")
        ax.set_xlim(-1, 13); ax.set_xlabel("hours from first charted HR"); ax.set_ylabel("bpm"); ax.legend(fontsize=7); ax.set_title(f"example entity {eid}", fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(out, "audit_alignment_raw.png"), dpi=130)
    print("wrote figure")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["extract", "analyze"]); ap.add_argument("--out", default="/projects/mwang80/staging/mimic3_audit")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--n-ecg", type=int, default=60)
    args = ap.parse_args()
    (extract if args.mode == "extract" else analyze)(args)


if __name__ == "__main__":
    main()
