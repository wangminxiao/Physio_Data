#!/usr/bin/env python3
"""
Stage F-3 - MOVER/EPIC demographics_4h.csv (augments demographics.csv).

Adds first-4-hour aggregates anchored at the first segment time_ms of each
entity.  Height/weight come straight from the cohort parquet (Stage A already
converts EPIC's feet-inches HEIGHT -> cm and ounces WEIGHT -> kg), so unlike
the MIMIC / Emory F-3 there is NO height/weight EHR scan here.

  height_cm      patient_information HEIGHT, converted to cm by Stage A
  weight_kg      patient_information WEIGHT (oz), converted to kg by Stage A
  avg_hr_4h      mean of var_id=100 in [t0, t1]
  avg_sbp_4h     mean of var_id=104 in [t0, t1]   (EPIC: always blank - BP is
  avg_dbp_4h     mean of var_id=105 in [t0, t1]    stored as "120/80" strings
                                                    and not parsed upstream)
  avg_map_4h     mean of var_id=106 (cuff MAP) in [t0, t1]
  bp_source      "cuff" if MAP present (EPIC has no arterial-line BP), else ""
  avg_spo2_4h    mean of var_id=101 in [t0, t1]
  avg_resp_4h    mean of var_id=102 in [t0, t1]
  avg_etco2_4h   mean of var_id=116 in [t0, t1]
  procedure      PRIMARY_PROCEDURE_NM (re-emitted from cohort parquet)
  anes_type      PRIMARY_ANES_TYPE_NM
  asa_rating     ASA_RATING
  t0_ms, t1_ms   recording-anchored window actually used
  n_*            raw event counts in window (post outlier filter)

t0 = time_ms[0] of the entity (first recording segment, UTC ms); t1 = t0 + 4 h.
Values outside the physiological range are dropped before averaging.

The shared core columns (entity_id, height_cm, weight_kg, avg_hr_4h,
avg_sbp_4h, avg_dbp_4h, avg_map_4h, bp_source, t0_ms, t1_ms, n_hr, n_sbp,
n_dbp, n_map) match Emory / MIMIC demographics_4h.csv so a cross-dataset
loader can read all three with one schema; EPIC adds SpO2 / Resp / ETCO2.

Run:
  python stage_f_demographics_extra.py --limit 5      # smoke
  python stage_f_demographics_extra.py                # full
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path

import numpy as np
import polars as pl

OUT_ROOT = "/opt/localdata100tb/physio_data/mover_epic"
COHORT_PARQUET = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/mover_epic/valid_cohort.parquet"
SUMMARY_JSON = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/mover_epic/stage_f_demographics_4h_summary.json"

FOUR_HOURS_MS = 4 * 3600 * 1000

# Var IDs (see indices/var_registry.json)
VAR_HR, VAR_SPO2, VAR_RESP, VAR_ETCO2 = 100, 101, 102, 116
VAR_NBP_S, VAR_NBP_D, VAR_NBP_M = 104, 105, 106  # EPIC: only MAP (106) ever present

# Physiological ranges (mirrors var_registry; values outside are dropped).
PHYSIO = {
    VAR_HR:    (10, 300),
    VAR_NBP_S: (30, 300), VAR_NBP_D: (10, 200), VAR_NBP_M: (20, 250),
    VAR_SPO2:  (50, 100),
    VAR_RESP:  (0, 80),
    VAR_ETCO2: (0, 100),
}

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _mean_in_range(values: np.ndarray, lo: float, hi: float) -> tuple[float, int]:
    if values.size == 0:
        return float("nan"), 0
    m = np.isfinite(values) & (values >= lo) & (values <= hi)
    n = int(m.sum())
    if n == 0:
        return float("nan"), 0
    return float(values[m].mean()), n


def _aggregate_one(entity_id: str, out_root: Path,
                   demo_lookup: dict[str, dict]) -> dict | None:
    """First-4h aggregates + carried demographics for one entity (None if missing)."""
    edir = out_root / entity_id
    tpath, epath = edir / "time_ms.npy", edir / "ehr_events.npy"
    if not tpath.exists() or not epath.exists():
        return None
    time_ms = np.load(tpath)
    if time_ms.size == 0:
        return None
    t0 = int(time_ms[0])
    t1 = t0 + FOUR_HOURS_MS

    events = np.load(epath)
    win = events[(events["time_ms"] >= t0) & (events["time_ms"] < t1)] if events.size else events

    def _mean_var(var_id: int) -> tuple[float, int]:
        sel = win[win["var_id"] == var_id]["value"]
        lo, hi = PHYSIO[var_id]
        return _mean_in_range(np.asarray(sel, dtype=np.float64), lo, hi)

    hr_mean,   n_hr   = _mean_var(VAR_HR)
    sbp_mean,  n_sbp  = _mean_var(VAR_NBP_S)
    dbp_mean,  n_dbp  = _mean_var(VAR_NBP_D)
    map_mean,  n_map  = _mean_var(VAR_NBP_M)
    spo2_mean, n_spo2 = _mean_var(VAR_SPO2)
    resp_mean, n_resp = _mean_var(VAR_RESP)
    etco2_mean, n_etco2 = _mean_var(VAR_ETCO2)

    # EPIC has only cuff measurements (no arterial line).
    bp_source = "cuff" if (n_sbp or n_dbp or n_map) else ""

    d = demo_lookup.get(entity_id, {})
    return {
        "entity_id":   entity_id,
        "height_cm":   d.get("height_cm", ""),
        "weight_kg":   d.get("weight_kg", ""),
        "avg_hr_4h":   round(hr_mean, 1)    if np.isfinite(hr_mean)    else "",
        "avg_sbp_4h":  round(sbp_mean, 1)   if np.isfinite(sbp_mean)   else "",
        "avg_dbp_4h":  round(dbp_mean, 1)   if np.isfinite(dbp_mean)   else "",
        "avg_map_4h":  round(map_mean, 1)   if np.isfinite(map_mean)   else "",
        "bp_source":   bp_source,
        "avg_spo2_4h": round(spo2_mean, 1)  if np.isfinite(spo2_mean)  else "",
        "avg_resp_4h": round(resp_mean, 1)  if np.isfinite(resp_mean)  else "",
        "avg_etco2_4h": round(etco2_mean, 1) if np.isfinite(etco2_mean) else "",
        "procedure":   d.get("procedure", ""),
        "anes_type":   d.get("anes_type", ""),
        "asa_rating":  d.get("asa_rating", ""),
        "t0_ms":       t0,
        "t1_ms":       t1,
        "n_hr":        n_hr,
        "n_sbp":       n_sbp,
        "n_dbp":       n_dbp,
        "n_map":       n_map,
        "n_spo2":      n_spo2,
        "n_resp":      n_resp,
        "n_etco2":     n_etco2,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=OUT_ROOT)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    t_start = time.time()
    out_root = Path(args.out_root)

    # 1. Entity universe (dirs with the required files).
    all_entity_ids = [
        d.name for d in sorted(out_root.iterdir())
        if d.is_dir() and (d / "time_ms.npy").exists() and (d / "ehr_events.npy").exists()
    ]
    entity_ids = all_entity_ids[:args.limit] if args.limit > 0 else all_entity_ids
    log.info(f"entities total={len(all_entity_ids)} to_aggregate={len(entity_ids)}")

    # 2. Demographics lookup from the cohort parquet (height/weight already
    #    converted to cm/kg by Stage A; procedure/anes_type/asa re-emitted).
    cohort = (pl.read_parquet(COHORT_PARQUET)
                .unique("entity_id", keep="first")
                .filter(pl.col("entity_id").is_in(all_entity_ids)))
    demo_lookup: dict[str, dict] = {}
    for r in cohort.iter_rows(named=True):
        demo_lookup[r["entity_id"]] = {
            "height_cm": round(float(r["height_cm"]), 1) if r.get("height_cm") is not None else "",
            "weight_kg": round(float(r["weight_kg"]), 2) if r.get("weight_kg") is not None else "",
            "procedure": r.get("procedure") or "",
            "anes_type": r.get("anes_type") or "",
            "asa_rating": r.get("asa_rating") or "",
        }
    log.info(f"cohort rows matched: {cohort.height}")

    # 3. Per-entity aggregation (sequential — small per-entity NPY reads).
    rows, n_err = [], 0
    for i, eid in enumerate(entity_ids, 1):
        try:
            r = _aggregate_one(eid, out_root, demo_lookup)
            if r is not None:
                rows.append(r)
        except Exception as e:
            n_err += 1
            log.warning(f"  {eid}: {e}")
        if i % 500 == 0:
            log.info(f"  processed {i}/{len(entity_ids)}")

    # 4. Emit demographics_4h.csv
    out_csv = out_root / "demographics_4h.csv"
    cols = ["entity_id", "height_cm", "weight_kg",
            "avg_hr_4h", "avg_sbp_4h", "avg_dbp_4h", "avg_map_4h", "bp_source",
            "avg_spo2_4h", "avg_resp_4h", "avg_etco2_4h",
            "procedure", "anes_type", "asa_rating",
            "t0_ms", "t1_ms",
            "n_hr", "n_sbp", "n_dbp", "n_map", "n_spo2", "n_resp", "n_etco2"]
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c, "") for c in cols])

    elapsed = time.time() - t_start
    summary = {
        "stage":                "f_demographics_4h",
        "ran_at_unix":          int(time.time()),
        "elapsed_sec":          round(elapsed, 1),
        "n_entities_processed": len(entity_ids),
        "n_rows_written":       len(rows),
        "n_errors":             n_err,
        "n_with_height_cm":     sum(1 for r in rows if r["height_cm"] != ""),
        "n_with_weight_kg":     sum(1 for r in rows if r["weight_kg"] != ""),
        "n_with_hr":            sum(1 for r in rows if r["avg_hr_4h"]   != ""),
        "n_with_map":           sum(1 for r in rows if r["avg_map_4h"]  != ""),
        "n_with_spo2":          sum(1 for r in rows if r["avg_spo2_4h"] != ""),
        "n_with_etco2":         sum(1 for r in rows if r["avg_etco2_4h"] != ""),
        "n_bp_source_cuff":     sum(1 for r in rows if r["bp_source"]   == "cuff"),
        "output_csv":           str(out_csv),
    }
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"wrote {out_csv}  rows={len(rows)}  elapsed={elapsed:.1f}s")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
