#!/usr/bin/env python3
"""
Stage F-3 — Emory demographics_4h.csv (augments demographics.csv).

Adds first-4-hour aggregates anchored at the first segment time_ms of each
entity, plus height/weight pulled from JGSEPSIS_VITALS2:

  height_cm          first non-null HEIGHT_CM in [t0, t1], else encounter-wide first non-null
  weight_kg          first non-null DAILY_WEIGHT_KG in [t0, t1], else encounter-wide first non-null
  avg_hr_4h          mean of var_id=100 in [t0, t1]
  avg_sbp_4h         mean of var_id=110 (line) if any, else var_id=104 (cuff)
  avg_dbp_4h         mean of var_id=111 (line) if any, else var_id=105 (cuff)
  avg_map_4h         mean of var_id=112 (line) if any, else var_id=106 (cuff)
  bp_source          "line" if any of SBP/DBP/MAP used arterial line, else "cuff", else ""
  admit_dx_icd10     re-emitted from cohort parquet for join convenience
  t0_ms, t1_ms       recording-anchored window actually used
  n_hr / n_sbp / n_dbp / n_map  raw event counts in window (post outlier filter)

t0 is time_ms[0] of the entity (first available recording segment in UTC ms),
t1 = t0 + 4 hours.  Values outside the physio range (see indices/var_registry.json)
are dropped before averaging.

Run:
  python stage_f_demographics_extra.py --limit 5            # smoke
  python stage_f_demographics_extra.py --workers 24         # full
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

EHR_ROOT = "/labs/hulab/Emory_EHR/CDW_Pull_ICU_Data_Siva_version"
VITALS2_CSV = f"{EHR_ROOT}/JGSEPSIS_VITALS2.csv"

OUT_ROOT = "/opt/localdata100tb/physio_data/emory"
COHORT_PARQUET = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/emory/valid_wave_window.parquet"
OUTPUTS_DIR = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/emory"
HW_CACHE_PARQUET = f"{OUTPUTS_DIR}/stage_f_height_weight_cache.parquet"
SUMMARY_JSON = f"{OUTPUTS_DIR}/stage_f_demographics_4h_summary.json"

NY_TZ = "America/New_York"
FOUR_HOURS_MS = 4 * 3600 * 1000

# Var IDs (see indices/var_registry.json)
VAR_HR = 100
VAR_NBP_S, VAR_NBP_D, VAR_NBP_M = 104, 105, 106
VAR_ABP_S, VAR_ABP_D, VAR_ABP_M = 110, 111, 112

# Physiological ranges (mirrors var_registry).
PHYSIO = {
    VAR_HR:    (10, 300),
    VAR_NBP_S: (30, 300), VAR_NBP_D: (10, 200), VAR_NBP_M: (20, 250),
    VAR_ABP_S: (40, 300), VAR_ABP_D: (20, 200), VAR_ABP_M: (30, 250),
}
# Plausible ranges for height/weight (anthropometric outlier guards).
HEIGHT_CM_RANGE = (50.0, 250.0)
WEIGHT_KG_RANGE = (1.0, 400.0)

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
                   hw_lookup: dict[int, tuple[float, float, float, float]]) -> dict | None:
    """Return one dict of extra demographics for `entity_id` (or None if missing files)."""
    edir = out_root / entity_id
    tpath = edir / "time_ms.npy"
    epath = edir / "ehr_events.npy"
    if not tpath.exists() or not epath.exists():
        return None
    time_ms = np.load(tpath)
    if time_ms.size == 0:
        return None
    t0 = int(time_ms[0])
    t1 = t0 + FOUR_HOURS_MS

    events = np.load(epath)
    if events.size:
        mask = (events["time_ms"] >= t0) & (events["time_ms"] < t1)
        win = events[mask]
    else:
        win = events  # empty

    def _mean_var(var_id: int) -> tuple[float, int]:
        sel = win[win["var_id"] == var_id]["value"]
        lo, hi = PHYSIO[var_id]
        return _mean_in_range(np.asarray(sel, dtype=np.float64), lo, hi)

    hr_mean,   n_hr   = _mean_var(VAR_HR)

    abp_s_mean, n_abp_s = _mean_var(VAR_ABP_S)
    abp_d_mean, n_abp_d = _mean_var(VAR_ABP_D)
    abp_m_mean, n_abp_m = _mean_var(VAR_ABP_M)
    nbp_s_mean, n_nbp_s = _mean_var(VAR_NBP_S)
    nbp_d_mean, n_nbp_d = _mean_var(VAR_NBP_D)
    nbp_m_mean, n_nbp_m = _mean_var(VAR_NBP_M)

    # Prefer arterial line; fall back to cuff per BP component.
    sbp_mean, n_sbp, sbp_src = (abp_s_mean, n_abp_s, "line") if n_abp_s else (nbp_s_mean, n_nbp_s, "cuff" if n_nbp_s else "")
    dbp_mean, n_dbp, dbp_src = (abp_d_mean, n_abp_d, "line") if n_abp_d else (nbp_d_mean, n_nbp_d, "cuff" if n_nbp_d else "")
    map_mean, n_map, map_src = (abp_m_mean, n_abp_m, "line") if n_abp_m else (nbp_m_mean, n_nbp_m, "cuff" if n_nbp_m else "")
    sources = {s for s in (sbp_src, dbp_src, map_src) if s}
    bp_source = "line" if "line" in sources else ("cuff" if "cuff" in sources else "")

    # entity_id is "{empi_nbr}_{encounter_nbr}". Look up height/weight by encounter_nbr.
    try:
        enc_nbr = int(entity_id.split("_", 1)[1])
    except (IndexError, ValueError):
        enc_nbr = None
    h_window, w_window, h_enc, w_enc = (float("nan"),) * 4
    if enc_nbr is not None and enc_nbr in hw_lookup:
        h_window, w_window, h_enc, w_enc = hw_lookup[enc_nbr]
    # Prefer the in-window value; fall back to the encounter-wide first non-null.
    height_cm = h_window if np.isfinite(h_window) else h_enc
    weight_kg = w_window if np.isfinite(w_window) else w_enc

    return {
        "entity_id":     entity_id,
        "height_cm":     round(float(height_cm), 1) if np.isfinite(height_cm) else "",
        "weight_kg":     round(float(weight_kg), 2) if np.isfinite(weight_kg) else "",
        "avg_hr_4h":     round(hr_mean, 1) if np.isfinite(hr_mean) else "",
        "avg_sbp_4h":    round(sbp_mean, 1) if np.isfinite(sbp_mean) else "",
        "avg_dbp_4h":    round(dbp_mean, 1) if np.isfinite(dbp_mean) else "",
        "avg_map_4h":    round(map_mean, 1) if np.isfinite(map_mean) else "",
        "bp_source":     bp_source,
        "t0_ms":         t0,
        "t1_ms":         t1,
        "n_hr":          n_hr,
        "n_sbp":         n_sbp,
        "n_dbp":         n_dbp,
        "n_map":         n_map,
    }


def _ny_to_utc_ms_expr(col: str) -> pl.Expr:
    """Polars expression: parse Emory RECORDED_TIME (NY local) -> UTC ms int64."""
    return (
        pl.col(col)
        .str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M:%S", strict=False)
        .dt.replace_time_zone(NY_TZ, ambiguous="earliest", non_existent="null")
        .dt.convert_time_zone("UTC")
        .dt.timestamp("ms")
    )


def build_height_weight_lookup(cohort_df: pl.DataFrame,
                               t0_map: dict[int, int]) -> dict[int, tuple[float, float, float, float]]:
    """Scan VITALS2 once. Return {encounter_nbr: (h_window, w_window, h_enc, w_enc)}.

    h_window/w_window: first non-null value within [t0, t0 + 4 h]
    h_enc/w_enc:       first non-null value across the entire encounter (fallback)
    """
    if os.path.exists(HW_CACHE_PARQUET):
        log.info(f"loading cached height/weight from {HW_CACHE_PARQUET}")
        hw = pl.read_parquet(HW_CACHE_PARQUET)
    else:
        encs = cohort_df["encounter_nbr"].unique().to_list()
        log.info(f"scanning {VITALS2_CSV} for {len(encs)} encounters")
        t_scan = time.time()
        vt_lf = pl.scan_csv(VITALS2_CSV, low_memory=True,
                            infer_schema_length=10000, ignore_errors=True)
        hw = (
            vt_lf
            .filter(pl.col("ENCOUNTER_NBR").cast(pl.Int64, strict=False).is_in(encs))
            .select([
                pl.col("ENCOUNTER_NBR").cast(pl.Int64, strict=False).alias("encounter_nbr"),
                _ny_to_utc_ms_expr("RECORDED_TIME").alias("recorded_utc_ms"),
                pl.col("HEIGHT_CM").cast(pl.Float64, strict=False).alias("height_cm"),
                pl.col("DAILY_WEIGHT_KG").cast(pl.Float64, strict=False).alias("weight_kg"),
            ])
            .filter(pl.col("encounter_nbr").is_not_null() & pl.col("recorded_utc_ms").is_not_null())
            .collect(engine="streaming")
        )
        log.info(f"  {hw.height} VITALS2 rows scanned in {time.time() - t_scan:.1f}s")
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        hw.write_parquet(HW_CACHE_PARQUET)
        log.info(f"  cached to {HW_CACHE_PARQUET}")

    # Anthropometric outlier filter
    h_lo, h_hi = HEIGHT_CM_RANGE
    w_lo, w_hi = WEIGHT_KG_RANGE
    hw = hw.with_columns([
        pl.when((pl.col("height_cm") >= h_lo) & (pl.col("height_cm") <= h_hi))
          .then(pl.col("height_cm")).otherwise(None).alias("height_cm"),
        pl.when((pl.col("weight_kg") >= w_lo) & (pl.col("weight_kg") <= w_hi))
          .then(pl.col("weight_kg")).otherwise(None).alias("weight_kg"),
    ])

    # Sort by recorded time so "first" = earliest.
    hw = hw.sort(["encounter_nbr", "recorded_utc_ms"])

    # Build {encounter_nbr: t0_ms} as a polars df for join.
    t0_df = pl.DataFrame({
        "encounter_nbr": list(t0_map.keys()),
        "t0_ms":         list(t0_map.values()),
    }, schema={"encounter_nbr": pl.Int64, "t0_ms": pl.Int64})
    hw = hw.join(t0_df, on="encounter_nbr", how="left")

    # First non-null in-window value per encounter.
    in_win = hw.filter(
        pl.col("t0_ms").is_not_null()
        & (pl.col("recorded_utc_ms") >= pl.col("t0_ms"))
        & (pl.col("recorded_utc_ms") < pl.col("t0_ms") + FOUR_HOURS_MS)
    )
    win_height = (in_win.filter(pl.col("height_cm").is_not_null())
                        .group_by("encounter_nbr").agg(pl.col("height_cm").first().alias("h_window")))
    win_weight = (in_win.filter(pl.col("weight_kg").is_not_null())
                        .group_by("encounter_nbr").agg(pl.col("weight_kg").first().alias("w_window")))
    enc_height = (hw.filter(pl.col("height_cm").is_not_null())
                    .group_by("encounter_nbr").agg(pl.col("height_cm").first().alias("h_enc")))
    enc_weight = (hw.filter(pl.col("weight_kg").is_not_null())
                    .group_by("encounter_nbr").agg(pl.col("weight_kg").first().alias("w_enc")))

    merged = (
        t0_df.select("encounter_nbr")
        .join(win_height, on="encounter_nbr", how="left")
        .join(win_weight, on="encounter_nbr", how="left")
        .join(enc_height, on="encounter_nbr", how="left")
        .join(enc_weight, on="encounter_nbr", how="left")
    )
    out: dict[int, tuple[float, float, float, float]] = {}
    for r in merged.iter_rows(named=True):
        out[int(r["encounter_nbr"])] = (
            float(r["h_window"]) if r["h_window"] is not None else float("nan"),
            float(r["w_window"]) if r["w_window"] is not None else float("nan"),
            float(r["h_enc"])    if r["h_enc"]    is not None else float("nan"),
            float(r["w_enc"])    if r["w_enc"]    is not None else float("nan"),
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=str, default=OUT_ROOT)
    ap.add_argument("--limit",    type=int, default=0,
                    help="if >0, only process this many entities (smoke test)")
    args = ap.parse_args()

    t0 = time.time()
    out_root = Path(args.out_root)

    # 1. Entity universe: directories that have meta.json AND ehr_events.npy.
    #    ALL such entities are scanned for the height/weight cache; --limit
    #    only restricts which entities we aggregate + write rows for.
    all_entity_ids: list[str] = []
    for d in sorted(out_root.iterdir()):
        if not d.is_dir(): continue
        if not (d / "meta.json").exists(): continue
        if not (d / "ehr_events.npy").exists(): continue
        if not (d / "time_ms.npy").exists(): continue
        all_entity_ids.append(d.name)
    entity_ids = all_entity_ids[:args.limit] if args.limit > 0 else all_entity_ids
    log.info(f"entities total={len(all_entity_ids)} to_aggregate={len(entity_ids)}")

    # 2. Cohort lookup (admit_dx_icd10 re-emit)
    cohort = (
        pl.read_parquet(COHORT_PARQUET)
          .unique("entity_id", keep="first")
          .filter(pl.col("entity_id").is_in(all_entity_ids))
    )
    log.info(f"cohort rows matched: {cohort.height}")
    icd_lookup: dict[str, str] = {
        r["entity_id"]: (r.get("admit_dx_icd10") or "")
        for r in cohort.iter_rows(named=True)
    }

    # 3. t0 per entity (over ALL entities, so cache covers full cohort)
    t0_map: dict[int, int] = {}
    for eid in all_entity_ids:
        tp = out_root / eid / "time_ms.npy"
        try:
            tm = np.load(tp, mmap_mode="r")
            if tm.size:
                t0_val = int(tm[0])
                try:
                    enc = int(eid.split("_", 1)[1])
                    t0_map[enc] = t0_val
                except (IndexError, ValueError):
                    pass
        except Exception as e:
            log.warning(f"  could not read time_ms for {eid}: {e}")

    # 4. Height/weight via one streaming scan of VITALS2
    hw_lookup = build_height_weight_lookup(cohort, t0_map)
    log.info(f"height/weight lookup encounters: {len(hw_lookup)}")

    # 5. Per-entity aggregation (sequential — small per-entity work, NPY read-bound)
    rows = []
    n_err = 0
    for i, eid in enumerate(entity_ids, 1):
        try:
            r = _aggregate_one(eid, out_root, hw_lookup)
            if r is None:
                continue
            r["admit_dx_icd10"] = icd_lookup.get(eid, "")
            rows.append(r)
        except Exception as e:
            n_err += 1
            log.warning(f"  {eid}: {e}")
        if i % 500 == 0:
            log.info(f"  processed {i}/{len(entity_ids)}")

    # 6. Emit demographics_4h.csv
    out_csv = out_root / "demographics_4h.csv"
    cols = ["entity_id", "height_cm", "weight_kg",
            "avg_hr_4h", "avg_sbp_4h", "avg_dbp_4h", "avg_map_4h", "bp_source",
            "admit_dx_icd10",
            "t0_ms", "t1_ms",
            "n_hr", "n_sbp", "n_dbp", "n_map"]
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c, "") for c in cols])

    elapsed = time.time() - t0
    n_with_height  = sum(1 for r in rows if r["height_cm"] != "")
    n_with_weight  = sum(1 for r in rows if r["weight_kg"] != "")
    n_with_hr      = sum(1 for r in rows if r["avg_hr_4h"]  != "")
    n_with_sbp     = sum(1 for r in rows if r["avg_sbp_4h"] != "")
    n_bp_line      = sum(1 for r in rows if r["bp_source"]  == "line")
    n_bp_cuff      = sum(1 for r in rows if r["bp_source"]  == "cuff")

    summary = {
        "stage":                "f_demographics_4h",
        "ran_at_unix":          int(time.time()),
        "elapsed_sec":          round(elapsed, 1),
        "n_entities_processed": len(entity_ids),
        "n_rows_written":       len(rows),
        "n_errors":             n_err,
        "n_with_height_cm":     n_with_height,
        "n_with_weight_kg":     n_with_weight,
        "n_with_hr":            n_with_hr,
        "n_with_sbp":           n_with_sbp,
        "n_bp_source_line":     n_bp_line,
        "n_bp_source_cuff":     n_bp_cuff,
        "output_csv":           str(out_csv),
    }
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"wrote {out_csv}  rows={len(rows)}  elapsed={elapsed:.1f}s")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
