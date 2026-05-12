#!/usr/bin/env python3
"""
Post-stage: MIMIC-III demographics_extra.csv (augments demographics.csv).

Adds first-4-hour aggregates anchored at the first segment time_ms of each
admission, plus height/weight pulled from CHARTEVENTS / ADMISSIONS:

  height_cm          first non-null height value in [t0, t1], else encounter-wide first non-null
  weight_kg          first non-null weight value in [t0, t1], else encounter-wide first non-null
  avg_hr_4h          mean of var_id=100 in [t0, t1]
  avg_sbp_4h         mean of var_id=110 (line) if any, else var_id=104 (cuff)
  avg_dbp_4h         mean of var_id=111 (line) if any, else var_id=105 (cuff)
  avg_map_4h         mean of var_id=112 (line) if any, else var_id=106 (cuff)
  bp_source          "line" if any of SBP/DBP/MAP used arterial line, else "cuff", else ""
  admission_diagnosis  ADMISSIONS.DIAGNOSIS free text (admission-time)
  icd9_primary       re-emitted from existing demographics.csv for join convenience
  t0_ms, t1_ms       recording-anchored window actually used
  n_hr / n_sbp / n_dbp / n_map  raw event counts in window (post outlier filter)

t0 is time_ms[0] of the entity (first available recording segment in shifted MIMIC ms),
t1 = t0 + 4 hours.  Values outside the physio range are dropped before averaging.

CHARTEVENTS is huge (~35 GB).  We do a single polars scan, filtered to the
relevant ITEMIDs only, then group by HADM_ID.  Result is cached to a parquet
so reruns are fast.

Run:
  python post_demographics_extra.py --limit 5
  python post_demographics_extra.py
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
with open(REPO_ROOT / "workzone" / "configs" / "server_paths.yaml") as f:
    cfg = yaml.safe_load(f)

EHR_ROOT      = cfg["mimic3"]["raw_ehr_dir"]
OUT_ROOT      = Path(cfg["mimic3"]["output_dir"])
OUTPUTS_DIR   = REPO_ROOT / "workzone" / "outputs" / "mimic3"
HW_CACHE      = OUTPUTS_DIR / "post_demographics_height_weight_cache.parquet"
SUMMARY_JSON  = OUTPUTS_DIR / "post_demographics_extra_summary.json"

FOUR_HOURS_MS = 4 * 3600 * 1000

# Var IDs (see indices/var_registry.json)
VAR_HR = 100
VAR_NBP_S, VAR_NBP_D, VAR_NBP_M = 104, 105, 106
VAR_ABP_S, VAR_ABP_D, VAR_ABP_M = 110, 111, 112

PHYSIO = {
    VAR_HR:    (10, 300),
    VAR_NBP_S: (30, 300), VAR_NBP_D: (10, 200), VAR_NBP_M: (20, 250),
    VAR_ABP_S: (40, 300), VAR_ABP_D: (20, 200), VAR_ABP_M: (30, 250),
}
HEIGHT_CM_RANGE = (50.0, 250.0)
WEIGHT_KG_RANGE = (1.0, 400.0)

# MIMIC ITEMIDs for height/weight (D_ITEMS confirmed).
# Values come back in the units listed; we convert to kg / cm.
WEIGHT_KG_ITEMIDS    = {762, 763, 3580, 3693, 224639, 226512}
WEIGHT_LB_ITEMIDS    = {3581, 226531}
HEIGHT_INCH_ITEMIDS  = {920, 1394, 4187, 3486, 226707}
HEIGHT_CM_ITEMIDS    = {3485, 4188, 226730}

ALL_HW_ITEMIDS = (WEIGHT_KG_ITEMIDS | WEIGHT_LB_ITEMIDS
                  | HEIGHT_INCH_ITEMIDS | HEIGHT_CM_ITEMIDS)

LB_TO_KG = 0.453592
INCH_TO_CM = 2.54

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _load_ehr(name: str, **kw) -> pd.DataFrame:
    p = os.path.join(EHR_ROOT, f"{name}.csv.gz")
    if not os.path.exists(p):
        p = os.path.join(EHR_ROOT, f"{name}.csv")
    return pd.read_csv(p, **kw)


def _mean_in_range(values: np.ndarray, lo: float, hi: float) -> tuple[float, int]:
    if values.size == 0:
        return float("nan"), 0
    m = np.isfinite(values) & (values >= lo) & (values <= hi)
    n = int(m.sum())
    if n == 0:
        return float("nan"), 0
    return float(values[m].mean()), n


def build_height_weight_lookup(hadm_ids: list[int],
                               t0_map: dict[int, int]) -> dict[int, tuple[float, float, float, float]]:
    """Scan CHARTEVENTS once. Return {hadm_id: (h_window, w_window, h_enc, w_enc)} in (cm, kg)."""
    if HW_CACHE.exists():
        log.info(f"loading cached height/weight from {HW_CACHE}")
        df = pl.read_parquet(HW_CACHE)
    else:
        # Prefer .csv.gz if uncompressed CHARTEVENTS.csv doesn't exist
        ce_csv = os.path.join(EHR_ROOT, "CHARTEVENTS.csv")
        ce_gz  = os.path.join(EHR_ROOT, "CHARTEVENTS.csv.gz")
        src = ce_csv if os.path.exists(ce_csv) else ce_gz
        log.info(f"scanning {src} (this is slow — ~35 GB)")
        t_scan = time.time()
        # Use scan_csv if uncompressed; for .gz, fall back to streaming pandas chunks.
        if src.endswith(".csv"):
            lf = pl.scan_csv(src, low_memory=True,
                             infer_schema_length=10000, ignore_errors=True)
            df = (
                lf.filter(pl.col("ITEMID").cast(pl.Int64, strict=False).is_in(list(ALL_HW_ITEMIDS)))
                  .filter(pl.col("HADM_ID").cast(pl.Int64, strict=False).is_in(hadm_ids))
                  .select([
                      pl.col("HADM_ID").cast(pl.Int64, strict=False).alias("hadm_id"),
                      pl.col("ITEMID").cast(pl.Int64, strict=False).alias("itemid"),
                      pl.col("CHARTTIME").str.strptime(pl.Datetime, strict=False).dt.timestamp("ms").alias("charttime_ms"),
                      pl.col("VALUENUM").cast(pl.Float64, strict=False).alias("valuenum"),
                  ])
                  .filter(pl.col("hadm_id").is_not_null()
                          & pl.col("charttime_ms").is_not_null()
                          & pl.col("valuenum").is_not_null())
                  .collect(engine="streaming")
            )
        else:
            log.info("  gzip source — streaming via pandas")
            chunks = []
            hadm_set = set(hadm_ids)
            for ch in pd.read_csv(src, usecols=["HADM_ID","ITEMID","CHARTTIME","VALUENUM"],
                                  chunksize=2_000_000, dtype={"HADM_ID":"Int64","ITEMID":"Int64","VALUENUM":"Float64"}):
                ch = ch[ch["ITEMID"].isin(ALL_HW_ITEMIDS) & ch["HADM_ID"].isin(hadm_set)]
                if not ch.empty:
                    ch["charttime_ms"] = pd.to_datetime(ch["CHARTTIME"], errors="coerce").astype("int64") // 1_000_000
                    chunks.append(ch[["HADM_ID","ITEMID","charttime_ms","VALUENUM"]]
                                  .rename(columns={"HADM_ID":"hadm_id","ITEMID":"itemid","VALUENUM":"valuenum"}))
            if chunks:
                pdf = pd.concat(chunks, ignore_index=True).dropna(subset=["hadm_id","charttime_ms","valuenum"])
                df = pl.from_pandas(pdf)
            else:
                df = pl.DataFrame(schema={"hadm_id": pl.Int64, "itemid": pl.Int64,
                                          "charttime_ms": pl.Int64, "valuenum": pl.Float64})
        log.info(f"  {df.height} rows after filter in {time.time() - t_scan:.1f}s")
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        df.write_parquet(HW_CACHE)
        log.info(f"  cached to {HW_CACHE}")

    # Normalize: convert all weight to kg, all height to cm.
    df = df.with_columns([
        pl.when(pl.col("itemid").is_in(list(WEIGHT_LB_ITEMIDS)))
          .then(pl.col("valuenum") * LB_TO_KG)
          .when(pl.col("itemid").is_in(list(WEIGHT_KG_ITEMIDS)))
          .then(pl.col("valuenum"))
          .otherwise(None).alias("weight_kg"),
        pl.when(pl.col("itemid").is_in(list(HEIGHT_INCH_ITEMIDS)))
          .then(pl.col("valuenum") * INCH_TO_CM)
          .when(pl.col("itemid").is_in(list(HEIGHT_CM_ITEMIDS)))
          .then(pl.col("valuenum"))
          .otherwise(None).alias("height_cm"),
    ])
    h_lo, h_hi = HEIGHT_CM_RANGE
    w_lo, w_hi = WEIGHT_KG_RANGE
    df = df.with_columns([
        pl.when((pl.col("height_cm") >= h_lo) & (pl.col("height_cm") <= h_hi))
          .then(pl.col("height_cm")).otherwise(None).alias("height_cm"),
        pl.when((pl.col("weight_kg") >= w_lo) & (pl.col("weight_kg") <= w_hi))
          .then(pl.col("weight_kg")).otherwise(None).alias("weight_kg"),
    ]).sort(["hadm_id", "charttime_ms"])

    t0_df = pl.DataFrame({
        "hadm_id": list(t0_map.keys()),
        "t0_ms":   list(t0_map.values()),
    }, schema={"hadm_id": pl.Int64, "t0_ms": pl.Int64})
    df = df.join(t0_df, on="hadm_id", how="left")

    in_win = df.filter(
        pl.col("t0_ms").is_not_null()
        & (pl.col("charttime_ms") >= pl.col("t0_ms"))
        & (pl.col("charttime_ms") < pl.col("t0_ms") + FOUR_HOURS_MS)
    )
    win_h = (in_win.filter(pl.col("height_cm").is_not_null())
                   .group_by("hadm_id").agg(pl.col("height_cm").first().alias("h_window")))
    win_w = (in_win.filter(pl.col("weight_kg").is_not_null())
                   .group_by("hadm_id").agg(pl.col("weight_kg").first().alias("w_window")))
    enc_h = (df.filter(pl.col("height_cm").is_not_null())
               .group_by("hadm_id").agg(pl.col("height_cm").first().alias("h_enc")))
    enc_w = (df.filter(pl.col("weight_kg").is_not_null())
               .group_by("hadm_id").agg(pl.col("weight_kg").first().alias("w_enc")))

    merged = (
        t0_df.select("hadm_id")
        .join(win_h, on="hadm_id", how="left")
        .join(win_w, on="hadm_id", how="left")
        .join(enc_h, on="hadm_id", how="left")
        .join(enc_w, on="hadm_id", how="left")
    )
    out: dict[int, tuple[float, float, float, float]] = {}
    for r in merged.iter_rows(named=True):
        out[int(r["hadm_id"])] = (
            float(r["h_window"]) if r["h_window"] is not None else float("nan"),
            float(r["w_window"]) if r["w_window"] is not None else float("nan"),
            float(r["h_enc"])    if r["h_enc"]    is not None else float("nan"),
            float(r["w_enc"])    if r["w_enc"]    is not None else float("nan"),
        )
    return out


def _aggregate_one(patient_id: str, out_root: Path,
                   hw_lookup: dict[int, tuple[float, float, float, float]]) -> dict | None:
    edir = out_root / patient_id
    tpath, epath = edir / "time_ms.npy", edir / "ehr_events.npy"
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
        win = events

    def _mean_var(var_id: int) -> tuple[float, int]:
        sel = win[win["var_id"] == var_id]["value"]
        lo, hi = PHYSIO[var_id]
        return _mean_in_range(np.asarray(sel, dtype=np.float64), lo, hi)

    hr_mean, n_hr = _mean_var(VAR_HR)
    abp_s, n_abp_s = _mean_var(VAR_ABP_S); abp_d, n_abp_d = _mean_var(VAR_ABP_D); abp_m, n_abp_m = _mean_var(VAR_ABP_M)
    nbp_s, n_nbp_s = _mean_var(VAR_NBP_S); nbp_d, n_nbp_d = _mean_var(VAR_NBP_D); nbp_m, n_nbp_m = _mean_var(VAR_NBP_M)

    sbp_mean, n_sbp, sbp_src = (abp_s, n_abp_s, "line") if n_abp_s else (nbp_s, n_nbp_s, "cuff" if n_nbp_s else "")
    dbp_mean, n_dbp, dbp_src = (abp_d, n_abp_d, "line") if n_abp_d else (nbp_d, n_nbp_d, "cuff" if n_nbp_d else "")
    map_mean, n_map, map_src = (abp_m, n_abp_m, "line") if n_abp_m else (nbp_m, n_nbp_m, "cuff" if n_nbp_m else "")
    sources = {s for s in (sbp_src, dbp_src, map_src) if s}
    bp_source = "line" if "line" in sources else ("cuff" if "cuff" in sources else "")

    try:
        hadm = int(patient_id.split("_", 1)[1])
    except (IndexError, ValueError):
        hadm = None
    h_w, w_w, h_e, w_e = (float("nan"),) * 4
    if hadm is not None and hadm in hw_lookup:
        h_w, w_w, h_e, w_e = hw_lookup[hadm]
    height_cm = h_w if np.isfinite(h_w) else h_e
    weight_kg = w_w if np.isfinite(w_w) else w_e

    return {
        "patient_id":  patient_id,
        "height_cm":   round(float(height_cm), 1) if np.isfinite(height_cm) else "",
        "weight_kg":   round(float(weight_kg), 2) if np.isfinite(weight_kg) else "",
        "avg_hr_4h":   round(hr_mean, 1)   if np.isfinite(hr_mean)  else "",
        "avg_sbp_4h":  round(sbp_mean, 1)  if np.isfinite(sbp_mean) else "",
        "avg_dbp_4h":  round(dbp_mean, 1)  if np.isfinite(dbp_mean) else "",
        "avg_map_4h":  round(map_mean, 1)  if np.isfinite(map_mean) else "",
        "bp_source":   bp_source,
        "t0_ms":       t0,
        "t1_ms":       t1,
        "n_hr":        n_hr,
        "n_sbp":       n_sbp,
        "n_dbp":       n_dbp,
        "n_map":       n_map,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    t_start = time.time()

    # 1. Patient universe (= directories with the required files)
    patient_ids: list[str] = []
    for d in sorted(OUT_ROOT.iterdir()):
        if not d.is_dir(): continue
        if not all((d / x).exists() for x in ("meta.json", "ehr_events.npy", "time_ms.npy")):
            continue
        patient_ids.append(d.name)
    if args.limit > 0:
        patient_ids = patient_ids[:args.limit]
    log.info(f"patients to process: {len(patient_ids)}")

    # 2. t0 per patient
    t0_map: dict[int, int] = {}
    for pid in patient_ids:
        try:
            tm = np.load(OUT_ROOT / pid / "time_ms.npy", mmap_mode="r")
            if tm.size:
                hadm = int(pid.split("_", 1)[1])
                t0_map[hadm] = int(tm[0])
        except Exception as e:
            log.warning(f"  could not read t0 for {pid}: {e}")

    # 3. Admission free-text + primary ICD9 (admission-time diagnosis)
    adm = _load_ehr("ADMISSIONS", usecols=["SUBJECT_ID","HADM_ID","DIAGNOSIS"])
    adm_lookup = {int(h): str(d) if pd.notna(d) else ""
                  for h, d in zip(adm["HADM_ID"], adm["DIAGNOSIS"])}
    diag = _load_ehr("DIAGNOSES_ICD", usecols=["SUBJECT_ID","HADM_ID","SEQ_NUM","ICD9_CODE"])
    diag = diag[diag["SEQ_NUM"] == 1]
    icd_lookup = {int(h): str(c) if pd.notna(c) else ""
                  for h, c in zip(diag["HADM_ID"], diag["ICD9_CODE"])}

    # 4. Height/weight via CHARTEVENTS scan
    hw_lookup = build_height_weight_lookup(list(t0_map.keys()), t0_map)
    log.info(f"height/weight lookup hadm_ids: {len(hw_lookup)}")

    # 5. Per-entity aggregation
    rows = []
    n_err = 0
    for i, pid in enumerate(patient_ids, 1):
        try:
            r = _aggregate_one(pid, OUT_ROOT, hw_lookup)
            if r is None: continue
            try:
                hadm = int(pid.split("_", 1)[1])
            except (IndexError, ValueError):
                hadm = None
            r["admission_diagnosis"] = adm_lookup.get(hadm, "") if hadm is not None else ""
            r["icd9_primary"]        = icd_lookup.get(hadm, "") if hadm is not None else ""
            rows.append(r)
        except Exception as e:
            n_err += 1
            log.warning(f"  {pid}: {e}")
        if i % 500 == 0:
            log.info(f"  processed {i}/{len(patient_ids)}")

    # 6. Emit demographics_extra.csv
    out_csv = OUT_ROOT / "demographics_extra.csv"
    cols = ["patient_id", "height_cm", "weight_kg",
            "avg_hr_4h", "avg_sbp_4h", "avg_dbp_4h", "avg_map_4h", "bp_source",
            "admission_diagnosis", "icd9_primary",
            "t0_ms", "t1_ms",
            "n_hr", "n_sbp", "n_dbp", "n_map"]
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c, "") for c in cols])

    elapsed = time.time() - t_start
    summary = {
        "stage":                "post_demographics_extra",
        "ran_at_unix":          int(time.time()),
        "elapsed_sec":          round(elapsed, 1),
        "n_patients_processed": len(patient_ids),
        "n_rows_written":       len(rows),
        "n_errors":             n_err,
        "n_with_height_cm":     sum(1 for r in rows if r["height_cm"] != ""),
        "n_with_weight_kg":     sum(1 for r in rows if r["weight_kg"] != ""),
        "n_with_hr":            sum(1 for r in rows if r["avg_hr_4h"]  != ""),
        "n_with_sbp":           sum(1 for r in rows if r["avg_sbp_4h"] != ""),
        "n_bp_source_line":     sum(1 for r in rows if r["bp_source"]  == "line"),
        "n_bp_source_cuff":     sum(1 for r in rows if r["bp_source"]  == "cuff"),
        "output_csv":           str(out_csv),
    }
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"wrote {out_csv}  rows={len(rows)}  elapsed={elapsed:.1f}s")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
