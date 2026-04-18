#!/usr/bin/env python3
"""
Stage D — Emory EHR events extraction.

Two independent phase-1 scans plus a shared phase-2 per-entity writer:

  Phase 1a (labs):
    Stream-scan JGSEPSIS_LABS.csv, filter ENCOUNTER_NBR to cohort,
    whitelist COMPONENT_ID → var_id 0..18, parse LAB_RESULT_TIME NY→UTC,
    cast LAB_RESULT to f64. Output:
      workzone/outputs/emory/stage_d_labs_combined.parquet

  Phase 1b (chart vitals):
    Stream-scan JGSEPSIS_VITALS2.csv, filter ENCOUNTER_NBR, parse
    RECORDED_TIME NY→UTC, melt wide columns → long (var_id 100..117).
    Output:
      workzone/outputs/emory/stage_d_chart_vitals_combined.parquet

  Phase 2 (per-entity):
    For each cohort entity with Stage B meta.json on disk:
      - labs:        write {entity}/labs_events.npy          (structured)
      - chart vitals: write {entity}/chart_vitals_events.npy  (structured)
      - update meta.json with `labs`, `chart_vitals` sections + stage_d_version=1

No episode clipping here — Stage E does the final partition using
HOSPITAL_ADMISSION/DISCHARGE_DATE_TIME.

Time convention: all EHR times are NY local → UTC with
`ambiguous='earliest'`, `non_existent='null'` (drops DST spring-forward holes).

Run modes:
  python stage_d_ehr.py --phase 1a --workers 1
  python stage_d_ehr.py --phase 1b --workers 1
  python stage_d_ehr.py --phase 2 --limit 50
  python stage_d_ehr.py                          # all phases, full cohort
"""
import os
import sys
import json
import time
import argparse
import logging
import traceback

import numpy as np
import polars as pl

EHR_ROOT = "/labs/hulab/Emory_EHR/CDW_Pull_ICU_Data_Siva_version"
LABS_CSV = f"{EHR_ROOT}/JGSEPSIS_LABS.csv"
VITALS2_CSV = f"{EHR_ROOT}/JGSEPSIS_VITALS2.csv"

OUT_ROOT = "/opt/localdata100tb/physio_data/emory"
COHORT_PARQUET = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/emory/valid_wave_window.parquet"
OUTPUTS_DIR = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/emory"
LABS_COMBINED = f"{OUTPUTS_DIR}/stage_d_labs_combined.parquet"
CHART_COMBINED = f"{OUTPUTS_DIR}/stage_d_chart_vitals_combined.parquet"
SUMMARY_JSON = f"{OUTPUTS_DIR}/stage_d_summary.json"
LOG_DIR = "/labs/hulab/mxwang/Physio_Data/workzone/emory/logs"

EVENT_DTYPE = np.dtype([
    ("time_ms", np.int64),
    ("var_id",  np.uint16),
    ("value",   np.float32),
])

# COMPONENT_ID → var_id (calibrated against real Emory JGSEPSIS_LABS; see
# datasets/emory/API.md §Labs). Whitelist approach (unambiguous).
LAB_COMPONENT_TO_VAR_ID = {
    # 0 Potassium
    1513627: 0, 1516154: 0, 665487134: 0, 1513616: 0,
    # 1 Calcium (total only — exclude Ionized)
    1514676: 1, 1516163: 1,
    # 2 Sodium
    1513370: 2, 1516135: 2, 665487126: 2, 1513368: 2,
    # 3 Glucose
    1513981: 3, 1516156: 3, 665487158: 3, 1513964: 3,
    # 4 Lactate (arterial or venous)
    1516153: 4, 1513613: 4, 665487166: 4, 1291848029: 4,
    # 5 Creatinine
    1514422: 5,
    # 6 Bilirubin Total
    1512680: 6,
    # 7 Platelet Count
    1513222: 7,
    # 8 White Blood Count
    1512214: 8,
    # 9 Hemoglobin (plain — not Meth/Carboxy/fetal variants)
    1513817: 9, 1513807: 9,
    # 10 INR
    1513646: 10,
    # 11 BUN
    1514698: 11,
    # 12 Albumin
    1515017: 12,
    # 13 pH (arterial)
    665490931: 13, 665490849: 13,
    # 14 paO2 (arterial)
    665486958: 14, 665486946: 14,
    # 15 paCO2 (arterial)
    665486922: 15, 665492721: 15,
    # 16 HCO3 (arterial)
    665486994: 16, 665486982: 16,
    # 17 AST (Aspartate Aminotransferase)
    1514885: 17,
    # 18 ALT (Alanine Aminotransferase)
    1514992: 18,
}

# Column → var_id for JGSEPSIS_VITALS2 (per API.md §Vitals chart table).
CHART_COL_TO_VAR_ID = {
    "PULSE":                100,
    "SPO2":                 101,
    "UNASSISTED_RESP_RATE": 102,
    "TEMPERATURE":          103,
    "SBP_CUFF":             104,
    "DBP_CUFF":             105,
    "MAP_CUFF":             106,
    "SBP_LINE":             110,
    "DBP_LINE":             111,
    "MAP_LINE":             112,
    "CVP":                  107,  # matches var_registry (UCSF uses CVP1-3 at 107)
    "END_TIDAL_CO2":        116,
    "O2_FLOW_RATE":         117,
}

NY_TZ = "America/New_York"
LOG = None


def _parse_ehr_dt(col_expr: pl.Expr) -> pl.Expr:
    """Parse naive EHR time string "MM/DD/YYYY HH:MM:SS" as NY-local → UTC ms.
    ambiguous='earliest' for DST fall-back; non_existent='null' drops
    DST spring-forward holes (e.g. 2010-03-14 02:57 seen in data)."""
    return (
        col_expr.cast(pl.Utf8)
        .str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M:%S", strict=False)
        .dt.replace_time_zone(NY_TZ, ambiguous="earliest", non_existent="null")
        .dt.convert_time_zone("UTC")
        .dt.timestamp("ms")
    )


def phase1_labs(cohort_encounters: list[int]) -> dict:
    LOG.info(f"Phase 1a: scan {LABS_CSV}")
    t0 = time.time()
    lf = pl.scan_csv(LABS_CSV, low_memory=True, infer_schema_length=10000,
                     ignore_errors=True)
    whitelist_ids = list(LAB_COMPONENT_TO_VAR_ID.keys())
    df = (
        lf.filter(pl.col("ENCOUNTER_NBR").cast(pl.Int64, strict=False).is_in(cohort_encounters))
          .filter(pl.col("COMPONENT_ID").cast(pl.Int64, strict=False).is_in(whitelist_ids))
          .select([
              pl.col("ENCOUNTER_NBR").cast(pl.Int64, strict=False).alias("encounter_nbr"),
              pl.col("COMPONENT_ID").cast(pl.Int64, strict=False).alias("component_id"),
              pl.col("LAB_RESULT_TIME"),
              pl.col("LAB_RESULT"),
          ])
          .with_columns([
              _parse_ehr_dt(pl.col("LAB_RESULT_TIME")).alias("time_ms"),
              pl.col("LAB_RESULT").cast(pl.Utf8).str.strip_chars()
                .str.strip_chars("<>=~%+ ")
                .cast(pl.Float64, strict=False).alias("value_f"),
              pl.col("component_id")
                .replace_strict(LAB_COMPONENT_TO_VAR_ID, return_dtype=pl.UInt16)
                .alias("var_id"),
          ])
          .filter(
              pl.col("time_ms").is_not_null()
              & pl.col("value_f").is_not_null()
              & pl.col("value_f").is_finite()
              & (pl.col("value_f") >= 0)
          )
          .select([
              "encounter_nbr",
              "time_ms",
              "var_id",
              pl.col("value_f").cast(pl.Float32).alias("value"),
          ])
    )
    out = df.collect(engine="streaming")
    out.write_parquet(LABS_COMBINED)
    info = {
        "rows": out.height,
        "elapsed_sec": round(time.time() - t0, 1),
        "unique_encounters": int(out["encounter_nbr"].n_unique()),
        "var_ids": sorted(int(v) for v in out["var_id"].unique()),
    }
    LOG.info(f"Phase 1a done: {info}")
    return info


def phase1_chart_vitals(cohort_encounters: list[int]) -> dict:
    LOG.info(f"Phase 1b: scan {VITALS2_CSV}")
    t0 = time.time()
    cols_needed = ["ENCOUNTER_NBR", "RECORDED_TIME"] + list(CHART_COL_TO_VAR_ID.keys())
    lf = pl.scan_csv(VITALS2_CSV, low_memory=True, infer_schema_length=10000,
                     ignore_errors=True)
    filtered = (
        lf.filter(pl.col("ENCOUNTER_NBR").cast(pl.Int64, strict=False).is_in(cohort_encounters))
          .select([
              pl.col("ENCOUNTER_NBR").cast(pl.Int64, strict=False).alias("encounter_nbr"),
              _parse_ehr_dt(pl.col("RECORDED_TIME")).alias("time_ms"),
              *[pl.col(c).cast(pl.Float64, strict=False) for c in CHART_COL_TO_VAR_ID],
          ])
          .filter(pl.col("time_ms").is_not_null())
          .collect(engine="streaming")
    )
    LOG.info(f"  filtered wide rows: {filtered.height:,}")

    # Melt wide → long. Each (encounter, time) row has up to len(cols) values.
    long = filtered.unpivot(
        index=["encounter_nbr", "time_ms"],
        on=list(CHART_COL_TO_VAR_ID.keys()),
        variable_name="col",
        value_name="value_f",
    ).filter(pl.col("value_f").is_not_null() & pl.col("value_f").is_finite())

    long = long.with_columns(
        pl.col("col").replace_strict(CHART_COL_TO_VAR_ID, return_dtype=pl.UInt16)
          .alias("var_id")
    )

    # Drop sentinel/sensor-error values — but preserve negatives where clinically
    # legitimate (none in our chart cols). Also drop exact 0 for SpO2 (API.md).
    long = long.filter(
        ~((pl.col("var_id") == 101) & (pl.col("value_f") == 0))
    ).filter(pl.col("value_f") >= 0)

    out = long.select([
        "encounter_nbr",
        "time_ms",
        "var_id",
        pl.col("value_f").cast(pl.Float32).alias("value"),
    ])
    out.write_parquet(CHART_COMBINED)
    info = {
        "rows": out.height,
        "elapsed_sec": round(time.time() - t0, 1),
        "unique_encounters": int(out["encounter_nbr"].n_unique()),
        "var_ids": sorted(int(v) for v in out["var_id"].unique()),
    }
    LOG.info(f"Phase 1b done: {info}")
    return info


def _write_events(out_dir: str, filename: str, sub: pl.DataFrame) -> np.ndarray:
    """Sort by time_ms and write structured npy. Returns the events array."""
    sub = sub.sort(["time_ms", "var_id"])
    events = np.empty(sub.height, dtype=EVENT_DTYPE)
    if sub.height > 0:
        events["time_ms"] = sub["time_ms"].to_numpy()
        events["var_id"] = sub["var_id"].to_numpy()
        events["value"] = sub["value"].to_numpy()
    np.save(f"{out_dir}/{filename}", events)
    return events


def phase2(entities_df: pl.DataFrame, out_root: str, resume: bool) -> list[dict]:
    LOG.info(f"Phase 2: reading combined parquets")
    labs = pl.read_parquet(LABS_COMBINED)
    chart = pl.read_parquet(CHART_COMBINED)
    LOG.info(f"  labs rows: {labs.height:,}  chart rows: {chart.height:,}")

    # Partition by encounter_nbr into dicts for O(1) lookup
    t0 = time.time()
    labs_by_enc = {int(k[0]) if isinstance(k, tuple) else int(k): g
                   for k, g in labs.partition_by("encounter_nbr",
                                                 as_dict=True).items()}
    chart_by_enc = {int(k[0]) if isinstance(k, tuple) else int(k): g
                    for k, g in chart.partition_by("encounter_nbr",
                                                   as_dict=True).items()}
    LOG.info(f"  partitioned: labs {len(labs_by_enc)} enc / chart {len(chart_by_enc)} enc "
             f"({time.time()-t0:.1f}s)")

    statuses: list[dict] = []
    n_total = entities_df.height
    done = 0
    t0 = time.time()

    for row in entities_df.iter_rows(named=True):
        eid = row["entity_id"]
        enc = int(row["encounter_nbr"])
        out_dir = f"{out_root}/{eid}"
        meta_path = f"{out_dir}/meta.json"
        labs_path = f"{out_dir}/labs_events.npy"
        chart_path = f"{out_dir}/chart_vitals_events.npy"
        status = {"entity_id": eid, "encounter_nbr": enc,
                  "status": "pending", "n_labs": 0, "n_chart": 0}
        try:
            if not os.path.exists(meta_path):
                status["status"] = "no_stage_b_meta"
                statuses.append(status); done += 1; continue

            if resume and os.path.exists(labs_path) and os.path.exists(chart_path):
                try:
                    with open(meta_path) as f:
                        m = json.load(f)
                    if m.get("stage_d_version", 0) >= 1:
                        status["status"] = "resumed"
                        status["n_labs"] = int(m.get("labs", {}).get("n_events", 0))
                        status["n_chart"] = int(m.get("chart_vitals", {}).get("n_events", 0))
                        statuses.append(status); done += 1; continue
                except Exception:
                    pass

            with open(meta_path) as f:
                meta = json.load(f)

            sub_labs = labs_by_enc.get(enc,
                                       labs.clear())  # empty same-schema df
            sub_chart = chart_by_enc.get(enc, chart.clear())

            lab_events = _write_events(out_dir, "labs_events.npy", sub_labs)
            chart_events = _write_events(out_dir, "chart_vitals_events.npy", sub_chart)

            # per-var counts
            def _per_var(ev):
                if ev.shape[0] == 0:
                    return {}
                v, c = np.unique(ev["var_id"], return_counts=True)
                return {str(int(vv)): int(cc) for vv, cc in zip(v, c)}

            meta["labs"] = {
                "n_events": int(lab_events.shape[0]),
                "per_var_count": _per_var(lab_events),
                "source": "JGSEPSIS_LABS.csv",
                "time_convention": "NY_local→UTC ms",
            }
            meta["chart_vitals"] = {
                "n_events": int(chart_events.shape[0]),
                "per_var_count": _per_var(chart_events),
                "source": "JGSEPSIS_VITALS2.csv",
                "time_convention": "NY_local→UTC ms",
            }
            meta["stage_d_version"] = 1
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2, default=str)

            status["status"] = "ok"
            status["n_labs"] = int(lab_events.shape[0])
            status["n_chart"] = int(chart_events.shape[0])
        except Exception as e:
            status["status"] = "error"
            status["error"] = f"{type(e).__name__}: {e}"
            status["traceback"] = traceback.format_exc()[-400:]

        statuses.append(status)
        done += 1
        if done % 500 == 0 or done == n_total:
            by = {}
            for s in statuses:
                by[s["status"]] = by.get(s["status"], 0) + 1
            LOG.info(f"  [{done}/{n_total}] elapsed={time.time()-t0:.1f}s  {by}")

    return statuses


def main():
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["1a", "1b", "2", "all"], default="all")
    ap.add_argument("--limit", type=int, default=0, help="phase 2 only")
    ap.add_argument("--entity-id", type=str, default=None)
    ap.add_argument("--entities", type=str, default=None,
                    help="comma-separated entity_ids")
    ap.add_argument("--out-root", type=str, default=OUT_ROOT)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--redo-phase1", action="store_true",
                    help="rebuild combined parquets even if they exist")
    args = ap.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(f"{LOG_DIR}/stage_d_ehr.log")])
    LOG = logging.getLogger(__name__)

    LOG.info(f"Loading cohort parquet: {COHORT_PARQUET}")
    cohort = pl.read_parquet(COHORT_PARQUET).unique("entity_id", keep="first")
    LOG.info(f"  cohort entities: {cohort.height}  unique encounters: "
             f"{cohort['encounter_nbr'].n_unique()}")
    cohort_encounters = cohort["encounter_nbr"].unique().to_list()

    info_labs = info_chart = None

    if args.phase in ("1a", "all"):
        if args.redo_phase1 or not os.path.exists(LABS_COMBINED):
            info_labs = phase1_labs(cohort_encounters)
        else:
            LOG.info(f"Phase 1a: skipping (combined exists: {LABS_COMBINED})")

    if args.phase in ("1b", "all"):
        if args.redo_phase1 or not os.path.exists(CHART_COMBINED):
            info_chart = phase1_chart_vitals(cohort_encounters)
        else:
            LOG.info(f"Phase 1b: skipping (combined exists: {CHART_COMBINED})")

    if args.phase in ("2", "all"):
        if not (os.path.exists(LABS_COMBINED) and os.path.exists(CHART_COMBINED)):
            LOG.error("Phase 2 needs both combined parquets. Run phase 1 first.")
            sys.exit(1)

        e_df = cohort
        if args.entity_id:
            e_df = e_df.filter(pl.col("entity_id") == args.entity_id)
        elif args.entities:
            ids = [s.strip() for s in args.entities.split(",") if s.strip()]
            e_df = e_df.filter(pl.col("entity_id").is_in(ids))
        elif args.limit:
            e_df = e_df.head(args.limit)

        LOG.info(f"Phase 2: {e_df.height} entities, resume={not args.no_resume}")
        t0 = time.time()
        statuses = phase2(e_df, args.out_root, resume=not args.no_resume)
        elapsed = time.time() - t0

        by_status: dict[str, int] = {}
        for s in statuses:
            by_status[s["status"]] = by_status.get(s["status"], 0) + 1
        summary = {
            "n_entities_processed": len(statuses),
            "elapsed_sec_phase2": round(elapsed, 1),
            "by_status": by_status,
            "total_labs_events": int(sum(s.get("n_labs", 0) for s in statuses)),
            "total_chart_events": int(sum(s.get("n_chart", 0) for s in statuses)),
            "phase1a": info_labs,
            "phase1b": info_chart,
        }
        with open(SUMMARY_JSON, "w") as f:
            json.dump({"summary": summary,
                       "errors": [s for s in statuses if s["status"] in
                                  {"error", "no_stage_b_meta"}][:50]},
                      f, indent=2, default=str)
        LOG.info(f"\n=== Stage D summary ===\n{json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
