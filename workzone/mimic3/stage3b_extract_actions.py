#!/usr/bin/env python3
"""
Stage 3b: Extract action variables and merge into existing ehr_events.npy.

Adds var_ids 200-206 (vasopressors, fluids, FiO2, PEEP, mechvent, urine output)
and 207-212 (per-drug vasopressor rates) from CHARTEVENTS, INPUTEVENTS_MV, and OUTPUTEVENTS.

This is an INCREMENTAL step: it reads each patient's existing ehr_events.npy,
appends new action events, re-sorts, and re-saves. Waveform .npy files are untouched.
Re-runs are idempotent -- merge_actions_for_patient() strips var_id 200-299 before merging.

var 200 (NE-equivalent sum) is UNCHANGED by the 207-212 addition: VASOPRESSOR_ITEMS still
holds exactly the same five drugs, so re-running reproduces it. Dobutamine (212) is
extracted per-drug only and deliberately kept out of the sum, per var_registry.json.

Run:  python workzone/mimic3/stage3b_extract_actions.py [--root DIR] [--limit N] [--dry-run]
Depends on: stage 3 output (patient dirs with time_ms.npy + ehr_events.npy)
"""
import os
import json
import time
import logging
import argparse
from pathlib import Path

import numpy as np
import polars as pl
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO_ROOT / "workzone" / "outputs" / "mimic3"

import yaml
with open(REPO_ROOT / "workzone" / "configs" / "server_paths.yaml") as f:
    cfg = yaml.safe_load(f)

EHR_ROOT = cfg["mimic3"]["raw_ehr_dir"]
PROCESSED_ROOT = cfg["mimic3"]["output_dir"]

EHR_EVENT_DTYPE = np.dtype([
    ('time_ms', 'int64'),
    ('seg_idx', 'int32'),
    ('var_id', 'uint16'),
    ('value', 'float32'),
])

# ========================================================================
# Vasopressor NE-equivalent conversion
# ========================================================================

# MIMIC-III INPUTEVENTS_MV ITEMIDs for vasopressors
VASOPRESSOR_ITEMS = {
    221906: {"drug": "Norepinephrine", "ne_factor": 1.0},     # mcg/kg/min -> 1:1
    221289: {"drug": "Epinephrine",    "ne_factor": 1.0},     # mcg/kg/min -> 1:1
    221662: {"drug": "Dopamine",       "ne_factor": 0.01},    # mcg/kg/min -> /100
    221749: {"drug": "Phenylephrine",  "ne_factor": 0.1/80},  # mcg/min -> /10/80kg
    222315: {"drug": "Vasopressin",    "ne_factor": 0.0},     # special: binary
}
VASOPRESSOR_ITEMIDS = list(VASOPRESSOR_ITEMS.keys())

# Per-drug channels (registry 207-212), emitted ALONGSIDE the NE-equivalent sum in var 200.
# var 200 is deliberately left untouched -- VASOPRESSOR_ITEMS above is unchanged, so re-running
# reproduces it exactly and this is purely additive.
#
# Why the split is needed: var 200 group_by's over STARTTIME and sums NE-eq across concurrent
# drugs, discarding drug identity. Phenylephrine (pure alpha -> peripheral vasoconstriction,
# the most direct route to a PPG effect) then becomes indistinguishable from dopamine
# (beta/dopaminergic -> inotropy + chronotropy), and an equipotent drug SWITCH looks like no
# change at all. That conflation breaks dose-graded analysis, where the whole argument is
# "identical charting process, graded physiological effect".
#
# Values are RAW rates in each drug's native unit (per the registry), not NE-equivalents, so
# no information is destroyed; ne_factor lives in the registry for anyone who wants the sum.
PER_DRUG_VAR = {221906: 207,    # Norepinephrine   mcg/kg/min
                221289: 208,    # Epinephrine      mcg/kg/min
                221749: 209,    # Phenylephrine    mcg/min
                221662: 210,    # Dopamine         mcg/kg/min
                222315: 211,    # Vasopressin      units/hr
                221653: 212}    # Dobutamine       mcg/kg/min -- inotrope, EXCLUDED from var 200
# Generous per-unit sanity caps: these reject transcription garbage, not clinical extremes.
PER_DRUG_MAX = {207: 100.0, 208: 100.0, 209: 1000.0, 210: 100.0, 211: 10.0, 212: 100.0}
STOP_GAP_MS = 5 * 60 * 1000     # an infusion that resumes within 5 min was not really stopped

# Crystalloid fluid ITEMIDs (INPUTEVENTS_MV)
FLUID_ITEMIDS = [225158, 225828, 225166]  # NS, LR, D5W

# FiO2 and PEEP (CHARTEVENTS)
FIO2_ITEMID = 223835   # Inspired O2 Fraction (%), convert to 0-1
PEEP_ITEMID = 220339   # PEEP set

# Urine output (OUTPUTEVENTS)
URINE_ITEMIDS = [226559, 226560, 226561, 226563, 226564, 226565,
                 226567, 226557, 226558, 227488, 227489]


# ========================================================================
# Extract from each source
# ========================================================================

def extract_fio2_peep():
    """Extract FiO2 and PEEP from CHARTEVENTS."""
    log.info("=== Extracting FiO2 + PEEP from CHARTEVENTS ===")
    chart_path = os.path.join(EHR_ROOT, "CHARTEVENTS.csv.gz")
    if not os.path.exists(chart_path):
        chart_path = os.path.join(EHR_ROOT, "CHARTEVENTS.csv")

    target_ids = [FIO2_ITEMID, PEEP_ITEMID]

    t0 = time.time()
    df = pl.scan_csv(chart_path, infer_schema_length=1000).filter(
        pl.col("ITEMID").is_in(target_ids) &
        pl.col("VALUENUM").is_not_null() &
        pl.col("VALUENUM").is_not_nan()
    ).select(["SUBJECT_ID", "HADM_ID", "ITEMID", "CHARTTIME", "VALUENUM"]).collect()
    log.info(f"  Loaded {len(df)} rows in {time.time()-t0:.1f}s")

    # Convert FiO2 from % to fraction (0-1)
    df = df.with_columns(
        pl.when(pl.col("ITEMID") == FIO2_ITEMID)
        .then(pl.col("VALUENUM") / 100.0)
        .otherwise(pl.col("VALUENUM"))
        .alias("VALUENUM")
    )

    # Assign var_ids
    df = df.with_columns(
        pl.when(pl.col("ITEMID") == FIO2_ITEMID).then(pl.lit(203))
        .when(pl.col("ITEMID") == PEEP_ITEMID).then(pl.lit(204))
        .alias("var_id")
    )

    # Range filter
    df = df.filter(
        ((pl.col("var_id") == 203) & (pl.col("VALUENUM") >= 0.21) & (pl.col("VALUENUM") <= 1.0)) |
        ((pl.col("var_id") == 204) & (pl.col("VALUENUM") >= 0) & (pl.col("VALUENUM") <= 30))
    )

    # Dedup
    df = df.sort("CHARTTIME").unique(subset=["SUBJECT_ID", "CHARTTIME", "var_id"], keep="first")

    df = df.with_columns(
        pl.col("CHARTTIME").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S").alias("charttime_dt")
    )

    log.info(f"  FiO2: {df.filter(pl.col('var_id')==203).height:,} events")
    log.info(f"  PEEP: {df.filter(pl.col('var_id')==204).height:,} events")
    return df


def extract_vasopressors_per_drug():
    """Per-drug raw infusion rates (vars 207-212), including explicit STOP events.

    Two things this does that the NE-equivalent path (var 200) does not:

    1. Keeps drug identity -- see PER_DRUG_VAR.
    2. Emits value=0.0 at ENDTIME when an infusion actually stops. The var 200 path filters
       `RATE > 0`, so a stop is represented only by ABSENCE of later rows. That is fine for a
       carry-forward state variable but wrong for change-point work, where a wean is as clean
       a haemodynamic transition as a start -- and there are more weans than starts. A stop is
       only emitted if the same drug does not resume within STOP_GAP_MS, so dose titrations
       charted as back-to-back intervals are not mistaken for stop/start pairs.
    """
    log.info("\n=== Extracting Per-Drug Vasopressor Rates (vars 207-212) ===")
    input_path = os.path.join(EHR_ROOT, "INPUTEVENTS_MV.csv.gz")
    if not os.path.exists(input_path):
        input_path = os.path.join(EHR_ROOT, "INPUTEVENTS_MV.csv")

    df = pl.scan_csv(input_path, infer_schema_length=1000).filter(
        pl.col("ITEMID").is_in(list(PER_DRUG_VAR.keys())) &
        pl.col("RATE").is_not_null() &
        pl.col("RATE").is_not_nan() &
        (pl.col("RATE") > 0) &
        (pl.col("STATUSDESCRIPTION") != "Rewritten")
    ).select(["SUBJECT_ID", "HADM_ID", "ITEMID", "STARTTIME", "ENDTIME", "RATE"]).collect()
    log.info(f"  Loaded {len(df)} per-drug rows")

    var_map = pl.DataFrame({"ITEMID": list(PER_DRUG_VAR.keys()),
                            "var_id": list(PER_DRUG_VAR.values())})
    df = df.join(var_map, on="ITEMID", how="inner").with_columns([
        pl.col("STARTTIME").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S").alias("t_start"),
        pl.col("ENDTIME").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S").alias("t_end"),
    ])

    cap = pl.DataFrame({"var_id": list(PER_DRUG_MAX.keys()),
                        "rate_max": list(PER_DRUG_MAX.values())})
    df = df.join(cap, on="var_id", how="inner").filter(pl.col("RATE") <= pl.col("rate_max"))

    # START rows: the rate as charted
    starts = df.select([
        "SUBJECT_ID", "HADM_ID", "var_id",
        pl.col("t_start").alias("charttime_dt"),
        pl.col("RATE").alias("VALUENUM"),
    ])

    # STOP rows: value 0 at t_end, only when the same drug does not resume promptly
    df = df.sort(["SUBJECT_ID", "HADM_ID", "var_id", "t_start"])
    nxt = pl.col("t_start").shift(-1).over(["SUBJECT_ID", "HADM_ID", "var_id"])
    stops = df.with_columns(nxt.alias("t_next")).filter(
        pl.col("t_next").is_null() |
        ((pl.col("t_next") - pl.col("t_end")).dt.total_milliseconds() > STOP_GAP_MS)
    ).select([
        "SUBJECT_ID", "HADM_ID", "var_id",
        pl.col("t_end").alias("charttime_dt"),
        pl.lit(0.0).cast(pl.Float64).alias("VALUENUM"),
    ]).filter(pl.col("charttime_dt").is_not_null())

    out = pl.concat([starts, stops]).with_columns([
        pl.col("charttime_dt").dt.strftime("%Y-%m-%d %H:%M:%S").alias("CHARTTIME"),
        pl.col("var_id").cast(pl.Int32),
    ])
    for vid in sorted(PER_DRUG_VAR.values()):
        n = out.filter(pl.col("var_id") == vid).height
        nz = out.filter((pl.col("var_id") == vid) & (pl.col("VALUENUM") == 0.0)).height
        log.info(f"  var_id={vid}: {n:,} events ({nz:,} stops)")
    return out


def extract_vasopressors():
    """Extract vasopressor rates from INPUTEVENTS_MV, convert to NE-equivalent."""
    log.info("\n=== Extracting Vasopressors from INPUTEVENTS_MV ===")
    input_path = os.path.join(EHR_ROOT, "INPUTEVENTS_MV.csv.gz")
    if not os.path.exists(input_path):
        input_path = os.path.join(EHR_ROOT, "INPUTEVENTS_MV.csv")
    log.info(f"  Reading: {input_path}")

    t0 = time.time()
    df = pl.scan_csv(input_path, infer_schema_length=1000).filter(
        pl.col("ITEMID").is_in(VASOPRESSOR_ITEMIDS) &
        pl.col("RATE").is_not_null() &
        pl.col("RATE").is_not_nan() &
        (pl.col("RATE") > 0) &
        (pl.col("STATUSDESCRIPTION") != "Rewritten")
    ).select(["SUBJECT_ID", "HADM_ID", "ITEMID", "STARTTIME", "RATE"]).collect()
    log.info(f"  Loaded {len(df)} rows in {time.time()-t0:.1f}s")

    # Convert each drug to NE-equivalent via join
    vaso_map = pl.DataFrame({
        "ITEMID": list(VASOPRESSOR_ITEMS.keys()),
        "ne_factor": [v["ne_factor"] for v in VASOPRESSOR_ITEMS.values()],
        "is_binary": [v["ne_factor"] == 0.0 for v in VASOPRESSOR_ITEMS.values()],
    })
    df = df.join(vaso_map, on="ITEMID", how="inner")

    df = df.with_columns(
        pl.when(pl.col("is_binary"))
        .then(pl.lit(0.1))  # Vasopressin: any dose = 0.1 mcg/kg/min NE-eq
        .otherwise(pl.col("RATE") * pl.col("ne_factor"))
        .alias("ne_rate")
    ).drop(["ne_factor", "is_binary"])

    # Aggregate per (SUBJECT_ID, HADM_ID, STARTTIME): sum NE-eq across concurrent pressors
    df = df.group_by(["SUBJECT_ID", "HADM_ID", "STARTTIME"]).agg(
        pl.col("ne_rate").sum().alias("VALUENUM")
    )

    # Range filter
    df = df.filter((pl.col("VALUENUM") >= 0) & (pl.col("VALUENUM") <= 10.0))

    df = df.with_columns([
        pl.lit(200).cast(pl.Int32).alias("var_id"),
        pl.col("STARTTIME").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S").alias("charttime_dt"),
        pl.col("STARTTIME").alias("CHARTTIME"),
    ])

    log.info(f"  Vasopressor events (NE-eq): {len(df):,}")
    return df


def extract_fluids():
    """Extract fluid infusion rate and boluses from INPUTEVENTS_MV."""
    log.info("\n=== Extracting Fluids from INPUTEVENTS_MV ===")
    input_path = os.path.join(EHR_ROOT, "INPUTEVENTS_MV.csv.gz")
    if not os.path.exists(input_path):
        input_path = os.path.join(EHR_ROOT, "INPUTEVENTS_MV.csv")

    t0 = time.time()
    df = pl.scan_csv(input_path, infer_schema_length=1000).filter(
        pl.col("ITEMID").is_in(FLUID_ITEMIDS) &
        pl.col("AMOUNT").is_not_null() &
        (pl.col("AMOUNT") > 0) &
        (pl.col("STATUSDESCRIPTION") != "Rewritten")
    ).select(["SUBJECT_ID", "HADM_ID", "ITEMID", "STARTTIME",
              "AMOUNT", "RATE", "ORDERCATEGORYDESCRIPTION"]).collect()
    log.info(f"  Loaded {len(df)} rows in {time.time()-t0:.1f}s")

    # Split into bolus vs infusion
    bolus = df.filter(
        pl.col("ORDERCATEGORYDESCRIPTION").str.contains("(?i)bolus")
    ).with_columns([
        pl.col("AMOUNT").alias("VALUENUM"),
        pl.lit(202).cast(pl.Int32).alias("var_id"),
    ])

    infusion = df.filter(
        ~pl.col("ORDERCATEGORYDESCRIPTION").str.contains("(?i)bolus") &
        pl.col("RATE").is_not_null() & (pl.col("RATE") > 0)
    ).with_columns([
        pl.col("RATE").alias("VALUENUM"),
        pl.lit(201).cast(pl.Int32).alias("var_id"),
    ])

    combined = pl.concat([
        bolus.select(["SUBJECT_ID", "HADM_ID", "STARTTIME", "VALUENUM", "var_id"]),
        infusion.select(["SUBJECT_ID", "HADM_ID", "STARTTIME", "VALUENUM", "var_id"]),
    ])

    # Range filter
    combined = combined.filter(
        ((pl.col("var_id") == 201) & (pl.col("VALUENUM") <= 2000)) |
        ((pl.col("var_id") == 202) & (pl.col("VALUENUM") <= 5000))
    )

    combined = combined.with_columns([
        pl.col("STARTTIME").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S").alias("charttime_dt"),
        pl.col("STARTTIME").alias("CHARTTIME"),
    ])

    log.info(f"  Fluid rate events: {combined.filter(pl.col('var_id')==201).height:,}")
    log.info(f"  Fluid bolus events: {combined.filter(pl.col('var_id')==202).height:,}")
    return combined


def extract_urine():
    """Extract urine output from OUTPUTEVENTS."""
    log.info("\n=== Extracting Urine Output from OUTPUTEVENTS ===")
    output_path = os.path.join(EHR_ROOT, "OUTPUTEVENTS.csv.gz")
    if not os.path.exists(output_path):
        output_path = os.path.join(EHR_ROOT, "OUTPUTEVENTS.csv")
    log.info(f"  Reading: {output_path}")

    t0 = time.time()
    df = pl.scan_csv(
        output_path, infer_schema_length=10000,
        schema_overrides={"VALUE": pl.Float64},
    ).filter(
        pl.col("ITEMID").is_in(URINE_ITEMIDS) &
        pl.col("VALUE").is_not_null() &
        pl.col("VALUE").is_not_nan()
    ).select(["SUBJECT_ID", "HADM_ID", "ITEMID", "CHARTTIME", "VALUE"]).collect()
    log.info(f"  Loaded {len(df)} rows in {time.time()-t0:.1f}s")

    df = df.with_columns([
        pl.col("VALUE").alias("VALUENUM"),
        pl.lit(206).cast(pl.Int32).alias("var_id"),
    ])

    # Range filter
    df = df.filter((pl.col("VALUENUM") >= 0) & (pl.col("VALUENUM") <= 2500))

    # Dedup
    df = df.sort("CHARTTIME").unique(subset=["SUBJECT_ID", "CHARTTIME", "var_id"], keep="first")

    df = df.with_columns(
        pl.col("CHARTTIME").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S").alias("charttime_dt")
    )

    log.info(f"  Urine output events: {len(df):,}")
    return df


# ========================================================================
# Merge into existing patient directories
# ========================================================================

def merge_actions_for_patient(patient_dir, patient_actions_df):
    """Merge new action events into existing ehr_events.npy for one patient.

    Returns n_new_events added, or -1 on error.
    """
    time_ms_path = os.path.join(patient_dir, "time_ms.npy")
    ehr_path = os.path.join(patient_dir, "ehr_events.npy")
    meta_path = os.path.join(patient_dir, "meta.json")

    if not os.path.exists(time_ms_path):
        return -1

    time_ms = np.load(time_ms_path)
    n_seg = len(time_ms)

    # Build new events with seg_idx alignment
    new_events = []
    for row in patient_actions_df.iter_rows(named=True):
        event_time_ms = int(row["charttime_dt"].timestamp() * 1000)
        seg_idx = np.searchsorted(time_ms, event_time_ms, side="right") - 1
        if 0 <= seg_idx < n_seg:
            new_events.append((event_time_ms, int(seg_idx), int(row["var_id"]), float(row["VALUENUM"])))

    if not new_events:
        return 0

    new_arr = np.array(new_events, dtype=EHR_EVENT_DTYPE)

    # Load existing events and merge
    if os.path.exists(ehr_path):
        existing = np.load(ehr_path)
        # Remove any old action events (var_id 200-299) to make re-runs idempotent
        if len(existing) > 0:
            keep_mask = (existing['var_id'] < 200) | (existing['var_id'] >= 300)
            existing = existing[keep_mask]
        merged = np.concatenate([existing, new_arr])
    else:
        merged = new_arr

    merged.sort(order="time_ms")

    # Save
    np.save(ehr_path, merged)

    # Update meta.json
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        meta["n_ehr_events"] = len(merged)
        n_action = int(np.sum((merged['var_id'] >= 200) & (merged['var_id'] < 300)))
        meta["n_action_events"] = n_action
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    return len(new_events)


# ========================================================================
# Infer mechanical ventilation
# ========================================================================

def infer_mechvent(fio2_peep_df):
    """Infer mechvent=1 at timestamps where FiO2 or PEEP is charted.

    If FiO2 is set (not room air 0.21) or PEEP > 0, patient is likely on ventilator.
    Returns DataFrame with var_id=205, value=1.0 at those timestamps.
    """
    log.info("\n=== Inferring Mechanical Ventilation from FiO2/PEEP ===")

    vent_events = fio2_peep_df.filter(
        ((pl.col("var_id") == 203) & (pl.col("VALUENUM") > 0.21)) |
        ((pl.col("var_id") == 204) & (pl.col("VALUENUM") > 0))
    ).select(["SUBJECT_ID", "HADM_ID", "CHARTTIME", "charttime_dt"]).unique(
        subset=["SUBJECT_ID", "CHARTTIME"]
    ).with_columns([
        pl.lit(205).cast(pl.Int32).alias("var_id"),
        pl.lit(1.0).alias("VALUENUM"),
    ])

    log.info(f"  Mechvent events (inferred): {len(vent_events):,}")
    return vent_events


# ========================================================================
# Main
# ========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Process only N patients (for testing)")
    parser.add_argument("--root", default=PROCESSED_ROOT,
                        help="entity root to merge into (default: mimic3.output_dir from "
                             "server_paths.yaml). Point this at a scratch mirror to leave the "
                             "canonical store untouched.")
    parser.add_argument("--dry-run", action="store_true",
                        help="extract + report counts, write nothing")
    args = parser.parse_args()
    root = args.root

    log.info("Stage 3b: Extract action variables -> merge into ehr_events.npy")
    log.info(f"  root={root}  {'DRY-RUN (no write)' if args.dry_run else 'WRITE'}")
    t0 = time.time()

    # 1. Extract all action events from raw MIMIC-III tables
    fio2_peep = extract_fio2_peep()
    vasopressors = extract_vasopressors()
    per_drug = extract_vasopressors_per_drug()
    fluids = extract_fluids()
    urine = extract_urine()
    mechvent = infer_mechvent(fio2_peep)

    # 2. Combine all into one DataFrame with uniform schema
    schema_cols = ["SUBJECT_ID", "HADM_ID", "CHARTTIME", "charttime_dt", "var_id", "VALUENUM"]
    all_actions = pl.concat([
        fio2_peep.select(schema_cols),
        vasopressors.select(schema_cols),
        per_drug.select(schema_cols),
        fluids.select(schema_cols),
        urine.select(schema_cols),
        mechvent.select(schema_cols),
    ])

    log.info(f"\n=== Total action events: {len(all_actions):,} ===")
    for vid in [200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212]:
        n = all_actions.filter(pl.col("var_id") == vid).height
        log.info(f"  var_id={vid}: {n:,}")

    # 3. Group by SUBJECT_ID for fast per-patient lookup
    actions_pd = all_actions.to_pandas()
    actions_pd["HADM_ID"] = pd.to_numeric(actions_pd["HADM_ID"], errors="coerce")
    actions_grouped = {sid: grp for sid, grp in actions_pd.groupby("SUBJECT_ID")}

    # 4. Iterate over processed patient directories and merge
    patient_dirs = sorted([
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
        and not d.startswith("tasks") and not d.startswith(".")
    ])
    log.info(f"\nProcessed patient dirs: {len(patient_dirs)}")

    if args.limit:
        patient_dirs = patient_dirs[:args.limit]
        log.info(f"  Limited to {len(patient_dirs)}")

    n_merged = 0
    n_skipped = 0
    n_no_actions = 0
    total_new_events = 0

    from tqdm import tqdm
    for dirname in tqdm(patient_dirs, desc="Stage 3b", unit="pat"):
        patient_dir = os.path.join(root, dirname)

        # Parse subject_id and hadm_id from dir name: "{subject_id}_{hadm_id}"
        parts = dirname.split("_")
        if len(parts) != 2:
            n_skipped += 1
            continue
        try:
            subject_id = int(parts[0])
            hadm_id = int(parts[1])
        except ValueError:
            n_skipped += 1
            continue

        # Get this patient's action events
        patient_actions = actions_grouped.get(subject_id)
        if patient_actions is None or len(patient_actions) == 0:
            n_no_actions += 1
            continue

        # Filter by HADM_ID
        mask = patient_actions["HADM_ID"] == hadm_id
        hadm_actions = patient_actions[mask]
        if len(hadm_actions) == 0:
            # Fall back to all events for this subject (HADM_ID might be NaN in some tables)
            hadm_actions = patient_actions

        # Convert back to polars for merge function
        hadm_pl = pl.from_pandas(hadm_actions[["charttime_dt", "var_id", "VALUENUM"]])

        if args.dry_run:
            n_new = len(hadm_pl)
        else:
            n_new = merge_actions_for_patient(patient_dir, hadm_pl)
        if n_new > 0:
            n_merged += 1
            total_new_events += n_new
        elif n_new == 0:
            n_no_actions += 1

    elapsed = time.time() - t0

    log.info(f"\n=== Stage 3b Complete ===")
    log.info(f"  Patients with new actions: {n_merged}")
    log.info(f"  Patients with no actions: {n_no_actions}")
    log.info(f"  Skipped: {n_skipped}")
    log.info(f"  Total new events added: {total_new_events:,}")
    log.info(f"  Time: {elapsed:.1f}s")

    # Save summary
    summary = {
        "n_merged": n_merged,
        "n_no_actions": n_no_actions,
        "n_skipped": n_skipped,
        "total_new_events": total_new_events,
        "total_time_sec": round(elapsed, 1),
        "per_variable": {
            str(vid): int(all_actions.filter(pl.col("var_id") == vid).height)
            for vid in [200, 201, 202, 203, 204, 205, 206]
        },
    }
    with open(OUT_DIR / "stage3b_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"\nNext: python workzone/mimic3/post_sepsis_cohort.py")


if __name__ == "__main__":
    main()
