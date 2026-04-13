#!/usr/bin/env python3
"""
Post-stage: Sepsis downstream task adaptation.

Takes the canonical processed data and the MedicalGYM sepsis cohort,
maps patients, adds sepsis-specific metadata, generates task-specific splits.

Steps:
  1. Load sepsis cohort (icustayid) + ICUSTAYS table -> map to SUBJECT_ID_HADM_ID
  2. Intersect with our processed patients
  3. For each matched patient:
     - Record sepsis onset time, mortality label
     - Add SOFA score sequence to ehr_events.npy (as new var_ids)
  4. Generate tasks/sepsis/cohort.json and tasks/sepsis/splits.json

Run:  python workzone/mimic3/post_sepsis_cohort.py
"""
import os
import json
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

import yaml
with open(REPO_ROOT / "workzone" / "configs" / "server_paths.yaml") as f:
    cfg = yaml.safe_load(f)

EHR_ROOT = cfg["mimic3"]["raw_ehr_dir"]
PROCESSED_ROOT = cfg["mimic3"]["output_dir"]

# Sepsis cohort source
SEPSIS_DATA = "/home/mxwan/workspace/MedicalGYM/data/sepsis"
SEPSIS_RL_CSV = os.path.join(SEPSIS_DATA, "sepsis_rl_dataset.csv")
SEPSIS_RAW_CSV = os.path.join(SEPSIS_DATA, "sepsis_raw_noagg_noimpute.csv")

# EHR event dtype
EHR_EVENT_DTYPE = np.dtype([
    ('time_ms', 'int64'),
    ('seg_idx', 'int32'),
    ('var_id', 'uint16'),
    ('value', 'float32'),
])

# SOFA-related var_ids (add to var_registry)
# Using IDs starting at 100 for task-specific variables
SEPSIS_VAR_IDS = {
    "sofa_total": 100,
    "sofa_resp": 101,
    "sofa_coag": 102,
    "sofa_liver": 103,
    "sofa_cardio": 104,
    "sofa_cns": 105,
    "sofa_renal": 106,
    "sepsis_onset": 107,  # binary marker: 1.0 at onset time
}

SPLIT_SEED = 42
TRAIN_FRACTION = 0.7
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15


def load_sepsis_cohort():
    """Load sepsis cohort and extract per-patient info."""
    log.info(f"Loading sepsis cohort from {SEPSIS_RL_CSV}")

    # Use the RL dataset (already filtered)
    df = pd.read_csv(SEPSIS_RL_CSV)

    # Clean column names (they have 'm: ', 'o: ', 'a: ', 'r: ' prefixes)
    df.columns = [c.strip() for c in df.columns]
    col_map = {}
    for c in df.columns:
        if ': ' in c:
            prefix, name = c.split(': ', 1)
            col_map[c] = name.strip()
        else:
            col_map[c] = c.strip()
    df = df.rename(columns=col_map)

    # Per-patient summary
    patients = df.groupby("icustayid").agg(
        onset_time=("presumed_onset", "first"),
        n_timesteps=("timestep", "max"),
        mortality=("reward", "last"),  # terminal reward: +1 survived, -1 died
    ).reset_index()

    # Convert mortality to binary label: 1 = died, 0 = survived
    patients["died"] = (patients["mortality"] < 0).astype(int)

    log.info(f"  Sepsis cohort: {len(patients)} patients")
    log.info(f"  Mortality: {patients['died'].sum()} died ({patients['died'].mean()*100:.1f}%)")

    return patients, df


def map_icustay_to_hadm(sepsis_patients):
    """Map icustayid -> (SUBJECT_ID, HADM_ID) via ICUSTAYS table."""
    log.info("Loading ICUSTAYS table for ID mapping...")
    icu_path = os.path.join(EHR_ROOT, "ICUSTAYS.csv.gz")
    if not os.path.exists(icu_path):
        icu_path = os.path.join(EHR_ROOT, "ICUSTAYS.csv")
    icustays = pd.read_csv(icu_path, usecols=["ICUSTAY_ID", "SUBJECT_ID", "HADM_ID"])

    # Merge
    mapped = sepsis_patients.merge(
        icustays, left_on="icustayid", right_on="ICUSTAY_ID", how="inner"
    )

    # Build our patient_id format: {SUBJECT_ID}_{HADM_ID}
    mapped["patient_id"] = mapped["SUBJECT_ID"].astype(str) + "_" + mapped["HADM_ID"].astype(str)

    log.info(f"  Mapped {len(mapped)}/{len(sepsis_patients)} to SUBJECT_ID_HADM_ID")
    return mapped


def compute_sofa_from_raw(raw_df, icustayid):
    """Compute SOFA component scores from raw sepsis data for one patient.

    SOFA components (simplified, from available variables):
      Resp:   paO2/FiO2 ratio
      Coag:   Platelets
      Liver:  Bilirubin
      Cardio: MeanBP + vasopressors
      CNS:    GCS (not in raw, skip)
      Renal:  Creatinine + urine output
    """
    pat = raw_df[raw_df["icustayid"] == icustayid].copy()
    if len(pat) == 0:
        return []

    sofa_events = []
    for _, row in pat.iterrows():
        charttime_ms = int(row["charttime"] * 1000)  # Unix seconds -> ms

        # Respiratory: PaO2/FiO2
        sofa_resp = 0
        if pd.notna(row.get("paO2")) and pd.notna(row.get("FiO2_1")) and row["FiO2_1"] > 0:
            pf = row["paO2"] / row["FiO2_1"]
            if pf < 100: sofa_resp = 4
            elif pf < 200: sofa_resp = 3
            elif pf < 300: sofa_resp = 2
            elif pf < 400: sofa_resp = 1

        # Coagulation: Platelets
        sofa_coag = 0
        if pd.notna(row.get("Platelets_count")):
            p = row["Platelets_count"]
            if p < 20: sofa_coag = 4
            elif p < 50: sofa_coag = 3
            elif p < 100: sofa_coag = 2
            elif p < 150: sofa_coag = 1

        # Liver: Bilirubin
        sofa_liver = 0
        if pd.notna(row.get("Total_bili")):
            b = row["Total_bili"]
            if b >= 12: sofa_liver = 4
            elif b >= 6: sofa_liver = 3
            elif b >= 2: sofa_liver = 2
            elif b >= 1.2: sofa_liver = 1

        # Cardiovascular: MAP + vasopressors
        sofa_cardio = 0
        vaso = row.get("vaso_rate_max", 0) or 0
        mbp = row.get("MeanBP", None)
        if vaso > 0.1: sofa_cardio = 4
        elif vaso > 0: sofa_cardio = 3
        elif pd.notna(mbp) and mbp < 70: sofa_cardio = 1

        # Renal: Creatinine
        sofa_renal = 0
        if pd.notna(row.get("Creatinine")):
            cr = row["Creatinine"]
            if cr >= 5: sofa_renal = 4
            elif cr >= 3.5: sofa_renal = 3
            elif cr >= 2: sofa_renal = 2
            elif cr >= 1.2: sofa_renal = 1

        sofa_total = sofa_resp + sofa_coag + sofa_liver + sofa_cardio + sofa_renal

        # Only add events when we have at least some non-zero data
        if any(pd.notna(row.get(v)) for v in ["paO2", "Platelets_count", "Total_bili", "Creatinine"]):
            sofa_events.append((charttime_ms, SEPSIS_VAR_IDS["sofa_total"], float(sofa_total)))
            sofa_events.append((charttime_ms, SEPSIS_VAR_IDS["sofa_resp"], float(sofa_resp)))
            sofa_events.append((charttime_ms, SEPSIS_VAR_IDS["sofa_coag"], float(sofa_coag)))
            sofa_events.append((charttime_ms, SEPSIS_VAR_IDS["sofa_liver"], float(sofa_liver)))
            sofa_events.append((charttime_ms, SEPSIS_VAR_IDS["sofa_cardio"], float(sofa_cardio)))
            sofa_events.append((charttime_ms, SEPSIS_VAR_IDS["sofa_renal"], float(sofa_renal)))

    return sofa_events


def add_sepsis_events_to_patient(patient_dir, onset_time_sec, sofa_events):
    """Merge sepsis-specific events into existing ehr_events.npy."""
    time_ms_path = os.path.join(patient_dir, "time_ms.npy")
    ehr_path = os.path.join(patient_dir, "ehr_events.npy")

    if not os.path.exists(time_ms_path) or not os.path.exists(ehr_path):
        return 0

    time_ms = np.load(time_ms_path)
    n_seg = len(time_ms)
    old_events = np.load(ehr_path)

    new_events = []

    # Add sepsis onset marker
    onset_ms = int(onset_time_sec * 1000)
    onset_seg_idx = np.searchsorted(time_ms, onset_ms, side="right") - 1
    if 0 <= onset_seg_idx < n_seg:
        new_events.append((onset_ms, onset_seg_idx, SEPSIS_VAR_IDS["sepsis_onset"], 1.0))

    # Add SOFA events with seg_idx alignment
    for event_time_ms, var_id, value in sofa_events:
        seg_idx = np.searchsorted(time_ms, event_time_ms, side="right") - 1
        if 0 <= seg_idx < n_seg:
            new_events.append((event_time_ms, seg_idx, var_id, value))

    if not new_events:
        return 0

    # Build new events array
    new_arr = np.array(new_events, dtype=EHR_EVENT_DTYPE)

    # Merge with existing
    if len(old_events) > 0:
        combined = np.concatenate([old_events, new_arr])
    else:
        combined = new_arr
    combined.sort(order="time_ms")

    # Save
    np.save(ehr_path, combined)

    # Update meta.json
    meta_path = os.path.join(patient_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        meta["n_ehr_events"] = len(combined)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    return len(new_events)


def main():
    log.info("Post-stage: Sepsis cohort adaptation")
    t0 = time.time()

    # 1. Load sepsis cohort
    sepsis_patients, raw_df = load_sepsis_cohort()

    # 2. Map icustayid -> SUBJECT_ID_HADM_ID
    mapped = map_icustay_to_hadm(sepsis_patients)

    # 3. Intersect with our processed patients
    processed_dirs = set(os.listdir(PROCESSED_ROOT))
    matched = mapped[mapped["patient_id"].isin(processed_dirs)]
    log.info(f"\n=== Cohort Overlap ===")
    log.info(f"  Sepsis cohort (mapped): {len(mapped)}")
    log.info(f"  Our processed patients: {len(processed_dirs)}")
    log.info(f"  Overlap: {len(matched)} patients")
    log.info(f"  Mortality in overlap: {matched['died'].sum()} ({matched['died'].mean()*100:.1f}%)")

    if len(matched) == 0:
        log.error("ABORT: No overlap between sepsis cohort and processed patients!")
        return

    # 4. Add SOFA + onset events to each matched patient
    log.info(f"\nAdding SOFA scores and sepsis onset markers...")
    n_events_added = 0
    for i, (_, row) in enumerate(matched.iterrows()):
        patient_dir = os.path.join(PROCESSED_ROOT, row["patient_id"])
        sofa_events = compute_sofa_from_raw(raw_df, row["icustayid"])
        n = add_sepsis_events_to_patient(patient_dir, row["onset_time"], sofa_events)
        n_events_added += n
        if (i + 1) % 200 == 0:
            log.info(f"  [{i+1}/{len(matched)}] {n_events_added} events added")

    log.info(f"  Total events added: {n_events_added}")

    # 5. Generate task-specific cohort + splits
    tasks_dir = os.path.join(PROCESSED_ROOT, "tasks", "sepsis")
    os.makedirs(tasks_dir, exist_ok=True)

    # Cohort file
    cohort = []
    for _, row in matched.iterrows():
        cohort.append({
            "patient_id": row["patient_id"],
            "subject_id": int(row["SUBJECT_ID"]),
            "hadm_id": int(row["HADM_ID"]),
            "icustayid": int(row["icustayid"]),
            "onset_time_sec": float(row["onset_time"]),
            "died": int(row["died"]),
        })

    with open(os.path.join(tasks_dir, "cohort.json"), "w") as f:
        json.dump(cohort, f, indent=2)
    log.info(f"Cohort: {os.path.join(tasks_dir, 'cohort.json')} ({len(cohort)} patients)")

    # Splits (by SUBJECT_ID, no leakage)
    rng = np.random.RandomState(SPLIT_SEED)
    unique_subjects = sorted(set(matched["SUBJECT_ID"].values))
    rng.shuffle(unique_subjects)

    n_test = int(len(unique_subjects) * TEST_FRACTION)
    n_val = int(len(unique_subjects) * VAL_FRACTION)
    test_subs = set(unique_subjects[:n_test])
    val_subs = set(unique_subjects[n_test:n_test + n_val])
    train_subs = set(unique_subjects[n_test + n_val:])

    splits = {"train": [], "val": [], "test": []}
    for entry in cohort:
        sid = entry["subject_id"]
        if sid in train_subs:
            splits["train"].append(entry["patient_id"])
        elif sid in val_subs:
            splits["val"].append(entry["patient_id"])
        else:
            splits["test"].append(entry["patient_id"])

    splits["n_train"] = len(splits["train"])
    splits["n_val"] = len(splits["val"])
    splits["n_test"] = len(splits["test"])
    splits["seed"] = SPLIT_SEED

    with open(os.path.join(tasks_dir, "splits.json"), "w") as f:
        json.dump(splits, f, indent=2)

    elapsed = time.time() - t0
    log.info(f"\n=== Sepsis Post-Stage Complete ===")
    log.info(f"  Matched patients: {len(matched)}")
    log.info(f"  Events added: {n_events_added}")
    log.info(f"  Train/Val/Test: {splits['n_train']}/{splits['n_val']}/{splits['n_test']}")
    log.info(f"  Output: {tasks_dir}")
    log.info(f"  Time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
