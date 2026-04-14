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

# Sepsis patient list (extracted from MedicalGYM, committed to repo -- 456 KB)
SEPSIS_PATIENT_LIST = os.path.join(REPO_ROOT, "workzone", "mimic3", "sepsis_patient_list.csv")

# EHR event dtype
EHR_EVENT_DTYPE = np.dtype([
    ('time_ms', 'int64'),
    ('seg_idx', 'int32'),
    ('var_id', 'uint16'),
    ('value', 'float32'),
])

# SOFA-related var_ids (from var_registry.json, scores 300-399)
SEPSIS_VAR_IDS = {
    "sofa_total": 300,
    "sofa_resp": 301,
    "sofa_coag": 302,
    "sofa_liver": 303,
    "sofa_cardio": 304,
    "sofa_cns": 305,
    "sofa_renal": 306,
    "sepsis_onset": 307,  # binary marker: 1.0 at onset time
}

SPLIT_SEED = 42
TRAIN_FRACTION = 0.7
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15


def load_sepsis_cohort():
    """Load sepsis patient list (icustayid + onset + mortality)."""
    log.info(f"Loading sepsis patient list from {SEPSIS_PATIENT_LIST}")

    patients = pd.read_csv(SEPSIS_PATIENT_LIST)
    patients["icustayid"] = patients["icustayid"].astype(float).astype(int)
    patients["onset_time"] = patients["presumed_onset"].astype(float)
    patients["died"] = (patients["mortality_reward"].astype(float) < 0).astype(int)

    log.info(f"  Sepsis cohort: {len(patients)} patients")
    log.info(f"  Mortality: {patients['died'].sum()} died ({patients['died'].mean()*100:.1f}%)")

    return patients


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


def compute_sofa_from_ehr_events(patient_dir):
    """Compute all 6 SOFA component scores from ehr_events.npy.

    Uses: Platelets (7), Bilirubin (6), Creatinine (5), paO2 (14), GCS (108),
          NBPm/MAP (106), FiO2 (203), vasopressor_rate (200).
    Returns list of (time_ms, var_id, value) tuples for SOFA scores.
    """
    ehr_path = os.path.join(patient_dir, "ehr_events.npy")
    if not os.path.exists(ehr_path):
        return []

    events = np.load(ehr_path)
    if len(events) == 0:
        return []

    sofa_events = []

    # --- SOFA Coagulation (var 302): Platelets ---
    for ev in events[events['var_id'] == 7]:
        plt = ev['value']
        score = 0
        if plt < 20: score = 4
        elif plt < 50: score = 3
        elif plt < 100: score = 2
        elif plt < 150: score = 1
        sofa_events.append((int(ev['time_ms']), SEPSIS_VAR_IDS["sofa_coag"], float(score)))

    # --- SOFA Liver (var 303): Bilirubin ---
    for ev in events[events['var_id'] == 6]:
        bili = ev['value']
        score = 0
        if bili >= 12: score = 4
        elif bili >= 6: score = 3
        elif bili >= 1.2: score = 2  # 1.2-5.9
        sofa_events.append((int(ev['time_ms']), SEPSIS_VAR_IDS["sofa_liver"], float(score)))

    # --- SOFA CNS (var 305): GCS ---
    for ev in events[events['var_id'] == 108]:
        gcs = ev['value']
        score = 0
        if gcs < 6: score = 4
        elif gcs < 10: score = 3
        elif gcs < 13: score = 2
        elif gcs < 15: score = 1
        sofa_events.append((int(ev['time_ms']), SEPSIS_VAR_IDS["sofa_cns"], float(score)))

    # --- SOFA Renal (var 306): Creatinine ---
    for ev in events[events['var_id'] == 5]:
        cr = ev['value']
        score = 0
        if cr >= 5: score = 4
        elif cr >= 3.5: score = 3
        elif cr >= 2: score = 2
        elif cr >= 1.2: score = 1
        sofa_events.append((int(ev['time_ms']), SEPSIS_VAR_IDS["sofa_renal"], float(score)))

    # --- SOFA Cardiovascular (var 304): MAP + vasopressor ---
    # Score based on vasopressor dose (NE-eq) or low MAP
    vaso_events = events[events['var_id'] == 200]
    map_events = events[events['var_id'] == 106]

    for ev in map_events:
        mapv = ev['value']
        # Find concurrent vasopressor (within 1 hour)
        nearby_vaso = vaso_events[
            (np.abs(vaso_events['time_ms'] - ev['time_ms']) < 3600000)
        ]
        max_ne = float(nearby_vaso['value'].max()) if len(nearby_vaso) > 0 else 0.0

        score = 0
        if max_ne > 0.1: score = 4
        elif max_ne > 0: score = 3  # any vasopressor ≤ 0.1
        elif mapv < 70: score = 1
        sofa_events.append((int(ev['time_ms']), SEPSIS_VAR_IDS["sofa_cardio"], float(score)))

    # --- SOFA Respiratory (var 301): PaO2/FiO2 ---
    pao2_events = events[events['var_id'] == 14]
    fio2_events = events[events['var_id'] == 203]

    for ev in pao2_events:
        pao2 = ev['value']
        # Find nearest FiO2 (within 4 hours)
        nearby_fio2 = fio2_events[
            (np.abs(fio2_events['time_ms'] - ev['time_ms']) < 14400000)
        ]
        if len(nearby_fio2) > 0:
            # Pick closest in time
            closest_idx = np.argmin(np.abs(nearby_fio2['time_ms'] - ev['time_ms']))
            fio2 = float(nearby_fio2[closest_idx]['value'])
        else:
            fio2 = 0.21  # assume room air

        if fio2 <= 0:
            continue
        pf_ratio = pao2 / fio2

        score = 0
        if pf_ratio < 100: score = 4
        elif pf_ratio < 200: score = 3
        elif pf_ratio < 300: score = 2
        elif pf_ratio < 400: score = 1
        sofa_events.append((int(ev['time_ms']), SEPSIS_VAR_IDS["sofa_resp"], float(score)))

    return sofa_events


def build_sepsis_extra_events(patient_dir, onset_time_sec, sofa_events, extra_events_dir):
    """Save sepsis-specific events to a SEPARATE file. Does NOT modify canonical ehr_events.npy.

    Saves to: tasks/sepsis/extra_events/{patient_id}.npy
    At training time, the adapter merges base + extra if needed.
    """
    time_ms_path = os.path.join(patient_dir, "time_ms.npy")
    if not os.path.exists(time_ms_path):
        return 0

    time_ms = np.load(time_ms_path)
    n_seg = len(time_ms)

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

    new_arr = np.array(new_events, dtype=EHR_EVENT_DTYPE)
    new_arr.sort(order="time_ms")

    # Save to extra_events directory (NOT modifying canonical files)
    patient_id = os.path.basename(patient_dir)
    os.makedirs(extra_events_dir, exist_ok=True)
    np.save(os.path.join(extra_events_dir, f"{patient_id}.npy"), new_arr)

    return len(new_events)


def extract_missing_patients(missing_df, labs_df, vitals_df, admissions_df):
    """Extract waveforms for sepsis patients that were dropped by stage 2b.

    These patients have waveform data but were filtered out because they didn't
    meet the main pipeline's EHR variable coverage threshold. We extract them
    now because they're needed for the sepsis task.
    """
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from workzone.mimic3.stage3_extract_waveforms import (
        parse_master_header, read_wfdb_blocks, resample_signal,
        segment_signal, build_ehr_events, save_patient,
        SOURCE_FS, SEGMENT_DUR_SEC, STRIDE_SEC, OVERLAP_SEC,
        WAVEFORM_DTYPE, TIME_DTYPE, TARGET_CHANNELS, BASE_CHANNEL,
    )

    WAV_ROOT = cfg["mimic3"]["raw_waveform_dir"]

    log.info(f"\n=== Extracting {len(missing_df)} missing sepsis patients ===")

    results = []
    for i, (_, row) in enumerate(missing_df.iterrows()):
        subject_id = int(row["SUBJECT_ID"])
        hadm_id = int(row["HADM_ID"])
        patient_id = f"{subject_id}_{hadm_id}"

        # Find waveform directory
        patient_dir_prefix = f"p{str(subject_id).zfill(6)[:3]}"
        patient_dir_name = f"p{str(subject_id).zfill(6)}"
        patient_path = os.path.join(WAV_ROOT, patient_dir_prefix, patient_dir_name)

        if not os.path.isdir(patient_path):
            results.append({"patient_id": patient_id, "status": "SKIP", "reason": "no waveform dir"})
            continue

        try:
            # Parse master header
            wav_start, source_fs, segment_list = parse_master_header(patient_path)
            if wav_start is None or segment_list is None:
                results.append({"patient_id": patient_id, "status": "SKIP", "reason": "no master header"})
                continue

            total_samples = sum(n for _, n in segment_list)
            wav_duration = total_samples / source_fs
            if wav_duration < 300:
                results.append({"patient_id": patient_id, "status": "SKIP", "reason": "too short"})
                continue

            # Read PLETH-anchored blocks
            blocks = read_wfdb_blocks(patient_path, segment_list, source_fs)
            if not blocks:
                results.append({"patient_id": patient_id, "status": "SKIP", "reason": "no PLETH blocks"})
                continue

            # Process blocks: resample + segment with overlap
            all_channel_segs = {ch: [] for ch in TARGET_CHANNELS}
            all_time_ms = []

            for block in blocks:
                block_start_ms = int((wav_start.timestamp() + block['start_sec']) * 1000)
                resampled = {}
                for ch, target_hz in TARGET_CHANNELS.items():
                    raw = block['channels'].get(ch)
                    if raw is not None and len(raw) > 0:
                        resampled[ch] = resample_signal(raw, source_fs, target_hz)
                    else:
                        base_raw = block['channels'][BASE_CHANNEL]
                        target_len = int(np.ceil(len(base_raw) * target_hz / source_fs))
                        resampled[ch] = np.full(target_len, np.nan)

                segmented = {}
                for ch, sig in resampled.items():
                    seg = segment_signal(sig, TARGET_CHANNELS[ch])
                    if seg is None:
                        break
                    segmented[ch] = seg

                if len(segmented) != len(TARGET_CHANNELS):
                    continue

                n_seg_block = min(s.shape[0] for s in segmented.values())
                for ch in segmented:
                    segmented[ch] = segmented[ch][:n_seg_block]

                stride_ms = STRIDE_SEC * 1000
                block_time_ms = np.array(
                    [block_start_ms + i * stride_ms for i in range(n_seg_block)],
                    dtype=TIME_DTYPE,
                )
                for ch in TARGET_CHANNELS:
                    all_channel_segs[ch].append(segmented[ch])
                all_time_ms.append(block_time_ms)

            if not all_time_ms:
                results.append({"patient_id": patient_id, "status": "SKIP", "reason": "blocks too short"})
                continue

            channels_out = {}
            for ch in TARGET_CHANNELS:
                ch_name = f"{ch}{TARGET_CHANNELS[ch]}"
                channels_out[ch_name] = np.concatenate(all_channel_segs[ch], axis=0)
            time_ms = np.concatenate(all_time_ms)
            n_seg = len(time_ms)

            # Build EHR events
            patient_labs = labs_df[labs_df["SUBJECT_ID"] == subject_id]
            patient_vitals = vitals_df[vitals_df["SUBJECT_ID"] == subject_id]
            ehr_events = build_ehr_events(subject_id, hadm_id, time_ms, patient_labs, patient_vitals)

            out_dir = os.path.join(PROCESSED_ROOT, patient_id)
            save_patient(
                out_dir=out_dir,
                channels=channels_out,
                time_ms=time_ms,
                ehr_events=ehr_events,
                meta_extra={
                    "patient_id": patient_id,
                    "subject_id": subject_id,
                    "hadm_id": hadm_id,
                    "source_dataset": "mimic3",
                    "source_path": patient_path,
                    "recording_start_ms": int(time_ms[0]),
                    "total_duration_hours": round(n_seg * STRIDE_SEC / 3600, 2),
                    "added_by": "post_sepsis_cohort",
                    "n_blocks": len(blocks),
                },
            )
            results.append({"patient_id": patient_id, "status": "OK", "n_segments": n_seg})

        except Exception as e:
            results.append({"patient_id": patient_id, "status": "ERROR", "reason": str(e)})

        if (i + 1) % 100 == 0:
            n_ok = sum(1 for r in results if r["status"] == "OK")
            log.info(f"  [{i+1}/{len(missing_df)}] OK={n_ok}")

    n_ok = sum(1 for r in results if r["status"] == "OK")
    n_skip = sum(1 for r in results if r["status"] == "SKIP")
    n_err = sum(1 for r in results if r["status"] == "ERROR")
    log.info(f"  Extraction done: OK={n_ok}, SKIP={n_skip}, ERR={n_err}")
    return results


def main():
    log.info("Post-stage: Sepsis cohort adaptation")
    t0 = time.time()

    # 1. Load sepsis cohort
    sepsis_patients = load_sepsis_cohort()

    # 2. Map icustayid -> SUBJECT_ID_HADM_ID
    mapped = map_icustay_to_hadm(sepsis_patients)

    # 3. Check overlap and find missing patients
    processed_dirs = set(os.listdir(PROCESSED_ROOT))
    already_processed = mapped[mapped["patient_id"].isin(processed_dirs)]
    missing = mapped[~mapped["patient_id"].isin(processed_dirs)]

    log.info(f"\n=== Cohort Overlap ===")
    log.info(f"  Sepsis cohort (mapped): {len(mapped)}")
    log.info(f"  Already processed:      {len(already_processed)}")
    log.info(f"  Missing (need extract):  {len(missing)}")

    # 4. Extract missing patients (waveforms + EHR)
    if len(missing) > 0:
        log.info("Loading EHR data for missing patient extraction...")
        OUT_DIR_OUTPUTS = REPO_ROOT / "workzone" / "outputs" / "mimic3"
        labs_df = pd.read_parquet(OUT_DIR_OUTPUTS / "labs_filtered.parquet")
        vitals_df = pd.read_parquet(OUT_DIR_OUTPUTS / "vitals_filtered.parquet")
        for df in [labs_df, vitals_df]:
            if "charttime_dt" not in df.columns:
                df["charttime_dt"] = pd.to_datetime(df["CHARTTIME"])
            if "HADM_ID" in df.columns:
                df["HADM_ID"] = pd.to_numeric(df["HADM_ID"], errors="coerce")

        adm_path = os.path.join(EHR_ROOT, "ADMISSIONS.csv.gz")
        if not os.path.exists(adm_path):
            adm_path = os.path.join(EHR_ROOT, "ADMISSIONS.csv")
        admissions_df = pd.read_csv(adm_path, usecols=["SUBJECT_ID", "HADM_ID", "ADMITTIME", "DISCHTIME"])
        admissions_df["ADMITTIME"] = pd.to_datetime(admissions_df["ADMITTIME"])
        admissions_df["DISCHTIME"] = pd.to_datetime(admissions_df["DISCHTIME"])

        extract_results = extract_missing_patients(missing, labs_df, vitals_df, admissions_df)

        # Update processed_dirs
        processed_dirs = set(os.listdir(PROCESSED_ROOT))

    # 5. Now get all matched (including newly extracted)
    matched = mapped[mapped["patient_id"].isin(processed_dirs)]
    log.info(f"\n=== After extraction ===")
    log.info(f"  Total matched: {len(matched)} patients")
    log.info(f"  Mortality: {matched['died'].sum()} ({matched['died'].mean()*100:.1f}%)")

    if len(matched) == 0:
        log.error("ABORT: No overlap between sepsis cohort and processed patients!")
        return

    # 4. Build sepsis-specific extra events (SEPARATE from canonical ehr_events.npy)
    tasks_dir = os.path.join(PROCESSED_ROOT, "tasks", "sepsis")
    extra_events_dir = os.path.join(tasks_dir, "extra_events")
    os.makedirs(extra_events_dir, exist_ok=True)

    log.info(f"\nBuilding sepsis extra events (SOFA + onset) -> {extra_events_dir}")
    log.info(f"  NOTE: canonical ehr_events.npy is NOT modified")
    n_events_added = 0
    for i, (_, row) in enumerate(matched.iterrows()):
        patient_dir = os.path.join(PROCESSED_ROOT, row["patient_id"])
        sofa_events = compute_sofa_from_ehr_events(patient_dir)
        n = build_sepsis_extra_events(patient_dir, row["onset_time"], sofa_events, extra_events_dir)
        n_events_added += n
        if (i + 1) % 200 == 0:
            log.info(f"  [{i+1}/{len(matched)}] {n_events_added} events added")

    log.info(f"  Total events added: {n_events_added}")

    # 5. Generate task-specific cohort + splits

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
