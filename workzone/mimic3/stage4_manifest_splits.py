#!/usr/bin/env python3
"""
Stage 4: Build manifest.json and train/test splits from processed patient directories.

Scans all processed patient directories, validates format, computes quality metrics,
and generates stratified train/test splits.

Run:  python workzone/mimic3/stage4_manifest_splits.py
Output:
  {output_dir}/manifest.json
  {output_dir}/pretrain_splits.json
  {output_dir}/downstream_splits.json
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
OUT_DIR_OUTPUTS = REPO_ROOT / "workzone" / "outputs" / "mimic3"

import yaml
with open(REPO_ROOT / "workzone" / "configs" / "server_paths.yaml") as f:
    cfg = yaml.safe_load(f)

PROCESSED_ROOT = cfg["mimic3"]["output_dir"]

SPLIT_SEED = 42
TEST_FRACTION = 0.2

EHR_EVENT_DTYPE = np.dtype([
    ('time_ms', 'int64'),
    ('seg_idx', 'int32'),
    ('var_id', 'uint16'),
    ('value', 'float32'),
])


def validate_patient(patient_dir):
    """Validate one patient directory. Returns (entry_dict, errors_list)."""
    entry = {"dir": os.path.basename(patient_dir)}
    errors = []

    # Check meta.json
    meta_path = os.path.join(patient_dir, "meta.json")
    if not os.path.exists(meta_path):
        errors.append("missing meta.json")
        return entry, errors

    with open(meta_path) as f:
        meta = json.load(f)

    entry.update({
        "subject_id": meta.get("subject_id"),
        "source_dataset": meta.get("source_dataset", "mimic3"),
        "n_segments": meta.get("n_segments", 0),
        "n_ehr_events": meta.get("n_ehr_events", 0),
        "duration_hours": meta.get("total_duration_hours", 0),
        "channels": list(meta.get("channels", {}).keys()),
    })

    n_seg = meta["n_segments"]

    # Validate each channel .npy
    for ch_name, ch_info in meta.get("channels", {}).items():
        npy_path = os.path.join(patient_dir, f"{ch_name}.npy")
        if not os.path.exists(npy_path):
            errors.append(f"missing {ch_name}.npy")
            continue
        arr = np.load(npy_path, mmap_mode='r')
        if arr.dtype != np.float16:
            errors.append(f"{ch_name}: dtype={arr.dtype}, expected float16")
        if not arr.flags['C_CONTIGUOUS']:
            errors.append(f"{ch_name}: not C-contiguous")
        if arr.shape[0] != n_seg:
            errors.append(f"{ch_name}: shape[0]={arr.shape[0]}, expected {n_seg}")

    # Validate time_ms
    time_path = os.path.join(patient_dir, "time_ms.npy")
    if not os.path.exists(time_path):
        errors.append("missing time_ms.npy")
    else:
        time_ms = np.load(time_path)
        if len(time_ms) != n_seg:
            errors.append(f"time_ms: len={len(time_ms)}, expected {n_seg}")
        if len(time_ms) > 1 and not np.all(np.diff(time_ms) > 0):
            errors.append("time_ms: not monotonically increasing")

    # Validate ehr_events
    ehr_path = os.path.join(patient_dir, "ehr_events.npy")
    if os.path.exists(ehr_path):
        events = np.load(ehr_path)
        if len(events) > 0:
            if events.dtype != EHR_EVENT_DTYPE:
                errors.append(f"ehr_events: dtype mismatch")
            if not np.all(np.diff(events['time_ms']) >= 0):
                errors.append("ehr_events: not sorted by time_ms")
            if np.any(events['seg_idx'] < 0) or np.any(events['seg_idx'] >= n_seg):
                errors.append(f"ehr_events: seg_idx out of bounds [0, {n_seg})")

    return entry, errors


def main():
    log.info(f"Stage 4: Build manifest and splits from {PROCESSED_ROOT}")
    t0 = time.time()

    # Find all patient directories
    patient_dirs = sorted([
        os.path.join(PROCESSED_ROOT, d)
        for d in os.listdir(PROCESSED_ROOT)
        if os.path.isdir(os.path.join(PROCESSED_ROOT, d))
    ])
    log.info(f"Found {len(patient_dirs)} patient directories")

    # Validate each
    manifest = []
    all_errors = {}
    n_pass = 0
    n_fail = 0

    for i, pd_path in enumerate(patient_dirs):
        entry, errors = validate_patient(pd_path)
        manifest.append(entry)

        if errors:
            all_errors[entry["dir"]] = errors
            n_fail += 1
        else:
            n_pass += 1

        if (i + 1) % 200 == 0:
            log.info(f"  Validated {i+1}/{len(patient_dirs)} (pass={n_pass}, fail={n_fail})")

    log.info(f"Validation: {n_pass} pass, {n_fail} fail")
    if all_errors:
        log.warning(f"Failed patients (first 10):")
        for k, v in list(all_errors.items())[:10]:
            log.warning(f"  {k}: {v}")

    # Filter to valid patients only
    valid_entries = [e for e in manifest if e["dir"] not in all_errors]

    # Save manifest
    manifest_path = os.path.join(PROCESSED_ROOT, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(valid_entries, f, indent=2)
    log.info(f"Manifest: {manifest_path} ({len(valid_entries)} patients)")

    # Generate splits -- split by SUBJECT_ID (not by dir)
    # All admissions from the same subject must be in the same set (no data leakage)
    rng = np.random.RandomState(SPLIT_SEED)

    # Get unique subject IDs
    subject_to_dirs = {}
    for e in valid_entries:
        sid = e.get("subject_id")
        if sid is None:
            # Parse from dir name: "154_102354" -> 154
            sid = int(e["dir"].split("_")[0])
        subject_to_dirs.setdefault(sid, []).append(e["dir"])

    unique_subjects = sorted(subject_to_dirs.keys())
    rng.shuffle(unique_subjects)

    n_test_subjects = int(len(unique_subjects) * TEST_FRACTION)
    test_subjects = set(unique_subjects[:n_test_subjects])
    train_subjects = set(unique_subjects[n_test_subjects:])

    # Map back to dirs
    train_ids = set()
    test_ids = set()
    for sid in train_subjects:
        train_ids.update(subject_to_dirs[sid])
    for sid in test_subjects:
        test_ids.update(subject_to_dirs[sid])

    # Verify no subject overlap
    train_sids = {int(d.split("_")[0]) for d in train_ids}
    test_sids = {int(d.split("_")[0]) for d in test_ids}
    overlap = train_sids & test_sids
    assert len(overlap) == 0, f"Subject overlap in splits: {overlap}"

    n_multi = sum(1 for dirs in subject_to_dirs.values() if len(dirs) > 1)
    log.info(f"  Unique subjects: {len(unique_subjects)} ({n_multi} with multiple admissions)")
    log.info(f"  Split by subject: train={len(train_subjects)} subjects, test={len(test_subjects)} subjects")
    log.info(f"  -> train={len(train_ids)} dirs, test={len(test_ids)} dirs")

    splits = {
        "train": sorted(train_ids),
        "test": sorted(test_ids),
        "seed": SPLIT_SEED,
        "test_fraction": TEST_FRACTION,
        "n_unique_subjects": len(unique_subjects),
        "n_train_subjects": len(train_subjects),
        "n_test_subjects": len(test_subjects),
        "n_train_dirs": len(train_ids),
        "n_test_dirs": len(test_ids),
        "n_subjects_with_multi_admissions": n_multi,
    }

    splits_path = os.path.join(PROCESSED_ROOT, "pretrain_splits.json")
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)
    log.info(f"Pretrain splits: {splits_path}")

    # Also save downstream splits in UNIPHY_Plus format
    downstream = {
        "train_control_list": [
            [os.path.join(PROCESSED_ROOT, pid), e.get("subject_id", int(pid.split("_")[0])),
             0, e["n_segments"], -1, 0]
            for e in valid_entries if e["dir"] in train_ids
            for pid in [e["dir"]]
        ],
        "test_control_list": [
            [os.path.join(PROCESSED_ROOT, pid), e.get("subject_id", int(pid.split("_")[0])),
             0, e["n_segments"], -1, 0]
            for e in valid_entries if e["dir"] in test_ids
            for pid in [e["dir"]]
        ],
    }

    ds_path = os.path.join(PROCESSED_ROOT, "downstream_splits.json")
    with open(ds_path, "w") as f:
        json.dump(downstream, f, indent=2)
    log.info(f"Downstream splits: {ds_path}")

    elapsed = time.time() - t0

    # Summary
    total_hours = sum(e.get("duration_hours", 0) for e in valid_entries)
    total_events = sum(e.get("n_ehr_events", 0) for e in valid_entries)

    log.info(f"\n=== Stage 4 Complete ===")
    log.info(f"  Valid patients: {len(valid_entries)}")
    log.info(f"  Total recording hours: {total_hours:.1f}")
    log.info(f"  Total EHR events: {total_events}")
    log.info(f"  Train/Test: {len(train_ids)}/{len(test_ids)} ({1-TEST_FRACTION:.0%}/{TEST_FRACTION:.0%})")
    log.info(f"  Time: {elapsed:.1f}s")

    # Save summary
    summary = {
        "n_valid": len(valid_entries),
        "n_failed": n_fail,
        "total_hours": round(total_hours, 1),
        "total_ehr_events": total_events,
        "n_train": len(train_ids),
        "n_test": len(test_ids),
        "failed_patients": dict(list(all_errors.items())[:20]),
    }
    with open(OUT_DIR_OUTPUTS / "stage4_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"\n=== All stages complete ===")
    log.info(f"Processed data at: {PROCESSED_ROOT}")
    log.info(f"Commit summaries: git add workzone/outputs/ && git commit -m 'MIMIC-III pipeline complete' && git push")


if __name__ == "__main__":
    main()
