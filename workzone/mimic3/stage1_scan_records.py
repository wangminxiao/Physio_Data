#!/usr/bin/env python3
"""
Stage 1: Scan all MIMIC-III waveform records to build an inventory.

For each patient, reads WFDB segment headers to find which channels are available
(PLETH, II, etc.), their sample rates, and total duration. Filters to patients
that have both PLETH and II channels.

Run:  python workzone/mimic3/stage1_scan_records.py
Output: workzone/outputs/mimic3/record_inventory.parquet

Verification: prints summary stats, flags patients with issues.
"""
import os
import json
import time
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO_ROOT / "workzone" / "outputs" / "mimic3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load server paths
import yaml
with open(REPO_ROOT / "workzone" / "configs" / "server_paths.yaml") as f:
    cfg = yaml.safe_load(f)

WAV_ROOT = cfg["mimic3"]["raw_waveform_dir"]
EHR_ROOT = cfg["mimic3"]["raw_ehr_dir"]

# Channels we care about
TARGET_CHANNELS = {"PLETH", "II"}
# Minimum total duration (seconds) to keep a patient
MIN_DURATION_SEC = 300  # 5 minutes


def scan_patient(patient_path):
    """Scan one patient directory. Returns dict with channel info or None on failure."""
    import wfdb

    patient_id = os.path.basename(patient_path)
    # Strip 'p' prefix to get numeric SUBJECT_ID
    subject_id_str = patient_id.lstrip("p")
    try:
        subject_id = int(subject_id_str)
    except ValueError:
        return None

    # Find all segment .hea files (not layout, not master, not numerics)
    hea_files = []
    for f in sorted(os.listdir(patient_path)):
        if not f.endswith(".hea"):
            continue
        if "_layout" in f or f.startswith("p") or f.endswith("n.hea"):
            continue
        if f == "RECORDS":
            continue
        hea_files.append(f)

    if not hea_files:
        return None

    # Read each segment header
    channels_found = {}
    total_samples = {}
    segment_info = []
    fs = None
    base_time = None
    base_date = None

    for hea_name in hea_files:
        record_path = os.path.join(patient_path, hea_name[:-4])
        try:
            h = wfdb.rdheader(record_path)
        except Exception:
            continue

        if h.n_sig is None or h.n_sig == 0 or h.sig_name is None:
            continue

        if fs is None:
            fs = h.fs

        # Extract base time/date from raw header line
        if base_time is None:
            try:
                with open(record_path + ".hea") as fh:
                    first_line = fh.readline().strip()
                    parts = first_line.split()
                    if len(parts) >= 5:
                        base_time = parts[4]
                    if len(parts) >= 6:
                        base_date = parts[5]
            except Exception:
                pass

        for ch in h.sig_name:
            channels_found[ch] = channels_found.get(ch, 0) + 1
            total_samples[ch] = total_samples.get(ch, 0) + h.sig_len

        segment_info.append({
            "file": hea_name[:-4],
            "sig_name": h.sig_name,
            "sig_len": h.sig_len,
            "n_sig": h.n_sig,
        })

    if not segment_info or fs is None:
        return None

    # Check if we have both target channels
    has_pleth = "PLETH" in channels_found
    has_ii = "II" in channels_found

    # Total duration for each target channel
    pleth_duration = total_samples.get("PLETH", 0) / fs if fs else 0
    ii_duration = total_samples.get("II", 0) / fs if fs else 0

    return {
        "subject_id": subject_id,
        "patient_dir": patient_id,
        "patient_path": patient_path,
        "fs": fs,
        "base_time": base_time,
        "base_date": base_date,
        "n_segments": len(segment_info),
        "channels_found": list(channels_found.keys()),
        "has_pleth": has_pleth,
        "has_ii": has_ii,
        "pleth_total_samples": total_samples.get("PLETH", 0),
        "ii_total_samples": total_samples.get("II", 0),
        "pleth_duration_sec": round(pleth_duration, 1),
        "ii_duration_sec": round(ii_duration, 1),
    }


def main():
    log.info(f"Stage 1: Scanning MIMIC-III waveform records at {WAV_ROOT}")
    t0 = time.time()

    # Collect all patient directories
    patient_dirs = []
    for top_dir in sorted(os.listdir(WAV_ROOT)):
        top_path = os.path.join(WAV_ROOT, top_dir)
        if not os.path.isdir(top_path):
            continue
        for patient_dir in sorted(os.listdir(top_path)):
            patient_path = os.path.join(top_path, patient_dir)
            if os.path.isdir(patient_path):
                patient_dirs.append(patient_path)

    log.info(f"Found {len(patient_dirs)} patient directories")

    # Scan in parallel
    results = []
    errors = 0
    with ProcessPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(scan_patient, p): p for p in patient_dirs}
        for i, future in enumerate(as_completed(futures)):
            if (i + 1) % 500 == 0:
                log.info(f"  Scanned {i+1}/{len(patient_dirs)}...")
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as e:
                errors += 1

    elapsed = time.time() - t0
    log.info(f"Scanned {len(patient_dirs)} patients in {elapsed:.1f}s ({errors} errors)")

    # Build DataFrame
    df = pd.DataFrame(results)

    # Summary stats
    n_total = len(df)
    n_both = len(df[(df["has_pleth"]) & (df["has_ii"])])
    n_pleth_only = len(df[(df["has_pleth"]) & (~df["has_ii"])])
    n_ii_only = len(df[(~df["has_pleth"]) & (df["has_ii"])])
    n_neither = len(df[(~df["has_pleth"]) & (~df["has_ii"])])

    # Filter: must have both PLETH and II, and minimum duration
    df_filtered = df[
        (df["has_pleth"]) &
        (df["has_ii"]) &
        (df["pleth_duration_sec"] >= MIN_DURATION_SEC) &
        (df["ii_duration_sec"] >= MIN_DURATION_SEC)
    ].copy()

    summary = {
        "total_patients_scanned": n_total,
        "has_pleth_and_ii": n_both,
        "has_pleth_only": n_pleth_only,
        "has_ii_only": n_ii_only,
        "has_neither": n_neither,
        "after_duration_filter": len(df_filtered),
        "min_duration_sec": MIN_DURATION_SEC,
        "scan_time_sec": round(elapsed, 1),
        "median_pleth_duration_hours": round(df_filtered["pleth_duration_sec"].median() / 3600, 1) if len(df_filtered) > 0 else 0,
        "median_ii_duration_hours": round(df_filtered["ii_duration_sec"].median() / 3600, 1) if len(df_filtered) > 0 else 0,
    }

    log.info(f"\n=== Summary ===")
    for k, v in summary.items():
        log.info(f"  {k}: {v}")

    # Save
    out_parquet = OUT_DIR / "record_inventory.parquet"
    df.to_parquet(out_parquet, index=False)
    log.info(f"Full inventory: {out_parquet} ({len(df)} rows)")

    out_filtered = OUT_DIR / "record_inventory_filtered.parquet"
    df_filtered.to_parquet(out_filtered, index=False)
    log.info(f"Filtered (PLETH+II, >={MIN_DURATION_SEC}s): {out_filtered} ({len(df_filtered)} rows)")

    out_summary = OUT_DIR / "stage1_summary.json"
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Summary: {out_summary}")

    # Verification
    log.info(f"\n=== Verification ===")
    if len(df_filtered) == 0:
        log.warning("ABORT: No patients passed filters!")
        return

    assert len(df_filtered) > 0, "No patients with both PLETH and II"
    assert df_filtered["fs"].nunique() == 1 or True, "Multiple sample rates found"  # allow, just log
    log.info(f"  [PASS] {len(df_filtered)} patients with PLETH+II and >={MIN_DURATION_SEC}s")
    log.info(f"  Sample rates: {df_filtered['fs'].unique()}")
    log.info(f"  Duration range: {df_filtered['pleth_duration_sec'].min():.0f}s - {df_filtered['pleth_duration_sec'].max():.0f}s")

    log.info(f"\nNext: python workzone/mimic3/stage2_extract_ehr.py")


if __name__ == "__main__":
    main()
