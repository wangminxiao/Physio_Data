#!/usr/bin/env python3
"""
Stage 3: Extract waveforms from raw WFDB records into canonical .npy format.

For each patient in the filtered inventory:
  1. Read all WFDB segments, concatenate PLETH and II channels
  2. Resample: PLETH 125Hz -> 40Hz, II 125Hz -> 120Hz and 500Hz
  3. Segment into 30-second windows
  4. Build ehr_events.npy from labs + vitals parquet
  5. Save as per-patient directory with verification

Run:  python workzone/mimic3/stage3_extract_waveforms.py [--workers 8] [--limit 10]
Output: /opt/localdata100tb/physio_data/mimic3/{SUBJECT}_{HADM}/

Depends on:
  - stage2b output: record_inventory_final.parquet (waveform + EHR cross-checked)
  - stage2 output: labs_filtered.parquet, vitals_filtered.parquet, normalization_stats.json
"""
import os
import sys
import json
import time
import argparse
import logging
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from math import gcd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR_OUTPUTS = REPO_ROOT / "workzone" / "outputs" / "mimic3"

import yaml
with open(REPO_ROOT / "workzone" / "configs" / "server_paths.yaml") as f:
    cfg = yaml.safe_load(f)

WAV_ROOT = cfg["mimic3"]["raw_waveform_dir"]
PROCESSED_ROOT = cfg["mimic3"]["output_dir"]

# Canonical format constants
SEGMENT_DUR_SEC = 30
WAVEFORM_DTYPE = np.float16
TIME_DTYPE = np.int64
EHR_EVENT_DTYPE = np.dtype([
    ('time_ms', 'int64'),
    ('seg_idx', 'int32'),
    ('var_id', 'uint16'),
    ('value', 'float32'),
])

# Resampling targets
TARGETS = {
    "PLETH": [("PLETH40", 40)],
    "II": [("II120", 120), ("II500", 500)],
}
SOURCE_FS = 125  # MIMIC-III waveform source rate


def resample_signal(signal, src_fs, target_fs):
    """Resample 1D signal using polyphase filtering."""
    if src_fs == target_fs:
        return signal
    g = gcd(int(src_fs), int(target_fs))
    up = int(target_fs) // g
    down = int(src_fs) // g
    return resample_poly(signal, up, down).astype(np.float64)


def read_wfdb_segments(patient_path, channel_name):
    """Read and concatenate all segments for a given channel from a patient directory."""
    import wfdb

    # Find segment .hea files (sorted for correct temporal order)
    hea_files = sorted([
        f[:-4] for f in os.listdir(patient_path)
        if f.endswith(".hea") and "_layout" not in f
        and not f.startswith("p") and not f.endswith("n.hea")
        and f != "RECORDS"
    ])

    signals = []
    start_time = None

    for seg_name in hea_files:
        record_path = os.path.join(patient_path, seg_name)
        try:
            h = wfdb.rdheader(record_path)
        except Exception:
            continue

        if h.sig_name is None or channel_name not in h.sig_name:
            continue

        ch_idx = h.sig_name.index(channel_name)

        # Get start time from first segment
        if start_time is None:
            try:
                with open(record_path + ".hea") as fh:
                    parts = fh.readline().strip().split()
                    time_str = parts[4] if len(parts) >= 5 else None
                    date_str = parts[5] if len(parts) >= 6 else None
                    if time_str and date_str:
                        # Parse "HH:MM:SS.mmm DD/MM/YYYY"
                        dt_str = f"{date_str} {time_str.split('.')[0]}"
                        start_time = datetime.strptime(dt_str, "%d/%m/%Y %H:%M:%S")
            except Exception:
                pass

        # Read signal data
        try:
            rec = wfdb.rdrecord(record_path, channels=[ch_idx])
            sig = rec.p_signal[:, 0]
            signals.append(sig)
        except Exception:
            continue

    if not signals:
        return None, None

    concatenated = np.concatenate(signals)
    return concatenated, start_time


def segment_signal(signal, sample_rate, seg_dur_sec=SEGMENT_DUR_SEC):
    """Split 1D signal into [N_seg, samples_per_seg] float16 C-contiguous array."""
    samples_per_seg = int(sample_rate * seg_dur_sec)
    n_seg = len(signal) // samples_per_seg

    if n_seg == 0:
        return None

    # Trim to exact multiple
    trimmed = signal[:n_seg * samples_per_seg]
    segmented = trimmed.reshape(n_seg, samples_per_seg)

    # Ensure float16 C-contiguous
    result = np.ascontiguousarray(segmented, dtype=WAVEFORM_DTYPE)
    return result


def build_ehr_events(subject_id, segment_times_ms, labs_df, vitals_df):
    """Build sparse ehr_events array from labs + vitals DataFrames."""
    events = []

    for source_df in [labs_df, vitals_df]:
        patient_events = source_df[source_df["SUBJECT_ID"] == subject_id]
        if len(patient_events) == 0:
            continue

        for _, row in patient_events.iterrows():
            event_time_ms = int(row["charttime_dt"].timestamp() * 1000)
            seg_idx = np.searchsorted(segment_times_ms, event_time_ms, side="right") - 1

            if seg_idx < 0 or seg_idx >= len(segment_times_ms):
                continue

            events.append((event_time_ms, int(seg_idx), int(row["var_id"]), float(row["VALUENUM"])))

    if not events:
        return np.array([], dtype=EHR_EVENT_DTYPE)

    result = np.array(events, dtype=EHR_EVENT_DTYPE)
    result.sort(order="time_ms")
    return result


def save_patient(out_dir, channels, time_ms, ehr_events, meta_extra):
    """Save one patient in canonical format with inline verification."""
    os.makedirs(out_dir, exist_ok=True)

    n_seg = len(time_ms)

    # Save and verify each channel
    for name, arr in channels.items():
        assert arr.dtype == WAVEFORM_DTYPE, f"{name}: expected {WAVEFORM_DTYPE}, got {arr.dtype}"
        assert arr.ndim == 2, f"{name}: expected 2D, got {arr.ndim}D"
        assert arr.flags['C_CONTIGUOUS'], f"{name}: not C-contiguous"
        assert arr.shape[0] == n_seg, f"{name}: {arr.shape[0]} rows != {n_seg} segments"
        np.save(os.path.join(out_dir, f"{name}.npy"), arr)

    # Save time
    assert time_ms.dtype == TIME_DTYPE
    assert np.all(np.diff(time_ms) > 0), "time_ms not monotonically increasing"
    np.save(os.path.join(out_dir, "time_ms.npy"), time_ms)

    # Save EHR events
    if len(ehr_events) > 0:
        assert ehr_events.dtype == EHR_EVENT_DTYPE
        assert np.all(np.diff(ehr_events['time_ms']) >= 0), "ehr_events not sorted"
        assert np.all(ehr_events['seg_idx'] >= 0), "negative seg_idx"
        assert np.all(ehr_events['seg_idx'] < n_seg), f"seg_idx >= {n_seg}"
    np.save(os.path.join(out_dir, "ehr_events.npy"), ehr_events)

    # Save meta.json
    meta = {
        "n_segments": n_seg,
        "segment_duration_sec": SEGMENT_DUR_SEC,
        "channels": {
            name: {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "sample_rate_hz": int(arr.shape[1] / SEGMENT_DUR_SEC),
            }
            for name, arr in channels.items()
        },
        "n_ehr_events": len(ehr_events),
        **meta_extra,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def process_patient(row, labs_df, vitals_df):
    """Process one patient: read WFDB -> resample -> segment -> align EHR -> save."""
    subject_id = row["subject_id"]
    patient_path = row["patient_path"]

    try:
        # Read raw waveform channels
        pleth_raw, start_time = read_wfdb_segments(patient_path, "PLETH")
        ii_raw, ii_start = read_wfdb_segments(patient_path, "II")

        if pleth_raw is None or ii_raw is None:
            return {"subject_id": subject_id, "status": "SKIP", "reason": "missing channel data"}

        if start_time is None:
            start_time = ii_start
        if start_time is None:
            return {"subject_id": subject_id, "status": "SKIP", "reason": "no start time"}

        # Resample
        pleth40 = resample_signal(pleth_raw, SOURCE_FS, 40)
        ii120 = resample_signal(ii_raw, SOURCE_FS, 120)
        ii500 = resample_signal(ii_raw, SOURCE_FS, 500)

        # Segment into 30s windows
        pleth40_seg = segment_signal(pleth40, 40)
        ii120_seg = segment_signal(ii120, 120)
        ii500_seg = segment_signal(ii500, 500)

        if pleth40_seg is None or ii120_seg is None or ii500_seg is None:
            return {"subject_id": subject_id, "status": "SKIP", "reason": "too short after resampling"}

        # Use minimum segment count across channels
        n_seg = min(pleth40_seg.shape[0], ii120_seg.shape[0], ii500_seg.shape[0])
        pleth40_seg = pleth40_seg[:n_seg]
        ii120_seg = ii120_seg[:n_seg]
        ii500_seg = ii500_seg[:n_seg]

        # Build time_ms array
        start_ms = int(start_time.timestamp() * 1000)
        time_ms = np.array(
            [start_ms + i * SEGMENT_DUR_SEC * 1000 for i in range(n_seg)],
            dtype=TIME_DTYPE,
        )

        # Build EHR events
        ehr_events = build_ehr_events(subject_id, time_ms, labs_df, vitals_df)

        # Determine output directory
        # Use SUBJECT_ID as patient_id (HADM_ID linkage done later via EHR)
        out_dir = os.path.join(PROCESSED_ROOT, str(subject_id))

        # NaN check
        pleth_nan_ratio = np.isnan(pleth40_seg.astype(np.float32)).mean()
        ii_nan_ratio = np.isnan(ii120_seg.astype(np.float32)).mean()

        channels = {"PLETH40": pleth40_seg, "II120": ii120_seg, "II500": ii500_seg}

        save_patient(
            out_dir=out_dir,
            channels=channels,
            time_ms=time_ms,
            ehr_events=ehr_events,
            meta_extra={
                "subject_id": int(subject_id),
                "source_dataset": "mimic3",
                "source_path": patient_path,
                "recording_start_ms": int(start_ms),
                "total_duration_hours": round(n_seg * SEGMENT_DUR_SEC / 3600, 2),
                "pleth_nan_ratio": round(float(pleth_nan_ratio), 4),
                "ii_nan_ratio": round(float(ii_nan_ratio), 4),
            },
        )

        return {
            "subject_id": subject_id,
            "status": "OK",
            "n_segments": n_seg,
            "n_ehr_events": len(ehr_events),
            "duration_hours": round(n_seg * SEGMENT_DUR_SEC / 3600, 2),
        }

    except Exception as e:
        return {
            "subject_id": subject_id,
            "status": "ERROR",
            "reason": str(e),
            "traceback": traceback.format_exc(),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Process only N patients (for testing)")
    args = parser.parse_args()

    log.info(f"Stage 3: Extract waveforms -> {PROCESSED_ROOT}")
    os.makedirs(PROCESSED_ROOT, exist_ok=True)

    # Load inventory (cross-checked: waveform + EHR confirmed)
    inv_path = OUT_DIR_OUTPUTS / "record_inventory_final.parquet"
    if not inv_path.exists():
        log.error("record_inventory_final.parquet not found. Run stage2b_cross_check.py first!")
        sys.exit(1)
    inventory = pd.read_parquet(inv_path)
    log.info(f"Loaded inventory: {len(inventory)} patients (waveform + EHR confirmed)")

    if args.limit:
        inventory = inventory.head(args.limit)
        log.info(f"  Limited to {len(inventory)} patients")

    # Load EHR data
    log.info("Loading EHR data...")
    labs_df = pd.read_parquet(OUT_DIR_OUTPUTS / "labs_filtered.parquet")
    vitals_df = pd.read_parquet(OUT_DIR_OUTPUTS / "vitals_filtered.parquet")

    # Convert charttime_dt to pandas datetime if needed
    for df in [labs_df, vitals_df]:
        if "charttime_dt" not in df.columns:
            df["charttime_dt"] = pd.to_datetime(df["CHARTTIME"])

    log.info(f"  Labs: {len(labs_df)} events, Vitals: {len(vitals_df)} events")

    # Process patients
    t0 = time.time()
    results = []

    # Sequential for now (wfdb doesn't always play nice with multiprocessing)
    for i, (_, row) in enumerate(inventory.iterrows()):
        result = process_patient(row, labs_df, vitals_df)
        results.append(result)

        if (i + 1) % 50 == 0 or (i + 1) == len(inventory):
            n_ok = sum(1 for r in results if r["status"] == "OK")
            n_skip = sum(1 for r in results if r["status"] == "SKIP")
            n_err = sum(1 for r in results if r["status"] == "ERROR")
            elapsed = time.time() - t0
            log.info(f"  [{i+1}/{len(inventory)}] OK={n_ok} SKIP={n_skip} ERR={n_err} ({elapsed:.0f}s)")

    # Summary
    elapsed = time.time() - t0
    n_ok = sum(1 for r in results if r["status"] == "OK")
    n_skip = sum(1 for r in results if r["status"] == "SKIP")
    n_err = sum(1 for r in results if r["status"] == "ERROR")

    summary = {
        "total_attempted": len(results),
        "ok": n_ok,
        "skipped": n_skip,
        "errors": n_err,
        "total_time_sec": round(elapsed, 1),
        "output_dir": PROCESSED_ROOT,
        "errors_detail": [r for r in results if r["status"] == "ERROR"][:20],
        "skips_detail": [r for r in results if r["status"] == "SKIP"][:20],
    }

    summary_path = OUT_DIR_OUTPUTS / "stage3_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    log.info(f"\n=== Stage 3 Complete ===")
    log.info(f"  OK: {n_ok}, SKIP: {n_skip}, ERROR: {n_err}")
    log.info(f"  Time: {elapsed:.0f}s")
    log.info(f"  Output: {PROCESSED_ROOT}")
    log.info(f"  Summary: {summary_path}")
    log.info(f"\nNext: python workzone/mimic3/stage4_manifest_splits.py")


if __name__ == "__main__":
    main()
