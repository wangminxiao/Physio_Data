#!/usr/bin/env python3
"""
Stage 3: Extract waveforms from raw WFDB records into canonical .npy format.

For each patient in the filtered inventory:
  1. Read master header to get waveform recording time
  2. Match to hospital admission (HADM_ID) via ADMISSIONS table
  3. Read all WFDB segments, concatenate PLETH and II channels
  4. Resample: PLETH 125Hz -> 40Hz, II 125Hz -> 120Hz and 500Hz
  5. Segment into 30-second windows
  6. Build ehr_events.npy from labs + vitals filtered by HADM_ID
  7. Save as per-patient directory: {SUBJECT_ID}_{HADM_ID}/

Run:  python workzone/mimic3/stage3_extract_waveforms.py [--limit 10]
Output: /opt/localdata100tb/physio_data/mimic3/{SUBJECT_ID}_{HADM_ID}/

Depends on:
  - stage2b output: record_inventory_final.parquet
  - stage2 output: labs_filtered.parquet, vitals_filtered.parquet
  - ADMISSIONS.csv.gz from raw MIMIC-III
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
from math import gcd

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR_OUTPUTS = REPO_ROOT / "workzone" / "outputs" / "mimic3"

import yaml
with open(REPO_ROOT / "workzone" / "configs" / "server_paths.yaml") as f:
    cfg = yaml.safe_load(f)

WAV_ROOT = cfg["mimic3"]["raw_waveform_dir"]
EHR_ROOT = cfg["mimic3"]["raw_ehr_dir"]
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
SOURCE_FS = 125  # MIMIC-III waveform source rate


# ========================================================================
# Waveform reading
# ========================================================================

def get_master_start_time(patient_path):
    """Read start time + total duration from master record header."""
    for f in os.listdir(patient_path):
        if f.startswith("p") and f.endswith(".hea") and not f.endswith("n.hea"):
            try:
                with open(os.path.join(patient_path, f)) as fh:
                    parts = fh.readline().strip().split()
                    time_str = None
                    date_str = None
                    total_samples = None
                    fs = None
                    for i, p in enumerate(parts):
                        if '/' in p and len(p) == 10 and p[2] == '/' and p[5] == '/':
                            date_str = p
                        elif ':' in p and '.' in p and len(p) > 8:
                            time_str = p
                    # Format: name/n_seg n_sig fs total_samples time date
                    # e.g.: p000020-2183-04-28-17-47/10 4 125 9862593 17:47:59.486 28/04/2183
                    try:
                        fs = float(parts[2])
                        total_samples = int(parts[3])
                    except (IndexError, ValueError):
                        pass

                    if time_str and date_str:
                        dt_str = f"{date_str} {time_str.split('.')[0]}"
                        start_dt = datetime.strptime(dt_str, "%d/%m/%Y %H:%M:%S")
                        duration_sec = total_samples / fs if fs and total_samples else None
                        return start_dt, duration_sec
            except Exception:
                continue
    return None, None


def read_wfdb_segments(patient_path, channel_name):
    """Read and concatenate all segments for a given channel."""
    import wfdb

    hea_files = sorted([
        f[:-4] for f in os.listdir(patient_path)
        if f.endswith(".hea") and "_layout" not in f
        and not f.startswith("p") and not f.endswith("n.hea")
        and f != "RECORDS"
    ])

    signals = []
    for seg_name in hea_files:
        record_path = os.path.join(patient_path, seg_name)
        try:
            h = wfdb.rdheader(record_path)
        except Exception:
            continue
        if h.sig_name is None or channel_name not in h.sig_name:
            continue
        ch_idx = h.sig_name.index(channel_name)
        try:
            rec = wfdb.rdrecord(record_path, channels=[ch_idx])
            signals.append(rec.p_signal[:, 0])
        except Exception:
            continue

    if not signals:
        return None
    return np.concatenate(signals)


def resample_signal(signal, src_fs, target_fs):
    """Resample 1D signal using polyphase filtering."""
    if src_fs == target_fs:
        return signal
    g = gcd(int(src_fs), int(target_fs))
    up = int(target_fs) // g
    down = int(src_fs) // g
    return resample_poly(signal, up, down).astype(np.float64)


def segment_signal(signal, sample_rate, seg_dur_sec=SEGMENT_DUR_SEC):
    """Split 1D signal into [N_seg, samples_per_seg] float16 C-contiguous."""
    samples_per_seg = int(sample_rate * seg_dur_sec)
    n_seg = len(signal) // samples_per_seg
    if n_seg == 0:
        return None
    trimmed = signal[:n_seg * samples_per_seg]
    return np.ascontiguousarray(trimmed.reshape(n_seg, samples_per_seg), dtype=WAVEFORM_DTYPE)


# ========================================================================
# Admission matching
# ========================================================================

def match_admission(subject_id, wav_start, wav_duration_sec, admissions_df):
    """Find the HADM_ID whose admission period best overlaps with the waveform recording.

    Returns (hadm_id, overlap_hours) or (None, 0) if no overlap.
    """
    wav_end = wav_start + timedelta(seconds=wav_duration_sec)

    subj_adm = admissions_df[admissions_df["SUBJECT_ID"] == subject_id]
    if len(subj_adm) == 0:
        return None, 0

    best_hadm = None
    best_overlap = 0

    for _, row in subj_adm.iterrows():
        adm_start = row["ADMITTIME"]
        adm_end = row["DISCHTIME"]
        if pd.isna(adm_start) or pd.isna(adm_end):
            continue

        # Compute overlap
        overlap_start = max(wav_start, adm_start.to_pydatetime())
        overlap_end = min(wav_end, adm_end.to_pydatetime())
        overlap_sec = max(0, (overlap_end - overlap_start).total_seconds())

        if overlap_sec > best_overlap:
            best_overlap = overlap_sec
            best_hadm = int(row["HADM_ID"])

    return best_hadm, best_overlap / 3600


# ========================================================================
# EHR event building (filtered by HADM_ID)
# ========================================================================

def build_ehr_events(subject_id, hadm_id, segment_times_ms, labs_df, vitals_df):
    """Build sparse ehr_events from labs + vitals, filtered by SUBJECT_ID AND HADM_ID."""
    events = []

    for source_df in [labs_df, vitals_df]:
        # Filter by SUBJECT_ID and HADM_ID
        mask = (source_df["SUBJECT_ID"] == subject_id)
        if hadm_id is not None and "HADM_ID" in source_df.columns:
            mask = mask & (source_df["HADM_ID"] == hadm_id)
        patient_events = source_df[mask]

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


# ========================================================================
# Save with verification
# ========================================================================

def save_patient(out_dir, channels, time_ms, ehr_events, meta_extra):
    """Save one patient in canonical format with inline verification."""
    os.makedirs(out_dir, exist_ok=True)
    n_seg = len(time_ms)

    for name, arr in channels.items():
        assert arr.dtype == WAVEFORM_DTYPE, f"{name}: expected {WAVEFORM_DTYPE}, got {arr.dtype}"
        assert arr.ndim == 2, f"{name}: expected 2D, got {arr.ndim}D"
        assert arr.flags['C_CONTIGUOUS'], f"{name}: not C-contiguous"
        assert arr.shape[0] == n_seg, f"{name}: {arr.shape[0]} rows != {n_seg} segments"
        np.save(os.path.join(out_dir, f"{name}.npy"), arr)

    assert time_ms.dtype == TIME_DTYPE
    assert np.all(np.diff(time_ms) > 0), "time_ms not monotonically increasing"
    np.save(os.path.join(out_dir, "time_ms.npy"), time_ms)

    if len(ehr_events) > 0:
        assert ehr_events.dtype == EHR_EVENT_DTYPE
        assert np.all(np.diff(ehr_events['time_ms']) >= 0), "ehr_events not sorted"
        assert np.all(ehr_events['seg_idx'] >= 0), "negative seg_idx"
        assert np.all(ehr_events['seg_idx'] < n_seg), f"seg_idx >= {n_seg}"
    np.save(os.path.join(out_dir, "ehr_events.npy"), ehr_events)

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


# ========================================================================
# Process one patient
# ========================================================================

def process_patient(row, labs_df, vitals_df, admissions_df):
    """Process one patient: match admission -> read WFDB -> resample -> align EHR -> save."""
    subject_id = row["subject_id"]
    patient_path = row["patient_path"]

    try:
        # 1. Get waveform recording time from master header
        wav_start, wav_duration = get_master_start_time(patient_path)
        if wav_start is None:
            return {"subject_id": subject_id, "status": "SKIP", "reason": "no master header start time"}
        if wav_duration is None or wav_duration < 300:
            return {"subject_id": subject_id, "status": "SKIP", "reason": f"duration too short: {wav_duration}s"}

        # 2. Match to hospital admission
        hadm_id, overlap_hours = match_admission(subject_id, wav_start, wav_duration, admissions_df)
        if hadm_id is None:
            return {"subject_id": subject_id, "status": "SKIP", "reason": "no matching admission"}
        if overlap_hours < 1.0:
            return {"subject_id": subject_id, "status": "SKIP",
                    "reason": f"admission overlap too short: {overlap_hours:.1f}h"}

        # 3. Read raw waveform channels
        pleth_raw = read_wfdb_segments(patient_path, "PLETH")
        ii_raw = read_wfdb_segments(patient_path, "II")

        if pleth_raw is None or ii_raw is None:
            return {"subject_id": subject_id, "hadm_id": hadm_id,
                    "status": "SKIP", "reason": "missing channel data"}

        # 4. Resample
        pleth40 = resample_signal(pleth_raw, SOURCE_FS, 40)
        ii120 = resample_signal(ii_raw, SOURCE_FS, 120)
        ii500 = resample_signal(ii_raw, SOURCE_FS, 500)

        # 5. Segment into 30s windows
        pleth40_seg = segment_signal(pleth40, 40)
        ii120_seg = segment_signal(ii120, 120)
        ii500_seg = segment_signal(ii500, 500)

        if pleth40_seg is None or ii120_seg is None or ii500_seg is None:
            return {"subject_id": subject_id, "hadm_id": hadm_id,
                    "status": "SKIP", "reason": "too short after resampling"}

        # Use minimum segment count across channels
        n_seg = min(pleth40_seg.shape[0], ii120_seg.shape[0], ii500_seg.shape[0])
        pleth40_seg = pleth40_seg[:n_seg]
        ii120_seg = ii120_seg[:n_seg]
        ii500_seg = ii500_seg[:n_seg]

        # 6. Build time_ms array
        start_ms = int(wav_start.timestamp() * 1000)
        time_ms = np.array(
            [start_ms + i * SEGMENT_DUR_SEC * 1000 for i in range(n_seg)],
            dtype=TIME_DTYPE,
        )

        # 7. Build EHR events (filtered by HADM_ID)
        ehr_events = build_ehr_events(subject_id, hadm_id, time_ms, labs_df, vitals_df)

        # 8. Save as {SUBJECT_ID}_{HADM_ID}/
        patient_id = f"{subject_id}_{hadm_id}"
        out_dir = os.path.join(PROCESSED_ROOT, patient_id)

        pleth_nan = np.isnan(pleth40_seg.astype(np.float32)).mean()
        ii_nan = np.isnan(ii120_seg.astype(np.float32)).mean()

        channels = {"PLETH40": pleth40_seg, "II120": ii120_seg, "II500": ii500_seg}

        save_patient(
            out_dir=out_dir,
            channels=channels,
            time_ms=time_ms,
            ehr_events=ehr_events,
            meta_extra={
                "patient_id": patient_id,
                "subject_id": int(subject_id),
                "hadm_id": int(hadm_id),
                "source_dataset": "mimic3",
                "source_path": patient_path,
                "recording_start_ms": int(start_ms),
                "total_duration_hours": round(n_seg * SEGMENT_DUR_SEC / 3600, 2),
                "admission_overlap_hours": round(overlap_hours, 2),
                "pleth_nan_ratio": round(float(pleth_nan), 4),
                "ii_nan_ratio": round(float(ii_nan), 4),
            },
        )

        return {
            "subject_id": subject_id,
            "hadm_id": hadm_id,
            "patient_id": patient_id,
            "status": "OK",
            "n_segments": n_seg,
            "n_ehr_events": len(ehr_events),
            "duration_hours": round(n_seg * SEGMENT_DUR_SEC / 3600, 2),
            "overlap_hours": round(overlap_hours, 2),
        }

    except Exception as e:
        return {
            "subject_id": subject_id,
            "status": "ERROR",
            "reason": str(e),
            "traceback": traceback.format_exc(),
        }


# ========================================================================
# Main
# ========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Process only N patients (for testing)")
    args = parser.parse_args()

    log.info(f"Stage 3: Extract waveforms -> {PROCESSED_ROOT}")
    os.makedirs(PROCESSED_ROOT, exist_ok=True)

    # Load inventory
    inv_path = OUT_DIR_OUTPUTS / "record_inventory_final.parquet"
    if not inv_path.exists():
        log.error("record_inventory_final.parquet not found. Run stage2b first!")
        sys.exit(1)
    inventory = pd.read_parquet(inv_path)
    log.info(f"Loaded inventory: {len(inventory)} patients")

    if args.limit:
        inventory = inventory.head(args.limit)
        log.info(f"  Limited to {len(inventory)} patients")

    # Load ADMISSIONS table for HADM_ID matching
    log.info("Loading ADMISSIONS...")
    adm_path = os.path.join(EHR_ROOT, "ADMISSIONS.csv.gz")
    if not os.path.exists(adm_path):
        adm_path = os.path.join(EHR_ROOT, "ADMISSIONS.csv")
    admissions_df = pd.read_csv(adm_path, usecols=["SUBJECT_ID", "HADM_ID", "ADMITTIME", "DISCHTIME"])
    admissions_df["ADMITTIME"] = pd.to_datetime(admissions_df["ADMITTIME"])
    admissions_df["DISCHTIME"] = pd.to_datetime(admissions_df["DISCHTIME"])
    log.info(f"  {len(admissions_df)} admissions loaded")

    # Load EHR data
    log.info("Loading EHR data...")
    labs_df = pd.read_parquet(OUT_DIR_OUTPUTS / "labs_filtered.parquet")
    vitals_df = pd.read_parquet(OUT_DIR_OUTPUTS / "vitals_filtered.parquet")
    for df in [labs_df, vitals_df]:
        if "charttime_dt" not in df.columns:
            df["charttime_dt"] = pd.to_datetime(df["CHARTTIME"])
        # Ensure HADM_ID is numeric (not float with NaN)
        if "HADM_ID" in df.columns:
            df["HADM_ID"] = pd.to_numeric(df["HADM_ID"], errors="coerce")
    log.info(f"  Labs: {len(labs_df)} events, Vitals: {len(vitals_df)} events")

    # Process patients
    t0 = time.time()
    results = []

    for i, (_, row) in enumerate(inventory.iterrows()):
        result = process_patient(row, labs_df, vitals_df, admissions_df)
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

    # Count skip reasons
    skip_reasons = {}
    for r in results:
        if r["status"] == "SKIP":
            reason = r.get("reason", "unknown")
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

    summary = {
        "total_attempted": len(results),
        "ok": n_ok,
        "skipped": n_skip,
        "errors": n_err,
        "total_time_sec": round(elapsed, 1),
        "output_dir": PROCESSED_ROOT,
        "skip_reasons": skip_reasons,
        "errors_detail": [r for r in results if r["status"] == "ERROR"][:20],
        "skips_detail": [r for r in results if r["status"] == "SKIP"][:20],
        "ok_sample": [r for r in results if r["status"] == "OK"][:5],
    }

    summary_path = OUT_DIR_OUTPUTS / "stage3_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    log.info(f"\n=== Stage 3 Complete ===")
    log.info(f"  OK: {n_ok}, SKIP: {n_skip}, ERROR: {n_err}")
    log.info(f"  Skip reasons: {skip_reasons}")
    log.info(f"  Time: {elapsed:.0f}s")
    log.info(f"  Output: {PROCESSED_ROOT}")

    if n_ok > 0:
        ok_results = [r for r in results if r["status"] == "OK"]
        avg_events = np.mean([r["n_ehr_events"] for r in ok_results])
        avg_hours = np.mean([r["duration_hours"] for r in ok_results])
        log.info(f"  Avg EHR events/patient: {avg_events:.0f}")
        log.info(f"  Avg duration: {avg_hours:.1f} hours")

    log.info(f"\nNext: python workzone/mimic3/stage4_manifest_splits.py")


if __name__ == "__main__":
    main()
