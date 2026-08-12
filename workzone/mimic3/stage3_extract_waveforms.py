#!/usr/bin/env python3
"""
Stage 3: Extract waveforms from raw WFDB records into canonical .npy format.

PLETH-anchored alignment: only include WFDB segments where PLETH is present.
Other channels (II) are NaN-filled when absent in a PLETH-present segment.
Gaps between recording blocks are reflected in time_ms jumps (no NaN-fill).
Segments use 30 s non-overlapping windows (30 s stride) — consistent with
the other physio_data datasets (Emory, MC_MED, UCSF, MOVER/SIS,
MOVER/EPIC, VitalDB).

Parameterized by a **channel list** and **seg_sec** (mirrors the MC_MED
stage_b_wave.py API) so one script produces any UNIPHY FM variant. A channel =
`name:source:target_fs[:src_fs]`; multiple channels may share a source sig_name
(read once, resampled to each target). Changing seg_sec re-aligns the seg_idx of
the inline ehr_events (expected). Defaults reproduce the original (PLETH40 @40,
II120 @120, 30 s) output exactly.

For each patient in the filtered inventory:
  1. Parse master header to get segment list + recording start time
  2. Match to hospital admission (HADM_ID) via ADMISSIONS table
  3. Read PLETH-anchored blocks (joint channel reading)
  4. Resample per channel spec (default: PLETH 125Hz -> 40Hz, II 125Hz -> 120Hz)
  5. Segment into seg_sec non-overlapping windows (default 30 s, 30 s stride)
  6. Build ehr_events.npy from labs + vitals filtered by HADM_ID
  7. Save as per-patient directory: {SUBJECT_ID}_{HADM_ID}/

Run:  python workzone/mimic3/stage3_extract_waveforms.py [--limit 10]
      # variant: 10 s @ PLETH125 + II500, separate out-root
      python workzone/mimic3/stage3_extract_waveforms.py --seg-sec 10 \\
          --channels PLETH125:PLETH:125,II500:II:500 --out-root /path/to/mimic3_seg10
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
from dataclasses import dataclass
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
SEGMENT_DUR_SEC = 30          # default segment duration (cfg.seg_sec overrides)
OVERLAP_SEC = 0
STRIDE_SEC = SEGMENT_DUR_SEC - OVERLAP_SEC  # 30 — matches the other datasets
WAVEFORM_DTYPE = np.float16
TIME_DTYPE = np.int64
EHR_EVENT_DTYPE = np.dtype([
    ('time_ms', 'int64'),
    ('seg_idx', 'int32'),
    ('var_id', 'uint16'),
    ('value', 'float32'),
])
# Native rate of each raw WFDB sig_name (used when a channel spec omits src_fs).
SOURCE_FS = {"PLETH": 125, "II": 125}  # MIMIC-III waveform source rate
MAX_NAN_RATIO = 0.20
BASE_CHANNEL = "PLETH"  # default anchor: only PLETH-present segments anchor a block


@dataclass(frozen=True)
class ChannelSpec:
    name: str          # canonical output name, e.g. "PLETH40"
    source: str        # raw WFDB sig_name, e.g. "PLETH"
    target_fs: int     # output sample rate, e.g. 40
    src_fs: int        # expected native rate of `source`, e.g. 125


@dataclass(frozen=True)
class ExtractConfig:
    seg_sec: int
    anchor: str                       # sig_name that defines the grid; must be present
    channels: tuple[ChannelSpec, ...]
    max_nan_ratio: float = MAX_NAN_RATIO  # carried for API parity with stage_b_wave;
    #                                       PLETH-anchored blocks are anchor-NaN-free so
    #                                       no window-drop is applied (original behavior).


# Default = the original hardcoded behavior (byte-identical output).
DEFAULT_CFG = ExtractConfig(
    seg_sec=30, anchor="PLETH",
    channels=(
        ChannelSpec("PLETH40", "PLETH", 40, 125),
        ChannelSpec("II120",   "II",    120, 125),
    ),
)


def resample_factors(target_fs, src_fs):
    """(up, down) for scipy.resample_poly. gcd-reduced."""
    g = gcd(int(target_fs), int(src_fs))
    return int(target_fs) // g, int(src_fs) // g


def _resample(sig, target_fs, src_fs):
    """Resample to target_fs; no-op (no FIR pass) when rates already match.

    Reference (stage_b_wave) API, float32. The mimic3 pipeline uses
    `resample_signal` (float64) so the default extraction stays byte-identical.
    """
    if int(target_fs) == int(src_fs):
        return np.asarray(sig, dtype=np.float32)
    up, down = resample_factors(target_fs, src_fs)
    return resample_poly(sig, up, down).astype(np.float32)


def parse_channels(spec):
    """Parse `name:source:target_fs[:src_fs]` comma-separated list.
    src_fs is inferred from SOURCE_FS when omitted."""
    chans = []
    for tok in (t.strip() for t in spec.split(",") if t.strip()):
        parts = tok.split(":")
        if len(parts) not in (3, 4):
            raise ValueError(f"bad channel spec {tok!r}; want name:source:target_fs[:src_fs]")
        name, source, target_fs = parts[0], parts[1], int(parts[2])
        if len(parts) == 4:
            src_fs = int(parts[3])
        elif source in SOURCE_FS:
            src_fs = SOURCE_FS[source]
        else:
            raise ValueError(f"src_fs for source {source!r} unknown; give it explicitly "
                             f"(known: {sorted(SOURCE_FS)})")
        chans.append(ChannelSpec(name, source, target_fs, src_fs))
    return tuple(chans)


# ========================================================================
# Master header parsing
# ========================================================================

def parse_master_header(patient_path):
    """Parse master multi-segment .hea header.

    Returns (start_dt, source_fs, segments) where:
        segments: [(seg_name | None, n_samples), ...]
        None = null segment (recording gap)
    Or (None, None, None) if no master header found.
    """
    # Find master header (starts with "p", not numerics "n")
    hea_path = None
    for f in os.listdir(patient_path):
        if f.startswith("p") and f.endswith(".hea") and not f.endswith("n.hea"):
            hea_path = os.path.join(patient_path, f)
            break
    if hea_path is None:
        return None, None, None

    with open(hea_path) as fh:
        lines = fh.read().strip().split('\n')

    # Line 1: record_name/n_seg n_sig fs total_samples [time] [date]
    parts = lines[0].split()
    try:
        source_fs = float(parts[2])
    except (IndexError, ValueError):
        return None, None, None

    # Parse date/time from extra fields
    start_dt = None
    time_str = date_str = None
    for p in parts:
        if '/' in p and len(p) == 10 and p[2] == '/' and p[5] == '/':
            date_str = p
        elif ':' in p and '.' in p and len(p) > 8:
            time_str = p
    if time_str and date_str:
        try:
            dt_str = f"{date_str} {time_str.split('.')[0]}"
            start_dt = datetime.strptime(dt_str, "%d/%m/%Y %H:%M:%S")
        except ValueError:
            pass

    # Parse segment lines (lines 2+)
    segments = []
    for line in lines[1:]:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        seg_parts = line.split()
        if len(seg_parts) >= 2:
            seg_name = seg_parts[0]
            try:
                seg_len = int(seg_parts[1])
            except ValueError:
                continue
            if seg_name == '~':
                segments.append((None, seg_len))
            else:
                segments.append((seg_name, seg_len))

    if not segments:
        return None, None, None

    return start_dt, source_fs, segments


# ========================================================================
# PLETH-anchored block reading
# ========================================================================

def read_wfdb_blocks(patient_path, segments, source_fs, source_names, base_channel=BASE_CHANNEL):
    """Read waveform as anchor-anchored blocks with joint channel reading.

    A 'block' is a maximal run of consecutive segments where `base_channel`
    (default PLETH) exists. Gaps (null segments or base absent) break blocks.

    `source_names` are the unique raw sig_names to read (e.g. ["PLETH", "II"]);
    multiple output channels may later share one source.

    Returns list of blocks:
        [{'start_sec': float, 'channels': {sig_name: 1D ndarray}}, ...]
    All channel arrays within a block have identical length.
    A non-base source absent in a base-present segment -> NaN-fill.
    """
    import wfdb

    target_names = list(source_names)
    blocks = []
    current_parts = None  # {channel: [arrays]} for current block
    current_start_sec = None
    cumulative_samples = 0

    for seg_name, seg_len in segments:
        seg_start_sec = cumulative_samples / source_fs

        if seg_name is None:
            # Null segment (recording gap) — close current block
            if current_parts is not None:
                blocks.append(_finalize_block(current_start_sec, current_parts))
                current_parts = None
            cumulative_samples += seg_len
            continue

        # Read segment header to check which channels exist
        seg_path = os.path.join(patient_path, seg_name)
        try:
            h = wfdb.rdheader(seg_path)
        except Exception:
            # Unreadable segment — treat as gap
            if current_parts is not None:
                blocks.append(_finalize_block(current_start_sec, current_parts))
                current_parts = None
            cumulative_samples += seg_len
            continue

        available = set(h.sig_name) if h.sig_name else set()

        if base_channel not in available:
            # Base channel absent — gap
            if current_parts is not None:
                blocks.append(_finalize_block(current_start_sec, current_parts))
                current_parts = None
            cumulative_samples += seg_len
            continue

        # PLETH present — include this segment
        if current_parts is None:
            current_parts = {ch: [] for ch in target_names}
            current_start_sec = seg_start_sec

        # Read all available target channels in one rdrecord call
        channels_to_read = [ch for ch in target_names if ch in available]
        channel_indices = [h.sig_name.index(ch) for ch in channels_to_read]

        try:
            rec = wfdb.rdrecord(seg_path, channels=channel_indices)
            sig_data = {ch: rec.p_signal[:, i] for i, ch in enumerate(channels_to_read)}
            # NU channels (PLETH): normalize by ADC full-scale. mimic3wdb-matched
            # headers keep gain=1023 even on 12-bit segments, so physical values
            # span [0,4] there vs [0,1] on 8/10-bit segments. gain/(2^res-1)
            # rescales every segment to a consistent NU [0,1].
            for i, ch in enumerate(channels_to_read):
                if rec.units[i] == 'NU' and rec.adc_res[i] and rec.adc_gain[i]:
                    sig_data[ch] = sig_data[ch] * (rec.adc_gain[i] / (2 ** rec.adc_res[i] - 1))
        except Exception:
            # Failed to read — treat as gap
            if current_parts is not None and any(len(v) > 0 for v in current_parts.values()):
                blocks.append(_finalize_block(current_start_sec, current_parts))
            current_parts = None
            cumulative_samples += seg_len
            continue

        actual_len = sig_data[base_channel].shape[0]

        # Append each channel (NaN-fill if absent in this segment)
        for ch in target_names:
            if ch in sig_data:
                current_parts[ch].append(sig_data[ch])
            else:
                current_parts[ch].append(np.full(actual_len, np.nan))

        cumulative_samples += seg_len

    # Close last block
    if current_parts is not None:
        blocks.append(_finalize_block(current_start_sec, current_parts))

    return blocks


def _finalize_block(start_sec, parts):
    """Concatenate segment arrays within a block."""
    channels = {}
    for ch, arrays in parts.items():
        if arrays:
            channels[ch] = np.concatenate(arrays)
    return {'start_sec': start_sec, 'channels': channels}


# ========================================================================
# Resampling and segmenting
# ========================================================================

def resample_signal(signal, src_fs, target_fs):
    """Resample 1D signal using polyphase filtering (float64; pipeline path).

    Routes through `resample_factors` (same gcd logic as the reference
    `_resample`) but keeps the original float64 dtype so default extraction is
    byte-identical to the pre-parameterization script.
    """
    if src_fs == target_fs:
        return signal
    up, down = resample_factors(target_fs, src_fs)
    return resample_poly(signal, up, down).astype(np.float64)


def segment_signal(signal, sample_rate, seg_dur_sec=SEGMENT_DUR_SEC, overlap_sec=OVERLAP_SEC):
    """Split 1D signal into [N_seg, samples_per_seg] float16 segments.

    Window: seg_dur_sec (30 s)
    Stride: seg_dur_sec - overlap_sec (30 s; overlap defaults to 0)
    """
    samples_per_seg = int(sample_rate * seg_dur_sec)
    stride_samples = int(sample_rate * (seg_dur_sec - overlap_sec))

    if len(signal) < samples_per_seg:
        return None

    n_seg = (len(signal) - samples_per_seg) // stride_samples + 1

    # as_strided builds a view (may overlap when overlap_sec>0); ascontiguousarray copies to C-contiguous
    view = np.lib.stride_tricks.as_strided(
        signal,
        shape=(n_seg, samples_per_seg),
        strides=(signal.strides[0] * stride_samples, signal.strides[0]),
    )
    return np.ascontiguousarray(view, dtype=WAVEFORM_DTYPE)


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

def save_patient(out_dir, channels, time_ms, ehr_events, meta_extra, seg_sec=SEGMENT_DUR_SEC):
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
        "segment_duration_sec": seg_sec,
        "overlap_sec": OVERLAP_SEC,
        "stride_sec": seg_sec - OVERLAP_SEC,
        "channels": {
            name: {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "sample_rate_hz": int(arr.shape[1] / seg_sec),
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

def process_patient(args):
    """Process one patient: parse header -> match admission -> read blocks -> align -> save.

    Takes a single tuple for multiprocessing compatibility:
    (row_dict, patient_labs, patient_vitals, patient_admissions, cfg, out_root)
    """
    row, patient_labs, patient_vitals, patient_admissions, cfg, out_root = args
    subject_id = row["subject_id"]
    patient_path = row["patient_path"]
    seg_sec = cfg.seg_sec
    # unique raw sig_names to read (multiple output channels may share a source)
    source_names = []
    for c in cfg.channels:
        if c.source not in source_names:
            source_names.append(c.source)

    try:
        # 1. Parse master header for segment list + start time
        wav_start, source_fs, segment_list = parse_master_header(patient_path)
        if wav_start is None or segment_list is None:
            return {"subject_id": subject_id, "status": "SKIP", "reason": "no master header"}

        total_samples = sum(n for _, n in segment_list)
        wav_duration = total_samples / source_fs
        if wav_duration < 300:
            return {"subject_id": subject_id, "status": "SKIP",
                    "reason": f"duration too short: {wav_duration:.0f}s"}

        # 2. Match to hospital admission
        hadm_id, overlap_hours = match_admission(
            subject_id, wav_start, wav_duration, patient_admissions)
        if hadm_id is None:
            return {"subject_id": subject_id, "status": "SKIP", "reason": "no matching admission"}
        if overlap_hours < 1.0:
            return {"subject_id": subject_id, "status": "SKIP",
                    "reason": f"admission overlap too short: {overlap_hours:.1f}h"}

        # 3. Read PLETH-anchored blocks (joint channel reading)
        blocks = read_wfdb_blocks(patient_path, segment_list, source_fs,
                                  source_names, cfg.anchor)
        if not blocks:
            return {"subject_id": subject_id, "hadm_id": hadm_id,
                    "status": "SKIP", "reason": "no blocks with PLETH"}

        # 4. Process each block: resample + segment (per channel spec)
        all_channel_segs = {c.name: [] for c in cfg.channels}
        all_time_ms = []

        for block in blocks:
            block_start_ms = int((wav_start.timestamp() + block['start_sec']) * 1000)

            # Resample each channel spec. Multiple specs may share a source; the
            # block is read once by read_wfdb_blocks, then resampled to each target.
            resampled = {}
            for c in cfg.channels:
                raw = block['channels'].get(c.source)
                if raw is not None and len(raw) > 0:
                    resampled[c.name] = resample_signal(raw, source_fs, c.target_fs)
                else:
                    # Source entirely missing in block — NaN at target rate
                    base_raw = block['channels'][cfg.anchor]
                    target_len = int(np.ceil(len(base_raw) * c.target_fs / source_fs))
                    resampled[c.name] = np.full(target_len, np.nan)

            # Segment each channel (seg_sec-second non-overlap windows)
            segmented = {}
            for c in cfg.channels:
                seg = segment_signal(resampled[c.name], c.target_fs, seg_dur_sec=seg_sec)
                if seg is None:
                    break  # block too short for even one window
                segmented[c.name] = seg

            if len(segmented) != len(cfg.channels):
                continue  # skip this block (too short)

            # Align segment counts (may differ by 1 due to resampling rounding)
            n_seg_block = min(s.shape[0] for s in segmented.values())
            for name in segmented:
                segmented[name] = segmented[name][:n_seg_block]

            # Build time_ms for this block
            stride_ms = seg_sec * 1000
            block_time_ms = np.array(
                [block_start_ms + i * stride_ms for i in range(n_seg_block)],
                dtype=TIME_DTYPE,
            )

            for c in cfg.channels:
                all_channel_segs[c.name].append(segmented[c.name])
            all_time_ms.append(block_time_ms)

        if not all_time_ms:
            return {"subject_id": subject_id, "hadm_id": hadm_id,
                    "status": "SKIP", "reason": "all blocks too short"}

        # 5. Concatenate across blocks (output keyed by ChannelSpec.name)
        channels_out = {}
        for c in cfg.channels:
            channels_out[c.name] = np.concatenate(all_channel_segs[c.name], axis=0)

        time_ms = np.concatenate(all_time_ms)
        n_seg = len(time_ms)

        # 6. Build EHR events (filtered by HADM_ID)
        ehr_events = build_ehr_events(subject_id, hadm_id, time_ms, patient_labs, patient_vitals)

        # 7. Compute NaN ratios per channel
        nan_ratios = {}
        valid_seg_ratios = {}
        for ch_name, arr in channels_out.items():
            arr32 = arr.astype(np.float32)
            nan_ratios[ch_name] = float(np.isnan(arr32).mean())
            # A segment is "valid" if not entirely NaN
            seg_nan = np.isnan(arr32).all(axis=1)
            valid_seg_ratios[ch_name] = float(1.0 - seg_nan.mean())

        # 8. Count recording blocks (gaps show up as time_ms jumps > stride)
        if len(time_ms) > 1:
            diffs = np.diff(time_ms)
            n_gaps = int(np.sum(diffs > seg_sec * 1000 * 1.5))  # >1.5x stride gap
        else:
            n_gaps = 0

        # 9. Save
        patient_id = f"{subject_id}_{hadm_id}"
        out_dir = os.path.join(out_root, patient_id)

        save_patient(
            out_dir=out_dir,
            channels=channels_out,
            time_ms=time_ms,
            ehr_events=ehr_events,
            seg_sec=seg_sec,
            meta_extra={
                "patient_id": patient_id,
                "subject_id": int(subject_id),
                "hadm_id": int(hadm_id),
                "source_dataset": "mimic3",
                "source_path": patient_path,
                "recording_start_ms": int(time_ms[0]),
                "total_duration_hours": round(n_seg * seg_sec / 3600, 2),
                "admission_overlap_hours": round(overlap_hours, 2),
                "n_blocks": len(blocks),
                "n_gaps": n_gaps,
                "per_channel": {
                    ch_name: {
                        "nan_ratio": round(nan_ratios[ch_name], 4),
                        "valid_seg_ratio": round(valid_seg_ratios[ch_name], 4),
                    }
                    for ch_name in channels_out
                },
            },
        )

        return {
            "subject_id": subject_id,
            "hadm_id": hadm_id,
            "patient_id": patient_id,
            "status": "OK",
            "n_segments": n_seg,
            "n_ehr_events": len(ehr_events),
            "duration_hours": round(n_seg * seg_sec / 3600, 2),
            "overlap_hours": round(overlap_hours, 2),
            "n_blocks": len(blocks),
            "n_gaps": n_gaps,
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
    import multiprocessing as mp

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Process only N patients (for testing)")
    parser.add_argument("--workers", type=int, default=12, help="Parallel workers (default 12, max 50%% of cores)")
    parser.add_argument("--out-root", type=str, default=None,
                        help="Output dir override (default = PROCESSED_ROOT from server_paths.yaml)")
    parser.add_argument("--seg-sec", type=int, default=None,
                        help="Segment duration (default 30 = the canonical variant)")
    parser.add_argument("--anchor", type=str, default=None,
                        help="Anchor sig_name that defines the grid (default PLETH)")
    parser.add_argument("--channels", type=str, default=None,
                        help="name:source:target_fs[:src_fs] comma list. Default: PLETH40+II120")
    parser.add_argument("--max-nan-ratio", type=float, default=None)
    args = parser.parse_args()

    if args.channels or args.seg_sec is not None or args.anchor or args.max_nan_ratio is not None:
        cfg = ExtractConfig(
            seg_sec=args.seg_sec if args.seg_sec is not None else DEFAULT_CFG.seg_sec,
            anchor=args.anchor or DEFAULT_CFG.anchor,
            channels=parse_channels(args.channels) if args.channels else DEFAULT_CFG.channels,
            max_nan_ratio=args.max_nan_ratio if args.max_nan_ratio is not None else MAX_NAN_RATIO,
        )
    else:
        cfg = DEFAULT_CFG
    out_root = args.out_root or PROCESSED_ROOT

    # Cap workers at 50% of cores (shared cluster)
    max_workers = os.cpu_count() // 2
    n_workers = min(args.workers, max_workers)

    log.info(f"Stage 3: Extract waveforms -> {out_root}")
    log.info(f"  Workers: {n_workers} (max {max_workers} = 50% of {os.cpu_count()} cores)")
    log.info(f"  Window: {cfg.seg_sec}s, overlap: {OVERLAP_SEC}s, stride: {cfg.seg_sec - OVERLAP_SEC}s")
    log.info(f"  Anchor: {cfg.anchor}, channels: "
             f"{[(c.name, c.source, c.target_fs, c.src_fs) for c in cfg.channels]}")
    os.makedirs(out_root, exist_ok=True)

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
        if "HADM_ID" in df.columns:
            df["HADM_ID"] = pd.to_numeric(df["HADM_ID"], errors="coerce")
    log.info(f"  Labs: {len(labs_df)} events, Vitals: {len(vitals_df)} events")

    # Pre-filter EHR per patient (avoid passing full DataFrames to workers)
    log.info("Pre-filtering EHR per patient...")
    subject_ids = set(inventory["subject_id"].values)
    labs_grouped = {sid: grp for sid, grp in labs_df.groupby("SUBJECT_ID") if sid in subject_ids}
    vitals_grouped = {sid: grp for sid, grp in vitals_df.groupby("SUBJECT_ID") if sid in subject_ids}
    adm_grouped = {sid: grp for sid, grp in admissions_df.groupby("SUBJECT_ID") if sid in subject_ids}
    empty_df_labs = labs_df.iloc[:0]
    empty_df_vitals = vitals_df.iloc[:0]
    empty_df_adm = admissions_df.iloc[:0]
    log.info(f"  Pre-filtered: {len(labs_grouped)} lab groups, {len(vitals_grouped)} vital groups")

    # Build argument tuples for multiprocessing
    task_args = []
    for _, row in inventory.iterrows():
        sid = row["subject_id"]
        task_args.append((
            row.to_dict(),
            labs_grouped.get(sid, empty_df_labs),
            vitals_grouped.get(sid, empty_df_vitals),
            adm_grouped.get(sid, empty_df_adm),
            cfg,
            out_root,
        ))

    # Process with multiprocessing
    t0 = time.time()
    results = []

    from tqdm import tqdm

    n_ok = n_skip = n_err = 0

    def update_bar(pbar, result):
        nonlocal n_ok, n_skip, n_err
        if result["status"] == "OK":
            n_ok += 1
        elif result["status"] == "SKIP":
            n_skip += 1
        else:
            n_err += 1
        pbar.set_postfix(OK=n_ok, SKIP=n_skip, ERR=n_err, refresh=False)
        pbar.update(1)

    if n_workers <= 1 or args.limit:
        with tqdm(total=len(task_args), desc="Stage 3", unit="pat") as pbar:
            for task in task_args:
                result = process_patient(task)
                results.append(result)
                update_bar(pbar, result)
    else:
        log.info(f"Starting {n_workers} workers...")
        with mp.Pool(n_workers) as pool:
            with tqdm(total=len(task_args), desc=f"Stage 3 ({n_workers}w)", unit="pat") as pbar:
                for result in pool.imap_unordered(process_patient, task_args, chunksize=4):
                    results.append(result)
                    update_bar(pbar, result)

    # Summary
    elapsed = time.time() - t0
    n_ok = sum(1 for r in results if r["status"] == "OK")
    n_skip = sum(1 for r in results if r["status"] == "SKIP")
    n_err = sum(1 for r in results if r["status"] == "ERROR")

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
        "output_dir": out_root,
        "channels": [c.name for c in cfg.channels],
        "segment_params": {
            "duration_sec": cfg.seg_sec,
            "overlap_sec": OVERLAP_SEC,
            "stride_sec": cfg.seg_sec - OVERLAP_SEC,
        },
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
    log.info(f"  Output: {out_root}")

    if n_ok > 0:
        ok_results = [r for r in results if r["status"] == "OK"]
        avg_events = np.mean([r["n_ehr_events"] for r in ok_results])
        avg_hours = np.mean([r["duration_hours"] for r in ok_results])
        avg_blocks = np.mean([r.get("n_blocks", 1) for r in ok_results])
        avg_gaps = np.mean([r.get("n_gaps", 0) for r in ok_results])
        log.info(f"  Avg EHR events/patient: {avg_events:.0f}")
        log.info(f"  Avg duration: {avg_hours:.1f} hours")
        log.info(f"  Avg blocks/patient: {avg_blocks:.1f}, avg gaps: {avg_gaps:.1f}")

    log.info(f"\nNext: python workzone/mimic3/stage4_manifest_splits.py")


if __name__ == "__main__":
    main()
