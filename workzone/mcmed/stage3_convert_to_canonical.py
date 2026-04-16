#!/usr/bin/env python3
"""
Stage 3: Convert MC_MED NPZ files to Physio_Data canonical per-patient format.

For each CSN:
  1. Load NPZ (waveforms already at target sample rates)
  2. Load EHR events from stage2 parquets
  3. Align EHR events to waveform segments (nearest seg_idx)
  4. Write canonical per-patient directory:
       {CSN}/PLETH40.npy, II120.npy, II500.npy, time_ms.npy, ehr_events.npy, meta.json
       {CSN}/emb_PLETH40_GPT19M.npy  (if available in NPZ)

Run:  python workzone/mcmed/stage3_convert_to_canonical.py
Output: {output_dir}/{CSN}/ directories
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
OUT_DIR = REPO_ROOT / "workzone" / "outputs" / "mcmed"

import yaml
with open(REPO_ROOT / "workzone" / "configs" / "server_paths.yaml") as f:
    cfg = yaml.safe_load(f)

OUTPUT_ROOT = Path(cfg["mcmed"]["output_dir"])
if not OUTPUT_ROOT.exists():
    # Fallback to local
    OUTPUT_ROOT = Path.home() / "workspace" / "physio_data_mcmed"

NPZ_DIR = cfg["mcmed"]["raw_npz_dir"]
if not os.path.exists(NPZ_DIR):
    NPZ_DIR = os.path.expanduser("~/workspace/MC_MED_ECG_PLETH_40hz_120hz_500hz_lab_vital")

# EHR event structured dtype (matches physio_data.schema)
EHR_EVENT_DTYPE = np.dtype([
    ("time_ms", "int64"),
    ("seg_idx", "int32"),
    ("var_id", "uint16"),
    ("value", "float32"),
])

SEGMENT_DUR_MS = 30_000  # 30 seconds in milliseconds


def load_ehr_events(csn: str, labs_df: pd.DataFrame, vitals_df: pd.DataFrame) -> pd.DataFrame:
    """Get all EHR events for a CSN from labs + vitals DataFrames."""
    lab_events = labs_df[labs_df["csn"] == csn] if labs_df is not None else pd.DataFrame()
    vital_events = vitals_df[vitals_df["csn"] == csn] if vitals_df is not None else pd.DataFrame()

    parts = []
    if len(lab_events) > 0:
        parts.append(lab_events[["time_ms", "var_id", "value"]])
    if len(vital_events) > 0:
        parts.append(vital_events[["time_ms", "var_id", "value"]])

    if not parts:
        return pd.DataFrame(columns=["time_ms", "var_id", "value"])

    return pd.concat(parts, ignore_index=True).sort_values("time_ms")


def convert_patient(
    npz_path: str,
    csn: str,
    labs_df: pd.DataFrame,
    vitals_df: pd.DataFrame,
    output_root: Path,
) -> dict | None:
    """Convert one NPZ file to canonical per-patient directory."""
    try:
        patient_dir = output_root / csn
        patient_dir.mkdir(parents=True, exist_ok=True)

        # Load NPZ
        data = np.load(npz_path, allow_pickle=True)

        # Waveforms
        pleth40 = data["PLETH40"] if "PLETH40" in data else None
        ii120 = data["II120"] if "II120" in data else None
        ii500 = data["II500"] if "II500" in data else None
        n_seg = pleth40.shape[0] if pleth40 is not None else (ii120.shape[0] if ii120 is not None else 0)

        if n_seg == 0:
            return None

        # Time array: convert datetime64 -> int64 milliseconds
        time_arr = data["time"] if "time" in data else None
        if time_arr is not None:
            # datetime64[ms] -> int64 epoch ms
            time_ms = time_arr.astype("datetime64[ms]").astype(np.int64)
        else:
            # Fallback: generate synthetic timestamps (30s apart)
            time_ms = np.arange(n_seg, dtype=np.int64) * SEGMENT_DUR_MS

        # Save waveforms as separate .npy files (float16, C-contiguous)
        if pleth40 is not None:
            np.save(str(patient_dir / "PLETH40.npy"), np.ascontiguousarray(pleth40.astype(np.float16)))
        if ii120 is not None:
            np.save(str(patient_dir / "II120.npy"), np.ascontiguousarray(ii120.astype(np.float16)))
        if ii500 is not None:
            np.save(str(patient_dir / "II500.npy"), np.ascontiguousarray(ii500.astype(np.float16)))

        # Save time
        np.save(str(patient_dir / "time_ms.npy"), time_ms)

        # Save pre-computed embeddings if available
        if "emb_PLETH40_GPT19M" in data:
            np.save(
                str(patient_dir / "emb_PLETH40_GPT19M.npy"),
                np.ascontiguousarray(data["emb_PLETH40_GPT19M"].astype(np.float32)),
            )

        # Build sparse EHR events
        ehr_df = load_ehr_events(csn, labs_df, vitals_df)
        if len(ehr_df) > 0:
            # Align each EHR event to nearest segment index
            event_times = ehr_df["time_ms"].values
            # Binary search: find nearest segment for each event
            seg_indices = np.searchsorted(time_ms, event_times, side="right") - 1
            seg_indices = np.clip(seg_indices, 0, n_seg - 1)

            # Build structured array
            n_events = len(ehr_df)
            ehr_events = np.zeros(n_events, dtype=EHR_EVENT_DTYPE)
            ehr_events["time_ms"] = event_times
            ehr_events["seg_idx"] = seg_indices.astype(np.int32)
            ehr_events["var_id"] = ehr_df["var_id"].values.astype(np.uint16)
            ehr_events["value"] = ehr_df["value"].values.astype(np.float32)

            # Sort by time_ms
            ehr_events.sort(order="time_ms")
        else:
            ehr_events = np.zeros(0, dtype=EHR_EVENT_DTYPE)

        np.save(str(patient_dir / "ehr_events.npy"), ehr_events)

        # Build meta.json
        channels = {}
        if pleth40 is not None:
            channels["PLETH40"] = {
                "sample_rate_hz": 40,
                "shape": list(pleth40.shape),
                "dtype": "float16",
            }
        if ii120 is not None:
            channels["II120"] = {
                "sample_rate_hz": 120,
                "shape": list(ii120.shape),
                "dtype": "float16",
            }
        if ii500 is not None:
            channels["II500"] = {
                "sample_rate_hz": 500,
                "shape": list(ii500.shape),
                "dtype": "float16",
            }

        meta = {
            "patient_id": csn,
            "source_dataset": "mcmed",
            "n_segments": int(n_seg),
            "segment_duration_sec": 30,
            "total_duration_hours": round(n_seg * 30 / 3600, 2),
            "recording_start_ms": int(time_ms[0]),
            "channels": channels,
            "n_ehr_events": int(len(ehr_events)),
            "has_embeddings": ["emb_PLETH40_GPT19M"] if "emb_PLETH40_GPT19M" in data else [],
        }

        with open(patient_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        return {
            "csn": csn,
            "n_segments": int(n_seg),
            "n_ehr_events": int(len(ehr_events)),
            "channels": list(channels.keys()),
        }

    except Exception as e:
        log.warning(f"Failed to convert {csn}: {e}")
        return None


def main():
    t0 = time.time()
    log.info(f"Stage 3: Converting MC_MED NPZ to canonical format")
    log.info(f"  NPZ source: {NPZ_DIR}")
    log.info(f"  Output: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Load inventory from stage 1
    inventory_path = OUT_DIR / "record_inventory_filtered.parquet"
    if not inventory_path.exists():
        log.error(f"Missing {inventory_path}. Run stage1 first.")
        return
    inv_df = pd.read_parquet(inventory_path)
    log.info(f"Loaded inventory: {len(inv_df)} patients")

    # Load EHR from stage 2
    labs_df = None
    vitals_df = None
    labs_path = OUT_DIR / "labs_filtered.parquet"
    vitals_path = OUT_DIR / "vitals_filtered.parquet"

    if labs_path.exists():
        labs_df = pd.read_parquet(labs_path)
        labs_df["csn"] = labs_df["csn"].astype(str)
        log.info(f"Loaded labs: {len(labs_df)} events")
    if vitals_path.exists():
        vitals_df = pd.read_parquet(vitals_path)
        vitals_df["csn"] = vitals_df["csn"].astype(str)
        log.info(f"Loaded vitals: {len(vitals_df)} events")

    # Process each patient
    results = []
    errors = 0

    for i, row in inv_df.iterrows():
        csn = str(row["csn"])
        npz_path = row["filepath"]

        if (i + 1) % 200 == 0:
            log.info(f"  Processing {i+1}/{len(inv_df)}...")

        result = convert_patient(npz_path, csn, labs_df, vitals_df, OUTPUT_ROOT)
        if result is not None:
            results.append(result)
        else:
            errors += 1

    elapsed = time.time() - t0
    log.info(f"\nConverted {len(results)} patients in {elapsed:.1f}s ({errors} errors)")

    # Summary
    summary = {
        "total_converted": len(results),
        "errors": errors,
        "total_segments": sum(r["n_segments"] for r in results),
        "total_ehr_events": sum(r["n_ehr_events"] for r in results),
        "conversion_time_sec": round(elapsed, 1),
    }

    with open(OUT_DIR / "stage3_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info("\n=== Summary ===")
    for k, v in summary.items():
        log.info(f"  {k}: {v}")

    log.info(f"\nNext: python workzone/mcmed/stage4_manifest_splits.py")


if __name__ == "__main__":
    main()
