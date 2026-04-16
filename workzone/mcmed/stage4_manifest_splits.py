#!/usr/bin/env python3
"""
Stage 4: Create manifest and train/val/test splits from canonical MC_MED data.

Validates all per-patient directories, computes quality metrics, and creates
stratified splits.

Run:  python workzone/mcmed/stage4_manifest_splits.py
Output:
  {output_dir}/manifest.json
  {output_dir}/splits.json
  workzone/outputs/mcmed/stage4_summary.json
"""
import json
import time
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO_ROOT / "workzone" / "outputs" / "mcmed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

import yaml
with open(REPO_ROOT / "workzone" / "configs" / "server_paths.yaml") as f:
    cfg = yaml.safe_load(f)

OUTPUT_ROOT = Path(cfg["mcmed"]["output_dir"])
if not OUTPUT_ROOT.exists():
    OUTPUT_ROOT = Path.home() / "workspace" / "physio_data_mcmed"

CSV_DIR = cfg["mcmed"]["raw_csv_dir"]
if not os.path.exists(CSV_DIR):
    CSV_DIR = os.path.expanduser("~/workspace/mc_med_csv")

# Minimum requirements for inclusion
MIN_SEGMENTS = 10
MIN_EHR_EVENTS = 1


def validate_patient(patient_dir: Path) -> dict | None:
    """Validate one patient directory, return metadata or None."""
    meta_path = patient_dir / "meta.json"
    if not meta_path.exists():
        return None

    try:
        with open(meta_path) as f:
            meta = json.load(f)

        csn = meta["patient_id"]
        n_seg = meta["n_segments"]
        n_ehr = meta["n_ehr_events"]

        # Check minimum requirements
        if n_seg < MIN_SEGMENTS:
            return None

        # Check required files exist
        time_path = patient_dir / "time_ms.npy"
        ehr_path = patient_dir / "ehr_events.npy"
        if not time_path.exists() or not ehr_path.exists():
            return None

        # Check at least one waveform channel
        channels = list(meta.get("channels", {}).keys())
        if not channels:
            return None

        return {
            "dir": csn,
            "source_dataset": "mcmed",
            "n_segments": n_seg,
            "duration_hours": meta.get("total_duration_hours", 0),
            "channels": channels,
            "n_ehr_events": n_ehr,
            "has_embeddings": meta.get("has_embeddings", []),
        }
    except Exception as e:
        log.warning(f"Failed to validate {patient_dir}: {e}")
        return None


def create_splits(manifest: list[dict], seed: int = 42) -> dict:
    """Create stratified train/val/test splits (70/15/15)."""
    rng = np.random.default_rng(seed)

    # Try to load existing chronological splits from MC_MED CSV
    chrono_train_path = Path(CSV_DIR) / "split_chrono_train.csv"
    chrono_val_path = Path(CSV_DIR) / "split_chrono_val.csv"
    chrono_test_path = Path(CSV_DIR) / "split_chrono_test.csv"

    all_csns = [m["dir"] for m in manifest]

    if chrono_train_path.exists() and chrono_val_path.exists() and chrono_test_path.exists():
        log.info("Using existing chronological splits from mc_med_csv/")
        train_csns = set(pd.read_csv(chrono_train_path, dtype=str).iloc[:, 0].astype(str))
        val_csns = set(pd.read_csv(chrono_val_path, dtype=str).iloc[:, 0].astype(str))
        test_csns = set(pd.read_csv(chrono_test_path, dtype=str).iloc[:, 0].astype(str))

        train = [c for c in all_csns if c in train_csns]
        val = [c for c in all_csns if c in val_csns]
        test = [c for c in all_csns if c in test_csns]

        # Any CSNs not in splits go to train
        assigned = set(train + val + test)
        unassigned = [c for c in all_csns if c not in assigned]
        if unassigned:
            log.info(f"  {len(unassigned)} CSNs not in chrono splits, added to train")
            train.extend(unassigned)
    else:
        log.info("No chrono splits found, creating random 70/15/15 split")
        indices = rng.permutation(len(all_csns))
        n_train = int(0.7 * len(all_csns))
        n_val = int(0.15 * len(all_csns))

        train = [all_csns[i] for i in indices[:n_train]]
        val = [all_csns[i] for i in indices[n_train:n_train + n_val]]
        test = [all_csns[i] for i in indices[n_train + n_val:]]

    return {
        "train_patients": sorted(train),
        "val_patients": sorted(val),
        "test_patients": sorted(test),
        "split_method": "chronological" if chrono_train_path.exists() else "random",
        "seed": seed,
    }


def main():
    t0 = time.time()
    log.info(f"Stage 4: Creating manifest and splits from {OUTPUT_ROOT}")

    # Validate all patient directories
    patient_dirs = sorted([
        OUTPUT_ROOT / d for d in os.listdir(OUTPUT_ROOT)
        if (OUTPUT_ROOT / d).is_dir() and (OUTPUT_ROOT / d / "meta.json").exists()
    ])
    log.info(f"Found {len(patient_dirs)} patient directories")

    manifest = []
    for i, pdir in enumerate(patient_dirs):
        if (i + 1) % 500 == 0:
            log.info(f"  Validating {i+1}/{len(patient_dirs)}...")
        entry = validate_patient(pdir)
        if entry is not None:
            manifest.append(entry)

    log.info(f"Valid patients: {len(manifest)}/{len(patient_dirs)}")

    # Create splits
    splits = create_splits(manifest)

    # Save manifest
    manifest_path = OUTPUT_ROOT / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info(f"Manifest: {manifest_path} ({len(manifest)} entries)")

    # Save splits
    splits_path = OUTPUT_ROOT / "splits.json"
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)
    log.info(f"Splits: {splits_path}")
    log.info(f"  train: {len(splits['train_patients'])}")
    log.info(f"  val: {len(splits['val_patients'])}")
    log.info(f"  test: {len(splits['test_patients'])}")

    elapsed = time.time() - t0

    # Summary
    total_seg = sum(m["n_segments"] for m in manifest)
    total_ehr = sum(m["n_ehr_events"] for m in manifest)
    summary = {
        "total_valid_patients": len(manifest),
        "total_segments": total_seg,
        "total_ehr_events": total_ehr,
        "train_patients": len(splits["train_patients"]),
        "val_patients": len(splits["val_patients"]),
        "test_patients": len(splits["test_patients"]),
        "split_method": splits["split_method"],
        "median_duration_hours": round(float(np.median([m["duration_hours"] for m in manifest])), 2) if manifest else 0,
        "validation_time_sec": round(elapsed, 1),
    }

    with open(OUT_DIR / "stage4_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info("\n=== Summary ===")
    for k, v in summary.items():
        log.info(f"  {k}: {v}")

    log.info(f"\nDone! Canonical data ready at {OUTPUT_ROOT}")
    log.info(f"Set data.root={OUTPUT_ROOT} and data.split_file={splits_path} in UNIPHY_Plus_v2 experiment config.")


if __name__ == "__main__":
    main()
