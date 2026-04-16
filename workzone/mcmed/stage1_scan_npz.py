#!/usr/bin/env python3
"""
Stage 1: Scan MC_MED NPZ files to build an inventory.

MC_MED waveforms are already preprocessed as NPZ files with PLETH40, II120, II500,
time, ehr_gt, ehr_mask, ehr_trend, and pre-computed embeddings.

This stage inventories all NPZ files, extracts metadata (channels, segment counts,
duration), and cross-checks against clinical CSVs.

Run:  python workzone/mcmed/stage1_scan_npz.py
Output: workzone/outputs/mcmed/record_inventory.parquet
        workzone/outputs/mcmed/stage1_summary.json
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
OUT_DIR.mkdir(parents=True, exist_ok=True)

import yaml
with open(REPO_ROOT / "workzone" / "configs" / "server_paths.yaml") as f:
    cfg = yaml.safe_load(f)

NPZ_DIR = cfg["mcmed"]["raw_npz_dir"]
CSV_DIR = cfg["mcmed"]["raw_csv_dir"]

# Also allow local workspace paths as fallback
if not os.path.exists(NPZ_DIR):
    NPZ_DIR = os.path.expanduser("~/workspace/MC_MED_ECG_PLETH_40hz_120hz_500hz_lab_vital")
if not os.path.exists(CSV_DIR):
    CSV_DIR = os.path.expanduser("~/workspace/mc_med_csv")

# Minimum segments to keep a patient
MIN_SEGMENTS = 10  # ~5 minutes at 30s per segment


def scan_npz(npz_path: str) -> dict | None:
    """Scan one NPZ file. Returns metadata dict or None."""
    try:
        fname = os.path.basename(npz_path)
        # Parse filename: {CSN}_P40_E120_E500_{n_segments}.npz
        parts = fname.replace(".npz", "").split("_")
        csn = parts[0]

        with np.load(npz_path, mmap_mode="r") as data:
            keys = list(data.keys())
            n_seg = data["PLETH40"].shape[0] if "PLETH40" in data else 0
            has_pleth = "PLETH40" in keys
            has_ii120 = "II120" in keys
            has_ii500 = "II500" in keys
            has_emb = "emb_PLETH40_GPT19M" in keys
            has_ehr = "ehr_gt" in keys

            # Extract time range
            time_arr = data["time"] if "time" in data else None
            time_start = str(time_arr[0]) if time_arr is not None and len(time_arr) > 0 else None
            time_end = str(time_arr[-1]) if time_arr is not None and len(time_arr) > 0 else None

            # EHR info
            n_ehr_vars = data["ehr_gt"].shape[1] if has_ehr else 0
            n_ehr_observed = int((data["ehr_mask"] > 1).sum()) if "ehr_mask" in data else 0

        return {
            "csn": csn,
            "filename": fname,
            "filepath": npz_path,
            "n_segments": n_seg,
            "duration_sec": n_seg * 30,
            "duration_hours": round(n_seg * 30 / 3600, 2),
            "has_pleth40": has_pleth,
            "has_ii120": has_ii120,
            "has_ii500": has_ii500,
            "has_emb_gpt19m": has_emb,
            "has_ehr": has_ehr,
            "n_ehr_vars": n_ehr_vars,
            "n_ehr_observed": n_ehr_observed,
            "time_start": time_start,
            "time_end": time_end,
            "keys": keys,
        }
    except Exception as e:
        log.warning(f"Failed to scan {npz_path}: {e}")
        return None


def main():
    log.info(f"Stage 1: Scanning MC_MED NPZ files at {NPZ_DIR}")
    t0 = time.time()

    # Collect all NPZ files
    npz_files = sorted([
        os.path.join(NPZ_DIR, f)
        for f in os.listdir(NPZ_DIR)
        if f.endswith(".npz")
    ])
    log.info(f"Found {len(npz_files)} NPZ files")

    # Scan in parallel
    results = []
    errors = 0
    with ProcessPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(scan_npz, p): p for p in npz_files}
        for i, future in enumerate(as_completed(futures)):
            if (i + 1) % 500 == 0:
                log.info(f"  Scanned {i+1}/{len(npz_files)}...")
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception:
                errors += 1

    elapsed = time.time() - t0
    log.info(f"Scanned {len(npz_files)} files in {elapsed:.1f}s ({errors} errors)")

    df = pd.DataFrame(results)

    # Cross-check with visits CSV
    visits_path = os.path.join(CSV_DIR, "visits.csv")
    csns_with_visits = set()
    if os.path.exists(visits_path):
        visits_df = pd.read_csv(visits_path, usecols=["CSN"], dtype={"CSN": str})
        csns_with_visits = set(visits_df["CSN"].astype(str))
        df["has_visits"] = df["csn"].isin(csns_with_visits)
        log.info(f"Visits CSV: {len(csns_with_visits)} CSNs, {df['has_visits'].sum()} matched to NPZ")

    # Cross-check with labs CSV
    labs_path = os.path.join(CSV_DIR, "labs.csv")
    csns_with_labs = set()
    if os.path.exists(labs_path):
        labs_df = pd.read_csv(labs_path, usecols=["CSN"], dtype={"CSN": str})
        csns_with_labs = set(labs_df["CSN"].astype(str))
        df["has_labs"] = df["csn"].isin(csns_with_labs)
        log.info(f"Labs CSV: {len(csns_with_labs)} unique CSNs, {df['has_labs'].sum()} matched to NPZ")

    # Cross-check with numerics CSV
    numerics_path = os.path.join(CSV_DIR, "numerics.csv")
    csns_with_vitals = set()
    if os.path.exists(numerics_path):
        numerics_df = pd.read_csv(numerics_path, usecols=["CSN"], dtype={"CSN": str})
        csns_with_vitals = set(numerics_df["CSN"].astype(str))
        df["has_vitals"] = df["csn"].isin(csns_with_vitals)
        log.info(f"Numerics CSV: {len(csns_with_vitals)} unique CSNs, {df['has_vitals'].sum()} matched to NPZ")

    # Filter: minimum segments
    df_filtered = df[df["n_segments"] >= MIN_SEGMENTS].copy()

    # Summary
    summary = {
        "total_npz_files": len(df),
        "unique_csns": int(df["csn"].nunique()),
        "with_pleth40": int(df["has_pleth40"].sum()),
        "with_ii120": int(df["has_ii120"].sum()),
        "with_ehr": int(df["has_ehr"].sum()),
        "with_emb_gpt19m": int(df["has_emb_gpt19m"].sum()),
        "with_visits": int(df.get("has_visits", pd.Series(dtype=bool)).sum()),
        "with_labs": int(df.get("has_labs", pd.Series(dtype=bool)).sum()),
        "with_vitals": int(df.get("has_vitals", pd.Series(dtype=bool)).sum()),
        "after_min_segment_filter": len(df_filtered),
        "min_segments": MIN_SEGMENTS,
        "total_segments": int(df["n_segments"].sum()),
        "median_duration_hours": round(float(df["duration_hours"].median()), 2),
        "scan_time_sec": round(elapsed, 1),
    }

    log.info("\n=== Summary ===")
    for k, v in summary.items():
        log.info(f"  {k}: {v}")

    # Save
    df.to_parquet(OUT_DIR / "record_inventory.parquet", index=False)
    df_filtered.to_parquet(OUT_DIR / "record_inventory_filtered.parquet", index=False)
    with open(OUT_DIR / "stage1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"\nSaved to {OUT_DIR}")
    log.info(f"Next: python workzone/mcmed/stage2_extract_ehr.py")


if __name__ == "__main__":
    main()
