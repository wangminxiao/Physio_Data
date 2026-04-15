#!/usr/bin/env python3
"""
Stage 3c: Convert per-patient EHR to 4-partition trajectory layout.

Reads each existing patient directory and writes four EHR files:
  ehr_baseline.npy  ehr_recent.npy  ehr_events.npy  ehr_future.npy

Labs + vitals are re-derived from the filtered parquets so baseline/recent/
future partitions gain events that the original in-waveform-only extraction
dropped. Action events (var_id 200-299) already in ehr_events.npy are
preserved in the new ehr_events.npy (in-waveform only, as that's where
stage3b originally emitted them).

Original ehr_events.npy is backed up to ehr_events.npy.bak on first run.
Safe to re-run: if .bak already exists we read from .bak as the source of
truth for preserved actions.

Usage:
  python workzone/mimic3/stage3c_ehr_trajectory.py --limit 5
  python workzone/mimic3/stage3c_ehr_trajectory.py --patient-ids path/to/ids.txt
  python workzone/mimic3/stage3c_ehr_trajectory.py --workers 16

Depends on:
  - stage3 output: per-patient dirs with time_ms.npy + ehr_events.npy + meta.json
  - stage2 output: workzone/outputs/mimic3/labs_filtered.parquet, vitals_filtered.parquet
  - ADMISSIONS.csv(.gz) from raw MIMIC-III
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from physio_data.ehr_trajectory import (
    EHR_EVENT_DTYPE,
    ALL_FNAMES,
    FNAME_BASELINE, FNAME_RECENT, FNAME_EVENTS, FNAME_FUTURE,
    CONTEXT_WINDOW_MS, BASELINE_CAP_MS, FUTURE_CAP_MS,
    split_events, merge_partition, validate_partition,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

with open(REPO_ROOT / "workzone" / "configs" / "server_paths.yaml") as f:
    cfg = yaml.safe_load(f)

EHR_ROOT = cfg["mimic3"]["raw_ehr_dir"]
PROCESSED_ROOT = Path(cfg["mimic3"]["output_dir"])
OUT_DIR_OUTPUTS = REPO_ROOT / "workzone" / "outputs" / "mimic3"

ACTION_VAR_MIN = 200   # actions + scores preserved from existing ehr_events.npy
ACTION_VAR_MAX = 399   # (300-399 = sepsis/SOFA, usually in task-specific file)


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------

def load_admissions() -> pd.DataFrame:
    p = os.path.join(EHR_ROOT, "ADMISSIONS.csv.gz")
    if not os.path.exists(p):
        p = os.path.join(EHR_ROOT, "ADMISSIONS.csv")
    df = pd.read_csv(p, usecols=["SUBJECT_ID", "HADM_ID", "ADMITTIME", "DISCHTIME"])
    df["ADMITTIME"] = pd.to_datetime(df["ADMITTIME"])
    df["DISCHTIME"] = pd.to_datetime(df["DISCHTIME"])
    df["admit_ms"] = (df["ADMITTIME"].astype("int64") // 10**6).astype("int64")
    df["disch_ms"] = (df["DISCHTIME"].astype("int64") // 10**6).astype("int64")
    return df.set_index("HADM_ID")[["SUBJECT_ID", "admit_ms", "disch_ms"]]


def load_events_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "charttime_dt" not in df.columns:
        df["charttime_dt"] = pd.to_datetime(df["CHARTTIME"])
    df["time_ms"] = (df["charttime_dt"].astype("int64") // 10**6).astype("int64")
    if "HADM_ID" in df.columns:
        df["HADM_ID"] = pd.to_numeric(df["HADM_ID"], errors="coerce")
    return df


# --------------------------------------------------------------------------
# Per-patient conversion
# --------------------------------------------------------------------------

def convert_patient(
    patient_dir: Path,
    labs_g: pd.api.typing.DataFrameGroupBy,
    vitals_g: pd.api.typing.DataFrameGroupBy,
    admissions: pd.DataFrame,
    *,
    dry_run: bool = False,
) -> dict:
    """Convert one patient dir to 4-file layout. Returns result dict."""
    pid = patient_dir.name
    meta_path = patient_dir / "meta.json"
    time_path = patient_dir / "time_ms.npy"
    events_path = patient_dir / FNAME_EVENTS
    events_bak_path = patient_dir / (FNAME_EVENTS + ".bak")

    if not meta_path.exists() or not time_path.exists():
        return {"patient_id": pid, "status": "SKIP", "reason": "missing meta.json or time_ms.npy"}

    with open(meta_path) as f:
        meta = json.load(f)

    time_ms = np.load(time_path)
    if len(time_ms) == 0:
        return {"patient_id": pid, "status": "SKIP", "reason": "empty time_ms"}

    subject_id = int(meta.get("subject_id", -1))
    hadm_id    = int(meta.get("hadm_id", -1))

    # Source of truth for existing events (actions are only in the old file)
    if events_bak_path.exists():
        existing = np.load(events_bak_path)
    elif events_path.exists():
        existing = np.load(events_path)
    else:
        existing = np.empty(0, dtype=EHR_EVENT_DTYPE)

    # Actions + scores to preserve (var_id 200-399 that were in ehr_events.npy)
    if len(existing) > 0:
        action_mask = (existing["var_id"] >= ACTION_VAR_MIN) & (existing["var_id"] <= ACTION_VAR_MAX)
        actions = existing[action_mask]
    else:
        actions = np.empty(0, dtype=EHR_EVENT_DTYPE)

    # Re-derive labs + vitals for the full admission window from the parquets.
    rows: list[tuple] = []
    for gdf_group in (labs_g, vitals_g):
        try:
            sub = gdf_group.get_group(subject_id)
        except KeyError:
            continue
        if hadm_id > 0 and "HADM_ID" in sub.columns:
            sub = sub[sub["HADM_ID"] == hadm_id]
        if len(sub) == 0:
            continue
        # columns: time_ms, var_id, VALUENUM
        for t, vid, val in zip(
            sub["time_ms"].to_numpy(),
            sub["var_id"].to_numpy(),
            sub["VALUENUM"].to_numpy(),
        ):
            rows.append((int(t), 0, int(vid), float(val)))
    lv_events = (
        np.array(rows, dtype=EHR_EVENT_DTYPE)
        if rows
        else np.empty(0, dtype=EHR_EVENT_DTYPE)
    )

    # Combine labs+vitals (full window) with preserved actions (in-waveform only)
    combined = merge_partition(lv_events, actions)

    # Admission bounds (optional)
    ep_start = ep_end = None
    if hadm_id in admissions.index:
        row = admissions.loc[hadm_id]
        ep_start = int(row["admit_ms"])
        ep_end   = int(row["disch_ms"])

    partitions = split_events(
        combined,
        time_ms,
        episode_start_ms=ep_start,
        episode_end_ms=ep_end,
    )

    # Validate
    n_seg = int(len(time_ms))
    errs: list[str] = []
    for kind in ("baseline", "recent", "events", "future"):
        errs.extend(validate_partition(partitions[kind], kind=kind, n_seg=n_seg))
    if errs:
        return {"patient_id": pid, "status": "ERROR", "reason": "; ".join(errs)}

    if dry_run:
        return {
            "patient_id": pid, "status": "DRY_OK",
            **{f"n_{k}": int(len(v)) for k, v in partitions.items()},
        }

    # Back up original ehr_events.npy once
    if events_path.exists() and not events_bak_path.exists():
        shutil.copy2(events_path, events_bak_path)

    # Write 4 files
    np.save(patient_dir / FNAME_BASELINE, partitions["baseline"])
    np.save(patient_dir / FNAME_RECENT,   partitions["recent"])
    np.save(patient_dir / FNAME_EVENTS,   partitions["events"])
    np.save(patient_dir / FNAME_FUTURE,   partitions["future"])

    # Update meta
    def _nvar(arr: np.ndarray) -> int:
        return int(len(np.unique(arr["var_id"]))) if len(arr) else 0

    meta.update({
        "n_ehr_events":   int(len(partitions["events"])),
        "n_events":       int(len(partitions["events"])),
        "n_baseline":     int(len(partitions["baseline"])),
        "n_recent":       int(len(partitions["recent"])),
        "n_future":       int(len(partitions["future"])),
        "n_baseline_vars": _nvar(partitions["baseline"]),
        "n_recent_vars":   _nvar(partitions["recent"]),
        "n_future_vars":   _nvar(partitions["future"]),
        "context_window_ms": CONTEXT_WINDOW_MS,
        "baseline_cap_ms":   BASELINE_CAP_MS,
        "future_cap_ms":     FUTURE_CAP_MS,
        "has_future_actions":      False,  # stage3b did not populate future actions
        "has_future_sofa":         False,
        "has_future_sepsis_onset": False,
        "ehr_layout_version": 2,
    })
    if ep_start is not None:
        meta["admission_start_ms"] = ep_start
        meta["admission_end_ms"]   = ep_end
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "patient_id": pid, "status": "OK",
        **{f"n_{k}": int(len(v)) for k, v in partitions.items()},
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only N patients (for testing)")
    ap.add_argument("--patient-ids", type=str, default=None,
                    help="Path to a text file with one patient_id per line")
    ap.add_argument("--dry-run", action="store_true",
                    help="Validate and report counts without writing files")
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel workers. Conversion is IO-light and parquet groupby "
                         "is already cached per worker; 1-4 is usually enough.")
    args = ap.parse_args()

    log.info(f"Stage 3c: EHR trajectory split -> {PROCESSED_ROOT}")
    log.info(f"  dry_run={args.dry_run}, limit={args.limit}, workers={args.workers}")

    # Select patient dirs
    if args.patient_ids:
        with open(args.patient_ids) as f:
            wanted = {ln.strip() for ln in f if ln.strip()}
        patient_dirs = sorted(
            PROCESSED_ROOT / pid for pid in wanted
            if (PROCESSED_ROOT / pid).is_dir()
        )
    else:
        patient_dirs = sorted(
            p for p in PROCESSED_ROOT.iterdir()
            if p.is_dir() and (p / "meta.json").exists()
        )
    if args.limit:
        patient_dirs = patient_dirs[: args.limit]
    log.info(f"  {len(patient_dirs)} patient dirs to process")

    # Load shared tables (once)
    log.info("Loading ADMISSIONS ...")
    admissions = load_admissions()
    log.info(f"  {len(admissions)} admissions")

    log.info("Loading labs_filtered + vitals_filtered ...")
    labs  = load_events_parquet(OUT_DIR_OUTPUTS / "labs_filtered.parquet")
    vitals = load_events_parquet(OUT_DIR_OUTPUTS / "vitals_filtered.parquet")
    log.info(f"  labs: {len(labs)} rows / vitals: {len(vitals)} rows")

    labs_g   = labs.groupby("SUBJECT_ID", sort=False)
    vitals_g = vitals.groupby("SUBJECT_ID", sort=False)

    # Process
    t0 = time.time()
    results: list[dict] = []
    for i, pdir in enumerate(patient_dirs, 1):
        results.append(convert_patient(
            pdir, labs_g, vitals_g, admissions, dry_run=args.dry_run,
        ))
        if i % 200 == 0 or i == len(patient_dirs):
            ok = sum(1 for r in results if r["status"] in ("OK", "DRY_OK"))
            sk = sum(1 for r in results if r["status"] == "SKIP")
            er = sum(1 for r in results if r["status"] == "ERROR")
            log.info(f"  [{i}/{len(patient_dirs)}] OK={ok} SKIP={sk} ERR={er}")

    # Summary
    ok_rows = [r for r in results if r["status"] in ("OK", "DRY_OK")]
    errors  = [r for r in results if r["status"] == "ERROR"]
    skips   = [r for r in results if r["status"] == "SKIP"]

    log.info(f"\n=== Stage 3c complete in {time.time() - t0:.1f}s ===")
    log.info(f"  OK:    {len(ok_rows)}")
    log.info(f"  SKIP:  {len(skips)}")
    log.info(f"  ERROR: {len(errors)}")
    if ok_rows:
        tot = {k: sum(r.get(f"n_{k}", 0) for r in ok_rows) for k in
               ("baseline", "recent", "events", "future")}
        log.info(f"  Events: baseline={tot['baseline']} recent={tot['recent']} "
                 f"events={tot['events']} future={tot['future']}")

    if errors:
        log.warning("First 10 errors:")
        for r in errors[:10]:
            log.warning(f"  {r['patient_id']}: {r['reason']}")

    OUT_DIR_OUTPUTS.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR_OUTPUTS / "stage3c_summary.json", "w") as f:
        json.dump({
            "n_processed": len(patient_dirs),
            "n_ok": len(ok_rows),
            "n_skip": len(skips),
            "n_error": len(errors),
            "dry_run": args.dry_run,
            "errors": errors[:50],
        }, f, indent=2)


if __name__ == "__main__":
    main()
