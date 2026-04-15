#!/usr/bin/env python3
"""
Post-stage: Convert tasks/sepsis/extra_events/{pid}.npy to 4-partition layout.

For each sepsis cohort patient that has a previous extra_events file:
  - Load extra_events (SOFA 301-306, sofa_total 300, sepsis_onset 307)
  - Split by time relative to the patient's waveform window
  - Write four files alongside the original:
      tasks/sepsis/extra_events/{pid}.baseline.npy
      tasks/sepsis/extra_events/{pid}.recent.npy
      tasks/sepsis/extra_events/{pid}.npy            (in-waveform, overwrites)
      tasks/sepsis/extra_events/{pid}.future.npy
  - Record has_future_sofa / has_future_sepsis_onset in the patient meta.json

Run:  python workzone/mimic3/post_sepsis_trajectory.py [--limit N] [--dry-run]
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
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from physio_data.ehr_trajectory import (
    EHR_EVENT_DTYPE,
    split_events, validate_partition,
    CONTEXT_WINDOW_MS, BASELINE_CAP_MS, FUTURE_CAP_MS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

with open(REPO_ROOT / "workzone" / "configs" / "server_paths.yaml") as f:
    cfg = yaml.safe_load(f)

PROCESSED_ROOT = Path(cfg["mimic3"]["output_dir"])
TASKS_SEPSIS = PROCESSED_ROOT / "tasks" / "sepsis"
EXTRA_EVENTS_DIR = TASKS_SEPSIS / "extra_events"

# var_id ranges
VID_SOFA_MIN, VID_SOFA_MAX = 300, 306    # sofa_total + 6 components
VID_SEPSIS_ONSET = 307


def convert_one(pid: str, *, dry_run: bool = False) -> dict:
    patient_dir = PROCESSED_ROOT / pid
    extra_path = EXTRA_EVENTS_DIR / f"{pid}.npy"
    bak_path   = EXTRA_EVENTS_DIR / f"{pid}.npy.bak"
    time_path  = patient_dir / "time_ms.npy"
    meta_path  = patient_dir / "meta.json"

    if not extra_path.exists() and not bak_path.exists():
        return {"patient_id": pid, "status": "SKIP", "reason": "no extra_events"}
    if not time_path.exists():
        return {"patient_id": pid, "status": "SKIP", "reason": "no time_ms.npy"}

    time_ms = np.load(time_path)
    if len(time_ms) == 0:
        return {"patient_id": pid, "status": "SKIP", "reason": "empty time_ms"}

    src = bak_path if bak_path.exists() else extra_path
    existing = np.load(src)
    if existing.dtype != EHR_EVENT_DTYPE:
        return {"patient_id": pid, "status": "ERROR",
                "reason": f"dtype {existing.dtype} != {EHR_EVENT_DTYPE}"}

    # Episode bounds: use admission bounds from meta.json if present
    ep_start = ep_end = None
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        if "admission_start_ms" in meta:
            ep_start = int(meta["admission_start_ms"])
            ep_end   = int(meta["admission_end_ms"])
    else:
        meta = {}

    partitions = split_events(
        existing, time_ms,
        episode_start_ms=ep_start, episode_end_ms=ep_end,
    )

    n_seg = int(len(time_ms))
    errs: list[str] = []
    for kind in ("baseline", "recent", "events", "future"):
        errs.extend(validate_partition(partitions[kind], kind=kind, n_seg=n_seg))
    if errs:
        return {"patient_id": pid, "status": "ERROR", "reason": "; ".join(errs)}

    future = partitions["future"]
    has_future_sofa = bool(
        len(future) and np.any(
            (future["var_id"] >= VID_SOFA_MIN) & (future["var_id"] <= VID_SOFA_MAX)
        )
    )
    has_future_onset = bool(
        len(future) and np.any(future["var_id"] == VID_SEPSIS_ONSET)
    )

    result = {
        "patient_id": pid,
        **{f"n_{k}": int(len(v)) for k, v in partitions.items()},
        "has_future_sofa": has_future_sofa,
        "has_future_sepsis_onset": has_future_onset,
    }
    if dry_run:
        result["status"] = "DRY_OK"
        return result

    if extra_path.exists() and not bak_path.exists():
        shutil.copy2(extra_path, bak_path)

    np.save(EXTRA_EVENTS_DIR / f"{pid}.baseline.npy", partitions["baseline"])
    np.save(EXTRA_EVENTS_DIR / f"{pid}.recent.npy",   partitions["recent"])
    np.save(EXTRA_EVENTS_DIR / f"{pid}.npy",          partitions["events"])
    np.save(EXTRA_EVENTS_DIR / f"{pid}.future.npy",   partitions["future"])

    if meta_path.exists():
        meta["has_future_sofa"]         = has_future_sofa
        meta["has_future_sepsis_onset"] = has_future_onset
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    result["status"] = "OK"
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--patient-ids", type=str, default=None,
                    help="Only process these patient_ids (one per line)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    log.info(f"Post-sepsis trajectory split -> {EXTRA_EVENTS_DIR}")

    if args.patient_ids:
        with open(args.patient_ids) as f:
            wanted = {ln.strip() for ln in f if ln.strip()}
        pids = [pid for pid in wanted if (EXTRA_EVENTS_DIR / f"{pid}.npy").exists()
                                     or (EXTRA_EVENTS_DIR / f"{pid}.npy.bak").exists()]
    else:
        pids = sorted({
            p.stem.split(".")[0]
            for p in EXTRA_EVENTS_DIR.glob("*.npy")
        })
    if args.limit:
        pids = pids[: args.limit]
    log.info(f"  {len(pids)} sepsis patients to convert")

    t0 = time.time()
    results = []
    for i, pid in enumerate(pids, 1):
        results.append(convert_one(pid, dry_run=args.dry_run))
        if i % 200 == 0 or i == len(pids):
            ok = sum(1 for r in results if r["status"] in ("OK", "DRY_OK"))
            sk = sum(1 for r in results if r["status"] == "SKIP")
            er = sum(1 for r in results if r["status"] == "ERROR")
            log.info(f"  [{i}/{len(pids)}] OK={ok} SKIP={sk} ERR={er}")

    ok_rows = [r for r in results if r["status"] in ("OK", "DRY_OK")]
    errors  = [r for r in results if r["status"] == "ERROR"]
    log.info(f"\n=== Sepsis trajectory complete in {time.time() - t0:.1f}s ===")
    log.info(f"  OK: {len(ok_rows)}, ERRORS: {len(errors)}")

    with_fut_sofa  = sum(1 for r in ok_rows if r.get("has_future_sofa"))
    with_fut_onset = sum(1 for r in ok_rows if r.get("has_future_sepsis_onset"))
    log.info(f"  Patients with future SOFA events:  {with_fut_sofa}")
    log.info(f"  Patients with future sepsis onset: {with_fut_onset}")

    OUT_DIR = REPO_ROOT / "workzone" / "outputs" / "mimic3"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "post_sepsis_trajectory_summary.json", "w") as f:
        json.dump({
            "n_ok": len(ok_rows),
            "n_error": len(errors),
            "n_with_future_sofa":  with_fut_sofa,
            "n_with_future_onset": with_fut_onset,
            "dry_run": args.dry_run,
            "errors": errors[:50],
        }, f, indent=2)


if __name__ == "__main__":
    main()
