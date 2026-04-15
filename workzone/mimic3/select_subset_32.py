#!/usr/bin/env python3
"""
Select 32 patients covering BOTH pretrain and sepsis task data.

Run this on the REMOTE server (where the processed data lives).

Strategy:
  - Sepsis cohort = patients in tasks/sepsis/cohort.json
  - Pretrain patients = patients listed in pretrain_splits.json (train+val+test)
  - "Covers both tasks" = patient must be in pretrain_splits AND sepsis cohort
    (so the same 32 patients can be used for pretrain dataloaders AND for the
    sepsis downstream task).
  - Among the intersection, pick a stratified sample of 32:
      * 16 died==1 (if available)
      * 16 died==0
    falling back to whatever is available if one class is short.
  - Also balance across pretrain train/val/test splits so the subset is
    usable as a tiny end-to-end smoke test.

Outputs (written next to the processed data, at output_dir):
  subset32/patient_ids.txt       - one patient_id per line
  subset32/rsync_files.txt       - relative paths to rsync (for --files-from)
  subset32/manifest.json         - per-patient metadata
"""
import os
import json
import random
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
with open(REPO_ROOT / "workzone" / "configs" / "server_paths.yaml") as f:
    cfg = yaml.safe_load(f)

PROCESSED_ROOT = Path(cfg["mimic3"]["output_dir"])
N_TARGET = 32
SEED = 42


def main():
    random.seed(SEED)

    pretrain_splits_path = PROCESSED_ROOT / "pretrain_splits.json"
    sepsis_cohort_path = PROCESSED_ROOT / "tasks" / "sepsis" / "cohort.json"
    sepsis_splits_path = PROCESSED_ROOT / "tasks" / "sepsis" / "splits.json"

    assert pretrain_splits_path.exists(), f"missing {pretrain_splits_path}"
    assert sepsis_cohort_path.exists(), f"missing {sepsis_cohort_path}"

    with open(pretrain_splits_path) as f:
        pre = json.load(f)
    with open(sepsis_cohort_path) as f:
        sepsis = json.load(f)

    pre_split_of = {}
    for sp in ("train", "val", "test"):
        for pid in pre.get(sp, []):
            pre_split_of[pid] = sp

    pre_ids = set(pre_split_of.keys())
    sepsis_by_pid = {e["patient_id"]: e for e in sepsis}

    intersection = sorted(set(sepsis_by_pid.keys()) & pre_ids)
    print(f"pretrain patients:   {len(pre_ids)}")
    print(f"sepsis patients:     {len(sepsis_by_pid)}")
    print(f"both tasks (intersect): {len(intersection)}")

    # Stratify by (died, pretrain_split)
    buckets = {}
    for pid in intersection:
        died = int(sepsis_by_pid[pid].get("died", 0))
        sp = pre_split_of[pid]
        buckets.setdefault((died, sp), []).append(pid)

    for k, v in buckets.items():
        random.shuffle(v)
        print(f"  bucket {k}: {len(v)}")

    # Round-robin pull from buckets until we hit N_TARGET
    bucket_order = sorted(buckets.keys())
    picked = []
    cursors = {k: 0 for k in bucket_order}
    while len(picked) < N_TARGET:
        progressed = False
        for k in bucket_order:
            if cursors[k] < len(buckets[k]):
                picked.append(buckets[k][cursors[k]])
                cursors[k] += 1
                progressed = True
                if len(picked) >= N_TARGET:
                    break
        if not progressed:
            break

    if len(picked) < N_TARGET:
        # Fall back: extend with more sepsis patients (even if not in pretrain_splits)
        extras = [p for p in sepsis_by_pid if p not in picked and (PROCESSED_ROOT / p).exists()]
        random.shuffle(extras)
        picked.extend(extras[: N_TARGET - len(picked)])

    picked = picked[:N_TARGET]
    print(f"picked: {len(picked)}")

    # Load sepsis splits for reference
    sepsis_split_of = {}
    if sepsis_splits_path.exists():
        with open(sepsis_splits_path) as f:
            ssp = json.load(f)
        for sp in ("train", "val", "test"):
            for pid in ssp.get(sp, []):
                sepsis_split_of[pid] = sp

    # Build manifest + rsync file list
    out_dir = PROCESSED_ROOT / "subset32"
    out_dir.mkdir(exist_ok=True)

    manifest = []
    rsync_lines = []

    # Top-level files every consumer needs
    top_level = [
        "manifest.json",
        "pretrain_splits.json",
        "downstream_splits.json",
        "tasks/sepsis/cohort.json",
        "tasks/sepsis/splits.json",
    ]
    for rel in top_level:
        if (PROCESSED_ROOT / rel).exists():
            rsync_lines.append(rel)

    for pid in picked:
        pdir = PROCESSED_ROOT / pid
        entry = {
            "patient_id": pid,
            "pretrain_split": pre_split_of.get(pid),
            "sepsis_split": sepsis_split_of.get(pid),
            "died": int(sepsis_by_pid[pid].get("died", 0)),
            "onset_time_sec": float(sepsis_by_pid[pid].get("onset_time_sec", 0.0)),
            "icustayid": int(sepsis_by_pid[pid].get("icustayid", 0)),
        }
        manifest.append(entry)

        # All files under the patient dir
        if pdir.is_dir():
            for root, _, files in os.walk(pdir):
                for fn in files:
                    full = Path(root) / fn
                    rsync_lines.append(str(full.relative_to(PROCESSED_ROOT)))

        # Sepsis extra events live outside the patient dir
        extra = PROCESSED_ROOT / "tasks" / "sepsis" / "extra_events" / f"{pid}.npy"
        if extra.exists():
            rsync_lines.append(str(extra.relative_to(PROCESSED_ROOT)))

    with open(out_dir / "patient_ids.txt", "w") as f:
        f.write("\n".join(picked) + "\n")
    with open(out_dir / "rsync_files.txt", "w") as f:
        f.write("\n".join(rsync_lines) + "\n")
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote:")
    print(f"  {out_dir / 'patient_ids.txt'}  ({len(picked)} ids)")
    print(f"  {out_dir / 'rsync_files.txt'}  ({len(rsync_lines)} files)")
    print(f"  {out_dir / 'manifest.json'}")

    # Breakdown
    died_n = sum(1 for e in manifest if e["died"] == 1)
    print(f"\nBreakdown: died={died_n}, survived={len(manifest) - died_n}")
    from collections import Counter
    print(f"  pretrain_split: {Counter(e['pretrain_split'] for e in manifest)}")
    print(f"  sepsis_split:   {Counter(e['sepsis_split'] for e in manifest)}")


if __name__ == "__main__":
    main()
