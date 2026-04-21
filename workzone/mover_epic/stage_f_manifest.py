#!/usr/bin/env python3
"""
Stage F-1 - MOVER/EPIC manifest + splits.

Splits are grouped by MRN (one MRN can have multiple LOG_ID encounters;
keep them together). 70/15/15, seed=42.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from physio_data.ehr_trajectory import (  # noqa: E402
    EHR_EVENT_DTYPE,
    FNAME_BASELINE, FNAME_RECENT, FNAME_EVENTS, FNAME_FUTURE,
    validate_partition,
)

OUT_ROOT = "/opt/localdata100tb/physio_data/mover_epic"
COHORT_PARQUET = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/mover_epic/valid_cohort.parquet"
OUTPUTS_DIR = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/mover_epic"

DEFAULT_SEED = 42
DEFAULT_RATIOS = (0.70, 0.15, 0.15)
EXPECTED_CHANNELS = {"PLETH40": 1200, "II120": 3600}

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def validate_entity(entity_dir: Path):
    entry = {"entity_id": entity_dir.name}
    errors: list[str] = []
    meta_path = entity_dir / "meta.json"
    if not meta_path.exists():
        return entry, ["missing meta.json"]
    try:
        meta = json.loads(meta_path.read_text())
    except Exception as e:
        return entry, [f"meta.json parse error: {e}"]
    n_seg = int(meta.get("n_segments") or 0)
    entry.update({
        "log_id": meta.get("log_id"),
        "mrn": meta.get("mrn"),
        "n_seg": n_seg,
        "source_dataset": meta.get("source_dataset", "mover_epic"),
    })
    for k in ("n_baseline", "n_recent", "n_events", "n_future",
              "n_events_vars", "ehr_layout_version",
              "stage_b_version", "stage_c_version",
              "stage_d_version", "stage_e_version",
              "admission_start_ms", "admission_end_ms"):
        if k in meta:
            entry[k] = meta[k]
    time_path = entity_dir / "time_ms.npy"
    if not time_path.exists():
        errors.append("missing time_ms.npy")
    else:
        t = np.load(time_path)
        if t.dtype != np.int64:
            errors.append(f"time_ms dtype {t.dtype}")
        if len(t) != n_seg:
            errors.append(f"time_ms len {len(t)} != n_seg {n_seg}")
        if len(t) > 1 and not np.all(np.diff(t) > 0):
            errors.append("time_ms not strictly monotonic")
        if len(t):
            entry["wave_start_ms"] = int(t[0])
            entry["wave_end_ms"]   = int(t[-1])
    for ch, samples in EXPECTED_CHANNELS.items():
        p = entity_dir / f"{ch}.npy"
        if not p.exists():
            errors.append(f"missing {ch}.npy"); continue
        arr = np.load(p, mmap_mode="r")
        if arr.dtype != np.float16:
            errors.append(f"{ch}: dtype {arr.dtype}")
        if not arr.flags["C_CONTIGUOUS"]:
            errors.append(f"{ch}: not C-contiguous")
        if arr.shape[0] != n_seg:
            errors.append(f"{ch}: shape[0]={arr.shape[0]} expected {n_seg}")
        if arr.ndim == 2 and arr.shape[1] != samples:
            errors.append(f"{ch}: shape[1]={arr.shape[1]} expected {samples}")
    for kind, fname in (("baseline", FNAME_BASELINE), ("recent", FNAME_RECENT),
                        ("events", FNAME_EVENTS), ("future", FNAME_FUTURE)):
        p = entity_dir / fname
        if not p.exists():
            errors.append(f"missing {fname}"); continue
        arr = np.load(p)
        if arr.dtype != EHR_EVENT_DTYPE:
            errors.append(f"{fname}: dtype mismatch"); continue
        errors.extend(validate_partition(arr, kind=kind, n_seg=n_seg))
    return entry, errors


def group_split(mrn_groups: dict[str, list[str]], ratios, seed: int):
    """Split LOG_IDs grouped by MRN to avoid patient leakage."""
    rng = np.random.default_rng(seed)
    mrns = sorted(mrn_groups.keys())
    rng.shuffle(mrns)
    n = len(mrns)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    buckets = {"train": mrns[:n_train],
               "val":   mrns[n_train:n_train + n_val],
               "test":  mrns[n_train + n_val:]}
    return {k: sorted(sum((mrn_groups[m] for m in v), [])) for k, v in buckets.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=OUT_ROOT)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--ratios", default=",".join(str(x) for x in DEFAULT_RATIOS))
    args = ap.parse_args()
    ratios = tuple(float(x) for x in args.ratios.split(","))
    assert len(ratios) == 3 and abs(sum(ratios) - 1.0) < 1e-6
    out_root = Path(args.out_root)
    log.info(f"out_root = {out_root}  seed={args.seed}  ratios={ratios}")
    t0 = time.time()
    entity_dirs = sorted(p for p in out_root.iterdir()
                         if p.is_dir() and (p / "meta.json").exists())
    log.info(f"entity dirs on disk: {len(entity_dirs)}")
    manifest, errors = [], {}
    for i, d in enumerate(entity_dirs, 1):
        entry, errs = validate_entity(d)
        manifest.append(entry)
        if errs:
            errors[d.name] = errs
        if i % 1000 == 0 or i == len(entity_dirs):
            log.info(f"  [{i}/{len(entity_dirs)}] pass={i-len(errors)} fail={len(errors)}")
    valid = [e for e in manifest if e["entity_id"] not in errors]
    (out_root / "manifest.json").write_text(json.dumps(valid, indent=2, default=str))
    log.info(f"wrote manifest.json ({len(valid)})")

    mrn_groups: dict[str, list[str]] = {}
    for e in valid:
        m = str(e.get("mrn") or "unknown")
        mrn_groups.setdefault(m, []).append(e["entity_id"])
    splits = group_split(mrn_groups, ratios, args.seed)

    pretrain_json = {"seed": args.seed, "ratios": list(ratios),
                     "group_by": "mrn", "stratify_by": None,
                     "n_unique_patients_mrn": len(mrn_groups),
                     "n_entities": len(valid),
                     "entities_by_split": {k: {"n_entities": len(v)} for k, v in splits.items()},
                     **splits}
    (out_root / "pretrain_splits.json").write_text(json.dumps(pretrain_json, indent=2))
    log.info(f"wrote pretrain_splits.json  " +
             str({k: len(v) for k, v in splits.items()}))

    entry_by_eid = {e["entity_id"]: e for e in valid}
    def build_list(eids):
        return [[str(out_root / e), str(entry_by_eid[e].get("mrn", "")),
                 0, int(entry_by_eid[e].get("n_seg", 0)), -1, 0] for e in eids]
    downstream = {
        "train_control_list": build_list(splits["train"]),
        "val_control_list":   build_list(splits["val"]),
        "test_control_list":  build_list(splits["test"]),
    }
    (out_root / "downstream_splits.json").write_text(json.dumps(downstream, indent=2))
    log.info(f"wrote downstream_splits.json")

    summary = {
        "stage": "f_manifest",
        "ran_at_unix": int(time.time()),
        "elapsed_sec": round(time.time() - t0, 1),
        "n_entity_dirs": len(entity_dirs),
        "n_valid": len(valid),
        "n_failed": len(errors),
        "n_unique_patients_mrn": len(mrn_groups),
        "total_segments": int(sum(e.get("n_seg", 0) for e in valid)),
        "total_ehr_events_in_wave": int(sum(e.get("n_events", 0) for e in valid)),
        "entities_by_split": {k: len(v) for k, v in splits.items()},
        "failed_first_20": dict(list(errors.items())[:20]),
    }
    (Path(OUTPUTS_DIR) / "stage_f_manifest_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "failed_first_20"},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
