#!/usr/bin/env python3
"""
Stage F-1 — MC_MED manifest + splits.

Scans every entity directory under OUT_ROOT, validates shapes/consistency,
and emits:

  {OUT_ROOT}/manifest.json           validated entities
  {OUT_ROOT}/pretrain_splits.json    adopted from MC_MED split_random_*.csv
                                     (patient-safe per README)
  {OUT_ROOT}/pretrain_splits_chrono.json   adopted from split_chrono_*.csv
  {OUT_ROOT}/downstream_splits.json  UNIPHY-compatible list format

We re-use MC_MED's pre-defined splits rather than shuffling ourselves, so
our results stay comparable to the ICML MC_MED paper / benchmarks.
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

OUT_ROOT = "/opt/localdata100tb/physio_data/mcmed"
RAW_ROOT = "/opt/localdata100tb/UNIPHY_Plus/raw_datasets/physionet.org/files/mc-med/1.0.1/data"
COHORT_PARQUET = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/mcmed/valid_cohort.parquet"
OUTPUTS_DIR = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/mcmed"

EXPECTED_CHANNELS = {"PLETH40": 1200, "II120": 3600}

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def validate_entity(entity_dir: Path) -> tuple[dict, list[str]]:
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
        "csn": meta.get("csn"),
        "mrn": meta.get("mrn"),
        "n_seg": n_seg,
        "source_dataset": meta.get("source_dataset", "mcmed"),
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
            errors.append(f"time_ms dtype {t.dtype}, expected int64")
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
            errors.append(f"{ch}: dtype {arr.dtype}, expected float16")
        if not arr.flags["C_CONTIGUOUS"]:
            errors.append(f"{ch}: not C-contiguous")
        if arr.shape[0] != n_seg:
            errors.append(f"{ch}: shape[0]={arr.shape[0]}, expected {n_seg}")
        if arr.ndim == 2 and arr.shape[1] != samples:
            errors.append(f"{ch}: shape[1]={arr.shape[1]}, expected {samples}")

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


def load_mcmed_split_csns(flavor: str, split: str) -> list[int]:
    """Read MC_MED pre-defined split csv (one CSN per line, no header)."""
    p = f"{RAW_ROOT}/split_{flavor}_{split}.csv"
    out = []
    with open(p) as f:
        for line in f:
            s = line.strip()
            if not s or s.lower() == "csn":
                continue
            try:
                out.append(int(s))
            except ValueError:
                continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=OUT_ROOT)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    log.info(f"out_root = {out_root}")

    cohort = pl.read_parquet(COHORT_PARQUET).unique("entity_id", keep="first")
    csn_to_eid = {int(r["csn"]): r["entity_id"] for r in cohort.iter_rows(named=True)
                  if r.get("csn") is not None}
    log.info(f"cohort entities: {cohort.height}")

    t0 = time.time()
    entity_dirs = sorted(p for p in out_root.iterdir()
                         if p.is_dir() and (p / "meta.json").exists())
    log.info(f"entity dirs on disk: {len(entity_dirs)}")

    manifest: list[dict] = []
    errors: dict[str, list[str]] = {}
    for i, d in enumerate(entity_dirs, 1):
        entry, errs = validate_entity(d)
        manifest.append(entry)
        if errs:
            errors[d.name] = errs
        if i % 1000 == 0 or i == len(entity_dirs):
            log.info(f"  [{i}/{len(entity_dirs)}] pass={i-len(errors)} fail={len(errors)}")

    n_pass = len(manifest) - len(errors)
    log.info(f"validation: pass={n_pass} fail={len(errors)}")
    if errors:
        log.warning("first 10 failing entities:")
        for k, v in list(errors.items())[:10]:
            log.warning(f"  {k}: {v}")

    valid_entries = [e for e in manifest if e["entity_id"] not in errors]
    (out_root / "manifest.json").write_text(json.dumps(valid_entries, indent=2, default=str))
    log.info(f"wrote manifest.json ({len(valid_entries)})")

    valid_set = {e["entity_id"] for e in valid_entries}

    def build_splits(flavor: str) -> dict:
        res = {"train": [], "val": [], "test": []}
        for split in ("train", "val", "test"):
            csns = load_mcmed_split_csns(flavor, split)
            eids = []
            for c in csns:
                eid = csn_to_eid.get(c)
                if eid is not None and eid in valid_set:
                    eids.append(eid)
            res[split] = sorted(eids)
        return res

    random_splits = build_splits("random")
    chrono_splits = build_splits("chrono")

    def stats(splits):
        return {k: {"n_entities": len(v),
                    "n_unique_mrn": len({next((e["mrn"] for e in valid_entries
                                               if e["entity_id"] == eid), None)
                                         for eid in v})}
                for k, v in splits.items()}

    pretrain_json = {
        "source": "MC_MED split_random_*.csv (80/10/10, patient-safe per README)",
        "entities_by_split": stats(random_splits),
        "train": random_splits["train"],
        "val":   random_splits["val"],
        "test":  random_splits["test"],
    }
    (out_root / "pretrain_splits.json").write_text(json.dumps(pretrain_json, indent=2))
    log.info(f"wrote pretrain_splits.json  {stats(random_splits)}")

    chrono_json = {
        "source": "MC_MED split_chrono_*.csv (time-ordered, no patient overlap)",
        "entities_by_split": stats(chrono_splits),
        **chrono_splits,
    }
    (out_root / "pretrain_splits_chrono.json").write_text(json.dumps(chrono_json, indent=2))
    log.info(f"wrote pretrain_splits_chrono.json  {stats(chrono_splits)}")

    entry_by_eid = {e["entity_id"]: e for e in valid_entries}

    def build_list(eids: list[str]) -> list[list]:
        out = []
        for eid in eids:
            e = entry_by_eid[eid]
            out.append([str(out_root / eid), str(e.get("mrn", "")),
                        0, int(e.get("n_seg", 0)), -1, 0])
        return out

    downstream = {
        "train_control_list": build_list(random_splits["train"]),
        "val_control_list":   build_list(random_splits["val"]),
        "test_control_list":  build_list(random_splits["test"]),
    }
    (out_root / "downstream_splits.json").write_text(json.dumps(downstream, indent=2))
    log.info(f"wrote downstream_splits.json")

    total_seg = sum(e.get("n_seg", 0) for e in valid_entries)
    total_events = sum(e.get("n_events", 0) for e in valid_entries)
    summary = {
        "stage": "f_manifest",
        "ran_at_unix": int(time.time()),
        "elapsed_sec": round(time.time() - t0, 1),
        "n_entity_dirs": len(entity_dirs),
        "n_valid": len(valid_entries),
        "n_failed": len(errors),
        "total_segments": int(total_seg),
        "total_ehr_events_in_wave": int(total_events),
        "random_entities_by_split": stats(random_splits),
        "chrono_entities_by_split": stats(chrono_splits),
        "failed_first_20": dict(list(errors.items())[:20]),
    }
    (Path(OUTPUTS_DIR) / "stage_f_manifest_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    log.info(f"wrote stage_f_manifest_summary.json")
    print(json.dumps({k: v for k, v in summary.items() if k != "failed_first_20"},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
