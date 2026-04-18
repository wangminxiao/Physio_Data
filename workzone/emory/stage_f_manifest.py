#!/usr/bin/env python3
"""
Stage F-1 — Emory manifest + splits.

Scans every entity directory under OUT_ROOT, validates file shapes and
consistency, and emits:

  {OUT_ROOT}/manifest.json           per-entity stats (valid entities only)
  {OUT_ROOT}/pretrain_splits.json    train/val/test grouped by empi_nbr,
                                     stratified by case/control type
  {OUT_ROOT}/downstream_splits.json  UNIPHY-compatible list format

Validation (per entity):
  - meta.json present + readable; stage_e_version >= 1
  - time_ms.npy: int64, strictly monotonic, len == n_segments
  - PLETH40.npy: float16, C-contig, shape (n_seg, 1200)
  - II120.npy:   float16, C-contig, shape (n_seg, 3600)
  - ehr_{baseline,recent,events,future}.npy: EHR_EVENT_DTYPE, sorted,
    sentinel seg_idx for outer partitions, real indices for events

Splits:
  - Group by empi_nbr (patient-level; multi-encounter patients stay together)
  - Stratified by type (case / control) → each split preserves ~6% case rate
  - Default ratios 70/15/15, seed 42
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
import logging
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

OUT_ROOT = "/opt/localdata100tb/physio_data/emory"
COHORT_PARQUET = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/emory/valid_wave_window.parquet"
OUTPUTS_DIR = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/emory"

DEFAULT_SEED = 42
DEFAULT_RATIOS = (0.70, 0.15, 0.15)

EXPECTED_CHANNELS = {
    "PLETH40": 1200,
    "II120":   3600,
}

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def validate_entity(entity_dir: Path) -> tuple[dict, list[str]]:
    entry: dict = {"entity_id": entity_dir.name}
    errors: list[str] = []

    meta_path = entity_dir / "meta.json"
    if not meta_path.exists():
        return entry, ["missing meta.json"]
    try:
        meta = json.loads(meta_path.read_text())
    except Exception as e:
        return entry, [f"meta.json parse error: {e}"]

    n_seg = int(meta.get("n_segments") or meta.get("n_seg") or 0)
    entry.update({
        "empi_nbr":       meta.get("empi_nbr"),
        "encounter_nbr":  meta.get("encounter_nbr"),
        "pat_id":         meta.get("pat_id"),
        "n_seg":          n_seg,
        "source_dataset": meta.get("source_dataset", "emory_sepsis"),
    })
    for k in ("n_baseline", "n_recent", "n_events", "n_future",
              "n_events_vars", "ehr_layout_version",
              "stage_b_version", "stage_c_version",
              "stage_d_version", "stage_e_version",
              "admission_start_ms", "admission_end_ms"):
        if k in meta:
            entry[k] = meta[k]

    # time_ms.npy
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
            entry["wave_end_ms"] = int(t[-1])

    for ch, samples in EXPECTED_CHANNELS.items():
        p = entity_dir / f"{ch}.npy"
        if not p.exists():
            errors.append(f"missing {ch}.npy")
            continue
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
            errors.append(f"missing {fname}")
            continue
        arr = np.load(p)
        if arr.dtype != EHR_EVENT_DTYPE:
            errors.append(f"{fname}: dtype mismatch")
            continue
        errors.extend(validate_partition(arr, kind=kind, n_seg=n_seg))

    return entry, errors


def stratified_group_split(groups_with_label: list[tuple[str, str]],
                           ratios: tuple[float, float, float],
                           seed: int) -> dict[str, list[str]]:
    """Greedy stratified patient-level split.

    Input: list of (group_key, label) pairs, one per group. Returns
    {split_name: [group_keys]}. Each label is assigned to splits in the
    desired ratio by shuffling within label then slicing.
    """
    by_label: dict[str, list[str]] = {}
    for g, lbl in groups_with_label:
        by_label.setdefault(lbl, []).append(g)
    rng = np.random.default_rng(seed)
    result: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for lbl, pids in by_label.items():
        pids = sorted(set(pids))
        rng.shuffle(pids)
        n = len(pids)
        n_train = int(round(n * ratios[0]))
        n_val = int(round(n * ratios[1]))
        result["train"].extend(pids[:n_train])
        result["val"].extend(pids[n_train:n_train + n_val])
        result["test"].extend(pids[n_train + n_val:])
    for k in result:
        result[k] = sorted(result[k])
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=str, default=OUT_ROOT)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--ratios", default=",".join(str(x) for x in DEFAULT_RATIOS))
    args = ap.parse_args()

    ratios = tuple(float(x) for x in args.ratios.split(","))
    assert len(ratios) == 3 and abs(sum(ratios) - 1.0) < 1e-6

    out_root = Path(args.out_root)
    log.info(f"out_root = {out_root}")
    log.info(f"seed={args.seed}  ratios={ratios}")

    cohort = pl.read_parquet(COHORT_PARQUET).unique("entity_id", keep="first")
    type_by_eid = {r["entity_id"]: r["type"] for r in cohort.to_dicts()}
    log.info(f"cohort entities: {cohort.height}")

    t0 = time.time()
    entity_dirs = sorted(
        p for p in out_root.iterdir()
        if p.is_dir() and (p / "meta.json").exists()
    )
    log.info(f"entity dirs on disk: {len(entity_dirs)}")

    manifest: list[dict] = []
    errors: dict[str, list[str]] = {}
    for i, d in enumerate(entity_dirs, 1):
        entry, errs = validate_entity(d)
        # attach cohort type (case/control)
        entry["type"] = type_by_eid.get(d.name)
        manifest.append(entry)
        if errs:
            errors[d.name] = errs
        if i % 1000 == 0 or i == len(entity_dirs):
            log.info(f"  [{i}/{len(entity_dirs)}] pass={i - len(errors)} fail={len(errors)}")

    n_pass = len(manifest) - len(errors)
    log.info(f"validation: pass={n_pass} fail={len(errors)}")
    if errors:
        log.warning("first 10 failing entities:")
        for k, v in list(errors.items())[:10]:
            log.warning(f"  {k}: {v}")

    valid_entries = [e for e in manifest if e["entity_id"] not in errors]

    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(valid_entries, indent=2, default=str))
    log.info(f"wrote {manifest_path}  ({len(valid_entries)} entries)")

    # --- Splits ---
    # Patient (empi_nbr) type = "case" if any encounter in cohort is case else "control"
    eid_by_empi: dict[str, list[dict]] = {}
    for e in valid_entries:
        empi = str(e["empi_nbr"])
        eid_by_empi.setdefault(empi, []).append(e)
    empi_label: dict[str, str] = {}
    for empi, entries in eid_by_empi.items():
        has_case = any(e.get("type") == "case" for e in entries)
        empi_label[empi] = "case" if has_case else "control"

    groups_with_label = [(k, v) for k, v in empi_label.items()]
    split_by_empi = stratified_group_split(groups_with_label, ratios, args.seed)

    def empis_to_entities(empis: list[str]) -> list[str]:
        out = []
        for empi in empis:
            for e in eid_by_empi.get(empi, []):
                out.append(e["entity_id"])
        return sorted(out)

    splits = {k: empis_to_entities(v) for k, v in split_by_empi.items()}

    # leakage sanity
    empi_by_eid = {e["entity_id"]: str(e["empi_nbr"]) for e in valid_entries}
    pats = {k: {empi_by_eid[e] for e in v} for k, v in splits.items()}
    assert not (pats["train"] & pats["val"]),  "train/val empi leakage"
    assert not (pats["train"] & pats["test"]), "train/test empi leakage"
    assert not (pats["val"] & pats["test"]),   "val/test empi leakage"

    def counts(entities):
        n_case = sum(1 for eid in entities if type_by_eid.get(eid) == "case")
        return {"n_entities": len(entities), "n_case": n_case,
                "n_control": len(entities) - n_case,
                "case_frac": round(n_case / max(1, len(entities)), 4)}

    split_stats = {k: counts(v) for k, v in splits.items()}

    pretrain_json = {
        "seed": args.seed,
        "ratios": list(ratios),
        "group_by": "empi_nbr",
        "stratify_by": "type",
        "n_unique_patients": len(empi_label),
        "patients_by_split": {k: len(v) for k, v in split_by_empi.items()},
        "entities_by_split": split_stats,
        "train": splits["train"],
        "val":   splits["val"],
        "test":  splits["test"],
    }
    (out_root / "pretrain_splits.json").write_text(
        json.dumps(pretrain_json, indent=2))
    log.info(f"wrote {out_root/'pretrain_splits.json'}")

    # UNIPHY downstream_splits.json
    entry_by_eid = {e["entity_id"]: e for e in valid_entries}

    def build_list(eids: list[str]) -> list[list]:
        out = []
        for eid in eids:
            e = entry_by_eid[eid]
            out.append([
                str(out_root / eid),
                str(e["empi_nbr"]),
                0,
                int(e.get("n_seg", 0)),
                -1,
                0,
            ])
        return out

    downstream = {
        "train_control_list": build_list(splits["train"]),
        "val_control_list":   build_list(splits["val"]),
        "test_control_list":  build_list(splits["test"]),
    }
    (out_root / "downstream_splits.json").write_text(
        json.dumps(downstream, indent=2))
    log.info(f"wrote {out_root/'downstream_splits.json'}")

    total_seg = sum(e.get("n_seg", 0) for e in valid_entries)
    total_events = sum(e.get("n_events", 0) for e in valid_entries)
    summary = {
        "stage": "f_manifest",
        "ran_at_unix": int(time.time()),
        "elapsed_sec": round(time.time() - t0, 1),
        "n_entity_dirs": len(entity_dirs),
        "n_valid": len(valid_entries),
        "n_failed": len(errors),
        "n_unique_patients": len(empi_label),
        "total_segments": int(total_seg),
        "total_ehr_events_in_wave": int(total_events),
        "entities_by_split": split_stats,
        "failed_first_20": dict(list(errors.items())[:20]),
    }
    summary_path = Path(OUTPUTS_DIR) / "stage_f_manifest_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    log.info(f"wrote {summary_path}")
    print(json.dumps({k: v for k, v in summary.items() if k != "failed_first_20"},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
