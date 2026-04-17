"""
Stage F - UCSF manifest & splits.

Scans every entity directory under {output_dir}/, validates shape and
consistency of every file, and emits:

  {output_dir}/manifest.json         per-entity stats (valid entities only)
  {output_dir}/pretrain_splits.json  train/val/test, grouped by patient_id_ge
  {output_dir}/downstream_splits.json UNIPHY-compatible list format

Validation (per entity):
  - meta.json present + readable
  - time_ms.npy: int64, monotone, length == n_seg in meta
  - PLETH40.npy + II120.npy: float16, C-contiguous, shape[0] == n_seg
  - ehr_{baseline,recent,events,future}.npy: dtype matches EHR_EVENT_DTYPE,
    sorted, seg_idx sentinels correct (ehr_events has real indices)

Split:
  - Group by patient_id_ge (one wave cycle == one entity; patients can have
    multiple wave cycles that MUST stay in the same split)
  - Shuffle patients with seed=42, deterministic
  - 70/15/15 unstratified (stratification left to per-task scripts)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from physio_data.ehr_trajectory import (  # noqa: E402
    EHR_EVENT_DTYPE,
    FNAME_BASELINE, FNAME_RECENT, FNAME_EVENTS, FNAME_FUTURE,
    validate_partition,
)

CONFIG_PATH = REPO_ROOT / "workzone" / "configs" / "server_paths.yaml"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_SEED = 42
DEFAULT_RATIOS = (0.70, 0.15, 0.15)

EXPECTED_CHANNELS = {
    "PLETH40": 1200,  # 40 Hz * 30 s
    "II120":   3600,  # 120 Hz * 30 s
}


def validate_entity(entity_dir: Path) -> tuple[dict, list[str]]:
    entry: dict = {"entity_id": entity_dir.name}
    errors: list[str] = []

    meta_path = entity_dir / "meta.json"
    if not meta_path.exists():
        errors.append("missing meta.json")
        return entry, errors
    try:
        meta = json.loads(meta_path.read_text())
    except Exception as e:
        errors.append(f"meta.json parse error: {e}")
        return entry, errors

    n_seg = int(meta.get("n_seg") or meta.get("n_segments") or 0)
    entry.update({
        "patient_id_ge":   meta.get("patient_id_ge"),
        "wave_cycle_uid":  meta.get("wave_cycle_uid"),
        "encounter_id":    meta.get("encounter_id"),
        "n_seg":           n_seg,
        "source_dataset":  meta.get("source_dataset", "ucsf"),
        "has_ca":          meta.get("has_ca"),
    })
    for k in ("n_baseline", "n_recent", "n_events", "n_future",
              "ehr_layout_version"):
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

    # Channels
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

    # EHR 4 partitions
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


def build_splits(valid_entries: list[dict], ratios: tuple[float, float, float],
                 seed: int) -> tuple[dict[str, list[str]], dict]:
    assert abs(sum(ratios) - 1.0) < 1e-6
    train_r, val_r, _ = ratios

    by_pat: dict[str, list[str]] = {}
    for e in valid_entries:
        pid = str(e["patient_id_ge"])
        by_pat.setdefault(pid, []).append(e["entity_id"])

    pids = sorted(by_pat.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(pids)

    n = len(pids)
    n_train = int(round(n * train_r))
    n_val = int(round(n * val_r))
    train_p = pids[:n_train]
    val_p = pids[n_train:n_train + n_val]
    test_p = pids[n_train + n_val:]

    def entities(pids_) -> list[str]:
        out: list[str] = []
        for p in pids_:
            out.extend(by_pat[p])
        return sorted(out)

    splits = {"train": entities(train_p), "val": entities(val_p),
              "test":  entities(test_p)}
    stats = {
        "n_unique_patients": n,
        "n_train_patients": len(train_p),
        "n_val_patients":   len(val_p),
        "n_test_patients":  len(test_p),
        "n_multi_cycle_patients": sum(1 for v in by_pat.values() if len(v) > 1),
    }
    return splits, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--ratios", default=",".join(str(x) for x in DEFAULT_RATIOS),
                    help="train,val,test fractions")
    args = ap.parse_args()

    ratios = tuple(float(x) for x in args.ratios.split(","))
    assert len(ratios) == 3

    cfg = yaml.safe_load(Path(args.config).read_text())["ucsf"]
    output_dir = Path(cfg["output_dir"])
    intermediate_dir = Path(cfg["intermediate_dir"])

    log.info(f"output_dir = {output_dir}")
    log.info(f"seed={args.seed} ratios={ratios}")

    t0 = time.time()
    entity_dirs = sorted(
        p for p in output_dir.iterdir()
        if p.is_dir() and (p / "meta.json").exists()
    )
    log.info(f"entity dirs: {len(entity_dirs)}")

    manifest: list[dict] = []
    errors: dict[str, list[str]] = {}
    for i, d in enumerate(entity_dirs, 1):
        entry, errs = validate_entity(d)
        manifest.append(entry)
        if errs:
            errors[d.name] = errs
        if i % 500 == 0 or i == len(entity_dirs):
            log.info(f"  [{i}/{len(entity_dirs)}] "
                     f"pass={i - len(errors)} fail={len(errors)}")

    n_pass = len(manifest) - len(errors)
    log.info(f"validation: pass={n_pass} fail={len(errors)}")
    if errors:
        log.warning("first 10 failing entities:")
        for k, v in list(errors.items())[:10]:
            log.warning(f"  {k}: {v}")

    valid_entries = [e for e in manifest if e["entity_id"] not in errors]
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(valid_entries, indent=2, default=str))
    log.info(f"wrote {manifest_path} ({len(valid_entries)} entries)")

    splits, split_stats = build_splits(valid_entries, ratios, args.seed)

    # Sanity: no patient leakage
    eid_to_pat = {e["entity_id"]: str(e["patient_id_ge"]) for e in valid_entries}
    pats = {k: {eid_to_pat[e] for e in v} for k, v in splits.items()}
    assert not (pats["train"] & pats["val"]),  "train/val patient leakage"
    assert not (pats["train"] & pats["test"]), "train/test patient leakage"
    assert not (pats["val"] & pats["test"]),   "val/test patient leakage"

    pretrain_json = {
        "seed": args.seed,
        "ratios": list(ratios),
        "group_by": "patient_id_ge",
        **split_stats,
        "n_train": len(splits["train"]),
        "n_val":   len(splits["val"]),
        "n_test":  len(splits["test"]),
        "train": splits["train"],
        "val":   splits["val"],
        "test":  splits["test"],
    }
    (output_dir / "pretrain_splits.json").write_text(
        json.dumps(pretrain_json, indent=2))
    log.info(f"wrote {output_dir/'pretrain_splits.json'}")

    # UNIPHY-compatible downstream splits (list of [dir, patient_id, 0, n_seg, -1, 0])
    entry_by_eid = {e["entity_id"]: e for e in valid_entries}

    def build_list(eids: list[str]) -> list[list]:
        out = []
        for eid in eids:
            e = entry_by_eid[eid]
            out.append([
                str(output_dir / eid),
                str(e["patient_id_ge"]),
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
    (output_dir / "downstream_splits.json").write_text(
        json.dumps(downstream, indent=2))
    log.info(f"wrote {output_dir/'downstream_splits.json'}")

    total_events = sum(e.get("n_events", 0) for e in valid_entries)
    total_seg = sum(e.get("n_seg", 0) for e in valid_entries)
    summary = {
        "stage": "f_manifest",
        "ran_at_unix": int(time.time()),
        "elapsed_sec": round(time.time() - t0, 1),
        "n_entity_dirs": len(entity_dirs),
        "n_valid": len(valid_entries),
        "n_failed": len(errors),
        "total_segments": total_seg,
        "total_ehr_events_in_wave": total_events,
        **split_stats,
        "n_train_entities": len(splits["train"]),
        "n_val_entities":   len(splits["val"]),
        "n_test_entities":  len(splits["test"]),
        "failed_first_20":  dict(list(errors.items())[:20]),
    }
    (intermediate_dir / "stage_f_manifest_summary.json").write_text(
        json.dumps(summary, indent=2))
    log.info(f"wrote {intermediate_dir/'stage_f_manifest_summary.json'}")
    log.info(f"done in {time.time()-t0:.1f}s")
    print(json.dumps({k: v for k, v in summary.items() if k != "failed_first_20"},
                     indent=2))


if __name__ == "__main__":
    main()
