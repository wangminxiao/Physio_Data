#!/usr/bin/env python3
"""
Stage G — Emory sepsis task post-stage.

Builds `{OUT_ROOT}/tasks/sepsis/{cohort.json, splits.json}`:

  cohort.json  per-entity record: label (case/control) + sepsis_time_zero +
               valid window + wave bounds + encounter/patient keys.

  splits.json  train/val/test lists. The pretraining splits were already
               patient-grouped (`empi_nbr`) and stratified by `type`
               (case/control), so they ARE the sepsis task splits; we copy
               them here with the task-specific schema.

Only entities with Stage B meta.json on disk are emitted.
"""
import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path

import numpy as np
import polars as pl

OUT_ROOT = "/opt/localdata100tb/physio_data/emory"
COHORT_PARQUET = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/emory/valid_wave_window.parquet"
OUTPUTS_DIR = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/emory"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _safe_int(x):
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _safe_float(x):
    try:
        if x is None:
            return None
        f = float(x)
        return f if np.isfinite(f) else None
    except Exception:
        return None


def build_cohort(cohort_df: pl.DataFrame, out_root: Path) -> list[dict]:
    rows: list[dict] = []
    n_no_meta = 0
    n_no_time = 0
    for r in cohort_df.iter_rows(named=True):
        eid = r["entity_id"]
        edir = out_root / eid
        meta_path = edir / "meta.json"
        time_path = edir / "time_ms.npy"
        if not meta_path.exists():
            n_no_meta += 1
            continue
        if not time_path.exists():
            n_no_time += 1
            continue
        time_ms = np.load(time_path, mmap_mode="r")
        if len(time_ms) == 0:
            n_no_time += 1
            continue
        wave_start = int(time_ms[0])
        wave_end = int(time_ms[-1])

        label = r["type"]  # "case" | "control"
        is_case = int(label == "case")
        sepsis_time_zero_ms = _safe_int(r.get("sepsis_time_zero_ms"))
        min_event_to_wave_end_minutes = None
        if sepsis_time_zero_ms is not None:
            min_event_to_wave_end_minutes = round(
                (sepsis_time_zero_ms - wave_end) / 60000.0, 2
            )

        rows.append({
            "entity_id": eid,
            "empi_nbr": _safe_int(r.get("empi_nbr")),
            "encounter_nbr": _safe_int(r.get("encounter_nbr")),
            "pat_id": r.get("pat_id"),
            "type": label,
            "is_case": is_case,
            "sepsis_time_zero_utc_ms": sepsis_time_zero_ms,
            "min_event_to_wave_end_minutes": min_event_to_wave_end_minutes,
            "valid_start_utc_ms": _safe_int(r.get("valid_start_ms")),
            "valid_end_utc_ms":   _safe_int(r.get("valid_end_ms")),
            "valid_duration_hour": _safe_float(r.get("valid_duration_hour_sum")),
            "valid_ratio": _safe_float(r.get("valid_ratio_mean")),
            "wave_start_ms": wave_start,
            "wave_end_ms":   wave_end,
            "admit_ms":      _safe_int(r.get("admit_ms")),
            "discharge_ms":  _safe_int(r.get("discharge_ms")),
        })
    log.info(f"cohort rows: {len(rows)}  (skipped no_meta={n_no_meta} no_time={n_no_time})")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=str, default=OUT_ROOT)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    task_dir = out_root / "tasks" / "sepsis"
    task_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"task_dir = {task_dir}")

    t0 = time.time()
    cohort_df = (
        pl.read_parquet(COHORT_PARQUET)
          .unique("entity_id", keep="first")
    )
    log.info(f"cohort parquet: {cohort_df.height}")

    cohort = build_cohort(cohort_df, out_root)
    if not cohort:
        log.error("empty cohort — abort")
        sys.exit(1)

    n_case = sum(1 for r in cohort if r["is_case"])
    n_control = len(cohort) - n_case

    # Sepsis_time_zero distribution relative to wave window
    before = during = after = none_ = 0
    for r in cohort:
        if not r["is_case"] or r["sepsis_time_zero_utc_ms"] is None:
            none_ += 1
            continue
        t0z = r["sepsis_time_zero_utc_ms"]
        if t0z < r["wave_start_ms"]:
            before += 1
        elif t0z <= r["wave_end_ms"]:
            during += 1
        else:
            after += 1

    cohort_json = {
        "task": "sepsis_prediction",
        "source_csv": "/labs/hulab/mxwang/data/sepsis/Wav/sepsis_cc_2025_06_13_all_collab_uniq_combine.csv",
        "built_at_unix": int(time.time()),
        "n_entities": len(cohort),
        "n_case": n_case,
        "n_control": n_control,
        "sepsis_time_zero_position": {
            "before_wave": before,
            "during_wave": during,
            "after_wave":  after,
            "null_for_case_or_control": none_,
        },
        "fields": list(cohort[0].keys()),
        "entities": cohort,
    }
    (task_dir / "cohort.json").write_text(
        json.dumps(cohort_json, indent=2, default=str))
    log.info(f"wrote {task_dir/'cohort.json'}")

    # Reuse pretrain_splits.json (already stratified by type + grouped by empi_nbr)
    pretrain = json.loads((out_root / "pretrain_splits.json").read_text())
    eid_set = {r["entity_id"] for r in cohort}

    splits = {}
    for k in ("train", "val", "test"):
        splits[k] = [e for e in pretrain[k] if e in eid_set]

    idx = {r["entity_id"]: r["is_case"] for r in cohort}

    def _counts(ents):
        n_pos = sum(1 for e in ents if idx.get(e, 0) == 1)
        return {"n": len(ents), "n_pos": n_pos, "n_neg": len(ents) - n_pos}

    entity_counts = {k: _counts(v) for k, v in splits.items()}

    # Patient-disjointness sanity
    eid_to_empi = {r["entity_id"]: r["empi_nbr"] for r in cohort}
    pats = {k: {eid_to_empi[e] for e in v if e in eid_to_empi} for k, v in splits.items()}
    assert not (pats["train"] & pats["val"]),  "train/val empi leakage"
    assert not (pats["train"] & pats["test"]), "train/test empi leakage"
    assert not (pats["val"] & pats["test"]),   "val/test empi leakage"

    splits_json = {
        "task": "sepsis_prediction",
        "seed": pretrain.get("seed"),
        "ratios": pretrain.get("ratios"),
        "group_by": "empi_nbr",
        "stratify_by": "type",
        "source": "inherits pretrain_splits.json",
        "entity_class_counts": entity_counts,
        "train": splits["train"],
        "val":   splits["val"],
        "test":  splits["test"],
    }
    (task_dir / "splits.json").write_text(json.dumps(splits_json, indent=2))
    log.info(f"wrote {task_dir/'splits.json'}")

    log.info(f"\nSepsis task cohort:")
    log.info(f"  entities: {len(cohort)}  (case={n_case}, control={n_control})")
    log.info(f"  sepsis_time_zero position: {cohort_json['sepsis_time_zero_position']}")
    log.info(f"  splits (entities): train={entity_counts['train']}  val={entity_counts['val']}  test={entity_counts['test']}")
    log.info(f"elapsed {time.time()-t0:.1f}s")
    print(json.dumps({
        "n_entities": len(cohort),
        "n_case": n_case,
        "n_control": n_control,
        "sepsis_time_zero_position": cohort_json["sepsis_time_zero_position"],
        "entity_class_counts": entity_counts,
    }, indent=2))


if __name__ == "__main__":
    main()
