"""
Stage F (ca_prediction) - UCSF cardiac-arrest cohort & splits.

Task: binary CA prediction per entity (wave cycle).

Per-entity fields emitted:
  - entity_id           {Patient_ID_GE}_{WaveCycleUID}
  - patient_id_ge       admission-grouping key (for split leakage avoidance)
  - has_ca              0/1 (1 = CA event recorded for this wave cycle)
  - event_time_ms       ms since epoch (GE time, ValidWaveTime CSV already
                        GE-shifted); null for controls
  - min_event_to_wave_end_min   (event_time - wave_end) / 60000, null for controls
  - episode_start_ms, episode_end_ms
  - wave_start_ms, wave_end_ms  (from time_ms.npy; first / last segment start)
  - admission_start_ms, admission_end_ms
  - wynton_folder, unit_bed

Outputs (under {output_dir}/tasks/ca_prediction/):
  - cohort.json   all entities with Stage B meta.json available
  - splits.json   train/val/test lists, stratified by has_ca, grouped by
                  patient_id_ge (no patient appears in multiple splits)

Reproducible: `--seed` controls the random split (default 42).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "workzone" / "configs" / "server_paths.yaml"


def build_cohort(parquet_path: Path, output_dir: Path) -> list[dict]:
    """Build per-entity CA cohort rows.

    Only keeps entities whose Stage B output dir has `meta.json` and
    `time_ms.npy` (i.e., Stage B ok). Picks up wave_start/wave_end from
    time_ms.npy for accurate bounds.
    """
    df = (
        pl.read_parquet(parquet_path)
        .unique("entity_id", keep="first")
        .with_columns(
            pl.col("event_time_raw").str.strptime(
                pl.Datetime, format="%Y-%m-%dT%H:%M:%S%.f", strict=False
            ).alias("event_dt")
        )
        .with_columns(
            pl.col("event_dt").dt.timestamp("ms").alias("event_time_ms")
        )
    )

    rows: list[dict] = []
    n_skip_no_meta = 0
    n_skip_no_time = 0
    for r in df.iter_rows(named=True):
        eid = r["entity_id"]
        edir = output_dir / eid
        meta_path = edir / "meta.json"
        time_path = edir / "time_ms.npy"
        if not meta_path.exists():
            n_skip_no_meta += 1
            continue
        if not time_path.exists():
            n_skip_no_time += 1
            continue
        time_ms = np.load(time_path, mmap_mode="r")
        if len(time_ms) == 0:
            n_skip_no_time += 1
            continue
        wave_start = int(time_ms[0])
        wave_end = int(time_ms[-1])

        ev_ms = r["event_time_ms"]
        min_after_wave_end = None
        if ev_ms is not None:
            min_after_wave_end = (int(ev_ms) - wave_end) / 60000.0

        rows.append({
            "entity_id": eid,
            "patient_id_ge": r["patient_id_ge"],
            "has_ca": int(r["has_ca"]),
            "event_time_ms": (int(ev_ms) if ev_ms is not None else None),
            "min_event_to_wave_end": (
                None if min_after_wave_end is None else round(min_after_wave_end, 2)
            ),
            "episode_start_ms": int(r["episode_start_ms"]),
            "episode_end_ms":   int(r["episode_end_ms"]),
            "wave_start_ms":    wave_start,
            "wave_end_ms":      wave_end,
            "admission_start_ms": int(r["admission_start_ms"]),
            "admission_end_ms":   int(r["admission_end_ms"]),
            "wynton_folder":    r["wynton_folder"],
            "unit_bed":         r["unit_bed"],
        })

    print(f"cohort rows: {len(rows)}  "
          f"(skipped: no_meta={n_skip_no_meta}, no_time={n_skip_no_time})",
          flush=True)
    return rows


def patient_split(cohort: list[dict], ratios: tuple[float, float, float],
                  seed: int) -> dict[str, list[str]]:
    """Patient-grouped stratified split.

    Strategy:
      1. Collapse to per-patient records with `has_any_ca` = any wave cycle positive.
      2. Stratify by `has_any_ca` to preserve class ratio across splits.
      3. Each patient's entities all go to the same split.

    Returns {"train": [eid,...], "val": [...], "test": [...]}.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1"
    train_r, val_r, test_r = ratios

    by_pat: dict[str, dict] = {}
    for row in cohort:
        pid = row["patient_id_ge"]
        entry = by_pat.setdefault(pid, {"entities": [], "has_any_ca": 0})
        entry["entities"].append(row["entity_id"])
        if row["has_ca"]:
            entry["has_any_ca"] = 1

    rng = np.random.default_rng(seed)

    def split_list(pids: list[str]) -> tuple[list[str], list[str], list[str]]:
        pids = list(pids)
        rng.shuffle(pids)
        n = len(pids)
        n_train = int(round(n * train_r))
        n_val   = int(round(n * val_r))
        # test gets the remainder (avoids drift from rounding)
        train = pids[:n_train]
        val   = pids[n_train:n_train + n_val]
        test  = pids[n_train + n_val:]
        return train, val, test

    pos_pids = [p for p, e in by_pat.items() if e["has_any_ca"] == 1]
    neg_pids = [p for p, e in by_pat.items() if e["has_any_ca"] == 0]

    pos_tr, pos_va, pos_te = split_list(pos_pids)
    neg_tr, neg_va, neg_te = split_list(neg_pids)

    def entities_for(pids: list[str]) -> list[str]:
        out: list[str] = []
        for p in pids:
            out.extend(by_pat[p]["entities"])
        return sorted(out)

    splits = {
        "train": entities_for(pos_tr + neg_tr),
        "val":   entities_for(pos_va + neg_va),
        "test":  entities_for(pos_te + neg_te),
    }
    return splits, {
        "n_patients_pos": len(pos_pids),
        "n_patients_neg": len(neg_pids),
        "patients_train": {"pos": len(pos_tr), "neg": len(neg_tr)},
        "patients_val":   {"pos": len(pos_va), "neg": len(neg_va)},
        "patients_test":  {"pos": len(pos_te), "neg": len(neg_te)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ratios", default="0.70,0.15,0.15",
                    help="train,val,test fractions")
    args = ap.parse_args()

    ratios = tuple(float(x) for x in args.ratios.split(","))
    assert len(ratios) == 3, "--ratios must be 3 comma-separated floats"

    cfg = yaml.safe_load(Path(args.config).read_text())["ucsf"]
    output_dir = Path(cfg["output_dir"])
    intermediate_dir = Path(cfg["intermediate_dir"])
    parquet = intermediate_dir / "valid_wave_window.parquet"

    task_dir = output_dir / "tasks" / "ca_prediction"
    task_dir.mkdir(parents=True, exist_ok=True)
    print(f"task_dir = {task_dir}")
    print(f"parquet  = {parquet}")
    print(f"seed     = {args.seed}, ratios={ratios}", flush=True)

    t0 = time.time()
    cohort = build_cohort(parquet, output_dir)
    if not cohort:
        print("ERROR: empty cohort; make sure Stage B has run first")
        sys.exit(1)

    n_pos_entities = sum(1 for r in cohort if r["has_ca"])
    n_neg_entities = len(cohort) - n_pos_entities

    splits, split_stats = patient_split(cohort, ratios, args.seed)

    def class_counts(eid_list: list[str]) -> dict[str, int]:
        idx = {r["entity_id"]: r["has_ca"] for r in cohort}
        n_pos = sum(1 for e in eid_list if idx.get(e) == 1)
        return {"n": len(eid_list), "n_pos": n_pos, "n_neg": len(eid_list) - n_pos}

    counts = {k: class_counts(v) for k, v in splits.items()}

    # Sanity: patient disjoint across splits
    eid_to_pat = {r["entity_id"]: r["patient_id_ge"] for r in cohort}
    train_pats = {eid_to_pat[e] for e in splits["train"]}
    val_pats   = {eid_to_pat[e] for e in splits["val"]}
    test_pats  = {eid_to_pat[e] for e in splits["test"]}
    assert not (train_pats & val_pats), "patient leakage train vs val"
    assert not (train_pats & test_pats), "patient leakage train vs test"
    assert not (val_pats & test_pats), "patient leakage val vs test"

    cohort_json = {
        "task": "ca_prediction",
        "source_csv": cfg["ca_eventtime_csv"],
        "built_at_unix": int(time.time()),
        "n_entities": len(cohort),
        "n_pos_entities": n_pos_entities,
        "n_neg_entities": n_neg_entities,
        "n_patients_pos": split_stats["n_patients_pos"],
        "n_patients_neg": split_stats["n_patients_neg"],
        "fields": [
            "entity_id", "patient_id_ge", "has_ca",
            "event_time_ms", "min_event_to_wave_end",
            "episode_start_ms", "episode_end_ms",
            "wave_start_ms", "wave_end_ms",
            "admission_start_ms", "admission_end_ms",
            "wynton_folder", "unit_bed",
        ],
        "entities": cohort,
    }
    (task_dir / "cohort.json").write_text(json.dumps(cohort_json, indent=2))

    splits_json = {
        "task": "ca_prediction",
        "seed": args.seed,
        "ratios": list(ratios),
        "group_by": "patient_id_ge",
        "stratify_by": "has_any_ca",
        "n_train": len(splits["train"]),
        "n_val":   len(splits["val"]),
        "n_test":  len(splits["test"]),
        "patient_counts": split_stats,
        "entity_class_counts": counts,
        "train": splits["train"],
        "val":   splits["val"],
        "test":  splits["test"],
    }
    (task_dir / "splits.json").write_text(json.dumps(splits_json, indent=2))

    print(f"\nCohort:  {len(cohort)} entities  (pos={n_pos_entities}, neg={n_neg_entities})")
    print(f"Patients: pos={split_stats['n_patients_pos']}, "
          f"neg={split_stats['n_patients_neg']}")
    print(f"Split entity counts:")
    for k in ("train", "val", "test"):
        c = counts[k]
        rate = c["n_pos"] / max(1, c["n"])
        print(f"  {k:5s}: n={c['n']:5d}  pos={c['n_pos']:4d}  "
              f"neg={c['n_neg']:5d}  pos_rate={rate:.3%}")
    print(f"Split patient counts: {split_stats}")
    print(f"\nwrote {task_dir/'cohort.json'}")
    print(f"wrote {task_dir/'splits.json'}")
    print(f"elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
