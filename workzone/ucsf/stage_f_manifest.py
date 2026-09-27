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
  - PLETH40.npy + II120.npy: float16, C-contiguous, shape[0] == n_seg, not all-NaN
    (an all-NaN channel = source lacked that title; such entities are excluded)
  - ehr_{baseline,recent,events,future}.npy: dtype matches EHR_EVENT_DTYPE,
    sorted, seg_idx sentinels correct (ehr_events has real indices).
    Optional: a store without EHR linkage (ucsf_all) has none of the four; a
    store with SOME of them is an error.
  - vitals_hf.npy (optional dense sidecar, see stage_c_vitals_hf.py): float32,
    C-contiguous, shape == (n_seg, slots_per_seg, n_var) declared in
    meta.vitals_hf; vitals_hf_abp_src.npy uint8 (n_seg, slots_per_seg);
    nbp_events.npy EHR_EVENT_DTYPE, sorted, seg_idx in [0, n_seg)

--dataset selects the config section (ucsf = CA cohort, ucsf_all = every raw
wave cycle).

Split:
  - Group by patient_id_ge (one wave cycle == one entity; patients can have
    multiple wave cycles that MUST stay in the same split)
  - Shuffle patients with seed=42, deterministic
  - 70/15/15 unstratified (stratification left to per-task scripts)
  - Patients of the cardiac-arrest study cohort (ValidWaveTime CSV, cases AND
    study controls; --exclude-from-train-csv) are NEVER placed in train: they
    are split val/test by --holdout-val-frac (default 0.5). Manifest entries
    carry in_ca_cohort and has_ca.

--splits-only: skip file validation, load the existing manifest.json, refresh the
cohort flags and rebuild the splits (seconds instead of hours).
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


def _unit_of(bed_subdir) -> str | None:
    """ICU unit from a bed sub-directory name like '13ICU_4-DE1234…' -> '13ICU'
    (13ICU, 9ICU, 10ICC, 8NICU/11NICU = UCSF Neuro ICU; all adult units)."""
    import re
    m = re.match(r"^\s*(\d*[A-Za-z]+)", str(bed_subdir or ""))
    return m.group(1).upper() if m else None


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
        "wynton_folder":   meta.get("wynton_folder"),
        "bed_subdir":      meta.get("bed_subdir"),
        # ICU unit = prefix of the bed sub-directory (13ICU, 9ICU, 10ICC, 8NICU/11NICU = UCSF Neuro ICU; all adult units)
        "unit":            _unit_of(meta.get("bed_subdir")),
        "episode_duration_sec": meta.get("episode_duration_sec"),
        "n_seg":           n_seg,
        "source_dataset":  meta.get("source_dataset", "ucsf"),
        "has_ca":          meta.get("has_ca"),
        # time base of time_ms (datasets/ucsf/ALIGNMENT.md): 'utc_continuous' (Stage B v2) or legacy 'wall_clock'
        "time_base":       meta.get("time_base", "wall_clock"),
        "dst_switch_in_cycle": meta.get("dst_switch_in_cycle"),
        "grid_utc_offset_min": meta.get("grid_utc_offset_min"),
        "stage_b_version": meta.get("stage_b_version", 1),
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
        # A paired store needs signal in every channel: an all-NaN channel (source files without
        # that title, e.g. no SPO2) makes the entity invalid. Sampled every k-th segment.
        if arr.ndim == 2 and arr.shape[0] > 0:
            step = max(1, arr.shape[0] // 200)
            nan_frac = float(np.isnan(np.asarray(arr[::step], dtype=np.float32)).mean())
            entry[f"{ch}_nan_frac"] = round(nan_frac, 4)
            if nan_frac >= 0.999:
                errors.append(f"{ch}: all NaN")

    # EHR 4 partitions (all present, or none for a waveform+numerics-only store)
    parts = (("baseline", FNAME_BASELINE), ("recent", FNAME_RECENT),
             ("events", FNAME_EVENTS), ("future", FNAME_FUTURE))
    present = [(entity_dir / f).exists() for _, f in parts]
    entry["has_ehr"] = all(present)
    if any(present):
        for (kind, fname), ok in zip(parts, present):
            if not ok:
                errors.append(f"missing {fname}")
                continue
            arr = np.load(entity_dir / fname)
            if arr.dtype != EHR_EVENT_DTYPE:
                errors.append(f"{fname}: dtype mismatch")
                continue
            errors.extend(validate_partition(arr, kind=kind, n_seg=n_seg))

    # Optional dense vitals sidecar (Stage C hf)
    vh = meta.get("vitals_hf")
    p = entity_dir / "vitals_hf.npy"
    entry["has_vitals_hf"] = bool(vh) and p.exists()
    if vh and not p.exists():
        errors.append("meta.vitals_hf present but vitals_hf.npy missing")
    if entry["has_vitals_hf"]:
        arr = np.load(p, mmap_mode="r")
        slots = int(vh.get("slots_per_seg", 15)); n_var = len(vh.get("var_ids", []))
        if arr.dtype != np.float32:
            errors.append(f"vitals_hf: dtype {arr.dtype}, expected float32")
        if not arr.flags["C_CONTIGUOUS"]:
            errors.append("vitals_hf: not C-contiguous")
        if arr.shape != (n_seg, slots, n_var):
            errors.append(f"vitals_hf: shape {arr.shape} != ({n_seg}, {slots}, {n_var})")
        entry["vitals_hf_var_ids"] = vh.get("var_ids")
        entry["vitals_hf_valid_frac"] = vh.get("valid_frac_per_var")
        src = entity_dir / "vitals_hf_abp_src.npy"
        if src.exists():
            a = np.load(src, mmap_mode="r")
            if a.dtype != np.uint8 or a.shape != (n_seg, slots):
                errors.append(f"vitals_hf_abp_src: dtype {a.dtype} shape {a.shape}")
        else:
            errors.append("missing vitals_hf_abp_src.npy")
        ev_p = entity_dir / "nbp_events.npy"
        if ev_p.exists():
            ev = np.load(ev_p)
            if ev.dtype != EHR_EVENT_DTYPE:
                errors.append("nbp_events: dtype mismatch")
            else:
                if ev.size and not np.all(np.diff(ev["time_ms"]) >= 0):
                    errors.append("nbp_events: not sorted")
                if ev.size and (ev["seg_idx"].min() < 0 or ev["seg_idx"].max() >= n_seg):
                    errors.append("nbp_events: seg_idx out of range")
                entry["n_nbp_events"] = int(ev.size)
        else:
            errors.append("missing nbp_events.npy")

    return entry, errors


def load_holdout_cohort(csv_path: Path) -> tuple[set[str], dict[str, int]]:
    """Cardiac-arrest study cohort from the ValidWaveTime CSV.

    Returns (patient ids never allowed in train, {entity_id: has_ca}) where has_ca = 1 when
    any row of that wave cycle has an EventTime other than -1 (cases), 0 for study controls.
    """
    df = pl.read_csv(csv_path, infer_schema_length=0, ignore_errors=True)
    df = df.rename({c: c.strip() for c in df.columns})
    need = {"Patient_ID_GE", "WaveCycleUID", "EventTime"}
    missing = need - set(df.columns)
    if missing:
        raise RuntimeError(f"cohort CSV {csv_path} lacks columns {missing}")
    pid = df["Patient_ID_GE"].cast(pl.Utf8).str.strip_chars().str.replace(r"^DE", "")
    uid = df["WaveCycleUID"].cast(pl.Utf8).str.strip_chars()
    ev = df["EventTime"].cast(pl.Utf8).fill_null("").str.strip_chars()
    has_ca: dict[str, int] = {}
    for p, u, e in zip(pid.to_list(), uid.to_list(), ev.to_list()):
        if not p:
            continue
        pos = int(e not in ("", "-1", "-1.0", "nan", "NaN"))
        key = f"{p}_{u}"
        has_ca[key] = max(has_ca.get(key, 0), pos)
    return set(pid.to_list()) - {""}, has_ca


def build_splits(valid_entries: list[dict], ratios: tuple[float, float, float],
                 seed: int, holdout_patients: set[str] | None = None,
                 holdout_val_frac: float = 0.5) -> tuple[dict[str, list[str]], dict]:
    assert abs(sum(ratios) - 1.0) < 1e-6
    train_r, val_r, _ = ratios
    holdout_patients = holdout_patients or set()

    by_pat: dict[str, list[str]] = {}
    for e in valid_entries:
        pid = str(e["patient_id_ge"])
        by_pat.setdefault(pid, []).append(e["entity_id"])

    rng = np.random.default_rng(seed)
    hold = sorted(p for p in by_pat if p in holdout_patients)
    rest = sorted(p for p in by_pat if p not in holdout_patients)
    rng.shuffle(rest); rng.shuffle(hold)

    n = len(rest)
    n_train = int(round(n * train_r))
    n_val = int(round(n * val_r))
    train_p = rest[:n_train]
    val_p = rest[n_train:n_train + n_val]
    test_p = rest[n_train + n_val:]
    # cohort patients: never train; val/test only
    n_hold_val = int(round(len(hold) * holdout_val_frac))
    hold_val_p, hold_test_p = hold[:n_hold_val], hold[n_hold_val:]
    val_p = val_p + hold_val_p
    test_p = test_p + hold_test_p

    def entities(pids_) -> list[str]:
        out: list[str] = []
        for p in pids_:
            out.extend(by_pat[p])
        return sorted(out)

    splits = {"train": entities(train_p), "val": entities(val_p),
              "test":  entities(test_p)}
    stats = {
        "n_unique_patients": len(by_pat),
        "n_train_patients": len(train_p),
        "n_val_patients":   len(val_p),
        "n_test_patients":  len(test_p),
        "n_multi_cycle_patients": sum(1 for v in by_pat.values() if len(v) > 1),
        "holdout_from_train": {
            "n_patients": len(hold),
            "n_entities": sum(len(by_pat[p]) for p in hold),
            "val_patients": len(hold_val_p), "test_patients": len(hold_test_p),
            "val_entities": sum(len(by_pat[p]) for p in hold_val_p),
            "test_entities": sum(len(by_pat[p]) for p in hold_test_p),
            "holdout_val_frac": holdout_val_frac,
        },
    }
    return splits, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--dataset", default="ucsf", help="config section: ucsf | ucsf_all")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--ratios", default=",".join(str(x) for x in DEFAULT_RATIOS),
                    help="train,val,test fractions")
    ap.add_argument("--exclude-from-train-csv", default=None,
                    help="ValidWaveTime CSV of the cardiac-arrest study cohort; its patients never enter train "
                         "(default: config key ca_eventtime_csv; pass '' to disable)")
    ap.add_argument("--holdout-val-frac", type=float, default=0.5,
                    help="share of the held-out cohort patients that go to val (rest to test)")
    ap.add_argument("--codeblue-patients-json", default=None,
                    help="JSON list or dict keyed by Patient_ID_GE of patients with a Code Blue CPA event "
                         "(workzone/ucsf/explore/ca_events_grid.py); added to the train holdout and flagged codeblue_cpa/has_ca=1")
    ap.add_argument("--keep-splits", action="store_true",
                    help="re-validate every entity and rewrite manifest.json, but reuse the existing split membership "
                         "(pretrain_splits.json); fails if the valid entity set changed")
    ap.add_argument("--splits-only", action="store_true",
                    help="skip validation; rebuild cohort flags + splits from the existing manifest.json")
    args = ap.parse_args()

    ratios = tuple(float(x) for x in args.ratios.split(","))
    assert len(ratios) == 3

    cfg = yaml.safe_load(Path(args.config).read_text())[args.dataset]
    output_dir = Path(cfg["output_dir"])
    intermediate_dir = Path(cfg["intermediate_dir"])
    holdout_csv = cfg.get("ca_eventtime_csv") if args.exclude_from_train_csv is None else args.exclude_from_train_csv
    holdout_patients: set[str] = set(); has_ca_map: dict[str, int] = {}
    if holdout_csv:
        holdout_patients, has_ca_map = load_holdout_cohort(Path(holdout_csv))
        log.info(f"holdout cohort: {len(holdout_patients)} patients, {len(has_ca_map)} wave cycles "
                 f"({sum(has_ca_map.values())} CA-positive) from {holdout_csv}")

    csv_cohort = set(holdout_patients)
    codeblue_pids: set[str] = set()
    if args.codeblue_patients_json:
        obj = json.loads(Path(args.codeblue_patients_json).read_text())
        codeblue_pids = {str(k).strip() for k in (obj.keys() if isinstance(obj, dict) else obj)}
        log.info(f"Code Blue CPA patients: {len(codeblue_pids)} ({len(codeblue_pids - holdout_patients)} outside the "
                 f"ValidWaveTime cohort) added to the train holdout")
        holdout_patients = holdout_patients | codeblue_pids
    log.info(f"output_dir = {output_dir}")
    log.info(f"seed={args.seed} ratios={ratios} splits_only={args.splits_only}")

    t0 = time.time()
    manifest_path = output_dir / "manifest.json"
    summary_path = intermediate_dir / "stage_f_manifest_summary.json"
    if args.splits_only:
        valid_entries = json.loads(manifest_path.read_text())
        prev = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        entity_dirs = []
        errors = {}
        n_pass = len(valid_entries)
        log.info(f"splits-only: {len(valid_entries)} valid entries from {manifest_path}")
    else:
        entity_dirs = sorted(
            p for p in output_dir.iterdir()
            if p.is_dir() and (p / "meta.json").exists()
        )
        log.info(f"entity dirs: {len(entity_dirs)}")

        manifest: list[dict] = []
        errors = {}
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
        prev = {}

    for e in valid_entries:
        pid = str(e.get("patient_id_ge"))
        e["in_ca_cohort"] = pid in csv_cohort            # ValidWaveTime study cohort (cases + controls)
        e["has_ca"] = has_ca_map.get(e["entity_id"], e.get("has_ca"))
        if codeblue_pids:
            e["codeblue_cpa"] = pid in codeblue_pids     # Code Blue TypeCode CPA (patient level)
            if e["codeblue_cpa"]:
                e["has_ca"] = 1
        e["held_out_of_train"] = pid in holdout_patients
        if not e.get("unit"):
            try:
                bs = json.loads((output_dir / e["entity_id"] / "meta.json").read_text()).get("bed_subdir") or ""
            except Exception:
                bs = ""
            e["bed_subdir"] = bs or e.get("bed_subdir"); e["unit"] = _unit_of(bs)
    manifest_path.write_text(json.dumps(valid_entries, indent=2, default=str))
    log.info(f"wrote {manifest_path} ({len(valid_entries)} entries)")

    if args.keep_splits:
        prev_splits = json.loads((output_dir / "pretrain_splits.json").read_text())
        splits = {k: list(prev_splits[k]) for k in ("train", "val", "test")}
        prev_ids = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
        cur_ids = {e["entity_id"] for e in valid_entries}
        if cur_ids - prev_ids:
            raise SystemExit(f"--keep-splits: {len(cur_ids - prev_ids)} valid entities are not in the existing splits; "
                             f"rerun without --keep-splits")
        if prev_ids - cur_ids:   # entities that no longer validate (e.g. all-NaN rejects): drop them, keep everyone else's split
            gone = prev_ids - cur_ids
            splits = {k: [e for e in v if e not in gone] for k, v in splits.items()}
            log.warning(f"keep-splits: {len(gone)} entities dropped from the existing splits (no longer valid); "
                        f"membership of the remaining {len(cur_ids)} unchanged")
        pid_of = {e["entity_id"]: str(e["patient_id_ge"]) for e in valid_entries}
        movers = sorted({pid_of[e] for e in splits["train"] if pid_of[e] in holdout_patients})
        if movers:   # holdout grew (e.g. Code Blue patients outside the CSV cohort): move them out of train, keep everyone else
            rs = np.random.RandomState(args.seed); rs.shuffle(movers)
            n_v = int(round(len(movers) * args.holdout_val_frac)); to_val = set(movers[:n_v]); to_test = set(movers[n_v:])
            moved = [e for e in splits["train"] if pid_of[e] in holdout_patients]
            splits["train"] = [e for e in splits["train"] if pid_of[e] not in holdout_patients]
            splits["val"] = splits["val"] + [e for e in moved if pid_of[e] in to_val]
            splits["test"] = splits["test"] + [e for e in moved if pid_of[e] in to_test]
            log.warning(f"keep-splits: {len(movers)} holdout patients ({len(moved)} entities) moved out of train -> "
                        f"val {len(to_val)} / test {len(to_test)} patients; all other memberships unchanged")
        split_stats = {k: v for k, v in prev_splits.items()
                       if k not in ("train", "val", "test", "seed", "ratios", "group_by", "holdout_from_train_source",
                                    "holdout_extra_source", "n_train", "n_val", "n_test")}
        log.info("keep-splits: membership unchanged; split lists reused, per-entity fields (n_seg, flags) refreshed")
    else:
        splits, split_stats = build_splits(valid_entries, ratios, args.seed,
                                           holdout_patients, args.holdout_val_frac)
    # Hard guarantee: no cohort patient in train
    train_ca = [e for e in splits["train"] if str({x["entity_id"]: x for x in valid_entries}[e]["patient_id_ge"]) in holdout_patients]
    assert not train_ca, f"{len(train_ca)} cohort entities leaked into train"

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
        "holdout_from_train_source": holdout_csv or None,
        "holdout_extra_source": args.codeblue_patients_json,
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
    n_failed_all_nan = sum(1 for errs in errors.values() if errs and all(e.endswith("all NaN") for e in errs))
    if args.splits_only:   # keep the validation bookkeeping of the full run
        n_failed_all_nan = prev.get("n_failed_all_nan_only", 0)
        entity_dirs = [None] * prev.get("n_entity_dirs", len(valid_entries))
        errors = {f"_prev_{i}": [] for i in range(prev.get("n_failed", 0))}
    summary = {
        "stage": "f_manifest",
        "dataset": args.dataset,
        "mode": "splits_only" if args.splits_only else "full",
        "n_failed_all_nan_only": n_failed_all_nan,
        "n_ca_cohort_entities": sum(1 for e in valid_entries if e.get("in_ca_cohort")),
        "n_ca_positive_entities": sum(1 for e in valid_entries if e.get("has_ca") == 1),
        "n_with_ehr": sum(1 for e in valid_entries if e.get("has_ehr")),
        "n_with_vitals_hf": sum(1 for e in valid_entries if e.get("has_vitals_hf")),
        "total_nbp_events": sum(e.get("n_nbp_events", 0) for e in valid_entries),
        "total_wave_hours": round(sum(e.get("n_seg", 0) for e in valid_entries) * 30 / 3600, 1),
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
        "failed_first_20":  dict(list(errors.items())[:20]) if not args.splits_only else prev.get("failed_first_20", {}),
    }
    (intermediate_dir / "stage_f_manifest_summary.json").write_text(
        json.dumps(summary, indent=2))
    log.info(f"wrote {intermediate_dir/'stage_f_manifest_summary.json'}")
    log.info(f"done in {time.time()-t0:.1f}s")
    print(json.dumps({k: v for k, v in summary.items() if k != "failed_first_20"},
                     indent=2))


if __name__ == "__main__":
    main()
