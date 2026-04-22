#!/usr/bin/env python3
"""
Build estimation task lists (cohort.json + splits.json) gated by per-entity
availability of target var_ids in `ehr_events.npy`.

Two modes:

  --coverage-only
      Scan all entities, emit one `coverage.json` per dataset reporting per-var
      eligibility / median events / event totals. No cohort or splits emitted.
      Use this first, then use the table to pick target sets.

  --spec PATH/TO/task_spec.yaml
      Use the spec to define `task_name`, `target_var_ids`,
      `min_events_per_target`, `eligibility` (`any` | `all` | `per_target`),
      `inherit_splits_from` (`pretrain` | `downstream`). Emits
      `<root>/tasks/<task_name>/{cohort.json, splits.json}`.

The script is dataset-agnostic: it reads `manifest.json`, `pretrain_splits.json`,
`downstream_splits.json`, and per-entity `ehr_events.npy` / `ehr_recent.npy` /
`ehr_baseline.npy`. Output schema mirrors the existing `tasks/sepsis/` pattern.

Usage examples:
    # Coverage scan (no spec needed)
    python build_estimation_task.py --root /opt/localdata100tb/physio_data/emory \
        --registry /labs/hulab/mxwang/Physio_Data/indices/var_registry.json \
        --coverage-only

    # Build a task from a spec
    python build_estimation_task.py --root /opt/localdata100tb/physio_data/emory \
        --registry /labs/hulab/mxwang/Physio_Data/indices/var_registry.json \
        --spec workzone/common/task_specs/lab_est_full.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_WORKERS = 16


# ---------- registry helpers ----------

def load_registry(path: str) -> dict[int, dict]:
    raw = json.loads(Path(path).read_text())
    entries = raw["variables"] if isinstance(raw, dict) else raw
    return {int(v["id"]): v for v in entries}


def resolve_targets(targets: list, registry: dict[int, dict]) -> list[int]:
    """Accept ints or names; return list of ints."""
    name_to_id = {v["name"].upper(): vid for vid, v in registry.items()}
    out: list[int] = []
    for t in targets:
        if isinstance(t, int):
            if t not in registry:
                raise ValueError(f"unknown var_id {t}")
            out.append(t)
        elif isinstance(t, str):
            key = t.upper()
            if key not in name_to_id:
                raise ValueError(f"unknown var name {t!r}")
            out.append(name_to_id[key])
        else:
            raise ValueError(f"target must be int or str, got {type(t)}")
    return out


# ---------- per-entity worker ----------

def _count_one_entity(args):
    """Count events per var_id in each partition for one entity.

    Returns: (entity_id, {partition: {var_id: count}}, n_seg, wave_start, wave_end)
    """
    entity_id, root_str = args
    edir = Path(root_str) / entity_id
    out = {"entity_id": entity_id, "ok": False}
    try:
        time_path = edir / "time_ms.npy"
        if not time_path.exists():
            out["reason"] = "no_time"
            return out
        t = np.load(time_path, mmap_mode="r")
        out["n_seg"] = int(len(t))
        out["wave_start_ms"] = int(t[0]) if len(t) else None
        out["wave_end_ms"]   = int(t[-1]) if len(t) else None

        per_partition = {}
        for fname, key in (("ehr_events.npy",   "events"),
                           ("ehr_recent.npy",   "recent"),
                           ("ehr_baseline.npy", "baseline")):
            p = edir / fname
            if not p.exists():
                per_partition[key] = {}
                continue
            arr = np.load(p, mmap_mode="r")
            if arr.size == 0:
                per_partition[key] = {}
                continue
            vids, counts = np.unique(arr["var_id"], return_counts=True)
            per_partition[key] = {int(v): int(c) for v, c in zip(vids, counts)}
        out["per_partition"] = per_partition
        out["ok"] = True
    except Exception as e:
        out["reason"] = f"{type(e).__name__}: {e}"
    return out


def scan_entities(root: Path, entity_ids: list[str], workers: int) -> list[dict]:
    args = [(eid, str(root)) for eid in entity_ids]
    results: list[dict] = []
    t0 = time.time()
    if workers <= 1:
        for i, a in enumerate(args, 1):
            results.append(_count_one_entity(a))
            if i % 1000 == 0 or i == len(args):
                log.info(f"  scanned {i}/{len(args)}  elapsed={time.time()-t0:.1f}s")
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers) as pool:
            for i, r in enumerate(pool.imap_unordered(_count_one_entity, args, chunksize=8), 1):
                results.append(r)
                if i % 1000 == 0 or i == len(args):
                    log.info(f"  scanned {i}/{len(args)}  elapsed={time.time()-t0:.1f}s")
    return results


# ---------- coverage report ----------

def build_coverage(scans: list[dict], registry: dict[int, dict],
                   thresholds=(1, 3, 5)) -> dict:
    n_total = sum(1 for s in scans if s["ok"])
    per_var_events:   dict[int, list[int]] = defaultdict(list)
    per_var_recent:   dict[int, list[int]] = defaultdict(list)
    per_var_baseline: dict[int, list[int]] = defaultdict(list)

    for s in scans:
        if not s["ok"]:
            continue
        ev = s["per_partition"].get("events", {})
        re_ = s["per_partition"].get("recent", {})
        ba = s["per_partition"].get("baseline", {})
        for v, c in ev.items():
            per_var_events[v].append(c)
        for v, c in re_.items():
            per_var_recent[v].append(c)
        for v, c in ba.items():
            per_var_baseline[v].append(c)

    out_per_var: dict[str, dict] = {}
    all_vids = sorted(set(per_var_events) | set(per_var_recent) | set(per_var_baseline))
    for vid in all_vids:
        ev_counts = per_var_events.get(vid, [])
        info = {
            "var_id": int(vid),
            "name": registry.get(vid, {}).get("name", "<unknown>"),
            "category": registry.get(vid, {}).get("category", "<unknown>"),
            "in_wave": {
                "n_eligible_entities_at_min": {
                    str(t): int(sum(1 for c in ev_counts if c >= t))
                    for t in thresholds
                },
                "frac_eligible_at_min": {
                    str(t): round(sum(1 for c in ev_counts if c >= t) / max(1, n_total), 4)
                    for t in thresholds
                },
                "median_events_per_eligible": int(np.median(ev_counts)) if ev_counts else 0,
                "p95_events_per_eligible":    int(np.percentile(ev_counts, 95)) if ev_counts else 0,
                "total_events": int(sum(ev_counts)),
            },
            "recent": {
                "n_entities_with_any": len(per_var_recent.get(vid, [])),
                "median_events":       int(np.median(per_var_recent[vid])) if per_var_recent.get(vid) else 0,
            },
            "baseline": {
                "n_entities_with_any": len(per_var_baseline.get(vid, [])),
                "median_events":       int(np.median(per_var_baseline[vid])) if per_var_baseline.get(vid) else 0,
            },
        }
        out_per_var[str(vid)] = info

    out_per_var_sorted = dict(
        sorted(out_per_var.items(),
               key=lambda kv: -kv[1]["in_wave"]["n_eligible_entities_at_min"]["1"])
    )
    return {
        "n_entities_scanned": n_total,
        "thresholds_min_events": list(thresholds),
        "per_var": out_per_var_sorted,
    }


# ---------- task builder ----------

def build_task(scans: list[dict], spec: dict, registry: dict[int, dict],
               root: Path) -> dict:
    target_ids = resolve_targets(spec["target_var_ids"], registry)
    min_n = int(spec.get("min_events_per_target", 1))
    mode = spec.get("eligibility", "any")
    assert mode in ("any", "all", "per_target"), f"bad eligibility {mode}"

    per_entity: dict[str, dict] = {}
    for s in scans:
        if not s["ok"]:
            continue
        ev = s["per_partition"].get("events", {})
        re_ = s["per_partition"].get("recent", {})
        ba = s["per_partition"].get("baseline", {})
        per_entity[s["entity_id"]] = {
            "entity_id": s["entity_id"],
            "n_seg": s.get("n_seg"),
            "wave_start_ms": s.get("wave_start_ms"),
            "wave_end_ms":   s.get("wave_end_ms"),
            "per_var_count":          {str(v): int(ev.get(v, 0)) for v in target_ids},
            "per_var_count_recent":   {str(v): int(re_.get(v, 0)) for v in target_ids},
            "per_var_count_baseline": {str(v): int(ba.get(v, 0)) for v in target_ids},
        }

    def _is_elig(per_var, mode_):
        if mode_ == "any":
            return any(per_var[str(v)] >= min_n for v in target_ids)
        elif mode_ == "all":
            return all(per_var[str(v)] >= min_n for v in target_ids)
        return None

    if mode != "per_target":
        eligible = {eid: row for eid, row in per_entity.items()
                    if _is_elig(row["per_var_count"], mode)}
        cohorts = {spec["task_name"]: (target_ids, eligible)}
    else:
        cohorts = {}
        for vid in target_ids:
            sub_name = registry.get(vid, {}).get("name", f"var{vid}").lower().replace(" ", "_")
            sub_eligible = {eid: row for eid, row in per_entity.items()
                            if row["per_var_count"][str(vid)] >= min_n}
            cohorts[f"{spec['task_name']}/{sub_name}__var{vid}"] = ([vid], sub_eligible)

    return {"per_entity": per_entity, "target_ids": target_ids,
            "mode": mode, "min_n": min_n, "cohorts": cohorts}


def write_task(task_name: str, target_ids: list[int], eligible: dict[str, dict],
               registry: dict[int, dict], spec: dict, root: Path) -> dict:
    """Write cohort.json + splits.json under <root>/tasks/<task_name>/."""
    task_dir = root / "tasks" / task_name
    task_dir.mkdir(parents=True, exist_ok=True)

    inherit_from = spec.get("inherit_splits_from", "pretrain")
    if inherit_from == "pretrain":
        sp = json.loads((root / "pretrain_splits.json").read_text())
        train, val, test = sp["train"], sp["val"], sp["test"]
        seed = sp.get("seed")
        ratios = sp.get("ratios")
    elif inherit_from == "downstream":
        ds = json.loads((root / "downstream_splits.json").read_text())
        train = [Path(r[0]).name for r in ds["train_control_list"]]
        val   = [Path(r[0]).name for r in ds["val_control_list"]]
        test  = [Path(r[0]).name for r in ds["test_control_list"]]
        seed = ratios = None
    else:
        raise ValueError(f"bad inherit_splits_from {inherit_from!r}")

    elig_set = set(eligible.keys())
    f_train = [e for e in train if e in elig_set]
    f_val   = [e for e in val   if e in elig_set]
    f_test  = [e for e in test  if e in elig_set]

    min_n = int(spec.get("min_events_per_target", 1))
    per_var_split: dict[str, dict[str, int]] = {}
    for vid in target_ids:
        per_var_split[str(vid)] = {
            "train": int(sum(1 for e in f_train if eligible[e]["per_var_count"][str(vid)] >= min_n)),
            "val":   int(sum(1 for e in f_val   if eligible[e]["per_var_count"][str(vid)] >= min_n)),
            "test":  int(sum(1 for e in f_test  if eligible[e]["per_var_count"][str(vid)] >= min_n)),
        }

    cohort_json = {
        "task": task_name,
        "task_kind": "estimation",
        "target_var_ids": target_ids,
        "target_var_names": {str(v): registry.get(v, {}).get("name") for v in target_ids},
        "min_events_per_target": min_n,
        "eligibility": spec.get("eligibility"),
        "n_entities": len(eligible),
        "per_var_eligible_in_cohort": {
            str(v): int(sum(1 for e in eligible.values()
                            if e["per_var_count"][str(v)] >= min_n))
            for v in target_ids
        },
        "per_var_total_events_in_cohort": {
            str(v): int(sum(e["per_var_count"][str(v)] for e in eligible.values()))
            for v in target_ids
        },
        "fields": ["entity_id", "n_seg", "wave_start_ms", "wave_end_ms",
                   "per_var_count", "per_var_count_recent", "per_var_count_baseline"],
        "entities": list(eligible.values()),
    }
    (task_dir / "cohort.json").write_text(json.dumps(cohort_json, indent=2, default=str))

    splits_json = {
        "task": task_name,
        "source": f"filtered from {inherit_from}_splits.json (eligibility-gated)",
        "seed": seed,
        "ratios": ratios,
        "n_train": len(f_train),
        "n_val":   len(f_val),
        "n_test":  len(f_test),
        "per_var_coverage_per_split": per_var_split,
        "train": sorted(f_train),
        "val":   sorted(f_val),
        "test":  sorted(f_test),
    }
    (task_dir / "splits.json").write_text(json.dumps(splits_json, indent=2))

    return {"task_dir": str(task_dir),
            "n_entities": len(eligible),
            "n_train": len(f_train), "n_val": len(f_val), "n_test": len(f_test)}


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="dataset root, e.g. /opt/localdata100tb/physio_data/emory")
    ap.add_argument("--registry", required=True,
                    help="path to indices/var_registry.json")
    ap.add_argument("--coverage-only", action="store_true",
                    help="emit <root>/tasks/coverage.json only; no cohort/splits")
    ap.add_argument("--spec", default=None,
                    help="YAML task spec; required unless --coverage-only")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--limit", type=int, default=0,
                    help="debug: scan first N entities only")
    ap.add_argument("--out-name", default=None,
                    help="override task_name from spec (used as subdir under tasks/)")
    args = ap.parse_args()

    root = Path(args.root)
    if not (root / "manifest.json").exists():
        log.error(f"manifest.json missing under {root}")
        sys.exit(1)
    if not args.coverage_only and not args.spec:
        log.error("must pass --coverage-only or --spec")
        sys.exit(1)

    registry = load_registry(args.registry)
    log.info(f"registry: {len(registry)} vars")

    manifest = json.loads((root / "manifest.json").read_text())
    # Accept both `entity_id` (Emory/UCSF convention) and `dir` (MIMIC-III).
    entity_ids = [m.get("entity_id") or m.get("dir") for m in manifest]
    entity_ids = [e for e in entity_ids if e]
    if args.limit:
        entity_ids = entity_ids[:args.limit]
    log.info(f"manifest entities: {len(entity_ids)}  workers={args.workers}")

    t0 = time.time()
    scans = scan_entities(root, entity_ids, args.workers)
    n_ok = sum(1 for s in scans if s["ok"])
    log.info(f"scan complete: {n_ok}/{len(scans)} ok  elapsed={time.time()-t0:.1f}s")

    if args.coverage_only:
        cov = build_coverage(scans, registry)
        out = root / "tasks" / "coverage.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(cov, indent=2))
        log.info(f"wrote {out}")
        log.info("Top 30 vars by in-wave eligibility (min=1 event):")
        log.info(f"  {'var_id':>6}  {'name':<22}  {'cat':<7}  "
                 f"{'elig@1':>7}  {'elig@3':>7}  {'elig@5':>7}  {'p50_n':>6}")
        for vid, info in list(cov["per_var"].items())[:30]:
            iw = info["in_wave"]
            log.info(f"  {vid:>6}  {info['name'][:22]:<22}  {info['category']:<7}  "
                     f"{iw['n_eligible_entities_at_min']['1']:>7}  "
                     f"{iw['n_eligible_entities_at_min']['3']:>7}  "
                     f"{iw['n_eligible_entities_at_min']['5']:>7}  "
                     f"{iw['median_events_per_eligible']:>6}")
        return

    spec = yaml.safe_load(Path(args.spec).read_text())
    if args.out_name:
        spec["task_name"] = args.out_name
    log.info(f"spec: task_name={spec['task_name']!r}  "
             f"targets={spec['target_var_ids']}  "
             f"min_events={spec.get('min_events_per_target', 1)}  "
             f"mode={spec.get('eligibility', 'any')}")

    bundle = build_task(scans, spec, registry, root)
    summary = []
    for sub_name, (tids, eligible) in bundle["cohorts"].items():
        sub_spec = dict(spec)
        sub_spec["target_var_ids"] = tids
        info = write_task(sub_name, tids, eligible, registry, sub_spec, root)
        summary.append({"task": sub_name, **info})
        log.info(f"wrote task {sub_name!r}: n={info['n_entities']}  "
                 f"train/val/test={info['n_train']}/{info['n_val']}/{info['n_test']}")

    out = root / "tasks" / spec["task_name"] / "build_summary.json"
    out.write_text(json.dumps({"spec": spec, "summary": summary}, indent=2))
    log.info(f"wrote {out}")


if __name__ == "__main__":
    main()
