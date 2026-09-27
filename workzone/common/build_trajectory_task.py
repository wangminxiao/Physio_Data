#!/usr/bin/env python3
"""
Two-stage trajectory task builder (cardio_traj / gas_traj / hgb_traj / kidney_traj).

Re-implementation of the original `build_task_cohort.py` (bedanalysis ~/workzone/physio_data/tasks, lost in the 2026-08
server migration). The rule was recovered from the spec recorded in every `tasks/<task>/build_summary.json` and verified
to reproduce all four MIMIC-III cohorts exactly from their `cohort.json` counts (2026-09-27):

  stage A  entity is in the cohort when ANY target has >= stage_a_min_events_per_target in-wave events      (cohort.json)
  stage B  entity is in the split lists when >= need targets each have >= stage_b_per_target_min events,
           need = stage_b_min_targets_passing (== ceil(stage_b_target_frac * n_targets) for every recorded spec)

Splitting (patient-disjoint, seed/ratios from the spec):
  --keep-splits [OLD]   default when tasks/<task>/splits.json exists: entities still passing stage B keep their old split,
                        entities that dropped out are removed, new entities follow their subject if it is already placed,
                        otherwise fill the split that is furthest below its ratio target (seeded order). Used after the
                        2026-09 clock fix so the trajectory splits stay comparable with earlier experiments.
  --resplit             fresh multilabel stratified split (Sechidis-style iterative stratification at subject level over
                        target presence, duration tertile, sex, age bin, admission type when demographics.csv has them).

Writes tasks/<task>/{cohort.json, splits.json, build_summary.json} (same schema as the original) and keeps the previous
splits.json under workzone/outputs/<dataset>/traj_rebuild/<task>_splits_before.json.

  python workzone/common/build_trajectory_task.py --root STORE --registry indices/var_registry.json \
      --from-summary STORE/tasks/kidney_traj/build_summary.json --keep-splits [--workers 8] [--dry-run]
  python workzone/common/build_trajectory_task.py --root STORE --registry ... --spec spec.yaml --resplit
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from build_estimation_task import load_registry, resolve_targets, scan_entities  # noqa: E402

SPLITS = ("train", "val", "test")


# ---------- spec ----------

def load_spec(args) -> dict:
    if args.from_summary:
        spec = json.loads(Path(args.from_summary).read_text())["spec"]
    else:
        p = Path(args.spec)
        if p.suffix in (".yaml", ".yml"):
            import yaml
            spec = yaml.safe_load(p.read_text())
        else:
            spec = json.loads(p.read_text())
    if args.task_name:
        spec["task_name"] = args.task_name
    spec.setdefault("stage_a_min_events_per_target", 1)
    spec.setdefault("stage_a_eligibility", "any")
    spec.setdefault("stage_b_per_target_min", 2)
    spec.setdefault("ratios", [0.7, 0.15, 0.15])
    spec.setdefault("seed", 42)
    n_t = len(spec["target_var_ids"])
    frac = spec.get("stage_b_target_frac")
    if spec.get("stage_b_min_targets_passing") is None:
        spec["stage_b_min_targets_passing"] = max(1, math.ceil(float(frac) * n_t)) if frac is not None else n_t
    return spec


# ---------- stages ----------

def stage_a_rows(scans: list[dict], target_ids: list[int], min_n: int, mode: str) -> tuple[dict[str, dict], int]:
    rows: dict[str, dict] = {}
    n_no_target = 0
    for s in scans:
        if not s.get("ok"):
            continue
        ev = s["per_partition"].get("events", {})
        re_ = s["per_partition"].get("recent", {})
        ba = s["per_partition"].get("baseline", {})
        cnt = {str(v): int(ev.get(v, 0)) for v in target_ids}
        if sum(cnt.values()) == 0:
            n_no_target += 1
        ok = any(c >= min_n for c in cnt.values()) if mode == "any" else all(c >= min_n for c in cnt.values())
        if not ok:
            continue
        rows[s["entity_id"]] = {"entity_id": s["entity_id"], "n_seg": s.get("n_seg"), "wave_start_ms": s.get("wave_start_ms"),
                                "wave_end_ms": s.get("wave_end_ms"), "per_var_count": cnt,
                                "per_var_count_recent": {str(v): int(re_.get(v, 0)) for v in target_ids},
                                "per_var_count_baseline": {str(v): int(ba.get(v, 0)) for v in target_ids}}
    return rows, n_no_target


def stage_b_filter(rows: dict[str, dict], per_target_min: int, need: int) -> set[str]:
    return {eid for eid, r in rows.items() if sum(1 for c in r["per_var_count"].values() if c >= per_target_min) >= need}


# ---------- labels for stratification / balance report ----------

def load_demographics(root: Path) -> dict[str, dict]:
    p = root / "demographics.csv"
    if not p.exists():
        return {}
    out = {}
    with p.open() as f:
        for r in csv.DictReader(f):
            key = r.get("patient_id") or r.get("entity_id") or r.get("pid")
            if key:
                out[key] = r
    return out


def entity_labels(eid: str, row: dict, demo: dict[str, dict], per_target_min: int, dur_edges: tuple[float, float]) -> set[str]:
    labs = {f"tgt_{v}" for v, c in row["per_var_count"].items() if c >= per_target_min}
    dur_h = (int(row["wave_end_ms"]) - int(row["wave_start_ms"])) / 3.6e6 if row.get("wave_end_ms") and row.get("wave_start_ms") else None
    if dur_h is not None:
        labs.add("dur_b0" if dur_h <= dur_edges[0] else "dur_b1" if dur_h <= dur_edges[1] else "dur_b2")
    d = demo.get(eid, {})
    sex = (d.get("gender") or d.get("sex") or "").strip().upper()[:1]
    if sex in ("M", "F"):
        labs.add(f"sex_{sex}")
    age = d.get("age_years") or d.get("age")
    try:
        a = float(age)
        labs.add("age_lt40" if a < 40 else "age_40_64" if a < 65 else "age_65_74" if a < 75 else "age_75plus")
    except (TypeError, ValueError):
        pass
    adm = (d.get("admission_type") or "").strip()
    if adm:
        labs.add(f"admit_{adm}")
    return labs


def duration_edges(rows: dict[str, dict], ids: set[str]) -> tuple[float, float]:
    d = [(int(rows[e]["wave_end_ms"]) - int(rows[e]["wave_start_ms"])) / 3.6e6 for e in ids
         if rows[e].get("wave_end_ms") and rows[e].get("wave_start_ms")]
    if len(d) < 3:
        return (float("inf"), float("inf"))
    return (float(np.percentile(d, 100 / 3)), float(np.percentile(d, 200 / 3)))


def balance_report(assign: dict[str, str], labels: dict[str, set[str]]) -> dict:
    used = sorted({l for s in labels.values() for l in s})
    n = Counter(assign.values())
    out = {}
    for l in used:
        out[l] = {s: round(sum(1 for e, sp in assign.items() if sp == s and l in labels[e]) / max(1, n[s]), 4) for s in SPLITS}
    return {"method": "multilabel", "labels_used": used, "balance_per_label_per_split": out}


# ---------- splitting ----------

def subject_of(eid: str, manifest_subject: dict[str, str]) -> str:
    return str(manifest_subject.get(eid) or eid.split("_")[0])


def keep_splits(old: dict, stage_b: set[str], subj: dict[str, str], ratios, seed: int) -> tuple[dict[str, str], dict]:
    old_assign = {e: s for s in SPLITS for e in old.get(s, [])}
    assign = {e: old_assign[e] for e in stage_b if e in old_assign}
    dropped = sorted(e for e in old_assign if e not in stage_b)
    new = sorted(e for e in stage_b if e not in old_assign)
    subj_split: dict[str, str] = {}
    for e, s in assign.items():
        subj_split.setdefault(subj[e], s)
    rng = random.Random(seed)
    rng.shuffle(new)
    placed_by_subject = 0
    by_subject: dict[str, list[str]] = defaultdict(list)
    for e in new:
        by_subject[subj[e]].append(e)
    for sid, ents in sorted(by_subject.items(), key=lambda kv: kv[0]):
        if sid in subj_split:
            for e in ents:
                assign[e] = subj_split[sid]; placed_by_subject += 1
    remaining = [sid for sid in by_subject if sid not in subj_split]
    rng.shuffle(remaining)
    total_target = len(assign) + sum(len(by_subject[s]) for s in remaining)
    for sid in remaining:
        n = Counter(assign.values())
        deficit = {s: ratios[i] * total_target - n.get(s, 0) for i, s in enumerate(SPLITS)}
        s_best = max(SPLITS, key=lambda s: deficit[s])
        for e in by_subject[sid]:
            assign[e] = s_best
        subj_split[sid] = s_best
    info = {"mode": "keep_splits", "n_kept": len([e for e in stage_b if e in old_assign]), "n_dropped": len(dropped),
            "n_added": len(new), "n_added_by_subject_rule": placed_by_subject, "dropped_entities": dropped,
            "added_entities": sorted(new)}
    return assign, info


def iterative_stratification(units: list[str], unit_labels: dict[str, set[str]], unit_size: dict[str, int], ratios, seed: int) -> dict[str, str]:
    """Sechidis et al. 2011 iterative stratification, weighted by unit size (entities per subject)."""
    rng = random.Random(seed)
    total = sum(unit_size[u] for u in units)
    desired = {s: ratios[i] * total for i, s in enumerate(SPLITS)}
    label_units: dict[str, set[str]] = defaultdict(set)
    for u in units:
        for l in unit_labels[u]:
            label_units[l].add(u)
    desired_label = {l: {s: ratios[i] * sum(unit_size[u] for u in us) for i, s in enumerate(SPLITS)} for l, us in label_units.items()}
    assign: dict[str, str] = {}
    remaining = set(units)
    while remaining:
        cand = [(sum(unit_size[u] for u in us if u in remaining), l) for l, us in label_units.items() if any(u in remaining for u in us)]
        if cand:
            _, l = min(cand, key=lambda x: (x[0], x[1]))   # rarest remaining label first
            pool = sorted(u for u in label_units[l] if u in remaining)
        else:
            l = None
            pool = sorted(remaining)
        rng.shuffle(pool)
        for u in pool:
            if u not in remaining:
                continue
            if l is not None:
                best = max(desired_label[l].values())
                cands = [s for s in SPLITS if desired_label[l][s] >= best - 1e-9]
            else:
                cands = list(SPLITS)
            if len(cands) > 1:
                best = max(desired[s] for s in cands)
                cands = [s for s in cands if desired[s] >= best - 1e-9]
            s = cands[0] if len(cands) == 1 else rng.choice(cands)
            assign[u] = s
            remaining.discard(u)
            desired[s] -= unit_size[u]
            for ll in unit_labels[u]:
                desired_label[ll][s] -= unit_size[u]
    return assign


def resplit(stage_b: set[str], subj: dict[str, str], labels: dict[str, set[str]], ratios, seed: int) -> tuple[dict[str, str], dict]:
    by_subject: dict[str, list[str]] = defaultdict(list)
    for e in sorted(stage_b):
        by_subject[subj[e]].append(e)
    units = sorted(by_subject)
    unit_labels = {u: set().union(*(labels[e] for e in by_subject[u])) for u in units}
    unit_size = {u: len(by_subject[u]) for u in units}
    ua = iterative_stratification(units, unit_labels, unit_size, ratios, seed)
    assign = {e: ua[u] for u in units for e in by_subject[u]}
    return assign, {"mode": "multilabel_stratified_patient_disjoint", "n_subjects": len(units)}


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True); ap.add_argument("--registry", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--from-summary", help="tasks/<task>/build_summary.json of the previous build (its 'spec' is reused)")
    g.add_argument("--spec", help="YAML/JSON spec with the two-stage keys")
    ap.add_argument("--task-name", default=None, help="override spec task_name (output dir under tasks/)")
    m = ap.add_mutually_exclusive_group()
    m.add_argument("--keep-splits", nargs="?", const="__auto__", default=None, metavar="OLD_SPLITS_JSON")
    m.add_argument("--resplit", action="store_true")
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dataset", default=None, help="label for outputs (default: spec.dataset or root name)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    root = Path(args.root)
    spec = load_spec(args)
    task = spec["task_name"]; dataset = args.dataset or spec.get("dataset") or root.name
    registry = load_registry(args.registry)
    target_ids = resolve_targets(spec["target_var_ids"], registry)
    spec["target_var_ids"] = target_ids
    spec.setdefault("target_var_names", [registry.get(v, {}).get("name") for v in target_ids])
    need = int(spec["stage_b_min_targets_passing"]); ptm = int(spec["stage_b_per_target_min"])
    ratios = [float(x) for x in spec["ratios"]]; seed = int(spec["seed"])
    task_dir = root / "tasks" / task
    old_splits_path = Path(args.keep_splits) if args.keep_splits and args.keep_splits != "__auto__" else task_dir / "splits.json"
    mode = "resplit" if args.resplit else ("keep" if (args.keep_splits or old_splits_path.exists()) else "resplit")
    if mode == "keep" and not old_splits_path.exists():
        sys.exit(f"--keep-splits: {old_splits_path} not found")
    print(f"task={task} dataset={dataset} targets={target_ids} stageA(min={spec['stage_a_min_events_per_target']},{spec['stage_a_eligibility']}) "
          f"stageB(per_target_min={ptm}, need={need}/{len(target_ids)}) split_mode={mode} seed={seed} ratios={ratios}", flush=True)

    manifest = json.loads((root / "manifest.json").read_text())
    ids = [m_.get("entity_id") or m_.get("dir") for m_ in manifest]
    ids = [e for e in ids if e]
    manifest_subject = {(m_.get("entity_id") or m_.get("dir")): (m_.get("subject_id") if m_.get("subject_id") is not None else m_.get("pid"))
                        for m_ in manifest}
    if args.limit:
        ids = ids[:args.limit]
    scans = scan_entities(root, ids, args.workers)
    rows, n_no_target = stage_a_rows(scans, target_ids, int(spec["stage_a_min_events_per_target"]), spec["stage_a_eligibility"])
    stage_b = stage_b_filter(rows, ptm, need)
    print(f"scanned={sum(1 for s in scans if s.get('ok'))} stageA={len(rows)} stageB={len(stage_b)} no_target_events={n_no_target}", flush=True)

    subj = {e: subject_of(e, manifest_subject) for e in stage_b}
    demo = load_demographics(root)
    edges = duration_edges(rows, stage_b)
    labels = {e: entity_labels(e, rows[e], demo, ptm, edges) for e in stage_b}
    if mode == "keep":
        old = json.loads(old_splits_path.read_text())
        assign, info = keep_splits(old, stage_b, subj, ratios, seed)
        source = f"keep_splits from {old_splits_path.name} (stage-B re-evaluated on the current partitions; new entities by subject rule / ratio fill)"
    else:
        assign, info = resplit(stage_b, subj, labels, ratios, seed)
        source = "multilabel_stratified_patient_disjoint"
    # gates
    per_split_subj = {s: {subj[e] for e, sp in assign.items() if sp == s} for s in SPLITS}
    assert not (per_split_subj["train"] & per_split_subj["val"]) and not (per_split_subj["train"] & per_split_subj["test"]) \
        and not (per_split_subj["val"] & per_split_subj["test"]), "subject leakage between splits"
    assert set(assign) == stage_b, "split lists != stage-B set"
    lists = {s: sorted(e for e, sp in assign.items() if sp == s) for s in SPLITS}
    per_var_cov = {str(v): {s: round(sum(1 for e in lists[s] if rows[e]["per_var_count"][str(v)] >= ptm) / max(1, len(lists[s])), 4) for s in SPLITS}
                   for v in target_ids}
    strat = balance_report(assign, labels)
    summary = {"task": task, "dataset": dataset, "task_dir": str(task_dir), "n_entities_scanned": sum(1 for s in scans if s.get("ok")),
               "n_no_target_events": n_no_target, "n_entities_stage_a": len(rows), "n_entities_stage_b": len(stage_b),
               "n_train": len(lists["train"]), "n_val": len(lists["val"]), "n_test": len(lists["test"]),
               "elapsed_sec": round(time.time() - t0, 1), "split_mode": info["mode"],
               **{k: v for k, v in info.items() if k not in ("dropped_entities", "added_entities", "mode")}}
    print(json.dumps(summary, indent=1), flush=True)
    if args.dry_run:
        return
    task_dir.mkdir(parents=True, exist_ok=True)
    bk = REPO_ROOT / "workzone" / "outputs" / dataset / "traj_rebuild"
    bk.mkdir(parents=True, exist_ok=True)
    if (task_dir / "splits.json").exists():
        (bk / f"{task}_splits_before.json").write_text((task_dir / "splits.json").read_text())
    cohort = {"task": task, "task_kind": "estimation", "target_var_ids": target_ids, "target_var_names": spec["target_var_names"],
              "min_events_per_target": int(spec["stage_a_min_events_per_target"]), "eligibility": spec["stage_a_eligibility"],
              "n_entities": len(rows),
              "per_var_eligible_in_cohort": {str(v): sum(1 for r in rows.values() if r["per_var_count"][str(v)] >= int(spec["stage_a_min_events_per_target"])) for v in target_ids},
              "per_var_total_events_in_cohort": {str(v): sum(r["per_var_count"][str(v)] for r in rows.values()) for v in target_ids},
              "fields": ["entity_id", "n_seg", "wave_start_ms", "wave_end_ms", "per_var_count", "per_var_count_recent", "per_var_count_baseline"],
              "entities": [rows[e] for e in sorted(rows)]}
    (task_dir / "cohort.json").write_text(json.dumps(cohort, indent=2, default=str))
    splits = {"task": task, "source": source, "seed": seed, "ratios": ratios, "n_train": len(lists["train"]), "n_val": len(lists["val"]),
              "n_test": len(lists["test"]), "per_var_coverage_per_split": per_var_cov, "stratification": strat,
              "stage_b_per_target_min": ptm, "stage_b_min_targets_passing": need, "stage_b_n_targets": len(target_ids),
              "rebuild": {k: v for k, v in info.items()}, "built_at": time.strftime("%Y-%m-%d %H:%M"), **lists}
    (task_dir / "splits.json").write_text(json.dumps(splits, indent=2))
    (task_dir / "build_summary.json").write_text(json.dumps({"spec": spec, "summary": [summary]}, indent=2, default=str))
    (bk / f"{task}_rebuild_summary.json").write_text(json.dumps({"spec": spec, "summary": summary, "stratification": strat}, indent=2, default=str))
    print(f"wrote {task_dir}/{{cohort,splits,build_summary}}.json", flush=True)


if __name__ == "__main__":
    main()
