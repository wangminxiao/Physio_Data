"""
Stage F (ca_risk) — cardiac-arrest risk-prediction task on the `ucsf_all` store, for models fine-tuned on top of the
FM pretrained on `ucsf_all` (PPG + ECG first; monitor vitals available through `vitals_hf`).

Design (see the README written next to the outputs):
  * Universe   every valid entity (wave cycle) of a patient held out of the pretrain TRAIN split
               (`held_out_of_train`: ValidWaveTime study cohort ∪ Code Blue CPA patients).  The FM never saw them.
  * Label      patient level: `has_ca = 1` iff the patient has a Code Blue TypeCode CPA event (`codeblue_cpa`).
  * Time zero  `event_grid_ms` = Code Blue CodeTime placed on THIS entity's UTC-continuous grid
               (`clock.real_wall_to_grid_ms`), so it is meaningful even when the arrest happened outside the cycle
               (negative `event_offset_from_wave_end_min` = event before the cycle ended).  Auxiliary physiological
               markers from the t0' detector (ECG collapse, t0', quality) are copied when available.
  * Coverage   per positive entity: PPG/ECG coverage in the 1/6/12/24 h before the event, gap from the last valid PPG
               segment to the event, PPG hours before the event, `usable_pre_event` (gap <= 30 min & 6-h PPG >= 50 %).
               Every entity: sampled PPG/ECG validity fractions.
  * Controls   (a) cohort controls — held-out patients without a CPA event (ValidWaveTime controls), matched at
               training time by unit / coverage as needed; (b) self-controls — windows of a positive entity that end
               more than `self_control_gap_h` before `event_grid_ms`.  Windows after the event must be excluded.
  * Split      train / test only (user decision: too few positives for a val set), grouped by patient, stratified by
               `has_ca`; the TEST patients are drawn from the pretrain TEST split only, so FM checkpoint selection on
               pretrain val can never touch task test.

Outputs  {output_dir}/tasks/ca_risk/{cohort.json, splits.json, README.md}
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import polars as pl
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "workzone" / "ucsf"))
from clock import real_wall_to_grid_ms  # noqa: E402
from stage_f_ca import load_codeblue  # noqa: E402

CONFIG_PATH = REPO_ROOT / "workzone" / "configs" / "server_paths.yaml"
SEG_MS = 30_000
HORIZONS_H = (1, 6, 12, 24)


def seg_valid(arr, i0: int, i1: int, stride: int = 1, min_frac: float = 0.5) -> np.ndarray:
    """bool per segment in [i0, i1) (stride-sampled): >= min_frac finite samples."""
    if i1 <= i0:
        return np.zeros(0, bool)
    idx = np.arange(i0, i1, stride)
    out = np.zeros(idx.size, bool)
    for j, i in enumerate(idx):
        x = np.asarray(arr[i], dtype=np.float32)
        out[j] = np.isfinite(x).mean() >= min_frac
    return out


def entity_stats(task: dict) -> dict:
    root, e, code_wall, og, gap_h = task["root"], task["entry"], task["code_wall"], task["og"], task["gap_h"]
    d = Path(root) / e["entity_id"]
    out = {"entity_id": e["entity_id"]}
    try:
        tm = np.load(d / "time_ms.npy")
        pl_ = np.load(d / "PLETH40.npy", mmap_mode="r")
        ii = np.load(d / "II120.npy", mmap_mode="r")
        n = len(tm)
        # sampled validity over the whole cycle (stride 5 = one segment per 2.5 min)
        out["ppg_valid_frac"] = round(float(seg_valid(pl_, 0, n, 5).mean()), 3) if n else None
        out["ecg_valid_frac"] = round(float(seg_valid(ii, 0, n, 10).mean()), 3) if n else None
        out["hours"] = round(n * SEG_MS / 3.6e6, 2)
        if code_wall is None or og is None:
            return out
        t = int(real_wall_to_grid_ms(code_wall, og, int(tm[0])))
        out["event_grid_ms"] = t
        out["event_in_cycle"] = bool(tm[0] <= t <= tm[-1] + SEG_MS)
        out["event_offset_from_wave_end_min"] = round((t - (int(tm[-1]) + SEG_MS)) / 60000.0, 1)
        out["event_offset_from_wave_start_min"] = round((t - int(tm[0])) / 60000.0, 1)
        k = int(np.searchsorted(tm, t, side="right"))          # segments [0, k) start before the event
        if k <= 0:
            out["pre_event_segments"] = 0
            return out
        k = min(k, n)
        k0 = max(0, k - 2880)                                     # 24 h before the event
        vp = seg_valid(pl_, k0, k); ve = seg_valid(ii, k0, k)
        for h in HORIZONS_H:
            m = h * 120
            out[f"ppg_cov_{h}h"] = round(float(vp[-m:].mean()), 3) if vp.size else None
            if h in (6, 24):
                out[f"ecg_cov_{h}h"] = round(float(ve[-m:].mean()), 3) if ve.size else None
        # last valid PPG segment before the event (searched over the whole pre-event part)
        vall = seg_valid(pl_, 0, k) if k <= 2880 else np.concatenate([seg_valid(pl_, 0, k0, 1), vp])
        nz = np.flatnonzero(vall)
        out["last_ppg_gap_min"] = round((t - int(tm[nz[-1]] + SEG_MS)) / 60000.0, 1) if nz.size else None
        out["ppg_hours_before_event"] = round(float(vall.sum()) * SEG_MS / 3.6e6, 2)
        out["pre_event_segments"] = int(k)
        out["usable_pre_event"] = bool(out["last_ppg_gap_min"] is not None and out["last_ppg_gap_min"] <= 30
                                       and (out.get("ppg_cov_6h") or 0) >= 0.5)
        out["self_control_end_ms"] = int(t - gap_h * 3.6e6)        # windows ending before this are within-patient negatives
        sc_end = min(out["self_control_end_ms"], int(tm[-1]) + SEG_MS)   # capped at the end of this cycle
        out["self_control_hours"] = round(max(0.0, (sc_end - int(tm[0])) / 3.6e6), 2)
    except Exception as ex:  # noqa: BLE001
        out["error"] = f"{type(ex).__name__}: {str(ex)[:120]}"
    return out


README = """# tasks/ca_risk — cardiac-arrest risk prediction on ucsf_all

Built {built} by `workzone/ucsf/stage_f_ca_risk.py` (seed {seed}, test fraction {test_frac}).

## What is in here
* `cohort.json` — every entity (wave cycle) of the {n_patients} held-out patients: labels, event time on the entity
  grid, pre-event coverage, auxiliary physiological markers, pretrain split, and a `patients` table.
* `splits.json` — `train` / `test` entity lists (no val set), grouped by `patient_id_ge`, stratified by `has_ca`.
  Every test patient belongs to the pretrain TEST split; train patients come from pretrain val + the rest of pretrain
  test.  The FM pretrained on ucsf_all never saw any of these patients.

## Labels and time zero
* `has_ca` (patient level) = Code Blue TypeCode CPA event (`SAUCSFCodeBlue_FirstEvent_2013_2018_final.xlsx`), 1 per
  patient.  `has_ca_csv` keeps the ValidWaveTime label for reference; `codeblue_only` marks the {n_codeblue_only}
  CPA patients absent from the ValidWaveTime cohort.
* `event_grid_ms` = CodeTime placed on THIS entity's UTC-continuous grid (`datasets/ucsf/ALIGNMENT.md`).  It can lie
  outside the cycle: use `event_in_cycle`, `event_offset_from_wave_end_min` (negative = the event happened while
  the cycle was still recording), `event_offset_from_wave_start_min`.
* Auxiliary markers from the t0' detector (`workzone/ucsf/explore/ca_t0_detect.py`, run on the fixed grid):
  `ecg_collapse_offset_min` (marker B), `t0_prime_offset_min`, `t0_prime_quality` — minutes relative to
  `event_grid_ms`; null when no cycle covered the event.  Use the Code Blue time as t0; markers are for QA.

## Windows
* Case windows: windows of a positive entity ending in `[event − H, event]` for a horizon H (1/6/12/24 h).
  Coverage fields `ppg_cov_{{1,6,12,24}}h`, `ecg_cov_{{6,24}}h`, `last_ppg_gap_min`, `ppg_hours_before_event`
  tell how much signal exists before the event; `usable_pre_event` = last PPG <= 30 min before the event and >= 50 %
  PPG in the prior 6 h ({n_usable} entities / {n_usable_patients} patients).
* Exclude everything after `event_grid_ms` (post-arrest physiology) from negatives.
* Self-controls: windows of a positive entity ending before `self_control_end_ms` (= event − {gap_h} h);
  `self_control_hours` is how much of that exists in the entity.
* Cohort controls: entities with `has_ca = 0` (ValidWaveTime study controls).  Match by `unit`, coverage or
  recording length at training time; `ppg_valid_frac` / `ecg_valid_frac` are sampled validity fractions.

## Counts
{counts}

## Provenance
* Store: {root}; manifest flags `held_out_of_train`, `codeblue_cpa`, `in_ca_cohort`, `dst_switch_in_cycle`.
* Code Blue → patient mapping via the offset table (`stage_f_ca.load_codeblue`), times via
  `clock.real_wall_to_grid_ms` (validated against ECG collapses and charted vitals, see ALIGNMENT.md §4).
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG_PATH)); ap.add_argument("--dataset", default="ucsf_all")
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--test-frac", type=float, default=0.30)
    ap.add_argument("--self-control-gap-h", type=float, default=24.0)
    ap.add_argument("--t0-summary", default="", help="summary.json of ca_t0_detect.py run with the grid event times")
    ap.add_argument("--task-name", default="ca_risk"); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit-neg-patients", type=int, default=0, help="smoke test: keep only N negative patients")
    ap.add_argument("--out-root", default="", help="write outputs under this root instead of the store (smoke test)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())[args.dataset]
    root = Path(cfg["output_dir"]); out_root = Path(args.out_root) if args.out_root else root
    man = json.loads((root / "manifest.json").read_text())
    sp = json.loads((root / "pretrain_splits.json").read_text())
    split_of = {e: k for k in ("train", "val", "test") for e in sp[k]}
    codeblue = load_codeblue(Path(cfg["codeblue_parquet"]), Path(cfg["offset_table_parquet"]))
    off = pl.read_parquet(cfg["offset_table_parquet"]).select([
        pl.col("Patient_ID_GE").cast(pl.Utf8).str.strip_chars().str.replace(r"^DE", "").alias("pid"),
        pl.col("offset_GE").cast(pl.Float64).alias("og")]).unique("pid", keep="first")
    og = dict(zip(off["pid"].to_list(), off["og"].to_list()))

    universe = [e for e in man if e.get("held_out_of_train") or e.get("in_ca_cohort") or e.get("codeblue_cpa")]
    pids = sorted({str(e["patient_id_ge"]) for e in universe})
    pos = sorted(p for p in pids if p in codeblue); neg = sorted(p for p in pids if p not in codeblue)
    if args.limit_neg_patients:
        rs = np.random.RandomState(args.seed); neg = sorted(rs.choice(neg, size=min(args.limit_neg_patients, len(neg)), replace=False).tolist())
        universe = [e for e in universe if str(e["patient_id_ge"]) in set(pos) | set(neg)]
    print(f"universe: {len(universe)} entities, {len(pids)} patients ({len(pos)} CPA positive, {len(neg)} negative)", flush=True)

    # ---- per-entity stats (coverage, event placement)
    tasks = [dict(root=str(root), entry=e, code_wall=codeblue.get(str(e["patient_id_ge"])), og=og.get(str(e["patient_id_ge"])),
                  gap_h=args.self_control_gap_h) for e in universe]
    t0 = time.time(); stats = {}
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
        futs = {ex.submit(entity_stats, t): t["entry"]["entity_id"] for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result(); stats[r["entity_id"]] = r
            if i % 500 == 0 or i == len(tasks):
                print(f"  [{i}/{len(tasks)}] {time.time()-t0:.0f}s", flush=True)

    # ---- t0' markers
    t0m = {}
    if args.t0_summary and Path(args.t0_summary).exists():
        for r in json.loads(Path(args.t0_summary).read_text()):
            t0m[r["entity"]] = {"ecg_collapse_offset_min": None if r.get("tB") is None else round(r["tB"], 1),
                                "t0_prime_offset_min": None if r.get("t0") is None else round(r["t0"], 1),
                                "t0_prime_quality": r.get("quality"), "t0_prime_method": r.get("method")}

    # ---- cohort entities
    ents = []
    for e in universe:
        pid = str(e["patient_id_ge"]); s = stats.get(e["entity_id"], {})
        row = {"entity_id": e["entity_id"], "patient_id_ge": pid, "unit": e.get("unit"), "n_seg": e.get("n_seg"),
               "wave_start_ms": e.get("wave_start_ms"), "wave_end_ms": e.get("wave_end_ms"),
               "has_ca": int(pid in codeblue), "has_ca_csv": e.get("has_ca") if e.get("in_ca_cohort") else None,
               "in_ca_cohort": bool(e.get("in_ca_cohort")), "codeblue_only": bool(pid in codeblue and not e.get("in_ca_cohort")),
               "pretrain_split": split_of.get(e["entity_id"]), "dst_switch_in_cycle": e.get("dst_switch_in_cycle"),
               "time_base": e.get("time_base")}
        row.update({k: v for k, v in s.items() if k != "entity_id"})
        row.update(t0m.get(e["entity_id"], {}))
        ents.append(row)

    # ---- split: patient-grouped, stratified; test patients only from pretrain TEST
    pat_split = {}
    for row in ents:
        pat_split.setdefault(row["patient_id_ge"], set()).add(row["pretrain_split"])
    mixed = [p for p, s in pat_split.items() if len(s) > 1]
    assert not mixed, f"{len(mixed)} patients span several pretrain splits"
    rs = np.random.RandomState(args.seed); train_p, test_p = set(), set(); notes = {}
    for label, group in (("pos", pos), ("neg", neg)):
        cands = sorted(p for p in group if pat_split.get(p) == {"test"})
        n_test = int(round(args.test_frac * len(group)))
        if n_test > len(cands):
            notes[label] = f"only {len(cands)} pretrain-test patients available for {n_test} requested"; n_test = len(cands)
        chosen = set(rs.choice(cands, size=n_test, replace=False).tolist()) if n_test else set()
        test_p |= chosen; train_p |= set(group) - chosen
    train = sorted(r["entity_id"] for r in ents if r["patient_id_ge"] in train_p)
    test = sorted(r["entity_id"] for r in ents if r["patient_id_ge"] in test_p)
    assert not (train_p & test_p)

    def cnt(eids):
        rows = [r for r in ents if r["entity_id"] in set(eids)]; pp = {r["patient_id_ge"] for r in rows}
        return {"entities": len(rows), "patients": len(pp), "pos_patients": len({r["patient_id_ge"] for r in rows if r["has_ca"]}),
                "pos_entities": sum(1 for r in rows if r["has_ca"]), "pos_entities_event_in_cycle": sum(1 for r in rows if r.get("event_in_cycle")),
                "pos_entities_usable_pre_event": sum(1 for r in rows if r.get("usable_pre_event")),
                "pos_patients_usable_pre_event": len({r["patient_id_ge"] for r in rows if r.get("usable_pre_event")}),
                "neg_entities": sum(1 for r in rows if not r["has_ca"])}
    counts = {"train": cnt(train), "test": cnt(test), "all": cnt([r["entity_id"] for r in ents])}
    patients = [{"patient_id_ge": p, "has_ca": int(p in codeblue), "split": "test" if p in test_p else "train",
                 "pretrain_split": next(iter(pat_split[p])), "n_entities": sum(1 for r in ents if r["patient_id_ge"] == p),
                 "codeblue_only": bool(p in codeblue and not any(r["in_ca_cohort"] for r in ents if r["patient_id_ge"] == p)),
                 "usable_pre_event": any(r.get("usable_pre_event") for r in ents if r["patient_id_ge"] == p)} for p in pids if p in train_p | test_p]

    task_dir = out_root / "tasks" / args.task_name; task_dir.mkdir(parents=True, exist_ok=True)
    built = time.strftime("%Y-%m-%d %H:%M")
    cohort = {"task": args.task_name, "task_kind": "event_risk_prediction", "built_at": built, "seed": args.seed,
              "label": "has_ca (patient level, Code Blue TypeCode CPA)", "time_zero": "event_grid_ms (CodeTime on the entity grid)",
              "self_control_gap_h": args.self_control_gap_h, "horizons_h": list(HORIZONS_H), "counts": counts, "split_notes": notes,
              "n_entities": len(ents), "n_patients": len(patients), "fields": sorted({k for r in ents for k in r}),
              "patients": patients, "entities": ents}
    (task_dir / "cohort.json").write_text(json.dumps(cohort, indent=1, default=str))
    splits = {"task": args.task_name, "source": "held-out patients of pretrain_splits.json (val + test); patient-grouped, stratified by has_ca; "
              "test patients drawn from pretrain test only", "seed": args.seed, "test_frac": args.test_frac, "group_by": "patient_id_ge",
              "n_train": len(train), "n_val": 0, "n_test": len(test), "counts": counts, "train": train, "val": [], "test": test}
    (task_dir / "splits.json").write_text(json.dumps(splits, indent=1))
    n_cb_only = sum(1 for p in patients if p["codeblue_only"])
    (task_dir / "README.md").write_text(README.format(
        built=built, seed=args.seed, test_frac=args.test_frac, n_patients=len(patients), n_codeblue_only=n_cb_only,
        n_usable=counts["all"]["pos_entities_usable_pre_event"], n_usable_patients=counts["all"]["pos_patients_usable_pre_event"],
        gap_h=args.self_control_gap_h, root=str(root), counts=json.dumps(counts, indent=2)))
    print(json.dumps({"counts": counts, "notes": notes, "errors": sum(1 for s in stats.values() if "error" in s)}, indent=1))
    print(f"wrote {task_dir}/cohort.json, splits.json, README.md")


if __name__ == "__main__":
    main()
