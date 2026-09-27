"""Verification gate after Stage F. Exit 0 = PASS, 1 = FAIL. Writes {intermediate_dir}/verify_stage_f.json.

Checks: manifest/splits present; n_failed <= 0.5 % of entity dirs; n_valid >= 99 % of Stage B ok; every manifest
entity has vitals_hf; splits disjoint by patient and by entity; downstream list sizes match; totals reported.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import polars as pl, yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "workzone" / "configs" / "server_paths.yaml"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default=str(CONFIG_PATH)); ap.add_argument("--dataset", default="ucsf_all")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())[args.dataset]
    inter = Path(cfg["intermediate_dir"]); out_dir = Path(cfg["output_dir"])
    checks = []; info = {}
    def check(name, ok, detail=""):
        checks.append({"check": name, "pass": bool(ok), "detail": detail}); print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    summ_p = inter / "stage_f_manifest_summary.json"
    for name in ("manifest.json", "pretrain_splits.json", "downstream_splits.json"):
        check(f"{name} present", (out_dir / name).exists())
    check("summary present", summ_p.exists())
    if not summ_p.exists() or not all((out_dir / n).exists() for n in ("manifest.json", "pretrain_splits.json", "downstream_splits.json")):
        return finish(inter, checks, info)
    summ = json.loads(summ_p.read_text()); man = json.loads((out_dir / "manifest.json").read_text()); spl = json.loads((out_dir / "pretrain_splits.json").read_text()); ds = json.loads((out_dir / "downstream_splits.json").read_text())
    n_allnan = int(summ.get("n_failed_all_nan_only", 0)); n_other_fail = summ["n_failed"] - n_allnan
    info["n_rejected_all_nan_channel"] = n_allnan
    check("structural failures <= 0.5 % of entity dirs (all-NaN rejects excluded)", n_other_fail <= 0.005 * max(summ["n_entity_dirs"], 1), f"structural={n_other_fail} all_nan={n_allnan} dirs={summ['n_entity_dirs']}")
    b_ok = json.loads((inter / "stage_b_summary.json").read_text())["by_status"].get("ok", 0) if (inter / "stage_b_summary.json").exists() else None
    if b_ok is not None:
        check("valid + all-NaN rejects >= 99 % of Stage B ok", summ["n_valid"] + n_allnan >= 0.99 * b_ok, f"valid={summ['n_valid']} all_nan={n_allnan} b_ok={b_ok}")
    check("every manifest entity has vitals_hf", all(e.get("has_vitals_hf") for e in man), f"missing={sum(1 for e in man if not e.get('has_vitals_hf'))}")
    pat = {e["entity_id"]: str(e["patient_id_ge"]) for e in man}
    sets = {k: set(spl[k]) for k in ("train", "val", "test")}; pats = {k: {pat[e] for e in v if e in pat} for k, v in sets.items()}
    check("splits disjoint by entity", not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"]) and not (sets["val"] & sets["test"]))
    check("splits disjoint by patient", not (pats["train"] & pats["val"]) and not (pats["train"] & pats["test"]) and not (pats["val"] & pats["test"]))
    check("splits cover all valid entities", len(sets["train"] | sets["val"] | sets["test"]) == len(man), f"{len(sets['train'] | sets['val'] | sets['test'])} vs {len(man)}")
    check("downstream list sizes match", all(len(ds[f"{k}_control_list"]) == len(spl[k]) for k in ("train", "val", "test")))
    # cardiac-arrest study cohort must never be in train
    ca_ent = {e["entity_id"] for e in man if e.get("in_ca_cohort")}
    ca_pos = {e["entity_id"] for e in man if e.get("has_ca") == 1}
    if spl.get("holdout_from_train_source"):
        check("CA study cohort absent from train", not (sets["train"] & ca_ent), f"cohort entities in train={len(sets['train'] & ca_ent)}")
        check("CA study cohort present in val and test", bool(sets["val"] & ca_ent) and bool(sets["test"] & ca_ent), f"val={len(sets['val'] & ca_ent)} test={len(sets['test'] & ca_ent)} of {len(ca_ent)}")
        check("CA-positive entities absent from train", not (sets["train"] & ca_pos), f"positives: train={len(sets['train'] & ca_pos)} val={len(sets['val'] & ca_pos)} test={len(sets['test'] & ca_pos)}")
        info["ca_cohort"] = {"entities": len(ca_ent), "positives": len(ca_pos), "val": len(sets["val"] & ca_ent), "test": len(sets["test"] & ca_ent)}
    else:
        check("CA study cohort holdout configured", False, "pretrain_splits.json has no holdout_from_train_source")
    info.update(n_valid=summ["n_valid"], n_failed=summ["n_failed"], total_segments=summ.get("total_segments"), total_wave_hours=summ.get("total_wave_hours"),
                n_patients=summ.get("n_unique_patients"), split_entities={k: len(spl[k]) for k in ("train", "val", "test")}, n_with_vitals_hf=summ.get("n_with_vitals_hf"), total_nbp_events=summ.get("total_nbp_events"))
    return finish(inter, checks, info)


def finish(inter, checks, info):
    ok = all(c["pass"] for c in checks)
    (inter / "verify_stage_f.json").write_text(json.dumps({"stage": "f", "pass": ok, "ran_at_unix": int(time.time()), "checks": checks, "info": info}, indent=2))
    print(json.dumps(info, indent=2)); print("VERIFY STAGE F:", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
