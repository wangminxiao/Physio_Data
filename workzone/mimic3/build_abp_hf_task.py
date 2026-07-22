#!/usr/bin/env python3
"""Build a dedicated abp_hf estimation cohort: vital_est_full filtered to patients
that actually have invasive ABP in the ehr_hf.npy sidecar, keeping the SAME
train/val/test split assignment (no leakage, comparable to the NBP BP task).

The stock build_estimation_task.py gates on ehr_events coverage; ABP_hf lives in
ehr_hf.npy, so this dedicated filter reads that sidecar instead.
"""
import numpy as np, json, os, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/opt/localdata100tb/physio_data/mimic3")
    ap.add_argument("--src", default="tasks/vital_est_full/splits.json",
                    help="source splits.json (relative to --root) to filter")
    ap.add_argument("--gate-var", type=int, default=155, help="ABPm_hf gate variable")
    ap.add_argument("--min-abp", type=int, default=30, help="min gate-var events to keep a patient")
    ap.add_argument("--out", default="tasks/abp_hf", help="output task dir (relative to --root)")
    ap.add_argument("--write", action="store_true", help="write splits/cohort (else report only)")
    args = ap.parse_args()

    src = json.load(open(os.path.join(args.root, args.src)))
    splits = {k: src[k] for k in ("train", "val", "test") if isinstance(src.get(k), list)}

    # count gate-var events per patient
    cnt = {}
    for sp, pids in splits.items():
        for pid in pids:
            hp = os.path.join(args.root, pid, "ehr_hf.npy")
            n = 0
            if os.path.exists(hp):
                hf = np.load(hp)
                n = int((hf["var_id"] == args.gate_var).sum())
            cnt[pid] = n

    print(f"source: {args.src}  gate=var{args.gate_var} (ABPm_hf)")
    print(f"{'thresh':>7s}" + "".join(f"{s:>9s}" for s in ("train","val","test","TOTAL")))
    for th in (1, 10, 30, 60, 120, 240):
        row = [sum(1 for p in splits[s] if cnt[p] >= th) for s in ("train","val","test")]
        print(f"{th:>7d}" + "".join(f"{v:>9d}" for v in row) + f"{sum(row):>9d}")
    print(f"(source cohort: " + "/".join(f"{len(splits[s])}" for s in ("train","val","test")) + ")")

    if not args.write:
        print("\n(report only; pass --write to emit the task)")
        return

    keep = {s: sorted(p for p in splits[s] if cnt[p] >= args.min_abp) for s in ("train","val","test")}
    outdir = os.path.join(args.root, args.out)
    os.makedirs(outdir, exist_ok=True)
    # NOTE: splits.json holds ONLY metadata + the train/val/test list keys (matches
    # build_estimation_task.py). target_var_ids lives in cohort.json, never here —
    # a list value in splits.json can be mis-read as a patient-id list by consumers.
    splits_json = {
        "task": os.path.basename(args.out),
        "source": f"filtered from {args.src} by ehr_hf var{args.gate_var} (ABPm_hf) >= {args.min_abp}",
        "gate_var": args.gate_var, "min_abp": args.min_abp,
        "n_train": len(keep["train"]), "n_val": len(keep["val"]), "n_test": len(keep["test"]),
        "train": keep["train"], "val": keep["val"], "test": keep["test"],
    }
    json.dump(splits_json, open(os.path.join(outdir, "splits.json"), "w"), indent=2)
    cohort = {
        "task": os.path.basename(args.out), "target_var_ids": [153, 154, 155],
        "source": splits_json["source"],
        "entities": {p: cnt[p] for s in ("train","val","test") for p in keep[s]},
    }
    json.dump(cohort, open(os.path.join(outdir, "cohort.json"), "w"), indent=2)
    print(f"\nWROTE {outdir}/splits.json  n={splits_json['n_train']}/{splits_json['n_val']}/{splits_json['n_test']}")


if __name__ == "__main__":
    main()
