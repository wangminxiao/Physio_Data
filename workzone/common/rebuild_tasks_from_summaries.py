"""
Rebuild every `tasks/<name>/` of a store that was produced by `build_estimation_task.py`, using the spec recorded in
its own `build_summary.json` (key "spec"). Tasks with a dedicated builder (abp_hf, sepsis, ca_*) are skipped and
listed. Used after a time-base change (clock fix) so cohorts/eligibility are recomputed on the corrected grid.

  python workzone/common/rebuild_tasks_from_summaries.py --root /mnt/localdata100tb/physio_data/mimic3 \
      --registry indices/var_registry.json --workers 8 [--only lab_est_full,vital_est_full] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
SKIP = {"abp_hf", "sepsis", "ca_prediction", "ca_risk"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--registry", required=True)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--only", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    root = Path(args.root); tasks = root / "tasks"; only = {s for s in args.only.split(",") if s}
    done, skipped, failed = [], [], []
    for td in sorted(p for p in tasks.iterdir() if p.is_dir()):
        name = td.name
        if only and name not in only:
            continue
        summ = td / "build_summary.json"
        if name in SKIP or not summ.exists():
            skipped.append(name); continue
        spec = json.loads(summ.read_text()).get("spec")
        if not spec:
            skipped.append(name + "(no spec)"); continue
        spec.setdefault("task_name", name)
        with tempfile.NamedTemporaryFile("w", suffix=f"_{name}.yaml", delete=False) as f:
            yaml.safe_dump(spec, f); spec_path = f.name
        cmd = [sys.executable, str(HERE / "build_estimation_task.py"), "--root", str(root), "--registry", args.registry,
               "--spec", spec_path, "--workers", str(args.workers)]
        print(f"== {name}: {' '.join(cmd)}", flush=True)
        if args.dry_run:
            done.append(name); continue
        r = subprocess.run(cmd)
        (done if r.returncode == 0 else failed).append(name)
    print(json.dumps({"rebuilt": done, "skipped": skipped, "failed": failed}, indent=1))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
