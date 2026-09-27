#!/bin/bash
# MIMIC-III clock-fix chain (run on the login node from .../workzone/mimic3/logs)
set -euo pipefail
S=../slurm/mimic3_fix_clock.sbatch; prev=""
for step in snapshot migrate numerics actions sepsis_cohort trajectory sepsis_traj demographics manifest tasks verify copy; do
  if [ -z "$prev" ]; then j=$(sbatch --parsable $S $step); else j=$(sbatch --parsable --dependency=afterok:$prev $S $step); fi
  echo "$step=$j"; prev=$j
done
squeue --me -o "%.7i %.20j %.8T %R"
