#!/bin/bash
# MOVER clock-fix chain (run on the login node from .../workzone/mover/logs)
set -euo pipefail
S=../slurm/mover_fix_clock.sbatch; prev=""
for step in snapshot b2_sis b2_epic verify_b2 assemble relink actions manifest tasks verify copy; do
  if [ -z "$prev" ]; then j=$(sbatch --parsable $S $step); else j=$(sbatch --parsable --dependency=afterok:$prev $S $step); fi
  echo "$step=$j"; prev=$j
done
squeue --me -o "%.7i %.18j %.8T %R"
