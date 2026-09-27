#!/bin/bash
# Resume the MIMIC-III clock-fix chain from a given step:  bash submit_fix_clock_from.sh <start_step> [extra args for the start step]
set -euo pipefail
START=${1:?start step}; shift || true; EXTRA="$*"
S=../slurm/mimic3_fix_clock.sbatch; prev=""; go=0
for step in snapshot migrate numerics actions sepsis_cohort trajectory sepsis_traj demographics manifest tasks traj lab6 verify copy; do
  [ "$step" = "$START" ] && go=1; [ $go = 1 ] || continue
  if [ -z "$prev" ]; then j=$(sbatch --parsable $S $step $EXTRA); else j=$(sbatch --parsable --dependency=afterok:$prev $S $step); fi
  echo "$step=$j"; prev=$j
done
squeue --me -o "%.7i %.20j %.8T %R"
