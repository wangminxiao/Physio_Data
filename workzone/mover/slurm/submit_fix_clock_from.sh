#!/bin/bash
# Resume the MOVER clock-fix chain from a given step:  bash submit_fix_clock_from.sh <start_step>
set -euo pipefail
START=${1:?start step}; S=../slurm/mover_fix_clock.sbatch; prev=""; go=0
for step in snapshot b2_sis b2_epic b2_validate b2_unverified verify_b2 assemble relink actions manifest tasks verify copy; do
  [ "$step" = "$START" ] && go=1; [ $go = 1 ] || continue
  if [ -z "$prev" ]; then j=$(sbatch --parsable $S $step); else j=$(sbatch --parsable --dependency=afterok:$prev $S $step); fi
  echo "$step=$j"; prev=$j
done
squeue --me -o "%.7i %.18j %.8T %R"
