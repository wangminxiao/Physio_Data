#!/bin/bash
# Submit the ucsf_all EHR chain from a given step:  bash submit_ehr_from.sh <start_step>   (steps: link d1 d2 e demo tasks verify copy)
set -euo pipefail
START=${1:?start step}; S=../slurm/ucsf_all_ehr.sbatch; prev=""; go=0
for step in link d1 d2 e demo tasks verify copy; do
  [ "$step" = "$START" ] && go=1; [ $go = 1 ] || continue
  if [ -z "$prev" ]; then j=$(sbatch --parsable $S $step); else j=$(sbatch --parsable --dependency=afterok:$prev $S $step); fi
  echo "$step=$j"; prev=$j
done
squeue --me -o "%.7i %.16j %.8T %R"
