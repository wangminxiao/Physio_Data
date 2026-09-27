#!/bin/bash
# Resume the clock-fix chains after Stage B v2 + gate B passed (gate B was re-run interactively on 2026-09-26).
set -euo pipefail
S=../slurm
jC=$(sbatch --parsable $S/ucsf_all_fix_dst.sbatch c)
jvC=$(sbatch --parsable -J ucsf_all_verify_c --dependency=afterok:$jC $S/ucsf_all_verify.sbatch c)
jG=$(sbatch --parsable --dependency=afterok:$jvC $S/ucsf_all_fix_dst.sbatch g)
jF=$(sbatch --parsable --dependency=afterok:$jG $S/ucsf_all_fix_dst.sbatch f)
jCp=$(sbatch --parsable --dependency=afterok:$jF $S/ucsf_all_fix_dst.sbatch copy)
ja=$(sbatch --parsable --dependency=afterok:$jF $S/ucsf_fix_clock.sbatch a)
jb=$(sbatch --parsable --dependency=afterok:$ja $S/ucsf_fix_clock.sbatch b)
jc=$(sbatch --parsable --dependency=afterok:$jb $S/ucsf_fix_clock.sbatch c)
jd=$(sbatch --parsable --dependency=afterok:$jc $S/ucsf_fix_clock.sbatch d)
je=$(sbatch --parsable --dependency=afterok:$jd $S/ucsf_fix_clock.sbatch e)
jf=$(sbatch --parsable --dependency=afterok:$je $S/ucsf_fix_clock.sbatch f)
jcp=$(sbatch --parsable --dependency=afterok:$jf $S/ucsf_fix_clock.sbatch copy)
echo "ucsf_all: C=$jC verify_c=$jvC G=$jG F=$jF copy=$jCp"
echo "ucsf:     a=$ja b=$jb c=$jc d=$jd e=$je f=$jf copy=$jcp"
squeue --me -o "%.7i %.22j %.8T %.10M %R"
