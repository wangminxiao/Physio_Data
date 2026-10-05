#!/usr/bin/env bash
#SBATCH --job-name=mladi_A
#SBATCH --account=med250003p
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=1900M
#SBATCH --time=08:00:00
#SBATCH --output=/jet/home/mwang11/Physio_Data/workzone/logs/mladi_A-%j.out
#SBATCH --error=/jet/home/mwang11/Physio_Data/workzone/logs/mladi_A-%j.out
# MLADI Stage A (workzone/mladi/stage_a_inventory.py) then Gate A (verify_stage_a.py, non-zero on FAIL).
# Resumable. Half a RM-shared node at most.
#   sbatch --export=ALL,LIMIT=5,OUT=<scratch> workzone/mladi/slurm/stage_a.sbatch     # trial
#   sbatch workzone/mladi/slurm/stage_a.sbatch                                        # full
set -euo pipefail
cd /jet/home/mwang11/Physio_Data
module purge; module load anaconda3
set +u; source "$(conda info --base)/etc/profile.d/conda.sh"
source activate /ocean/projects/med250003p/mwang11/envs/wav_hnet; set -u
export OMP_NUM_THREADS=1
echo "[$(date)] mladi stage A commit=$(git rev-parse --short HEAD) on $(hostname)"
OUTARG=(); [[ -n "${OUT:-}" ]] && OUTARG=(--out "$OUT")
python -u workzone/mladi/stage_a_inventory.py --limit "${LIMIT:-0}" --workers "${SLURM_CPUS_PER_TASK:-32}" "${OUTARG[@]}"
if [[ -z "${LIMIT:-}" || "${LIMIT}" == "0" ]]; then
  python -u workzone/mladi/verify_stage_a.py
fi
