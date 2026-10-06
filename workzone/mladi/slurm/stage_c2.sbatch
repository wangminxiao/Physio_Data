#!/usr/bin/env bash
#SBATCH --job-name=mladi_C2
#SBATCH --account=med250003p
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=1900M
#SBATCH --time=06:00:00
#SBATCH --output=/jet/home/mwang11/Physio_Data/workzone/logs/mladi_C2-%j.out
#SBATCH --error=/jet/home/mwang11/Physio_Data/workzone/logs/mladi_C2-%j.out
# MLADI Stage C2 (workzone/mladi/stage_c2_nbp_twins.py): NBP twin copies. Resumable.
#   sbatch workzone/mladi/slurm/stage_c2.sbatch
set -euo pipefail
cd /jet/home/mwang11/Physio_Data
module purge; module load anaconda3
set +u; source "$(conda info --base)/etc/profile.d/conda.sh"
source activate /ocean/projects/med250003p/mwang11/envs/wav_hnet; set -u
export OMP_NUM_THREADS=1
echo "[$(date)] mladi stage C2 commit=$(git rev-parse --short HEAD) shard=${SHARD:-0/1} on $(hostname)"
OUTARG=(); [[ -n "${OUT:-}" ]] && OUTARG=(--out "$OUT")
python -u workzone/mladi/stage_c2_nbp_twins.py --workers "${SLURM_CPUS_PER_TASK:-32}" "${OUTARG[@]}"
