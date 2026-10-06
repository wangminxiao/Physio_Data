#!/usr/bin/env bash
#SBATCH --job-name=mladi_C4
#SBATCH --account=med250003p
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=1900M
#SBATCH --time=48:00:00
#SBATCH --output=/jet/home/mwang11/Physio_Data/workzone/logs/mladi_C4-%j.out
#SBATCH --error=/jet/home/mwang11/Physio_Data/workzone/logs/mladi_C4-%j.out
# MLADI Stage C4 (workzone/mladi/stage_c4_pleth_aligned.py: PLETH40_aligned) then Gate C4 (VERIFY=1). Resumable; SHARD=i/n.
#   sbatch --export=ALL,SHARD=0/2 workzone/mladi/slurm/stage_c4.sbatch; gate alone: VERIFY_ONLY=1
set -euo pipefail
cd /jet/home/mwang11/Physio_Data
module purge; module load anaconda3
set +u; source "$(conda info --base)/etc/profile.d/conda.sh"
source activate /ocean/projects/med250003p/mwang11/envs/wav_hnet; set -u
export OMP_NUM_THREADS=1
echo "[$(date)] mladi stage C4 commit=$(git rev-parse --short HEAD) shard=${SHARD:-0/1} on $(hostname)"
OUTARG=(); [[ -n "${OUT:-}" ]] && OUTARG=(--out "$OUT")
if [[ "${VERIFY_ONLY:-0}" != "1" ]]; then
  python -u workzone/mladi/stage_c4_pleth_aligned.py --limit "${LIMIT:-0}" --shard "${SHARD:-0/1}" --workers "${SLURM_CPUS_PER_TASK:-32}" "${OUTARG[@]}"
fi
if [[ "${VERIFY:-0}" == "1" || "${VERIFY_ONLY:-0}" == "1" ]]; then python -u workzone/mladi/verify_stage_c4.py "${OUTARG[@]}"; fi
