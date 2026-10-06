#!/usr/bin/env bash
#SBATCH --job-name=mladi_D
#SBATCH --account=med250003p
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=1900M
#SBATCH --time=24:00:00
#SBATCH --output=/jet/home/mwang11/Physio_Data/workzone/logs/mladi_D-%j.out
#SBATCH --error=/jet/home/mwang11/Physio_Data/workzone/logs/mladi_D-%j.out
# MLADI Stage D (EHR, stage_d_ehr.py) + D2 (actions, stage_d2_actions.py) then Gate D (VERIFY=1). Resumable; SHARD=i/n.
#   sbatch --export=ALL,VERIFY=1 workzone/mladi/slurm/stage_d.sbatch
set -euo pipefail
cd /jet/home/mwang11/Physio_Data
module purge; module load anaconda3
set +u; source "$(conda info --base)/etc/profile.d/conda.sh"
source activate /ocean/projects/med250003p/mwang11/envs/wav_hnet; set -u
export OMP_NUM_THREADS=1
echo "[$(date)] mladi stage D+D2 commit=$(git rev-parse --short HEAD) shard=${SHARD:-0/1} on $(hostname)"
OUTARG=(); [[ -n "${OUT:-}" ]] && OUTARG=(--out "$OUT")
python -u workzone/mladi/stage_d_ehr.py --limit "${LIMIT:-0}" --shard "${SHARD:-0/1}" --workers "${SLURM_CPUS_PER_TASK:-32}" "${OUTARG[@]}"
python -u workzone/mladi/stage_d2_actions.py --limit "${LIMIT:-0}" --shard "${SHARD:-0/1}" --workers "${SLURM_CPUS_PER_TASK:-32}" "${OUTARG[@]}"
if [[ "${VERIFY:-0}" == "1" ]]; then python -u workzone/mladi/verify_stage_d.py "${OUTARG[@]}"; fi
