#!/usr/bin/env bash
#SBATCH --job-name=mladi_C
#SBATCH --account=med250003p
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=1900M
#SBATCH --time=24:00:00
#SBATCH --output=/jet/home/mwang11/Physio_Data/workzone/logs/mladi_C-%j.out
#SBATCH --error=/jet/home/mwang11/Physio_Data/workzone/logs/mladi_C-%j.out
# MLADI Stage C (workzone/mladi/stage_c_vitals_hf.py) then Gate C (VERIFY=1). Resumable; SHARD=i/n.
#   sbatch --export=ALL,VERIFY=1 workzone/mladi/slurm/stage_c.sbatch
set -euo pipefail
cd /jet/home/mwang11/Physio_Data
module purge; module load anaconda3
set +u; source "$(conda info --base)/etc/profile.d/conda.sh"
source activate /ocean/projects/med250003p/mwang11/envs/wav_hnet; set -u
export OMP_NUM_THREADS=1
echo "[$(date)] mladi stage C commit=$(git rev-parse --short HEAD) shard=${SHARD:-0/1} on $(hostname)"
OUTARG=(); [[ -n "${OUT:-}" ]] && OUTARG=(--out "$OUT")
python -u workzone/mladi/stage_c_vitals_hf.py --limit "${LIMIT:-0}" --shard "${SHARD:-0/1}" --workers "${SLURM_CPUS_PER_TASK:-32}" "${OUTARG[@]}"
if [[ "${VERIFY:-0}" == "1" ]]; then python -u workzone/mladi/verify_stage_c.py "${OUTARG[@]}"; fi
