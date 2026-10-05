#!/usr/bin/env bash
#SBATCH --job-name=mladi_B
#SBATCH --account=med250003p
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=1900M
#SBATCH --time=48:00:00
#SBATCH --output=/jet/home/mwang11/Physio_Data/workzone/logs/mladi_B-%j.out
#SBATCH --error=/jet/home/mwang11/Physio_Data/workzone/logs/mladi_B-%j.out
# MLADI Stage B (workzone/mladi/stage_b_wave.py): PLETH40 / II120 / time_ms / meta.json from raw H5.
# Resumable per entity; SHARD=i/n splits the entity list. Gate B (verify_stage_b.py) runs when
# VERIFY=1 (the last shard / a separate afterok job).
#   sbatch --export=ALL,LIMIT=20,OUT=<scratch> workzone/mladi/slurm/stage_b.sbatch     # trial
#   sbatch --export=ALL,SHARD=0/4 workzone/mladi/slurm/stage_b.sbatch                 # full, one of 4
set -euo pipefail
cd /jet/home/mwang11/Physio_Data
module purge; module load anaconda3
set +u; source "$(conda info --base)/etc/profile.d/conda.sh"
source activate /ocean/projects/med250003p/mwang11/envs/wav_hnet; set -u
export OMP_NUM_THREADS=1
echo "[$(date)] mladi stage B commit=$(git rev-parse --short HEAD) shard=${SHARD:-0/1} on $(hostname)"
OUTARG=(); [[ -n "${OUT:-}" ]] && OUTARG=(--out "$OUT")
python -u workzone/mladi/stage_b_wave.py --limit "${LIMIT:-0}" --shard "${SHARD:-0/1}" \
    --workers "${SLURM_CPUS_PER_TASK:-32}" "${OUTARG[@]}"
if [[ "${VERIFY:-0}" == "1" ]]; then
  python -u workzone/mladi/verify_stage_b.py "${OUTARG[@]}"
fi
