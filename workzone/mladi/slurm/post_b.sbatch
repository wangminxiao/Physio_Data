#!/usr/bin/env bash
#SBATCH --job-name=mladi_postB
#SBATCH --account=med250003p
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=1900M
#SBATCH --time=12:00:00
#SBATCH --output=/jet/home/mwang11/Physio_Data/workzone/logs/mladi_postB-%j.out
#SBATCH --error=/jet/home/mwang11/Physio_Data/workzone/logs/mladi_postB-%j.out
# After the Stage B shards: (1) Stage B again (resumable -> only entities that failed, e.g. the
# DST-crossing runs the per-row wall grid could not order), (2) regrid_time_ms.py on every entity
# (run-wise grid + dst_crossing_runs flags), (3) Gate B.
set -euo pipefail
cd /jet/home/mwang11/Physio_Data
module purge; module load anaconda3
set +u; source "$(conda info --base)/etc/profile.d/conda.sh"
source activate /ocean/projects/med250003p/mwang11/envs/wav_hnet; set -u
export OMP_NUM_THREADS=1
echo "[$(date)] mladi post-B commit=$(git rev-parse --short HEAD) on $(hostname)"
python -u workzone/mladi/stage_b_wave.py --workers "${SLURM_CPUS_PER_TASK:-32}"
python -u workzone/mladi/regrid_time_ms.py --workers "${SLURM_CPUS_PER_TASK:-32}"
python -u workzone/mladi/verify_stage_b.py
