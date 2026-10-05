#!/usr/bin/env bash
#SBATCH --job-name=mladi_explore
#SBATCH --account=med250003p
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1900M
#SBATCH --time=06:00:00
#SBATCH --output=/jet/home/mwang11/Physio_Data/workzone/logs/mladi_explore-%j.out
#SBATCH --error=/jet/home/mwang11/Physio_Data/workzone/logs/mladi_explore-%j.out
# MLADI Step 0b (workzone/mladi/explore/profile_h5.py): census of all raw DWC H5 files + a
# year-stratified detail sample. Read-only on the raw data; writes dataset_profile.json to the
# intermediate dir. RM-shared, half a node at most (shared-cluster rule).
#   sbatch workzone/mladi/slurm/explore.sbatch
set -euo pipefail
cd /jet/home/mwang11/Physio_Data
module purge; module load anaconda3
set +u; source "$(conda info --base)/etc/profile.d/conda.sh"
source activate /ocean/projects/med250003p/mwang11/envs/wav_hnet; set -u
export OMP_NUM_THREADS=1
echo "[$(date)] mladi explore commit=$(git rev-parse --short HEAD) on $(hostname)"
python -u workzone/mladi/explore/profile_h5.py --n-detail "${N_DETAIL:-60}" --workers "${SLURM_CPUS_PER_TASK:-16}"
