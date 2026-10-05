#!/usr/bin/env bash
#SBATCH --job-name=mladi_demo
#SBATCH --account=med250003p
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1900M
#SBATCH --time=02:00:00
#SBATCH --output=/jet/home/mwang11/Physio_Data/workzone/logs/mladi_demo-%j.out
#SBATCH --error=/jet/home/mwang11/Physio_Data/workzone/logs/mladi_demo-%j.out
# MLADI Step 0c (workzone/mladi/explore/demo_entity.py): one encounter -> canonical + alignment plot.
#   sbatch workzone/mladi/slurm/demo.sbatch                       # auto-picked train encounter
#   sbatch --export=ALL,BASE=<base> workzone/mladi/slurm/demo.sbatch
set -euo pipefail
cd /jet/home/mwang11/Physio_Data
module purge; module load anaconda3
set +u; source "$(conda info --base)/etc/profile.d/conda.sh"
source activate /ocean/projects/med250003p/mwang11/envs/wav_hnet; set -u
echo "[$(date)] mladi demo commit=$(git rev-parse --short HEAD) on $(hostname)"
python -u workzone/mladi/explore/demo_entity.py ${BASE:+--base $BASE}
