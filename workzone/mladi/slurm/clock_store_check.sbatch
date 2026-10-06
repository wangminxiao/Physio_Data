#!/usr/bin/env bash
#SBATCH --job-name=mladi_clkchk
#SBATCH --account=med250003p
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1900M
#SBATCH --time=06:00:00
#SBATCH --output=/jet/home/mwang11/Physio_Data/workzone/logs/mladi_clkchk-%j.out
#SBATCH --error=/jet/home/mwang11/Physio_Data/workzone/logs/mladi_clkchk-%j.out
set -euo pipefail
cd /jet/home/mwang11/Physio_Data
module purge; module load anaconda3
set +u; source "$(conda info --base)/etc/profile.d/conda.sh"
source activate /ocean/projects/med250003p/mwang11/envs/wav_hnet; set -u
python -u workzone/mladi/explore/clock_store_check.py --n "${N:-300}"
