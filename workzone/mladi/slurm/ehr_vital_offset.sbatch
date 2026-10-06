#!/usr/bin/env bash
#SBATCH --job-name=mladi_vofs
#SBATCH --account=med250003p
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1900M
#SBATCH --time=04:00:00
#SBATCH --output=/jet/home/mwang11/Physio_Data/workzone/logs/mladi_vofs-%j.out
#SBATCH --error=/jet/home/mwang11/Physio_Data/workzone/logs/mladi_vofs-%j.out
set -euo pipefail
cd /jet/home/mwang11/Physio_Data
module purge; module load anaconda3
set +u; source "$(conda info --base)/etc/profile.d/conda.sh"
source activate /ocean/projects/med250003p/mwang11/envs/wav_hnet; set -u
python -u workzone/mladi/explore/ehr_vital_offset.py ${ARGS:-}
