#!/usr/bin/env bash
#SBATCH --job-name=mladi_names
#SBATCH --account=med250003p
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1900M
#SBATCH --time=04:00:00
#SBATCH --output=/jet/home/mwang11/Physio_Data/workzone/logs/mladi_names-%j.out
#SBATCH --error=/jet/home/mwang11/Physio_Data/workzone/logs/mladi_names-%j.out
# Value census for /ehr item names (workzone/mladi/explore/name_values.py). TABLE NAMECOL VALCOLS NAMES FILES via --export.
set -euo pipefail
cd /jet/home/mwang11/Physio_Data
module purge; module load anaconda3
set +u; source "$(conda info --base)/etc/profile.d/conda.sh"
source activate /ocean/projects/med250003p/mwang11/envs/wav_hnet; set -u
python -u workzone/mladi/explore/name_values.py --table "${TABLE:-low_rate}" --name-col "${NAMECOL:-eventName}" \
    --val-cols "${VALCOLS:-resultVal,resultUnit}" --names "${NAMES:?}" --files "${FILES:-400}"
