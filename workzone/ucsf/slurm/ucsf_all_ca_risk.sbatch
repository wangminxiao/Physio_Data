#!/bin/bash
# ucsf_all — build tasks/ca_risk (cohort.json + splits.json + README) and sync it to the BeeGFS copy.
# Submit (user):  cd /projects/mwang80/staging/Physio_Data/workzone/ucsf/logs && sbatch ../slurm/ucsf_all_ca_risk.sbatch
#SBATCH --job-name=ucsf_all_ca_risk
#SBATCH --partition=xhu40-b.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -euo pipefail
MIRROR=/projects/mwang80/staging/Physio_Data
REPO=/mnt/localdata/storage/mxwang/Physio_Data
PY=/mnt/localdata/storage/mxwang/anaconda3/envs/physio_data/bin/python
STORE=/mnt/localdata100tb/physio_data/ucsf_all
COPY=/projects/xhu40-cdsfm/physio_data/ucsf_all
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
rsync -rt --no-g --exclude '__pycache__' --exclude 'logs/' $MIRROR/workzone/ucsf/ $REPO/workzone/ucsf/
cd "$REPO"
echo "[ca_risk] $(date) host=$(hostname)"
$PY workzone/ucsf/stage_f_ca_risk.py --dataset ucsf_all --workers 8 --t0-summary /projects/mwang80/staging/ca_t0_plots_grid/summary.json "$@"
rsync -rlt --no-g --info=stats1 "$STORE/tasks/ca_risk/" "$COPY/tasks/ca_risk/"
echo "[ca_risk] done $(date)"
