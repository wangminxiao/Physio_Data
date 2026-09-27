#!/bin/bash
# UCSF all-raw pipeline — stage g_ehr_hf (MIMIC-compatible ehr_hf.npy from vitals_hf) — lab node xhu40-n01 only.
# Submit (user):  cd /projects/mwang80/staging/Physio_Data/workzone/ucsf/logs && sbatch ../slurm/ucsf_all_g_ehr_hf.sbatch [extra args]
#SBATCH --job-name=ucsf_all_g_ehr_hf
#SBATCH --partition=xhu40-b.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -euo pipefail
REPO=/mnt/localdata/storage/mxwang/Physio_Data
PY=/mnt/localdata/storage/mxwang/anaconda3/envs/physio_data/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
echo "[job] $(date) host=$(hostname) stage=g_ehr_hf git=$(git rev-parse --short HEAD 2>/dev/null || echo ?)"
$PY workzone/ucsf/stage_g_ehr_hf.py --dataset ucsf_all --cadence pair --include-nbp --workers 8 "$@"
echo "[job] done $(date)"
