#!/bin/bash
# UCSF all-raw pipeline — stage F manifest + splits — runs on xhu40-n01 (xhu40-b.q), validates every entity dir.
# Submit (user):  cd /projects/mwang80/staging/Physio_Data/workzone/ucsf/logs && sbatch ../slurm/ucsf_all_f_manifest.sbatch
#SBATCH --job-name=ucsf_all_f_manifest
#SBATCH --partition=xhu40-b.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -euo pipefail
REPO=/mnt/localdata/storage/mxwang/Physio_Data
PY=/mnt/localdata/storage/mxwang/anaconda3/envs/physio_data/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
echo "[job] $(date) host=$(hostname) stage=f_manifest git=$(git rev-parse --short HEAD 2>/dev/null || echo ?)"
$PY workzone/ucsf/stage_f_manifest.py --dataset ucsf_all "$@"
echo "[job] done $(date)"
