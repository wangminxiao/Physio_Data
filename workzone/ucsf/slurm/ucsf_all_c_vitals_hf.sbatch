#!/bin/bash
# UCSF all-raw pipeline — stage c_vitals_hf — runs ONLY on the lab node xhu40-n01 (xhu40-b.q), the sole node
# that mounts /mnt/localdata and /mnt/localdata100tb. Node is shared by the lab: stay <= 22 cores / 88 GB.
# The clone lives on a volume that only xhu40-n01 mounts, and sbatch reads the script on the login node, so
# submit from the BeeGFS mirror (visible everywhere); the job cd's into the clone at run time.
# Submit (user):  cd /projects/mwang80/staging/Physio_Data/workzone/ucsf/logs && sbatch ../slurm/ucsf_all_c_vitals_hf.sbatch [extra args]
# Validate only:  sbatch --test-only ../slurm/ucsf_all_c_vitals_hf.sbatch
#SBATCH --job-name=ucsf_all_c_vitals_hf
#SBATCH --partition=xhu40-b.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -euo pipefail
REPO=/mnt/localdata/storage/mxwang/Physio_Data
PY=/mnt/localdata/storage/mxwang/anaconda3/envs/physio_data/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 POLARS_MAX_THREADS=${SLURM_CPUS_PER_TASK}
cd "$REPO"
echo "[job] $(date) host=$(hostname) cpus=$SLURM_CPUS_PER_TASK mem=$SLURM_MEM_PER_NODE stage=c_vitals_hf git=$(git rev-parse --short HEAD 2>/dev/null || echo ?)"
df -h /mnt/localdata100tb | tail -1
$PY workzone/ucsf/stage_c_vitals_hf.py --workers 16 --zero-anchor adibin_first --abs-tail reanchor "$@"
echo "[job] done $(date)"
