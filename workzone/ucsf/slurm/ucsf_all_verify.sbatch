#!/bin/bash
# Verification gate for the UCSF all-raw pipeline. Usage: sbatch -J ucsf_all_verify_<s> [--dependency=afterok:<jobid>] ../slurm/ucsf_all_verify.sbatch <s>
# where <s> in a|b|c|f. Exits non-zero on FAIL so an `afterok` dependency stops the chain.
#SBATCH --job-name=ucsf_all_verify
#SBATCH --partition=xhu40-b.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -euo pipefail
STAGE=${1:?stage letter a|b|c|f}
REPO=/mnt/localdata/storage/mxwang/Physio_Data
PY=/mnt/localdata/storage/mxwang/anaconda3/envs/physio_data/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$REPO"
echo "[verify $STAGE] $(date) host=$(hostname)"
$PY workzone/ucsf/verify_stage_${STAGE}.py --dataset ucsf_all
echo "[verify $STAGE] done $(date)"
