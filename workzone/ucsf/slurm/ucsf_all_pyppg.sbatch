#!/bin/bash
# ucsf_all — pyPPG biomarker sidecars (UNIPHY_Plus_v2 scripts/extract_pyppg_v3.py: pyPPG fiducials + numpy biomarker layer,
# bit-identical to v2 output, ~10x faster; self-check against pyPPG on the first 20 segments) on the lab node xhu40-n01.
# One array task = one shard of the entity list (entities = union of the task cohorts in TASKS); shards run one after
# another on the node (32 of its 44 cores each) and are resumable (--skip-existing). Reads/writes the BeeGFS copy
# /projects/xhu40-cdsfm/physio_data/ucsf_all/<entity>/pyPPG_raw.npy; run scripts/slurm_pyppg_standardize.sh afterwards.
# Submit (user): cd /projects/mwang80/staging/Physio_Data/workzone/ucsf/logs && \
#   [MAX_SEGMENTS=240] [TASKS="lab6_any_min2 nbp_est_hf vital_est_hf abp_hf"] sbatch --array=0-7 ../slurm/ucsf_all_pyppg.sbatch
#   (MAX_SEGMENTS unset = full-length recordings, the cache convention of the NMI runbook; 240 = 2-h prefix, ~30x cheaper)
#   V4=1 (default in the submit commands): numpy fiducial frame + correction + cached filters + vectorised statistics
#   (scripts/validate_pyppg_v4.py: fiducials identical, float16 sidecars identical); FAST_JPG=1 = p1/p2 patch only
#SBATCH --job-name=ucsf_all_pyppg
#SBATCH --partition=xhu40-b.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=7-00:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
set -euo pipefail
PROJECT_DIR=/home/mwang80/workspace/UNIPHY_Plus_v2
DATA_ROOT=/projects/xhu40-cdsfm/physio_data/ucsf_all
TASKS="${TASKS:-lab6_any_min2 nbp_est_hf vital_est_hf abp_hf}"
NSHARDS="${NSHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}"   # set NSHARDS=40 with --array=0-7 when ucsf_all_pyppg_bq.sbatch takes shards 8-39
SHARD_ID="${SLURM_ARRAY_TASK_ID:-0}"
EXTRA=()
[ -n "${MAX_SEGMENTS:-}" ] && EXTRA+=(--max-segments "$MAX_SEGMENTS")

set +eu
spack load /quucnxi 2>/dev/null
source activate ppg_feat_ext 2>/dev/null
set -eu
python -c "import pyPPG, joblib, scipy" 2>/dev/null || { echo "ERROR: conda env ppg_feat_ext not usable on $(hostname) (spack conda + env must be visible here)"; exit 1; }
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
ulimit -n 65536 2>/dev/null || true
cd "$PROJECT_DIR"
echo "=== $(date) | pyPPG extract ucsf_all shard ${SHARD_ID}/${NSHARDS} on $(hostname) cpus=${SLURM_CPUS_PER_TASK} tasks='$TASKS' max_segments=${MAX_SEGMENTS:-full} ==="
python scripts/extract_pyppg_v3.py \
  --dataset-root "$DATA_ROOT" --tasks $TASKS --channel PLETH40 --fs 40 \
  --num-shards "$NSHARDS" --shard-id "$SHARD_ID" --workers "${SLURM_CPUS_PER_TASK}" --chunk 32 --self-check 20 ${FAST_JPG:+--fast-jpg} ${V4:+--v4} \
  --skip-existing "${EXTRA[@]}" "$@"
echo "=== Done shard ${SHARD_ID} $(date) ==="
