#!/bin/bash
# ucsf_all — pyPPG sidecars, companion of ucsf_all_pyppg.sbatch for the shared CPU nodes of the b.q partition
# (CPU-only on any b.q node: node01-03/41/81 GPU nodes have 60-64 idle CPUs, node191-193 CPU nodes). Same extractor (v3), same entity list; the two arrays split the work by a COMMON
# shard count so no entity is touched twice:
#   NSHARDS=40  xhu40-n01 (ucsf_all_pyppg.sbatch)  --array=0-7     (8 shards, 32 cores each, one at a time)
#               b.q       (this file)              --array=8-39    (32 shards, 28 cores each, 5 concurrent on node191/192/193)
# Submit (user): cd /projects/mwang80/staging/Physio_Data/workzone/ucsf/logs && \
#   NSHARDS=40 FAST_JPG=1 sbatch --array=8-39 ../slurm/ucsf_all_pyppg_bq.sbatch
#SBATCH --job-name=ucsf_all_pyppg_bq
#SBATCH --partition=b.q
# (no --nodelist: any b.q node; CPU only, no GPU requested. 40 cores => at most ONE shard per 60/64-core GPU node, leaving >= 20 cores for GPU training jobs)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=96G
#SBATCH --time=7-00:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
set -euo pipefail
PROJECT_DIR=/home/mwang80/workspace/UNIPHY_Plus_v2
DATA_ROOT=/projects/xhu40-cdsfm/physio_data/ucsf_all
TASKS="${TASKS:-lab6_any_min2 nbp_est_hf vital_est_hf abp_hf}"
NSHARDS="${NSHARDS:?set NSHARDS to the common shard count (e.g. 40)}"
SHARD_ID="${SLURM_ARRAY_TASK_ID:?array job only}"
EXTRA=()
[ -n "${MAX_SEGMENTS:-}" ] && EXTRA+=(--max-segments "$MAX_SEGMENTS")
set +eu
spack load /quucnxi 2>/dev/null
source activate ppg_feat_ext 2>/dev/null
set -eu
python -c "import pyPPG, joblib, scipy" 2>/dev/null || { echo "ERROR: conda env ppg_feat_ext not usable on $(hostname)"; exit 1; }
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
ulimit -n 65536 2>/dev/null || true
cd "$PROJECT_DIR"
echo "=== $(date) | pyPPG v3 extract ucsf_all shard ${SHARD_ID}/${NSHARDS} on $(hostname) cpus=${SLURM_CPUS_PER_TASK} max_segments=${MAX_SEGMENTS:-full} ==="
python scripts/extract_pyppg_v3.py \
  --dataset-root "$DATA_ROOT" --tasks $TASKS --channel PLETH40 --fs 40 \
  --num-shards "$NSHARDS" --shard-id "$SHARD_ID" --workers "${SLURM_CPUS_PER_TASK}" --chunk 32 --self-check 20 ${FAST_JPG:+--fast-jpg} ${V4:+--v4} \
  --skip-existing "${EXTRA[@]}" "$@"
echo "=== Done shard ${SHARD_ID} $(date) ==="
