#!/usr/bin/env bash
# Full VitalDB pipeline runner. Runs Stages A (if needed) through G sequentially.
# Stops on first stage error so you can debug without cascading failures.
#
# Usage (on bedanalysis):
#   tmux new -s vitaldb_run
#   bash /labs/hulab/mxwang/Physio_Data/workzone/vitaldb/run_full.sh
#   # detach with Ctrl-b d; reattach with `tmux attach -t vitaldb_run`
#
# Skip Stage A re-run (cohort parquet already produced by smoke).
# Skip the wipe-smoke-artifacts if you already cleaned manually.
#
# Logs land in workzone/vitaldb/logs/ and per-stage .out files in the same dir.

set -euo pipefail

REPO=/labs/hulab/mxwang/Physio_Data
WZ=$REPO/workzone/vitaldb
OUT=/opt/localdata100tb/physio_data/vitaldb
COMMON=$REPO/workzone/common
REG=$REPO/indices/var_registry.json

cd "$WZ"
mkdir -p logs

source /labs/hulab/mxwang/anaconda3/etc/profile.d/conda.sh
conda activate physio_data

stage() {
    local name="$1"; shift
    local logf="$WZ/logs/${name}.out"
    echo ""
    echo "================================================================"
    echo "  STAGE $name  ---  $(date)"
    echo "  cmd: $*"
    echo "  log: $logf"
    echo "================================================================"
    local t0
    t0=$(date +%s)
    if ! "$@" 2>&1 | tee "$logf"; then
        echo "STAGE $name FAILED - see $logf" >&2
        exit 1
    fi
    local elapsed=$(( $(date +%s) - t0 ))
    echo "STAGE $name done in ${elapsed}s"
}

echo "=== VitalDB full run starting at $(date) ==="
echo "Target output: $OUT"

# --- Step 0: wipe smoke-test artifacts + stale Stage D combined ---
echo "--- cleanup smoke artifacts ---"
rm -rf "$OUT"/0001 "$OUT"/0002 "$OUT"/0003 \
       "$OUT"/demographics.csv "$OUT"/downstream_splits.json \
       "$OUT"/manifest.json "$OUT"/pretrain_splits.json
rm -f "$REPO/workzone/outputs/vitaldb/stage_d_labs_combined.parquet"

# --- Stage A: re-run only if cohort parquet missing ---
if [ ! -s "$REPO/workzone/outputs/vitaldb/valid_cohort.parquet" ]; then
    stage A_cohort python stage_a_cohort.py
else
    echo "STAGE A already has cohort parquet - skipping"
fi

# --- Stage B: waveforms (long pole) ---
stage B_wave python stage_b_wave.py --workers 16

# --- Stage C: 1-Hz vitals from .vital Solar8000/* tracks ---
stage C_vitals python stage_c_vitals.py --workers 16

# --- Stage D: labs from lab_data.csv ---
stage D_labs python stage_d_labs.py

# --- Stage E: 4-partition EHR trajectory ---
stage E_assemble python stage_e_assemble.py --workers 8

# --- Stage F: demographics + manifest + splits ---
stage F_demographics python stage_f_demographics.py
stage F_manifest python stage_f_manifest.py

# --- Stage G: estimation-task cohorts (shared builder) ---
stage G_lab python "$COMMON/build_estimation_task.py" \
    --root "$OUT" --registry "$REG" \
    --spec "$COMMON/task_specs/lab_est_full.yaml" --workers 16

stage G_vital python "$COMMON/build_estimation_task.py" \
    --root "$OUT" --registry "$REG" \
    --spec "$COMMON/task_specs/vital_est_full.yaml" --workers 16

echo ""
echo "================================================================"
echo "  VitalDB run COMPLETE at $(date)"
echo "================================================================"
ls "$OUT" | head
echo "  manifest: $(jq length "$OUT/manifest.json" 2>/dev/null || echo '?') entities"
echo "  tasks:"
ls "$OUT/tasks" 2>/dev/null
