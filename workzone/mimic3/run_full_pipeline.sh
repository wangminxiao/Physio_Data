#!/usr/bin/env bash
# Full MIMIC-III pipeline, clean run from raw WFDB + CSV to canonical format
# with 4-partition EHR + demographics + sepsis task.
#
# Run ON THE SERVER (bedanalysis), inside the physio_data conda env:
#   conda activate physio_data
#   cd /labs/hulab/mxwang/Physio_Data
#   git pull
#   bash workzone/mimic3/run_full_pipeline.sh [--wipe]
#
# Flags:
#   --wipe           rm -rf processed output dir first (DESTRUCTIVE)
#   --stage3-limit N cap stage3 to N patients for smoke run (debug)
#   --workers N      waveform extraction workers (default 12, capped at 50% cores)
#   --skip-wave      skip stage3 (waveform extraction, the slow one) if already done
#
# Expected wall time on bedanalysis:
#   stage1  ~10 min   (signal header scan)
#   stage2  ~10 min   (EHR parquet filter)
#   stage2b ~1  min   (cross-check)
#   stage3  ~4-6 h    (WFDB read + resample, 5621 patients, 12 workers)
#   stage3b ~20 min   (actions merge)
#   post_sepsis ~10 min (SOFA + extract missing sepsis patients)
#   stage3c ~10 min   (EHR trajectory split)
#   post_sepsis_trajectory ~2 min
#   post_demographics ~1 s
#   stage4  ~5 min    (manifest + splits)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

WIPE=0
STAGE3_LIMIT=""
WORKERS=12
SKIP_WAVE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --wipe)          WIPE=1; shift ;;
        --stage3-limit)  STAGE3_LIMIT="--limit $2"; shift 2 ;;
        --workers)       WORKERS="$2"; shift 2 ;;
        --skip-wave)     SKIP_WAVE=1; shift ;;
        *)  echo "Unknown arg: $1"; exit 1 ;;
    esac
done

OUTDIR=$(python -c "
import yaml
cfg = yaml.safe_load(open('workzone/configs/server_paths.yaml'))
print(cfg['mimic3']['output_dir'])")

echo "================================================================"
echo "  MIMIC-III FULL pipeline"
echo "  output_dir: ${OUTDIR}"
echo "  workers:    ${WORKERS}"
echo "  wipe:       ${WIPE}"
echo "  skip_wave:  ${SKIP_WAVE}"
echo "================================================================"

if [[ "${WIPE}" == "1" ]]; then
    echo ""
    echo "WIPE requested: rm -rf ${OUTDIR}"
    read -r -p "Really delete all processed data? [y/N] " ans
    case "${ans:-N}" in
        y|Y|yes|YES) rm -rf "${OUTDIR}"; mkdir -p "${OUTDIR}" ;;
        *) echo "Aborted."; exit 0 ;;
    esac
fi

mkdir -p workzone/outputs/mimic3

# --- 1. Signal scan ---
echo ""
echo "[1/10] stage1_scan_records.py ..."
python workzone/mimic3/stage1_scan_records.py

# --- 2. EHR filter ---
echo ""
echo "[2/10] stage2_extract_ehr.py ..."
python workzone/mimic3/stage2_extract_ehr.py

# --- 2b. Cross-check ---
echo ""
echo "[3/10] stage2b_cross_check.py ..."
python workzone/mimic3/stage2b_cross_check.py

# --- 3. Waveform extraction (SLOW) ---
if [[ "${SKIP_WAVE}" == "0" ]]; then
    echo ""
    echo "[4/10] stage3_extract_waveforms.py ..."
    python workzone/mimic3/stage3_extract_waveforms.py \
        --workers "${WORKERS}" ${STAGE3_LIMIT}
else
    echo ""
    echo "[4/10] SKIPPED stage3_extract_waveforms.py (--skip-wave)"
fi

# --- 3b. Actions merge ---
echo ""
echo "[5/10] stage3b_extract_actions.py ..."
python workzone/mimic3/stage3b_extract_actions.py

# --- 4. Sepsis cohort + extract missing ---
echo ""
echo "[6/10] post_sepsis_cohort.py ..."
python workzone/mimic3/post_sepsis_cohort.py

# --- 5. EHR trajectory split (THIS is the new canonical layout) ---
echo ""
echo "[7/10] stage3c_ehr_trajectory.py ..."
python workzone/mimic3/stage3c_ehr_trajectory.py

# --- 6. Sepsis extras trajectory split ---
echo ""
echo "[8/10] post_sepsis_trajectory.py ..."
python workzone/mimic3/post_sepsis_trajectory.py

# --- 7. Demographics ---
echo ""
echo "[9/10] post_demographics.py ..."
python workzone/mimic3/post_demographics.py

# --- 8. Manifest + splits ---
echo ""
echo "[10/10] stage4_manifest_splits.py ..."
python workzone/mimic3/stage4_manifest_splits.py

echo ""
echo "================================================================"
echo "  FULL pipeline done."
echo "  Processed: ${OUTDIR}"
echo "  Manifest:  ${OUTDIR}/manifest.json"
echo "  Splits:    ${OUTDIR}/pretrain_splits.json"
echo "  Sepsis:    ${OUTDIR}/tasks/sepsis/"
echo "  Demog:     ${OUTDIR}/demographics.csv"
echo ""
echo "  Per-patient layout: {pid}/"
echo "    PLETH40.npy  II120.npy  time_ms.npy"
echo "    ehr_baseline.npy  ehr_recent.npy  ehr_events.npy  ehr_future.npy"
echo "    meta.json"
echo "================================================================"

# Summary artifacts:
ls -lh workzone/outputs/mimic3/*.json 2>/dev/null | tail -15 || true
