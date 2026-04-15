#!/usr/bin/env bash
# Run the EHR trajectory update on the MIMIC-III processed data.
#
# Run ON THE SERVER (bedanalysis), inside the physio_data conda env:
#   conda activate physio_data
#   cd /labs/hulab/mxwang/Physio_Data
#   git pull
#   bash workzone/mimic3/run_trajectory_update.sh
#
# What it does (in order):
#   1. Dry-run stage3c on a 32-patient smoke subset
#   2. Real run of stage3c on that subset; back up old ehr_events.npy → .bak
#   3. PROMPT the user to verify with spot checks before touching all 5,621
#   4. Full stage3c run
#   5. post_sepsis_trajectory.py for SOFA/onset split
#   6. post_demographics.py for demographics.csv
#   7. stage4_manifest_splits.py to rebuild manifest with new validation
#
# Flags:
#   --smoke-only     Stop after step 2 (don't touch the other 5,589 patients)
#   --no-prompt      Skip confirmation between smoke and full pass
#   --workers N      Workers for stage3c (default 4)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

SMOKE_ONLY=0
NO_PROMPT=0
WORKERS=4

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-only)  SMOKE_ONLY=1;       shift ;;
        --no-prompt)   NO_PROMPT=1;        shift ;;
        --workers)     WORKERS="$2";       shift 2 ;;
        *)  echo "Unknown arg: $1"; exit 1 ;;
    esac
done

OUTDIR=$(python -c "
import yaml, pathlib
cfg = yaml.safe_load(open('workzone/configs/server_paths.yaml'))
print(cfg['mimic3']['output_dir'])")

SMOKE_IDS="${OUTDIR}/subset32/patient_ids.txt"

echo "================================================================"
echo "  MIMIC-III EHR trajectory update"
echo "  output_dir: ${OUTDIR}"
echo "  smoke_ids:  ${SMOKE_IDS}"
echo "  workers:    ${WORKERS}"
echo "================================================================"

if [[ ! -f "${SMOKE_IDS}" ]]; then
    echo ""
    echo "[prep] subset32/patient_ids.txt not found. Running select_subset_32.py first."
    python workzone/mimic3/select_subset_32.py
fi

# --- 1. Dry-run on 32-patient subset ---
echo ""
echo "[1/7] stage3c DRY RUN on 32-patient smoke subset ..."
python workzone/mimic3/stage3c_ehr_trajectory.py \
    --patient-ids "${SMOKE_IDS}" \
    --dry-run

# --- 2. Real run on 32-patient subset ---
echo ""
echo "[2/7] stage3c REAL RUN on 32-patient smoke subset ..."
python workzone/mimic3/stage3c_ehr_trajectory.py \
    --patient-ids "${SMOKE_IDS}"

echo ""
echo "Smoke pass done. Spot-check a patient:"
SMOKE_PID=$(head -1 "${SMOKE_IDS}")
echo "  ls ${OUTDIR}/${SMOKE_PID}/"
ls -lh "${OUTDIR}/${SMOKE_PID}/" | grep -E '(ehr_|time_ms|meta)' || true
echo ""
python -c "
import json
m = json.load(open('${OUTDIR}/${SMOKE_PID}/meta.json'))
for k in ('n_events','n_baseline','n_recent','n_future',
         'n_baseline_vars','n_recent_vars','n_future_vars',
         'context_window_ms','baseline_cap_ms','future_cap_ms','ehr_layout_version'):
    if k in m: print(f'  {k}: {m[k]}')
"

if [[ "${SMOKE_ONLY}" == "1" ]]; then
    echo ""
    echo "--smoke-only set. Stopping. Re-run without --smoke-only to process all patients."
    exit 0
fi

if [[ "${NO_PROMPT}" == "0" ]]; then
    echo ""
    read -r -p "Proceed with FULL run on all patients? [y/N] " ans
    case "${ans:-N}" in
        y|Y|yes|YES) ;;
        *) echo "Aborted."; exit 0 ;;
    esac
fi

# --- 3. Full run on every patient ---
echo ""
echo "[3/7] stage3c FULL RUN on all patients ..."
python workzone/mimic3/stage3c_ehr_trajectory.py --workers "${WORKERS}"

# --- 4. Sepsis post-stage trajectory split ---
echo ""
echo "[4/7] post_sepsis_trajectory.py ..."
python workzone/mimic3/post_sepsis_trajectory.py

# --- 5. Demographics ---
echo ""
echo "[5/7] post_demographics.py ..."
python workzone/mimic3/post_demographics.py

# --- 6. Rebuild manifest + splits with new validation ---
echo ""
echo "[6/7] stage4_manifest_splits.py ..."
python workzone/mimic3/stage4_manifest_splits.py

# --- 7. Report ---
echo ""
echo "[7/7] Done. Summary artifacts at workzone/outputs/mimic3/:"
ls -lh workzone/outputs/mimic3/*.json | tail -10 || true
echo ""
echo "New per-patient layout:"
echo "  {pid}/ehr_baseline.npy  ehr_recent.npy  ehr_events.npy  ehr_future.npy"
echo "  ehr_events.npy.bak kept as safety backup"
echo "Top-level additions:"
echo "  ${OUTDIR}/demographics.csv"
