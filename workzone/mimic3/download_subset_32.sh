#!/usr/bin/env bash
# Download the 32-patient subset from the remote server to local.
#
# Prerequisite: on the server run
#   python workzone/mimic3/select_subset_32.py
# which creates /opt/localdata100tb/physio_data/mimic3/subset32/rsync_files.txt
#
# Then run this script LOCALLY (not on the server):
#   bash workzone/mimic3/download_subset_32.sh
#
# Env overrides:
#   REMOTE_USER   (default: mwang80)
#   REMOTE_HOST   (default: bedanalysis.priv.bmi.emory.edu)
#   REMOTE_ROOT   (default: /opt/localdata100tb/physio_data/mimic3)
#   LOCAL_ROOT    (default: <repo>/datasets/mimic3/processed_subset32)
#   SSH_KEY       (optional: path to ssh private key)

set -euo pipefail

REMOTE_USER="${REMOTE_USER:-mwang80}"
REMOTE_HOST="${REMOTE_HOST:-bedanalysis.priv.bmi.emory.edu}"
REMOTE_ROOT="${REMOTE_ROOT:-/opt/localdata100tb/physio_data/mimic3}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL_ROOT="${LOCAL_ROOT:-${REPO_ROOT}/datasets/mimic3/processed_subset32}"

SSH_OPTS="-o StrictHostKeyChecking=no"
if [[ -n "${SSH_KEY:-}" ]]; then
    SSH_OPTS="${SSH_OPTS} -i ${SSH_KEY}"
fi

mkdir -p "${LOCAL_ROOT}"

echo "Remote: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ROOT}"
echo "Local:  ${LOCAL_ROOT}"

# 1) Fetch the file list produced by select_subset_32.py
echo ""
echo "[1/2] Fetching file list..."
scp ${SSH_OPTS} \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ROOT}/subset32/rsync_files.txt" \
    "${LOCAL_ROOT}/rsync_files.txt"
scp ${SSH_OPTS} \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ROOT}/subset32/patient_ids.txt" \
    "${LOCAL_ROOT}/patient_ids.txt"
scp ${SSH_OPTS} \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ROOT}/subset32/manifest.json" \
    "${LOCAL_ROOT}/manifest.json"

N_FILES=$(wc -l < "${LOCAL_ROOT}/rsync_files.txt")
echo "    ${N_FILES} files to pull"

# 2) Rsync listed files (preserves relative dir structure)
echo ""
echo "[2/2] rsyncing files..."
rsync -avzP \
    --files-from="${LOCAL_ROOT}/rsync_files.txt" \
    -e "ssh ${SSH_OPTS}" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ROOT}/" \
    "${LOCAL_ROOT}/"

echo ""
echo "Done. Subset at: ${LOCAL_ROOT}"
echo "Patients:"
head "${LOCAL_ROOT}/patient_ids.txt"
