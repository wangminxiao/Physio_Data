#!/usr/bin/env bash
# Push the phase-4 EHR extraction from bedanalysis to DREAM.
#
# RUN THIS ON BEDANALYSIS. bedanalysis reaches dream.nursing.emory.edu:22 directly, but has
# no ssh key of its own -- so connect with agent forwarding and let your laptop's key do the
# auth. Nothing private is left on the shared server:
#
#     ssh -A bedanalysis
#     tmux new -s push          # the transfer outlives an SSH drop
#     bash /labs/hulab/mxwang/Physio_Data/workzone/mimic3/push_mimic3_to_dream.sh
#
# MODES
#   ehr   (default)  only what this extraction changed: ehr_hf.npy + ehr_events.npy +
#                    meta.json. ~1.7 GB. Correct when DREAM already has the mimic3 store --
#                    time_ms.npy and the waveform .npy are untouched by stage3b and are NOT
#                    re-sent.
#   full             the whole canonical store (waveforms included, ~40 GB) and THEN the
#                    updated EHR on top. Use only if DREAM has no mimic3 store yet.
#
# The EHR files come from the scratch mirror, not from the canonical store: the canonical
# store was deliberately left untouched by this run, so it does NOT contain vars 156-164 or
# 207-212. Sending the canonical EHR would silently ship the old channel set.
set -euo pipefail

MODE="${MODE:-ehr}"
DREAM_HOST="${DREAM_HOST:-mwang80@dream.nursing.emory.edu}"
DEST="${DEST:-/projects/mwang80/physio_data/mimic3}"     # <-- CONFIRM before the first run
MIRROR="${MIRROR:-/labs/hulab/mxwang/tmp_local/mimic3}"
CANON="${CANON:-/opt/localdata100tb/physio_data/mimic3}"
REG="${REG:-/labs/hulab/mxwang/Physio_Data/indices/var_registry.json}"

# -rlt not -a: owner/group preservation is meaningless across clusters and only produces
# warnings. --partial so a dropped WAN link resumes instead of restarting.
RS=(rsync -rltvz --partial --info=progress2 --human-readable)
[[ -n "${DRY:-}" ]] && RS+=(--dry-run) && echo "*** DRY RUN ***"

echo "[push] mode=${MODE}  ->  ${DREAM_HOST}:${DEST}"
ssh -o ConnectTimeout=20 "${DREAM_HOST}" "mkdir -p '${DEST}'"

if [[ "${MODE}" == "full" ]]; then
  echo "[push] 1/3 canonical store (waveforms + time_ms) ..."
  "${RS[@]}" --exclude='tasks/' --exclude='subset32/' \
    "${CANON}/" "${DREAM_HOST}:${DEST}/"
fi

echo "[push] EHR files updated by this extraction ..."
"${RS[@]}" \
  --include='*/' \
  --include='ehr_hf.npy' --include='ehr_events.npy' --include='meta.json' \
  --exclude='*' \
  "${MIRROR}/" "${DREAM_HOST}:${DEST}/"

echo "[push] var_registry.json (ids 156-164, 207-212 are new; without it the new var_ids are unnamed) ..."
"${RS[@]}" "${REG}" "${DREAM_HOST}:${DEST}/var_registry.json"

echo "[push] done. Verify on DREAM:"
cat <<'VERIFY'
  python - <<'PY'
import numpy as np, glob, json
from collections import Counter
root = "/projects/mwang80/physio_data/mimic3"
c = Counter()
for d in sorted(glob.glob(root + "/*/"))[:300]:
    for f in ("ehr_hf.npy", "ehr_events.npy"):
        try: a = np.load(d + f)
        except OSError: continue
        c.update(a["var_id"].tolist())
reg = {v["id"]: v["name"] for v in json.load(open(root + "/var_registry.json"))["variables"]}
for v in sorted(c):
    if v >= 150: print(v, reg.get(int(v), "?"), c[v])
PY
  # expect 156-164 and 207-212 to be present; if only 150-155/200-206 appear, the EHR leg
  # of the push did not land.
VERIFY
