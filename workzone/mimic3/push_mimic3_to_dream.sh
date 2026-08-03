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
# MODES  (MODE=auto by default -- probes the destination and picks)
#   ehr    only what this extraction changed: ehr_hf.npy + ehr_events.npy + meta.json.
#          ~1.7 GB. Correct when DREAM already has the mimic3 store -- time_ms.npy and the
#          waveform .npy are untouched by stage3b and are NOT re-sent.
#   full   the whole canonical store (waveforms included, ~40 GB) and THEN the updated EHR
#          on top. Used when DREAM has no mimic3 store yet.
#
# auto picks `ehr` only if the destination already holds entity dirs WITH time_ms.npy. A
# destination that exists but is empty still gets `full`, so a half-made directory cannot
# silently produce an EHR-only store with no waveforms to align against.
#
# The EHR files come from the scratch mirror, not from the canonical store: the canonical
# store was deliberately left untouched by this run, so it does NOT contain vars 156-164 or
# 207-212. Sending the canonical EHR would silently ship the old channel set.
set -euo pipefail

MODE="${MODE:-auto}"
DREAM_HOST="${DREAM_HOST:-mwang80@dream.nursing.emory.edu}"
DEST="${DEST:-/projects/xhu40-cdsfm/physio_data/mimic3}"
MIRROR="${MIRROR:-/labs/hulab/mxwang/tmp_local/mimic3}"
CANON="${CANON:-/opt/localdata100tb/physio_data/mimic3}"
REG="${REG:-/labs/hulab/mxwang/Physio_Data/indices/var_registry.json}"

# ONE shared ssh connection for every ssh/rsync call below. Without multiplexing this script
# authenticates 4-5 separate times; DREAM offers password/keyboard-interactive, so that would
# be 4-5 password prompts (and would break entirely under a non-interactive runner).
CTL="${TMPDIR:-/tmp}/push_dream_$$"
SSH=(ssh -o ControlMaster=auto -o ControlPath="${CTL}" -o ControlPersist=900 -o ConnectTimeout=20)
trap 'ssh -o ControlPath="${CTL}" -O exit "${DREAM_HOST}" 2>/dev/null || true' EXIT

echo "[push] opening the shared connection (this is the ONLY auth prompt) ..."
"${SSH[@]}" "${DREAM_HOST}" true

# -rlt not -a: owner/group preservation is meaningless across clusters and only produces
# warnings. --partial so a dropped WAN link resumes instead of restarting.
RS=(rsync -rltvz --partial --info=progress2 --human-readable -e "${SSH[*]}")
[[ -n "${DRY:-}" ]] && RS+=(--dry-run) && echo "*** DRY RUN ***"

"${SSH[@]}" "${DREAM_HOST}" "mkdir -p '${DEST}'"

if [[ "${MODE}" == "auto" ]]; then
  n_ent=$("${SSH[@]}" "${DREAM_HOST}" "ls -d '${DEST}'/*/time_ms.npy 2>/dev/null | wc -l" || echo 0)
  if [[ "${n_ent}" -gt 0 ]]; then
    MODE=ehr
    echo "[push] auto: destination has ${n_ent} entity dirs with time_ms.npy -> MODE=ehr"
  else
    MODE=full
    echo "[push] auto: destination has no entity dirs with time_ms.npy -> MODE=full (~40 GB)"
  fi
fi

echo "[push] mode=${MODE}  ->  ${DREAM_HOST}:${DEST}"

if [[ "${MODE}" == "full" ]]; then
  echo "[push] 1/3 canonical store (waveforms + time_ms) ..."
  "${RS[@]}" --exclude='tasks/' --exclude='subset32/' \
    "${CANON}/" "${DREAM_HOST}:${DEST}/"
fi

echo "[push] EHR files updated by this extraction ..."
if [[ "${MODE}" == "full" ]]; then
  # Everything was just sent, so every entity dir exists; a plain filtered sync is safe.
  "${RS[@]}" \
    --include='*/' \
    --include='ehr_hf.npy' --include='ehr_events.npy' --include='meta.json' \
    --exclude='*' \
    "${MIRROR}/" "${DREAM_HOST}:${DEST}/"
else
  # Sync ONLY entities DREAM already has. `--include='*/'` creates a directory for every
  # mirror entity, so an entity DREAM lacks would end up holding ehr_hf.npy + ehr_events.npy
  # + meta.json with no time_ms.npy and no waveform -- and seg_idx is meaningless without
  # time_ms. An explicit intersection makes that impossible instead of merely unlikely.
  L="${TMPDIR:-/tmp}/push_dream_$$.list"
  trap 'rm -f "${L}" "${L}".* ; ssh -o ControlPath="${CTL}" -O exit "${DREAM_HOST}" 2>/dev/null || true' EXIT
  "${SSH[@]}" "${DREAM_HOST}" "cd '${DEST}' && ls -d */time_ms.npy 2>/dev/null | cut -d/ -f1" \
    | sort > "${L}.dream"
  find "${MIRROR}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort > "${L}.mirror"
  comm -12 "${L}.dream" "${L}.mirror" > "${L}.both"
  n_both=$(wc -l < "${L}.both"); n_mir=$(wc -l < "${L}.mirror"); n_dre=$(wc -l < "${L}.dream")
  n_skip=$(comm -13 "${L}.dream" "${L}.mirror" | wc -l)
  echo "[push]   mirror=${n_mir}  dream=${n_dre}  intersection=${n_both}"
  if [[ "${n_skip}" -gt 0 ]]; then
    echo "[push]   SKIPPING ${n_skip} entity(ies) absent from DREAM (would be waveform-less):"
    comm -13 "${L}.dream" "${L}.mirror" | head -10 | sed 's/^/[push]     /'
    [[ "${n_skip}" -gt 10 ]] && echo "[push]     ... and $((n_skip - 10)) more"
  fi
  if [[ "${n_both}" -eq 0 ]]; then
    echo "[push] ERROR: no entity in common -- wrong DEST, or DREAM's store has a different layout." >&2
    exit 1
  fi
  awk '{print $1"/ehr_hf.npy"; print $1"/ehr_events.npy"; print $1"/meta.json"}' \
    "${L}.both" > "${L}"
  "${RS[@]}" --files-from="${L}" "${MIRROR}/" "${DREAM_HOST}:${DEST}/"
fi

echo "[push] var_registry.json (ids 156-164, 207-212 are new; without it the new var_ids are unnamed) ..."
"${RS[@]}" "${REG}" "${DREAM_HOST}:${DEST}/var_registry.json"

echo "[push] done. Verify on DREAM:"
cat <<VERIFY
  python - <<'PY'
import numpy as np, glob, json
from collections import Counter
root = "${DEST}"
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
