#!/bin/bash
# Sync the ehr_actions.npy sidecars (+ var_registry.json) from bedanalysis -> dream, ONE hop.
#
# >>> RUN THIS ON bedanalysis <<<
#     ssh bedanalysis
#     bash /labs/hulab/mxwang/tmp_local/sync_actions_to_dream.sh
#
# bedanalysis reaches dream directly (dream.nursing.emory.edu:22 open), so the data never touches
# your laptop. dream asks for a password ONCE — SSH connection multiplexing (ControlMaster) reuses
# that session for every later step.
#
# WHY A TAR: the payload is ~46k tiny files (7.2 MB gzipped). Per-file rsync over 46k files is all
# round-trip overhead; one archive moves in seconds.
#
# MOVER PATH ASYMMETRY (the subtle bit):
#   bedanalysis: physio_data/mover_combine/<id> is a SYMLINK -> physio_data/mover/<id> | mover_epic/<id>
#                (sidecars physically live in mover/ and mover_epic/)
#   dream      : has NO mover/ or mover_epic/ — only a REAL mover_combine/ directory
#   => we archive through the mover_combine VIEW (`find -L` follows the symlinks, `tar -h`
#      dereferences), so paths read mover_combine/<id>/ehr_actions.npy and drop straight into
#      dream's real mover_combine/. mimic3 and mcmed are plain dirs on both sides.
#
# Path translation (authoritative, from UNIPHY scripts/slurm_run.sh):
#   /opt/localdata100tb/physio_data/ -> /projects/xhu40-cdsfm/physio_data/      (data)
#   /labs/hulab/mxwang/Physio_Data/  -> /home/mwang80/workspace/Physio_Data/    (var_registry)
#
# Idempotent: the archive is rebuilt from bedanalysis each run and tar overwrites in place.
set -euo pipefail

DREAM=mwang80@dream.nursing.emory.edu
BED_ROOT=/opt/localdata100tb/physio_data
DREAM_ROOT=/projects/xhu40-cdsfm/physio_data
BED_REG=/labs/hulab/mxwang/Physio_Data/indices/var_registry.json
DREAM_REG=/home/mwang80/workspace/Physio_Data/indices/var_registry.json
TMP=/labs/hulab/mxwang/tmp_local
TARBALL=ehr_actions.tar.gz
CM=(-o ControlMaster=auto -o ControlPath="$TMP/cm-%r@%h:%p" -o ControlPersist=15m)

echo "===================== STEP 1: build archive (bedanalysis-local) ====================="
cd "$BED_ROOT"
{ find mimic3 mcmed -mindepth 2 -maxdepth 2 -name ehr_actions.npy -print0 2>/dev/null;
  find -L mover_combine -mindepth 2 -maxdepth 2 -name ehr_actions.npy -print0 2>/dev/null; } \
  | tar --null -czhf "$TMP/$TARBALL" -T - 2>/dev/null
ls -lh "$TMP/$TARBALL"
echo "files in archive: $(tar -tzf "$TMP/$TARBALL" | wc -l)   (expect 46257)"
tar -tzf "$TMP/$TARBALL" | cut -d/ -f1 | sort | uniq -c

echo "===================== STEP 2: dream pre-flight (password prompt here) ====================="
ssh "${CM[@]}" "$DREAM" "echo '-- datasets --'; ls -d $DREAM_ROOT/*/ 2>/dev/null | head -8
echo -n '-- ehr_actions already on dream (0 = first sync): '; find -L $DREAM_ROOT -mindepth 2 -maxdepth 2 -name ehr_actions.npy 2>/dev/null | wc -l"
echo
read -r -p "Proceed with push? [y/N] " ok
[[ "$ok" == "y" || "$ok" == "Y" ]] || { echo "Aborted."; exit 1; }

echo "===================== STEP 3: push archive + registry (reuses the session) ====================="
rsync -avzh --progress -e "ssh ${CM[*]}" "$TMP/$TARBALL" "$DREAM:/tmp/"
rsync -avzh          -e "ssh ${CM[*]}" "$BED_REG"      "$DREAM:$DREAM_REG"

echo "===================== STEP 4: extract + verify on dream ====================="
ssh "${CM[@]}" "$DREAM" "set -e; cd $DREAM_ROOT
  tar -xzf /tmp/$TARBALL && echo 'extracted OK'
  for d in mimic3 mcmed mover_combine; do
    printf '  %-14s %s\n' \"\$d\" \"\$(find -L \$d -mindepth 2 -maxdepth 2 -name ehr_actions.npy 2>/dev/null | wc -l)\"
  done
  echo '-- spot-check a sidecar loads --'
  python3 -c \"import numpy as np,glob; f=glob.glob('$DREAM_ROOT/mimic3/*/ehr_actions.npy')[0]; a=np.load(f); print('  ',f.split('/')[-2], a.dtype.names, len(a),'events')\"
  echo '-- registry action ids on dream --'
  python3 -c \"import json; V=json.load(open('$DREAM_REG'))['variables']; print('  ',sorted(v['id'] for v in V if 200<=v.get('id',0)<=220))\"
  rm -f /tmp/$TARBALL"

echo
echo "DONE.  Expected:  mimic3 4957 | mcmed 33507 | mover_combine 7793   (total 46257)"
echo "       registry action ids 200..220 present on dream."
echo "Note: ehr_events.npy is NOT touched — actions live only in ehr_actions.npy."
