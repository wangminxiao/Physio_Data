#!/bin/bash
# MOVER (SIS + EPIC) clock fix (datasets/CLOCK_FIX_PLAN_MIMIC_MOVER.md §2) — one step per job, chain with afterok:
#   snapshot | b2_sis | b2_epic | verify_b2 | assemble | actions | manifest | tasks | verify | copy
# Submit (user): cd /projects/mwang80/staging/Physio_Data/workzone/mover/logs && sbatch ../slurm/mover_fix_clock.sbatch <step>
#SBATCH --job-name=mover_fix_clock
#SBATCH --partition=xhu40-b.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -euo pipefail
STEP=${1:?step}; shift || true
MIRROR=/projects/mwang80/staging/Physio_Data
REPO=/mnt/localdata/storage/mxwang/Physio_Data
PY=/mnt/localdata/storage/mxwang/anaconda3/envs/physio_data/bin/python
SIS=/mnt/localdata100tb/physio_data/mover
EPIC=/mnt/localdata100tb/physio_data/mover_epic
COMB=/mnt/localdata100tb/physio_data/mover_combine
COPY=/projects/xhu40-cdsfm/physio_data
SNAP=/projects/mwang80/staging/mover_before
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 POLARS_MAX_THREADS=${SLURM_CPUS_PER_TASK}
for d in mover mover_epic mover_combine common; do rsync -rt --no-g --exclude '__pycache__' --exclude 'logs/' $MIRROR/workzone/$d/ $REPO/workzone/$d/; done
cd "$REPO"
echo "[mover fix $STEP] $(date) host=$(hostname) cpus=$SLURM_CPUS_PER_TASK"
case "$STEP" in
  snapshot)  mkdir -p $SNAP/sis $SNAP/epic
             ls $SIS | grep -vE '\.json$|^tasks$' | shuf -n 300 --random-source=<(yes) | while read e; do mkdir -p $SNAP/sis/$e; cp $SIS/$e/time_ms.npy $SIS/$e/meta.json $SNAP/sis/$e/; cp $SIS/$e/vitals_events.npy $SIS/$e/ehr_events.npy $SNAP/sis/$e/ 2>/dev/null || true; done
             ls $EPIC | grep -vE '\.json$|^tasks$' | shuf -n 100 --random-source=<(yes) | while read e; do mkdir -p $SNAP/epic/$e; cp $EPIC/$e/time_ms.npy $EPIC/$e/meta.json $SNAP/epic/$e/; cp $EPIC/$e/vitals_events.npy $EPIC/$e/ehr_events.npy $SNAP/epic/$e/ 2>/dev/null || true; done
             for s in $SIS $EPIC $COMB; do n=$(basename $s); cp $s/pretrain_splits.json $SNAP/${n}_pretrain_splits.json 2>/dev/null || true; for t in $s/tasks/*/; do tn=$(basename $t); cp $t/splits.json $SNAP/${n}_task_${tn}_splits.json 2>/dev/null || true; done; done
             echo "snapshot: sis $(ls -d $SNAP/sis/*/ | wc -l) epic $(ls -d $SNAP/epic/*/ | wc -l)" ;;
  b2_sis)    $PY workzone/mover/stage_b2_clock.py --dataset mover --workers 8 "$@" ;;
  b2_epic)   $PY workzone/mover/stage_b2_clock.py --dataset mover_epic --workers 8 "$@" ;;
  verify_b2) $PY workzone/mover/verify_mover_clock.py --dataset mover --skip-structural --snapshot $SNAP/sis --out workzone/outputs/mover/verify_clock_b2.json
             $PY workzone/mover/verify_mover_clock.py --dataset mover_epic --skip-structural --snapshot $SNAP/epic --g3-envelope=-30,90 --g3-min-frac 0.8 --out workzone/outputs/mover_epic/verify_clock_b2.json ;;
  assemble)  $PY workzone/mover/stage_e_assemble.py --no-resume --workers 8; $PY workzone/mover_epic/stage_e_assemble.py --no-resume --workers 8 ;;
  actions)   $PY workzone/mover_combine/stage3b_actions.py "$@" ;;
  manifest)  $PY workzone/mover/stage_f_manifest.py; $PY workzone/mover/stage_f_demographics.py; $PY workzone/mover_epic/stage_f_manifest.py; $PY workzone/mover_epic/stage_f_demographics.py
             for n in mover mover_epic; do $PY - <<PYEOF
import json; a=json.load(open("$SNAP/${n}_pretrain_splits.json")); b=json.load(open("/mnt/localdata100tb/physio_data/$n/pretrain_splits.json"))
same = all(sorted(a[k]) == sorted(b[k]) for k in ("train","val","test")); print("$n pretrain splits identical to snapshot:", same, {k: (len(a[k]), len(b[k])) for k in ("train","val","test")})
PYEOF
             done ;;
  relink)    # mover_combine entity dirs are symlinks to the old /opt/localdata100tb paths (broken since the 2026-08 migration)
             n=0; for l in $COMB/*; do [ -L "$l" ] || continue; t=$(readlink "$l"); case "$t" in /opt/localdata100tb/*) ln -sfn "/mnt/localdata100tb/${t#/opt/localdata100tb/}" "$l"; n=$((n+1));; esac; done
             echo "relinked $n symlinks; broken now: $(for l in $COMB/*; do [ -L "$l" ] && [ ! -e "$l" ] && echo x; done | wc -l)" ;;
  tasks)     for s in $SIS $EPIC $COMB; do $PY workzone/common/rebuild_tasks_from_summaries.py --root $s --registry $REPO/indices/var_registry.json --workers 8; done
             for s in $SIS $EPIC $COMB; do n=$(basename $s); for t in $s/tasks/*/; do tn=$(basename $t); if [ -f $SNAP/${n}_task_${tn}_splits.json ]; then echo "$n/$tn: before $(grep -o '"n_train": [0-9]*' $SNAP/${n}_task_${tn}_splits.json | head -1) after $(grep -o '"n_train": [0-9]*' $t/splits.json | head -1)"; fi; done; done ;;
  verify)    $PY workzone/mover/verify_mover_clock.py --dataset mover --snapshot $SNAP/sis; $PY workzone/mover/verify_mover_clock.py --dataset mover_epic --snapshot $SNAP/epic --g3-envelope=-30,90 --g3-min-frac 0.8 ;;
  copy)      for n in mover mover_epic mover_combine; do if [ -d $COPY/$n ]; then rsync -rlt --no-g --info=stats1 --include='*/' --include='time_ms.npy' --include='meta.json' --include='ehr_*.npy' --include='demographics*.csv' --include='manifest.json' --include='*splits.json' --include='tasks/**' --exclude='*' /mnt/localdata100tb/physio_data/$n/ $COPY/$n/; echo "copied $n"; else echo "no BeeGFS copy for $n (skipped)"; fi; done ;;
  *) echo "unknown step $STEP"; exit 2 ;;
esac
echo "[mover fix $STEP] done $(date)"
