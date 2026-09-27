#!/bin/bash
# MOVER (SIS + EPIC) clock fix (datasets/CLOCK_FIX_PLAN_MIMIC_MOVER.md §2) — one step per job, chain with afterok:
#   snapshot | b2_sis | b2_epic | b2_validate | b2_unverified | verify_b2 | assemble | relink | actions | manifest | tasks | verify | copy
#   b2_validate / b2_unverified = v4 season-constrained binary test (2026-09-27): validate on decided cases (fails the chain
#   below 98 % agreement), then re-test the unverified cases only. tasks/ exclude what is still unverified (exclude_meta).
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
  b2_validate) $PY workzone/mover/stage_b2_clock.py --dataset mover --workers 8 --validate-decided "$@"
             $PY workzone/mover/stage_b2_clock.py --dataset mover_epic --workers 8 --validate-decided "$@" ;;
  b2_unverified) $PY workzone/mover/stage_b2_clock.py --dataset mover --workers 8 --only-unverified "$@"
             $PY workzone/mover/stage_b2_clock.py --dataset mover_epic --workers 8 --only-unverified "$@" ;;
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
  tasks)     for s in $SIS $EPIC $COMB; do $PY workzone/common/rebuild_tasks_from_summaries.py --root $s --registry $REPO/indices/var_registry.json --workers 8 --builder-args=--exclude-clock-unverified; done
             for s in $SIS $EPIC $COMB; do n=$(basename $s); for t in $s/tasks/*/; do tn=$(basename $t); if [ -f $SNAP/${n}_task_${tn}_splits.json ]; then echo "$n/$tn: before $(grep -o '"n_train": [0-9]*' $SNAP/${n}_task_${tn}_splits.json | head -1) after $(grep -o '"n_train": [0-9]*' $t/splits.json | head -1)"; fi; done; done ;;
  verify)    $PY workzone/mover/verify_mover_clock.py --dataset mover --snapshot $SNAP/sis; $PY workzone/mover/verify_mover_clock.py --dataset mover_epic --snapshot $SNAP/epic --g3-envelope=-30,90 --g3-min-frac 0.8 ;;
  copy)      mkdir -p $COPY/mover_combine $COPY/mover $COPY/mover_epic
             # mover_combine: entity dirs are symlinks on the lab node -> dereference (-L); waveforms + FM sidecars already live on BeeGFS
             rsync -rLt --no-g --info=stats1 --include='*/' --include='time_ms.npy' --include='meta.json' --include='ehr_*.npy' --include='vitals_events.npy' --include='labs_events.npy' --include='coverage_s.npy' --include='ii_coverage_s.npy' --include='demographics*.csv' --include='manifest.json' --include='*splits.json' --include='tasks/**' --exclude='*' $COMB/ $COPY/mover_combine/
             echo "mover_combine BeeGFS copy: $(ls -d $COPY/mover_combine/*/ | wc -l) dirs; stale time_ms (older than store): $(n=0; for e in $(ls $SIS | grep -E '^[0-9a-f]{16}$' | head -200); do [ $COPY/mover_combine/$e/time_ms.npy -ot $SIS/$e/time_ms.npy ] && n=$((n+1)); done; echo $n)/200 sampled"
             # mover / mover_epic: every entity is also a mover_combine entity -> per-entity symlinks into the BeeGFS mover_combine copy
             # (valid on every node, no 38 GB duplicate), plus their own top-level tables and tasks/
             for n in mover mover_epic; do src=/mnt/localdata100tb/physio_data/$n; dst=$COPY/$n; miss=0; k=0
               for e in $(ls $src | grep -E '^[0-9a-f]{16}$'); do if [ -d $COPY/mover_combine/$e ]; then ln -sfn ../mover_combine/$e $dst/$e; k=$((k+1)); else miss=$((miss+1)); echo "missing in combine copy: $e"; fi; done
               rsync -rLt --no-g --info=stats1 --include='/tasks/***' --include='/*.json' --include='/*.csv' --include='/*.npz' --exclude='*' $src/ $dst/
               echo "$n BeeGFS copy: $k entity links, $miss missing, top-level: $(ls $dst | grep -vE '^[0-9a-f]{16}$' | tr '\n' ' ')"; done ;;
  *) echo "unknown step $STEP"; exit 2 ;;
esac
echo "[mover fix $STEP] done $(date)"
