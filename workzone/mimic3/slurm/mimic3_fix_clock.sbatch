#!/bin/bash
# MIMIC-III clock fix (datasets/CLOCK_FIX_PLAN_MIMIC_MOVER.md §1) — one step per job, chain with --dependency=afterok.
#   snapshot | migrate | numerics | actions | sepsis_cohort | trajectory | sepsis_traj | demographics | manifest | tasks | traj | verify | copy
#   traj = rebuild cardio/gas/hgb/kidney_traj with the two-stage builder (workzone/common/build_trajectory_task.py, --keep-splits)
# Submit (user): cd /projects/mwang80/staging/Physio_Data/workzone/mimic3/logs && sbatch ../slurm/mimic3_fix_clock.sbatch <step>
#SBATCH --job-name=mimic3_fix_clock
#SBATCH --partition=xhu40-b.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -euo pipefail
STEP=${1:?step}; shift || true
MIRROR=/projects/mwang80/staging/Physio_Data
REPO=/mnt/localdata/storage/mxwang/Physio_Data
PY=/mnt/localdata/storage/mxwang/anaconda3/envs/physio_data/bin/python
STORE=/mnt/localdata100tb/physio_data/mimic3
COPY=/projects/xhu40-cdsfm/physio_data/mimic3
SNAP=/projects/mwang80/staging/mimic3_before
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 POLARS_MAX_THREADS=${SLURM_CPUS_PER_TASK}
rsync -rt --no-g --exclude '__pycache__' --exclude 'logs/' $MIRROR/workzone/mimic3/ $REPO/workzone/mimic3/
rsync -rt --no-g --exclude '__pycache__' $MIRROR/workzone/common/ $REPO/workzone/common/
cd "$REPO"
echo "[mimic3 fix $STEP] $(date) host=$(hostname) cpus=$SLURM_CPUS_PER_TASK"
case "$STEP" in
  snapshot)  mkdir -p $SNAP; ls $STORE | grep -E '^[0-9]+_[0-9]+$' | shuf -n 300 --random-source=<(yes) > $SNAP/entities.txt
             while read e; do mkdir -p $SNAP/$e; cp $STORE/$e/time_ms.npy $STORE/$e/meta.json $SNAP/$e/; cp $STORE/$e/ehr_events.npy $STORE/$e/ehr_hf.npy $SNAP/$e/ 2>/dev/null || true; done < $SNAP/entities.txt
             cp $STORE/pretrain_splits.json $STORE/manifest.json $SNAP/; for t in $STORE/tasks/*/; do n=$(basename $t); cp $t/splits.json $SNAP/tasks_${n}_splits.json 2>/dev/null || true; done
             echo "snapshot: $(ls -d $SNAP/*/ | wc -l) entities" ;;
  migrate)   $PY workzone/mimic3/fix_clock_migrate.py --dry-run | tail -12; $PY workzone/mimic3/fix_clock_migrate.py | tail -12 ;;
  numerics)  $PY workzone/mimic3/stage3b_extract_numerics.py --root $STORE --workers 16 "$@" ;;
  actions)   $PY workzone/mimic3/stage3b_extract_actions.py "$@"; $PY workzone/mimic3/stage3b_actions_v2.py "$@" ;;
  sepsis_cohort) $PY workzone/mimic3/post_sepsis_cohort.py "$@" ;;
  trajectory) $PY workzone/mimic3/stage3c_ehr_trajectory.py --workers 8 "$@" ;;
  sepsis_traj) $PY workzone/mimic3/post_sepsis_trajectory.py "$@" ;;
  demographics) $PY workzone/mimic3/post_demographics.py; $PY workzone/mimic3/post_demographics_extra.py ;;
  manifest)  $PY workzone/mimic3/stage4_manifest_splits.py --keep-splits ;;
  tasks)     $PY workzone/common/rebuild_tasks_from_summaries.py --root $STORE --registry $REPO/indices/var_registry.json --workers 8
             $PY workzone/mimic3/build_abp_hf_task.py --root $STORE --src tasks/vital_est_full/splits.json --gate-var 155 --min-abp 30 --out tasks/abp_hf --write
             for t in $STORE/tasks/*/; do n=$(basename $t); if [ -f $SNAP/tasks_${n}_splits.json ]; then echo "task $n: before $(grep -o '"n_train": [0-9]*' $SNAP/tasks_${n}_splits.json | head -1) after $(grep -o '"n_train": [0-9]*' $t/splits.json | head -1)"; fi; done ;;
  traj)      for t in cardio_traj gas_traj hgb_traj kidney_traj; do $PY workzone/common/build_trajectory_task.py --root $STORE --registry $REPO/indices/var_registry.json --from-summary $STORE/tasks/$t/build_summary.json --keep-splits --workers 8 "$@"; done
             for t in cardio_traj gas_traj hgb_traj kidney_traj; do echo "$t: before $(grep -o '"n_train": [0-9]*' $SNAP/tasks_${t}_splits.json | head -1) after $(grep -o '"n_train": [0-9]*' $STORE/tasks/$t/splits.json | head -1) | rebuild $(grep -o '"n_kept": [0-9]*\|"n_dropped": [0-9]*\|"n_added": [0-9]*' $STORE/tasks/$t/splits.json | tr '\n' ' ')"; done ;;
  verify)    $PY workzone/mimic3/verify_mimic3_clock.py --root $STORE --snapshot $SNAP "$@" ;;
  copy)      mkdir -p $COPY; rsync -rlt --no-g --info=stats1 --include='*/' --include='time_ms.npy' --include='meta.json' --include='ehr_*.npy' --include='demographics*.csv' --include='manifest.json' --include='pretrain_splits.json' --include='downstream_splits.json' --include='tasks/**' --exclude='*' "$STORE/" "$COPY/"
             echo "copy: $(ls -d $COPY/*/ | wc -l) dirs" ;;
  *) echo "unknown step $STEP"; exit 2 ;;
esac
echo "[mimic3 fix $STEP] done $(date)"
