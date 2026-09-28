#!/bin/bash
# ucsf_all — attach the EHR side (labs, 4-partition trajectory, demographics) and build the NMI-parity tasks
# (lab_est_full, lab6_any_min2, nbp_est_hf; vital_est_hf / abp_hf already exist). One step per job, chain with afterok:
#   link | d1 | d2 | e | demo | tasks | verify | copy
# Submit (user): cd /projects/mwang80/staging/Physio_Data/workzone/ucsf/logs && bash ../slurm/submit_ehr_from.sh link
#SBATCH --job-name=ucsf_all_ehr
#SBATCH --partition=xhu40-b.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -euo pipefail
STEP=${1:?step}; shift || true
MIRROR=/projects/mwang80/staging/Physio_Data
REPO=/mnt/localdata/storage/mxwang/Physio_Data
PY=/mnt/localdata/storage/mxwang/anaconda3/envs/physio_data/bin/python
STORE=/mnt/localdata100tb/physio_data/ucsf_all
COPY=/projects/xhu40-cdsfm/physio_data/ucsf_all
REG=$REPO/indices/var_registry.json
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 POLARS_MAX_THREADS=${SLURM_CPUS_PER_TASK}
for d in ucsf common configs; do rsync -rt --no-g --exclude '__pycache__' --exclude 'logs/' $MIRROR/workzone/$d/ $REPO/workzone/$d/; done
cd "$REPO"
echo "[ucsf_all ehr $STEP] $(date) host=$(hostname) cpus=$SLURM_CPUS_PER_TASK"
case "$STEP" in
  link)   $PY workzone/ucsf/stage_a2_ehr_link.py --dataset ucsf_all ;;
  d1)     $PY workzone/ucsf/stage_d_labs.py --dataset ucsf_all --phase 1 --workers 16 "$@" ;;
  d2)     $PY workzone/ucsf/stage_d_labs.py --dataset ucsf_all --phase 2 --no-resume "$@" ;;   # no-resume: smoke-test entities carry partial labs
  e)      $PY workzone/ucsf/stage_e_assemble.py --dataset ucsf_all --workers 16 "$@" ;;
  demo)   $PY workzone/ucsf/stage_f_demographics.py --dataset ucsf_all ;;
  tasks)  for spec in lab_est_full lab6_any_min2 nbp_est_hf; do
            $PY workzone/common/build_estimation_task.py --root $STORE --registry $REG --spec workzone/common/task_specs/$spec.yaml --workers 16
          done
          for t in lab_est_full lab6_any_min2 nbp_est_hf vital_est_hf abp_hf; do echo "$t: $(grep -o '"n_train": [0-9]*\|"n_val": [0-9]*\|"n_test": [0-9]*' $STORE/tasks/$t/splits.json | head -3 | tr '\n' ' ')"; done ;;
  verify) $PY workzone/ucsf/verify_stage_de.py --dataset ucsf_all ;;
  copy)   rsync -rlt --no-g --info=stats1 --include='*/' --include='labs_events.npy' --include='ehr_baseline.npy' --include='ehr_recent.npy' --include='ehr_events.npy' --include='ehr_future.npy' --include='meta.json' --include='demographics.csv' --include='manifest.json' --include='tasks/**' --exclude='*' $STORE/ $COPY/
          echo "copy: tasks on BeeGFS: $(ls $COPY/tasks | tr '\n' ' ')" ;;
  *) echo "unknown step $STEP"; exit 2 ;;
esac
echo "[ucsf_all ehr $STEP] done $(date)"
