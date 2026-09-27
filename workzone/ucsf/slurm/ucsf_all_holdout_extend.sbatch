#!/bin/bash
# ucsf_all — extend the pretrain-train holdout to every Code Blue CPA patient (43 outside the ValidWaveTime cohort),
# keeping all other split memberships, then rebuild the hand-off tasks from the new splits and sync the BeeGFS copy.
# Submit (user):  cd /projects/mwang80/staging/Physio_Data/workzone/ucsf/logs && sbatch ../slurm/ucsf_all_holdout_extend.sbatch
#SBATCH --job-name=ucsf_all_holdout_extend
#SBATCH --partition=xhu40-b.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -euo pipefail
MIRROR=/projects/mwang80/staging/Physio_Data
REPO=/mnt/localdata/storage/mxwang/Physio_Data
PY=/mnt/localdata/storage/mxwang/anaconda3/envs/physio_data/bin/python
STORE=/mnt/localdata100tb/physio_data/ucsf_all
COPY=/projects/xhu40-cdsfm/physio_data/ucsf_all
CB=/projects/mwang80/staging/ca_events_grid.json
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
rsync -rt --no-g --exclude '__pycache__' --exclude 'logs/' $MIRROR/workzone/ucsf/ $REPO/workzone/ucsf/
cd "$REPO"
echo "[holdout] $(date) host=$(hostname)"
cp -f "$CB" "$REPO/workzone/outputs/ucsf_all/codeblue_cpa_patients.json"
echo "[holdout] 1/5 manifest + splits: keep memberships, move Code Blue CPA patients out of train"
$PY workzone/ucsf/stage_f_manifest.py --dataset ucsf_all --keep-splits --codeblue-patients-json "$REPO/workzone/outputs/ucsf_all/codeblue_cpa_patients.json"
echo "[holdout] 2/5 tasks/vital_est_hf"
$PY workzone/common/build_estimation_task.py --root "$STORE" --registry "$REPO/indices/var_registry.json" --spec workzone/common/task_specs/vital_est_hf.yaml --workers 8
echo "[holdout] 3/5 tasks/abp_hf"
$PY workzone/mimic3/build_abp_hf_task.py --root "$STORE" --src tasks/vital_est_hf/splits.json --gate-var 155 --min-abp 30 --out tasks/abp_hf --write
echo "[holdout] 4/5 gate F"
$PY workzone/ucsf/verify_stage_f.py --dataset ucsf_all
echo "[holdout] 5/5 sync manifest/splits/tasks to the BeeGFS copy"
rsync -rlt --no-g --info=stats1 --include='*/' --include='manifest.json' --include='pretrain_splits.json' --include='downstream_splits.json' --include='var_registry.json' --include='tasks/**' --exclude='*' "$STORE/" "$COPY/"
cp -f "$CB" "$STORE/codeblue_cpa_patients.json"; cp -f "$CB" "$COPY/codeblue_cpa_patients.json"
echo "[holdout] done $(date)"
