#!/bin/bash
# Phase-4 hand-off for the ucsf_all store: manifest refresh (unit, cohort flags) -> Stage G ehr_hf.npy (MIMIC pair
# convention, + NBP events) -> tasks/vital_est_hf -> tasks/abp_hf -> gate F -> sync additions to the BeeGFS copy.
# Submit from /projects/mwang80/staging/Physio_Data/workzone/ucsf/logs:  sbatch ../slurm/ucsf_all_handoff.sbatch
#SBATCH --job-name=ucsf_all_handoff
#SBATCH --partition=xhu40-b.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -euo pipefail
REPO=/mnt/localdata/storage/mxwang/Physio_Data
PY=/mnt/localdata/storage/mxwang/anaconda3/envs/physio_data/bin/python
STORE=/mnt/localdata100tb/physio_data/ucsf_all
COPY=/projects/xhu40-cdsfm/physio_data/ucsf_all
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
echo "[handoff] $(date) host=$(hostname)"
echo "[handoff] 1/6 manifest refresh (unit + cohort flags), splits unchanged (same seed)"
$PY workzone/ucsf/stage_f_manifest.py --dataset ucsf_all --splits-only
echo "[handoff] 2/6 Stage G ehr_hf.npy (pair cadence, + NBP events)"
$PY workzone/ucsf/stage_g_ehr_hf.py --dataset ucsf_all --cadence pair --include-nbp --workers 8
echo "[handoff] 3/6 tasks/vital_est_hf"
$PY workzone/common/build_estimation_task.py --root "$STORE" --registry "$REPO/indices/var_registry.json" --spec workzone/common/task_specs/vital_est_hf.yaml --workers 8
echo "[handoff] 4/6 tasks/abp_hf (ABPm_hf >= 30 ticks, same rule as MIMIC-III)"
$PY workzone/mimic3/build_abp_hf_task.py --root "$STORE" --src tasks/vital_est_hf/splits.json --gate-var 155 --min-abp 30 --out tasks/abp_hf --write
echo "[handoff] 5/6 gate F"
$PY workzone/ucsf/verify_stage_f.py --dataset ucsf_all
echo "[handoff] 6/6 sync additions to the BeeGFS copy"
rsync -rlt --no-g --info=stats1 --include='*/' --include='ehr_hf.npy' --include='meta.json' --include='manifest.json' --include='pretrain_splits.json' --include='downstream_splits.json' --include='var_registry.json' --include='tasks/**' --exclude='*' "$STORE/" "$COPY/"
cp -f "$REPO/indices/var_registry.json" "$STORE/var_registry.json"; cp -f "$REPO/indices/var_registry.json" "$COPY/var_registry.json"
echo "[handoff] copy check: ehr_hf files in copy = $(find "$COPY" -maxdepth 2 -name ehr_hf.npy | wc -l); tasks: $(ls "$COPY/tasks")"
echo "[handoff] done $(date)"
