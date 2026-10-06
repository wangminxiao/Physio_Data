#!/usr/bin/env bash
#SBATCH --job-name=mladi_FG
#SBATCH --account=med250003p
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=1900M
#SBATCH --time=24:00:00
#SBATCH --output=/jet/home/mwang11/Physio_Data/workzone/logs/mladi_FG-%j.out
#SBATCH --error=/jet/home/mwang11/Physio_Data/workzone/logs/mladi_FG-%j.out
# MLADI phase-4 hand-off, as workzone/ucsf/slurm/ucsf_all_handoff.sbatch: Stage G ehr_hf.npy (pair cadence + NBP)
# -> Stage F manifest / splits / demographics (gate F) -> tasks vital_est_hf, abp_hf, nbp_est_hf, lab_est_full,
# lab6_any_min2 -> var_registry.json into the store. Resumable (G skips entities at the current version).
#   sbatch workzone/mladi/slurm/handoff.sbatch
set -euo pipefail
cd /jet/home/mwang11/Physio_Data
module purge; module load anaconda3
set +u; source "$(conda info --base)/etc/profile.d/conda.sh"
source activate /ocean/projects/med250003p/mwang11/envs/wav_hnet; set -u
export OMP_NUM_THREADS=1
STORE=/ocean/projects/med250003p/shared/physio_data/mladi
REG=$PWD/indices/var_registry.json
W=${SLURM_CPUS_PER_TASK:-32}
echo "[$(date)] mladi hand-off commit=$(git rev-parse --short HEAD) on $(hostname)"
echo "[handoff] 1/4 Stage G ehr_hf.npy"
python -u workzone/mladi/stage_g_ehr_hf.py --workers "$W"
echo "[handoff] 2/4 Stage F manifest + splits + demographics (gate F)"
python -u workzone/mladi/stage_f_manifest.py --workers "$W"
echo "[handoff] 3/4 tasks"
for spec in vital_est_hf nbp_est_hf lab_est_full lab6_any_min2; do
  python -u workzone/common/build_estimation_task.py --root "$STORE" --registry "$REG" --spec workzone/common/task_specs/$spec.yaml --workers 16
done
python -u workzone/mimic3/build_abp_hf_task.py --root "$STORE" --src tasks/vital_est_hf/splits.json --gate-var 155 --min-abp 30 --out tasks/abp_hf --write
for t in vital_est_hf abp_hf nbp_est_hf lab_est_full lab6_any_min2; do
  echo "$t: $(grep -o '"n_train": [0-9]*\|"n_val": [0-9]*\|"n_test": [0-9]*' $STORE/tasks/$t/splits.json | head -3 | tr '\n' ' ')"
done
echo "[handoff] 4/4 registry"
cp -f "$REG" "$STORE/var_registry.json"
echo "[$(date)] done"
