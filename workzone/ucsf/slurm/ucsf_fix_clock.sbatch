#!/bin/bash
# UCSF CA-cohort store (`ucsf`, /mnt/localdata100tb/physio_data/ucsf) — clock-convention fix (datasets/ucsf/ALIGNMENT.md).
# One step per job, chained with --dependency=afterok:  a | b | c | d | e | f | copy
#   a    Stage A  wave windows + admission windows on the UTC-continuous grid (offset table parquet)
#   b    Stage B  waveforms, ONLY cycles straddling a DST switch (UTC-continuous grid, no 1-h gap/overlap)
#   c    Stage C  .vital events, same subset (headers -> grid)
#   d    Stage D  labs phase 2 for ALL entities (EHR -> grid with the corrected rule)
#   e    Stage E  4-partition EHR assembly for ALL entities
#   f    Stage F  ca_prediction task (Code Blue times on the grid), demographics, manifest (splits kept)
#   copy rsync the store to the BeeGFS mirror /projects/xhu40-cdsfm/physio_data/ucsf
# Submit (user):  cd /projects/mwang80/staging/Physio_Data/workzone/ucsf/logs && sbatch ../slurm/ucsf_fix_clock.sbatch <step>
#SBATCH --job-name=ucsf_fix_clock
#SBATCH --partition=xhu40-b.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -euo pipefail
STEP=${1:?step a|b|c|d|e|f|copy}; shift || true
REPO=/mnt/localdata/storage/mxwang/Physio_Data
PY=/mnt/localdata/storage/mxwang/anaconda3/envs/physio_data/bin/python
STORE=/mnt/localdata100tb/physio_data/ucsf
COPY=/projects/xhu40-cdsfm/physio_data/ucsf
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 POLARS_MAX_THREADS=${SLURM_CPUS_PER_TASK}
cd "$REPO"
echo "[fix_clock $STEP] $(date) host=$(hostname) cpus=$SLURM_CPUS_PER_TASK git=$(git rev-parse --short HEAD 2>/dev/null || echo ?)"
case "$STEP" in
  a)    $PY workzone/ucsf/stage_a_wave_windows.py "$@" ;;
  b)    $PY workzone/ucsf/stage_b_adibin.py --dataset ucsf --workers 16 --dst-switch-only --no-resume "$@" ;;
  c)    $PY workzone/ucsf/stage_c_vital.py --workers 16 --dst-switch-only --no-resume "$@" ;;
  d)    $PY workzone/ucsf/stage_d_labs.py --phase 2 --no-resume --workers 16 "$@" ;;
  e)    $PY workzone/ucsf/stage_e_assemble.py --no-resume --workers 8 "$@" ;;
  fm)   rsync -rt --no-g /projects/mwang80/staging/Physio_Data/workzone/ucsf/stage_f_manifest.py $REPO/workzone/ucsf/stage_f_manifest.py
        $PY workzone/ucsf/stage_f_manifest.py --dataset ucsf --keep-splits --exclude-from-train-csv '' "$@"
        cp -f "$REPO/indices/var_registry.json" "$STORE/var_registry.json" ;;
  f)    $PY workzone/ucsf/stage_f_ca.py "$@"
        $PY workzone/ucsf/stage_f_demographics.py
        $PY workzone/ucsf/stage_f_manifest.py --dataset ucsf --keep-splits --exclude-from-train-csv ''
        cp -f "$REPO/indices/var_registry.json" "$STORE/var_registry.json" ;;
  copy) mkdir -p "$COPY"
        rsync -rlt --no-g --partial --info=stats2 "$STORE/" "$COPY/"
        echo "src dirs: $(ls -d ${STORE}/*/ | wc -l)  dest dirs: $(ls -d ${COPY}/*/ | wc -l)" ;;
  *)    echo "unknown step $STEP"; exit 2 ;;
esac
echo "[fix_clock $STEP] done $(date)"
