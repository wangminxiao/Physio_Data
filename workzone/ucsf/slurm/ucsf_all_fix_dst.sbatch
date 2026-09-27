#!/bin/bash
# UCSF all-raw store — rerun of the cycles that straddle a DST switch on the GE calendar (610 of 37,985) with the
# UTC-continuous grid (Stage B v2 / C v2), then ehr_hf, manifest refresh (splits kept) and BeeGFS sync.
# Steps (one job each, chain with afterok):  b | c | g | f | copy
# Submit (user):  cd /projects/mwang80/staging/Physio_Data/workzone/ucsf/logs && sbatch ../slurm/ucsf_all_fix_dst.sbatch <step>
#SBATCH --job-name=ucsf_all_fix_dst
#SBATCH --partition=xhu40-b.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -euo pipefail
STEP=${1:?step b|c|g|f|copy}; shift || true
REPO=/mnt/localdata/storage/mxwang/Physio_Data
PY=/mnt/localdata/storage/mxwang/anaconda3/envs/physio_data/bin/python
STORE=/mnt/localdata100tb/physio_data/ucsf_all
COPY=/projects/xhu40-cdsfm/physio_data/ucsf_all
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 POLARS_MAX_THREADS=${SLURM_CPUS_PER_TASK}
cd "$REPO"
echo "[fix_dst $STEP] $(date) host=$(hostname) cpus=$SLURM_CPUS_PER_TASK git=$(git rev-parse --short HEAD 2>/dev/null || echo ?)"
case "$STEP" in
  b)    $PY workzone/ucsf/stage_b_adibin.py --dataset ucsf_all --workers 16 --dst-switch-only --no-resume "$@" ;;
  c)    $PY workzone/ucsf/stage_c_vitals_hf.py --dataset ucsf_all --workers 16 --zero-anchor adibin_first --abs-tail reanchor --dst-switch-only --no-resume "$@" ;;
  g)    $PY workzone/ucsf/stage_g_ehr_hf.py --dataset ucsf_all --cadence pair --include-nbp --workers 8 --dst-switch-only --no-resume "$@" ;;
  f)    $PY workzone/ucsf/stage_f_manifest.py --dataset ucsf_all --keep-splits "$@"
        $PY workzone/ucsf/verify_stage_f.py --dataset ucsf_all
        cp -f "$REPO/indices/var_registry.json" "$STORE/var_registry.json" ;;
  copy) rsync -rlt --no-g --partial --info=stats2 "$STORE/" "$COPY/"
        cp -f "$REPO/indices/var_registry.json" "$COPY/var_registry.json"
        echo "src dirs: $(ls -d ${STORE}/*/ | wc -l)  dest dirs: $(ls -d ${COPY}/*/ | wc -l)" ;;
  *)    echo "unknown step $STEP"; exit 2 ;;
esac
echo "[fix_dst $STEP] done $(date)"
