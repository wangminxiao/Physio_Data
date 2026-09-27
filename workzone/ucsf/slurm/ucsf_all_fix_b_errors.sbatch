#!/bin/bash
# Re-run Stage B (then C) for the entities that errored in the main Stage B run, writing into the REAL store
# but with a separate intermediate dir so the main run's status table / summary are not overwritten.
# Submit from /projects/mwang80/staging/Physio_Data/workzone/ucsf/logs:  sbatch ../slurm/ucsf_all_fix_b_errors.sbatch
#SBATCH --job-name=ucsf_all_fix_b
#SBATCH --partition=xhu40-b.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -euo pipefail
REPO=/mnt/localdata/storage/mxwang/Physio_Data
PY=/mnt/localdata/storage/mxwang/anaconda3/envs/physio_data/bin/python
REAL=$REPO/workzone/outputs/ucsf_all
FIX=$REPO/workzone/outputs/ucsf_all_fix
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
mkdir -p "$FIX"; cp -f "$REAL/valid_wave_window.parquet" "$FIX/"
CFG=$FIX/fix_paths.yaml
printf 'ucsf_all:\n  raw_waveform_dir: /mnt/localdata/storage/UCSF\n  intermediate_dir: %s\n  output_dir: /mnt/localdata100tb/physio_data/ucsf_all\n' "$FIX" > "$CFG"
IDS=$($PY -c "
import polars as pl
st = pl.read_parquet('$REAL/stage_b_status.parquet')
print(','.join(st.filter(pl.col('status') == 'error')['entity_id'].to_list()))")
N=$(echo "$IDS" | tr ',' '\n' | grep -c .)
echo "[fix] $(date) host=$(hostname) errored entities to rerun: $N"
$PY workzone/ucsf/stage_b_adibin.py --config "$CFG" --dataset ucsf_all --entities "$IDS" --workers 4 --no-resume
$PY workzone/ucsf/stage_c_vitals_hf.py --config "$CFG" --dataset ucsf_all --entities "$IDS" --workers 4 --no-resume --zero-anchor adibin_first --abs-tail reanchor
echo "[fix] statuses:"; $PY -c "
import polars as pl
for f in ('stage_b_status.parquet', 'stage_c_hf_status.parquet'):
    st = pl.read_parquet('$FIX/' + f); print(f, st.group_by('status').len().to_dicts())
    if 'error' in st.columns:
        for e in st.filter(pl.col('status') == 'error')['error'].to_list()[:5]: print('   ', str(e)[:150])"
echo "[fix] done $(date)"
