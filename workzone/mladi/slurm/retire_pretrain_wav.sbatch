#!/usr/bin/env bash
#SBATCH --job-name=mladi_retire
#SBATCH --account=med250003p
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1900M
#SBATCH --time=08:00:00
#SBATCH --output=/jet/home/mwang11/Physio_Data/workzone/logs/mladi_retire-%j.out
#SBATCH --error=/jet/home/mwang11/Physio_Data/workzone/logs/mladi_retire-%j.out
# Retire pretrain_wav_v2 (superseded by the canonical store; segment i == mmap row i).
#   MODE=inventory  counts / sizes / owners by file type (read-only)
#   MODE=archive    copy every file that is NOT a *_mmap.npy (seg_list __meta.json etc.) to $ARCH (read-only on the source)
#   MODE=delete     remove *_mmap.npy only, and only if $ARCH/ARCHIVE_OK exists and $VERIFIED_FLAG exists
set -euo pipefail
SRC=/ocean/projects/med250003p/shared/pretrain_wav_v2
ARCH=/ocean/projects/med250003p/mwang11/Physio_Data/workzone/outputs/mladi/pretrain_wav_v2_meta
VERIFIED_FLAG=${VERIFIED_FLAG:-$ARCH/READER_PARITY_OK}
echo "[$(date)] mode=${MODE:?} src=$SRC"
case "$MODE" in
  inventory)
    find "$SRC" -maxdepth 1 -type f -printf "%u %s %f\n" | awk '{
        t = ($3 ~ /_mmap\.npy$/) ? "mmap" : (($3 ~ /__meta\.json$/) ? "meta" : "other");
        n[t]++; s[t] += $2; o[$1]++ } END {
        for (k in n) printf "%-6s files %8d  size %.3f TB\n", k, n[k], s[k] / 1e12;
        for (u in o) printf "owner %s %d\n", u, o[u] }'
    find "$SRC" -maxdepth 1 -type f ! -name "*_mmap.npy" ! -name "*__meta.json" -printf "%f\n" | head -20
    find "$SRC" -mindepth 1 -maxdepth 1 -type d -printf "dir %f\n" | head ;;
  archive)
    mkdir -p "$ARCH"
    rsync -rlt --no-g --exclude "*_mmap.npy" "$SRC/" "$ARCH/"
    n_src=$(find "$SRC" -maxdepth 1 -type f ! -name "*_mmap.npy" | wc -l); n_arc=$(find "$ARCH" -maxdepth 1 -type f ! -name "ARCHIVE_OK" ! -name "READER_PARITY_OK" | wc -l)
    echo "non-mmap files: source $n_src, archive $n_arc"
    if [[ "$n_src" == "$n_arc" ]]; then date > "$ARCH/ARCHIVE_OK"; echo ARCHIVE_OK; else echo "ARCHIVE MISMATCH"; exit 1; fi ;;
  delete)
    [[ -f "$ARCH/ARCHIVE_OK" ]] || { echo "no ARCHIVE_OK"; exit 1; }
    [[ -f "$VERIFIED_FLAG" ]] || { echo "no reader parity flag $VERIFIED_FLAG"; exit 1; }
    n=$(find "$SRC" -maxdepth 1 -type f -name "*_mmap.npy" | wc -l); echo "deleting $n mmap files"
    find "$SRC" -maxdepth 1 -type f -name "*_mmap.npy" -delete
    echo "left: $(find "$SRC" -maxdepth 1 -type f | wc -l) files" ;;
  *) echo "MODE?"; exit 2 ;;
esac
echo "[$(date)] done"
