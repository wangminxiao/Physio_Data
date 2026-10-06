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
#   MODE=delete     remove *_mmap.npy only (1.76 TB), and only if $VERIFIED_FLAG exists (written after the
#                   Physio_HNET reader parity: same windows from mmap and store). __meta.json (seg_list, read by
#                   Stages B/D/D2) and __numerics.npz stay where they are, so nothing needs archiving.
set -euo pipefail
SRC=/ocean/projects/med250003p/shared/pretrain_wav_v2
VERIFIED_FLAG=${VERIFIED_FLAG:-/ocean/projects/med250003p/mwang11/Physio_Data/workzone/outputs/mladi/READER_PARITY_OK}
echo "[$(date)] mode=${MODE:?} src=$SRC"
case "$MODE" in
  inventory)
    find "$SRC" -maxdepth 1 -type f -printf "%u %s %f\n" | awk '{
        t = ($3 ~ /_mmap\.npy$/) ? "mmap" : (($3 ~ /__meta\.json$/) ? "meta" : "other");
        n[t]++; s[t] += $2; o[$1]++ } END {
        for (k in n) printf "%-6s files %8d  size %.3f TB\n", k, n[k], s[k] / 1e12;
        for (u in o) printf "owner %s %d\n", u, o[u] }'
    find "$SRC" -maxdepth 1 -type f ! -name "*_mmap.npy" ! -name "*__meta.json" -printf "%f\n" | sed -n 1,20p
    find "$SRC" -mindepth 1 -maxdepth 1 -type d -printf "dir %f\n" | sed -n 1,10p ;;
  delete)
    [[ -f "$VERIFIED_FLAG" ]] || { echo "no reader parity flag $VERIFIED_FLAG"; exit 1; }
    n=$(find "$SRC" -maxdepth 1 -type f -name "*_mmap.npy" | wc -l); echo "deleting $n mmap files"
    find "$SRC" -maxdepth 1 -type f -name "*_mmap.npy" -delete
    echo "left: $(find "$SRC" -maxdepth 1 -type f | wc -l) files ($(find "$SRC" -maxdepth 1 -name "*__meta.json" | wc -l) __meta.json)" ;;
  *) echo "MODE?"; exit 2 ;;
esac
echo "[$(date)] done"
