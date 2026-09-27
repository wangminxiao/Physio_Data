#!/bin/bash
# OPTIONAL backup/copy of the finished ucsf_all store from the lab-node volume (no backup, node-local) to BeeGFS,
# where every DREAM node can read it for training. ~3 TB; run only after verify F2 passed and only if the user
# decided the destination (default below: the PI project share; ~6.4 TiB were free on 2026-09-10).
# Submit from /projects/mwang80/staging/Physio_Data/workzone/ucsf/logs:  sbatch ../slurm/ucsf_all_copy_to_projects.sbatch [DEST]
#SBATCH --job-name=ucsf_all_copy
#SBATCH --partition=xhu40-b.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -euo pipefail
SRC=/mnt/localdata100tb/physio_data/ucsf_all/
DEST=${1:-/projects/xhu40-cdsfm/physio_data/ucsf_all/}
echo "[copy] $(date) host=$(hostname) $SRC -> $DEST"
df -h "$(dirname "${DEST%/}")" | tail -1
mkdir -p "$DEST"
# -rlt (no -a/-p/-g: BeeGFS group quotas follow ownership; keep the destination group), --partial to resume, no compression on the local link
rsync -rlt --no-g --partial --info=progress2,stats2 "$SRC" "$DEST"
echo "[copy] verifying entity counts"; echo "src dirs: $(ls -d ${SRC}*/ | wc -l)  dest dirs: $(ls -d ${DEST}*/ | wc -l)"
echo "[copy] done $(date)"
