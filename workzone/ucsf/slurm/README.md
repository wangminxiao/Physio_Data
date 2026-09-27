# UCSF all-raw pipeline — Slurm scripts

All jobs run on the lab node `xhu40-n01` (partition `xhu40-b.q`), the only DREAM node that mounts
`/mnt/localdata` (raw data, repo clone) and `/mnt/localdata100tb` (output store). The clone volume is
not visible on the login node, so the scripts are **submitted from the BeeGFS mirror** and `cd` into the
clone at run time:

```
MIRROR=/projects/mwang80/staging/Physio_Data          # rsync target of the local repo (workzone/ucsf, configs, physio_data, indices, datasets/ucsf)
cd $MIRROR/workzone/ucsf/logs                          # %x_%j.out/.err land here (BeeGFS, readable from the login node)
```

| Script | Stage | Resources | Notes |
|---|---|---|---|
| `ucsf_all_a_all_raw.sbatch` | A enumerate wave cycles | 16 c / 16 GB / 4 h | ~22 min for 70 folders; per-folder shards resume |
| `ucsf_all_b_adibin.sbatch` | B waveforms | 16 c / 80 GB / 7 d | `--dataset ucsf_all --workers 16`; resume skips entities with `meta.json` |
| `ucsf_all_c_vitals_hf.sbatch` | C dense vitals | 16 c / 32 GB / 12 h | `--zero-anchor adibin_first --abs-tail reanchor` |
| `ucsf_all_f_manifest.sbatch` | F manifest + splits | 2 c / 16 GB / 6 h | validates every entity |
| `ucsf_all_verify.sbatch <a|b|c|f>` | gate after each stage | 2 c / 16 GB / 3 h | exits non-zero on FAIL |

Chain (submitted once A's gate passed):

```bash
jB=$(sbatch --parsable ../slurm/ucsf_all_b_adibin.sbatch)
jvB=$(sbatch --parsable -J ucsf_all_verify_b --dependency=afterok:$jB  ../slurm/ucsf_all_verify.sbatch b)
jC=$(sbatch --parsable --dependency=afterok:$jvB ../slurm/ucsf_all_c_vitals_hf.sbatch)
jvC=$(sbatch --parsable -J ucsf_all_verify_c --dependency=afterok:$jC  ../slurm/ucsf_all_verify.sbatch c)
jF=$(sbatch --parsable --dependency=afterok:$jvC ../slurm/ucsf_all_f_manifest.sbatch)
jvF=$(sbatch --parsable -J ucsf_all_verify_f --dependency=afterok:$jF  ../slurm/ucsf_all_verify.sbatch f)
```

No e-mail: the scripts carry no `--mail-type` (user preference 2026-09-09).

Node etiquette: at most half of the node per job (≤ 22 cores / 88 GB), no GPU request, `rc check dream`
before resizing anything. Gate reports: `workzone/outputs/ucsf_all/verify_stage_<s>.json` in the clone.

## Clock-convention fix (2026-09-26, `datasets/ucsf/ALIGNMENT.md`)

| Script | Steps | Notes |
|---|---|---|
| `ucsf_all_fix_dst.sbatch <b|c|g|f|copy>` | Stage B v2 / C v2 on the 610 cycles that straddle a DST switch (`--dst-switch-only --no-resume`), ehr_hf for them, manifest refresh with `--keep-splits`, rsync to BeeGFS | 16 c / 64 GB; chain with `afterok`; gate C (`ucsf_all_verify.sbatch c`) after step c |
| `ucsf_fix_clock.sbatch <a|b|c|d|e|f|copy>` | CA-cohort store: Stage A (admission windows), B/C on straddling cycles, labs phase 2 + assembly for all entities with `clock.ehr_wall_to_grid_ms`, `ca_prediction` task from Code Blue times, manifest (`--keep-splits`), rsync | needs `workzone/outputs/ucsf_all/codeblue_first_event.parquet` (`explore/convert_codeblue_xlsx.py`, S4M env) |
| `ucsf_all_g_ehr_hf.sbatch` | Stage G stand-alone | accepts `--entity-file`, `--dst-switch-only` |

