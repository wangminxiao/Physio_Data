# MC_MED pipeline (workzone/mcmed)

Target output: `/opt/localdata100tb/physio_data/mcmed/{CSN}/`

Raw source: `/opt/localdata100tb/UNIPHY_Plus/raw_datasets/physionet.org/files/mc-med/1.0.1/data/`

## Stage order

| Stage | Script | Input | Output | Est. wall |
|-------|--------|-------|--------|-----------|
| A     | `stage_a_cohort.py` | `waveform_summary.csv` + `visits.csv` | `workzone/outputs/mcmed/valid_cohort.parquet` | < 1 min |
| B     | `stage_b_wave.py` (24 workers) | Pleth / II .dat per CSN | `PLETH40.npy`, `II120.npy`, `time_ms.npy`, `meta.json` | ~8–12 h full |
| C     | `stage_c_vitals.py` | `numerics.csv` (2.6 GB) | per-CSN `vitals_events.npy` | ~20–30 min |
| D     | `stage_d_labs.py` | `labs.csv` (752 MB) | per-CSN `labs_events.npy` | ~10–15 min |
| E     | `stage_e_assemble.py` (8 workers) | time_ms + vitals + labs | `ehr_{baseline,recent,events,future}.npy` | ~15 min |
| F-2   | `stage_f_demographics.py` | cohort parquet + meta | `demographics.csv` | < 1 min |
| F-1   | `stage_f_manifest.py` | all entity dirs + MC_MED split csvs | `manifest.json`, `pretrain_splits{,_chrono}.json`, `downstream_splits.json` | ~5 min |
| G     | `workzone/common/build_estimation_task.py --root /opt/localdata100tb/physio_data/mcmed --registry ../../indices/var_registry.json --coverage-only` then `--spec workzone/common/task_specs/lab_est_full.yaml` | manifest + per-entity EHR | `tasks/coverage.json`, `tasks/lab_est_full/*`, `tasks/vital_est_full/*` | < 5 min |

## Smoke test (do this before any full run)

```bash
cd /labs/hulab/mxwang/Physio_Data/workzone/mcmed

# Stage A: full (fast)
python stage_a_cohort.py

# Stage B: 3 entities, 2 workers
python stage_b_wave.py --limit 3 --workers 2

# Stage C phase 1 is a full table scan — skip for smoke (expensive).
# Instead, build a tiny combined parquet by hand OR run full phase 1
# once and then iterate phase 2 only with --limit.
python stage_c_vitals.py --phase 2 --limit 3   # requires phase 1 already run

# Stage D: phase 2 only on same 3 entities
python stage_d_labs.py --phase 2 --limit 3

# Stage E on the 3
python stage_e_assemble.py --limit 3 --workers 2

# F-2 + F-1 (fine on partial output — manifest only reports what exists)
python stage_f_demographics.py
python stage_f_manifest.py
```

## Full run

```bash
# Assume all stages are ready; run in tmux.
tmux new -s mcmed_pipe
cd /labs/hulab/mxwang/Physio_Data/workzone/mcmed

python stage_a_cohort.py                                     # ~30 s
python stage_b_wave.py --workers 24 2>&1 | tee logs/stage_b.out  # ~8–12 h
python stage_c_vitals.py --phase 1 2>&1 | tee logs/stage_c1.out  # ~15 min
python stage_c_vitals.py --phase 2 2>&1 | tee logs/stage_c2.out  # ~10 min
python stage_d_labs.py --phase 1 2>&1 | tee logs/stage_d1.out    # ~10 min
python stage_d_labs.py --phase 2 2>&1 | tee logs/stage_d2.out    # ~5  min
python stage_e_assemble.py --workers 8 2>&1 | tee logs/stage_e.out   # ~15 min
python stage_f_demographics.py
python stage_f_manifest.py 2>&1 | tee logs/stage_f.out

# Downstream task (Stage G): shared script
python ../common/build_estimation_task.py \
    --root /opt/localdata100tb/physio_data/mcmed \
    --registry ../../indices/var_registry.json \
    --coverage-only    # emits tasks/coverage.json

# Then pick a spec (or write new) and build the task:
python ../common/build_estimation_task.py \
    --root /opt/localdata100tb/physio_data/mcmed \
    --registry ../../indices/var_registry.json \
    --spec ../common/task_specs/lab_est_full.yaml
```

## Splits

MC_MED ships pre-defined **patient-safe** splits. We adopt them verbatim:
- `split_random_{train,val,test}.csv` (80/10/10) → `pretrain_splits.json`
- `split_chrono_{train,val,test}.csv`            → `pretrain_splits_chrono.json`

No new shuffling, so results stay comparable to benchmarks in the MC_MED ICML paper.
