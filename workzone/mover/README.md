# MOVER/SIS pipeline (workzone/mover)

Target output: `/opt/localdata100tb/physio_data/mover/{PID}/`

Raw source: `/opt/localdata100tb/UNIPHY_Plus/raw_datasets/MOVER/`

Scope: **SIS only** (sis_wave_v2 + EMR/*.csv). EPIC waves 1/2/4 are deferred.

## Stage order

| Stage | Script | Input | Output | Est. wall |
|-------|--------|-------|--------|-----------|
| A | `stage_a_cohort.py` | `EMR/patient_information.csv` + scan Waveforms tree | `workzone/outputs/mover/valid_cohort.parquet` | ~2 min (tree scan is I/O-bound) |
| B | `stage_b_wave.py` (16 workers) | SIS XML per PID | `PLETH40.npy`, `II120.npy`, `time_ms.npy`, `meta.json` | ~2–4 h (18k PIDs × ~18 XMLs each) |
| C | `stage_c_vitals.py` | `EMR/patient_vitals.csv` (3.6 M rows, wide) | per-PID `vitals_events.npy` | ~10 min |
| D | `stage_d_labs.py` | `EMR/patient_labs.csv` (14k rows, wide) | per-PID `labs_events.npy` | <2 min |
| E | `stage_e_assemble.py` (8 workers) | time_ms + vitals + labs | `ehr_{baseline,recent,events,future}.npy` | ~5 min |
| F-2 | `stage_f_demographics.py` | cohort parquet + meta | `demographics.csv` | <1 min |
| F-1 | `stage_f_manifest.py` | entity dirs | `manifest.json`, `pretrain_splits.json`, `downstream_splits.json` | ~2 min |
| G | `workzone/common/build_estimation_task.py --spec {lab_est_full, vital_est_full}.yaml` | manifest + per-entity EHR | `tasks/*` | <1 min each |

## Full run (in tmux)

```bash
source /labs/hulab/mxwang/anaconda3/etc/profile.d/conda.sh && conda activate physio_data
cd /labs/hulab/mxwang/Physio_Data/workzone/mover

python stage_a_cohort.py                                           # 2 min
python stage_b_wave.py --workers 16 2>&1 | tee logs/stage_b.out    # ~2–4 h
python stage_c_vitals.py 2>&1 | tee logs/stage_c.out               # ~10 min
python stage_d_labs.py 2>&1 | tee logs/stage_d.out                 # <2 min
python stage_e_assemble.py --workers 8 2>&1 | tee logs/stage_e.out # ~5 min
python stage_f_demographics.py
python stage_f_manifest.py 2>&1 | tee logs/stage_f.out

# Downstream tasks via shared builder
python ../common/build_estimation_task.py --root /opt/localdata100tb/physio_data/mover \
    --registry ../../indices/var_registry.json --spec ../common/task_specs/lab_est_full.yaml --workers 16
python ../common/build_estimation_task.py --root /opt/localdata100tb/physio_data/mover \
    --registry ../../indices/var_registry.json --spec ../common/task_specs/vital_est_full.yaml --workers 16
```

## Notes

- **Entity = PID**. One PID = one surgery per `patient_information.csv`. No MRN grouping needed — PID is patient-level and we have 1 encounter/patient in SIS.
- **Time convention**: XML `<cpc datetime="Z">` is authoritative UTC. `patient_information.csv` naive times assumed America/Los_Angeles → UTC.
- **Channels extracted**: `PLETH` (100 Hz → 40 Hz) and `ECG1` (300 Hz → 120 Hz). The GE_ART / GE_ECG / INVP1 duplicates are ignored in v1 (would need new var_registry IDs).
- **Vitals**: 5 var_ids (100, 101, 104, 105, 106). No RR or Temperature in SIS `patient_vitals.csv`.
- **Labs**: 9 var_ids (0, 1, 2, 3, 9, 13, 14, 15, 16). BE (base excess) dropped — not in registry.
- **Episode bounds**: OR_start / OR_end from `patient_information.csv`. Surgery_start / Surgery_end are narrower and stored in meta.json for reference.
