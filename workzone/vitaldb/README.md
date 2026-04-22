# VitalDB pipeline (workzone/vitaldb)

Target output: `/opt/localdata100tb/physio_data/vitaldb/{caseid:04d}/`

Raw source: `/opt/localdata100tb/UNIPHY_Plus/raw_datasets/vitalDB/`

## Stage order

| Stage | Script | Input | Output | Est. wall |
|-------|--------|-------|--------|-----------|
| A | `stage_a_cohort.py` | `clinical_data.csv` + `.vital` presence | `valid_cohort.parquet` | ~30 s |
| B | `stage_b_wave.py` (16 workers) | `.vital` via `vitaldb` library | `PLETH40.npy`, `II120.npy`, `time_ms.npy`, `meta.json` | ~20–40 min |
| C | `stage_c_vitals.py` (16 workers) | `.vital` Solar8000/* 1-Hz numerics | per-entity `vitals_events.npy` | ~15–30 min |
| D | `stage_d_labs.py` | `lab_data.csv` (928k rows) | per-entity `labs_events.npy` | <2 min |
| E | `stage_e_assemble.py` (8 workers) | time_ms + vitals + labs | `ehr_{baseline,recent,events,future}.npy` | ~5 min |
| F-2 | `stage_f_demographics.py` | cohort + meta | `demographics.csv` | <1 min |
| F-1 | `stage_f_manifest.py` | entity dirs | `manifest.json`, `pretrain_splits.json` (70/15/15 by subjectid), `downstream_splits.json` | ~2 min |
| G | `workzone/common/build_estimation_task.py --spec {lab,vital}_est_full.yaml` | manifest + per-entity EHR | `tasks/*` | <1 min |

## Full run (in tmux)

```bash
source /labs/hulab/mxwang/anaconda3/etc/profile.d/conda.sh && conda activate physio_data
cd /labs/hulab/mxwang/Physio_Data/workzone/vitaldb

python stage_a_cohort.py
python stage_b_wave.py --workers 16 2>&1 | tee logs/stage_b.out
python stage_c_vitals.py --workers 16 2>&1 | tee logs/stage_c.out
python stage_d_labs.py 2>&1 | tee logs/stage_d.out
python stage_e_assemble.py --workers 8 2>&1 | tee logs/stage_e.out
python stage_f_demographics.py
python stage_f_manifest.py 2>&1 | tee logs/stage_f.out
```

## Notes

- **Entity = caseid** (zero-padded to 4 chars, e.g. `0001`). 1 caseid = 1 surgical case.
- **Patient key = subjectid**; splits grouped by subjectid so no patient leaks across train/val/test.
- **Time**: VitalDB uses a fake epoch per case (`dtstart` in `.vital`). We multiply by 1000 for canonical `time_ms`. All EHR events shift by `dtstart`. `casestart_s=0` in `clinical_data.csv` is the relative anchor.
- **Channels**: `SNUADC/PLETH` (500 Hz) → PLETH40 via `resample_poly(2, 25)`. `SNUADC/ECG_II` (500 Hz) → II120 via `resample_poly(6, 25)`.
- **Strict anchor**: `MIN_SECONDS_PRESENT = 30` (entire 30-s window must have 0 % NaN). VitalDB data is clean, so yield stays high (~97 %).
- **Episode bounds**: `anestart_ms` / `aneend_ms` from `clinical_data.csv` (anesthesia window, converted using `dtstart`).
