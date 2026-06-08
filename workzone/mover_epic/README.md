# MOVER/EPIC pipeline (workzone/mover_epic)

Target output: `/opt/localdata100tb/physio_data/mover_epic/{LOG_ID}/`

Raw source: `/opt/localdata100tb/UNIPHY_Plus/raw_datasets/MOVER/` (EPIC subset).

Scope: **EPIC only**, peer to `workzone/mover/` (SIS). Same canonical format.

## Stage order

| Stage | Script | Input | Output | Est. wall |
|-------|--------|-------|--------|-----------|
| A | `stage_a_cohort.py` | `EPIC_MRN_PAT_ID.csv` + `EPIC_EMR/EMR/patient_information.csv` + file enumeration across 3 wave dirs | `valid_cohort.parquet` | ~15-30 min (XML enumeration) |
| B | `stage_b_wave.py` (16 workers) | per-LOG_ID XML list | per-LOG_ID `PLETH40.npy` + `II120.npy` + `time_ms.npy` + `meta.json` | ~2-4 h |
| C | `stage_c_flowsheets.py` (v2: parses BP) | `flowsheets_cleaned/flowsheet_part*.csv` (142 GB, 1.44 B rows) | per-LOG_ID `vitals_events.npy` (HR/SpO2/RR/Temp/ETCO2 + SBP/DBP/MAP) | ~5 min (phase 1 streaming) |
| D | `stage_d_labs.py` | `EPIC_EMR/EMR/patient_labs.csv` (29 M rows) | per-LOG_ID `labs_events.npy` | ~10-15 min |
| E | `stage_e_assemble.py` (8 workers) | time_ms + vitals + labs | `ehr_{baseline,recent,events,future}.npy` | ~10-15 min |
| F-2 | `stage_f_demographics.py` | cohort + meta | `demographics.csv` | < 1 min |
| F-1 | `stage_f_manifest.py` | entity dirs | `manifest.json` + `pretrain_splits.json` (MRN-grouped 70/15/15) + `downstream_splits.json` | ~5 min |
| F-3 | `stage_f_demographics_extra.py` | cohort parquet + per-entity `ehr_events.npy` | `demographics_4h.csv` (height/weight + first-4h HR/SpO2/Resp/MAP/ETCO2 + procedure) | < 1 min |
| G | `workzone/common/build_estimation_task.py --spec {lab_est_full,vital_est_full}.yaml` | manifest + per-entity EHR | `tasks/*` | < 1 min each |

## Full run (in tmux)

```bash
source /labs/hulab/mxwang/anaconda3/etc/profile.d/conda.sh && conda activate physio_data
cd /labs/hulab/mxwang/Physio_Data/workzone/mover_epic

python stage_a_cohort.py                                           # 15-30 min
python stage_b_wave.py --workers 16 2>&1 | tee logs/stage_b.out    # 2-4 h
python stage_c_flowsheets.py 2>&1 | tee logs/stage_c.out           # 2-4 h
python stage_d_labs.py 2>&1 | tee logs/stage_d.out                 # 10-15 min
python stage_e_assemble.py --workers 8 2>&1 | tee logs/stage_e.out # 10-15 min
python stage_f_demographics.py
python stage_f_manifest.py 2>&1 | tee logs/stage_f.out
python stage_f_demographics_extra.py                              # F-3: first-4h vitals + height/weight

python ../common/build_estimation_task.py --root /opt/localdata100tb/physio_data/mover_epic \
    --registry ../../indices/var_registry.json --spec ../common/task_specs/lab_est_full.yaml --workers 16
python ../common/build_estimation_task.py --root /opt/localdata100tb/physio_data/mover_epic \
    --registry ../../indices/var_registry.json --spec ../common/task_specs/vital_est_full.yaml --workers 16
```

## Notes

- **Entity = LOG_ID** (1 row per surgical encounter in `patient_information.csv`).
- **Filename → LOG_ID**: XML filename prefix is the 16-hex MRN. Match via `EPIC_MRN_PAT_ID.csv` crosswalk, then pick the LOG_ID whose anesthesia window (`AN_START_DATETIME` ± 1 h buffer) contains the XML file's timestamp.
- **XML schema = SIS**. Same `<cpcArchive><cpc><mg>...</mg></cpc>` structure; same decoder (base64 → int16 → sentinel-mask → gain+offset). About 40 % of EPIC XMLs are DATADOWN placeholders (empty `<measurements/>`) — correctly yield 0 blocks.
- **PLETH prevalence similar to SIS** (~18-40 % of LOG_IDs have the PLETH channel; rest have only GE_ECG / GE_ART). Expected yield from the ~65,729 LOG_IDs ≈ 10-25k entities.
- **BP is parsed (Stage C v2)** — EPIC flowsheet stores BP as `"120/80"` strings; Stage C splits them into SBP (104) / DBP (105) cuff, with arterial-line BP → 110/111/112. Cuff MAP (106) comes from the numeric `MAP (mmHg)` field. Coverage ≈ 54 % of entities (vs SIS 96 %); F-3 fills `avg_sbp_4h` / `avg_dbp_4h` / `avg_map_4h` accordingly.
- **Splits are MRN-grouped** to avoid patient leakage (one MRN can have multiple LOG_IDs = multiple surgical encounters).
