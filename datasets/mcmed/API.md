# MC_MED Data Source Specification

## Source Data

### Raw Clinical Tables (`mc_med_csv/`)
| File | Key Columns | Notes |
|------|-------------|-------|
| `labs.csv` | CSN, Result_time, Component_name, Component_value, Component_units | Lab measurements |
| `numerics.csv` | CSN, Measure, Value, Time | Vital signs (HR, RR, SpO2, SBP, DBP, MAP, Temp) |
| `visits.csv` | MRN, CSN, Age, Gender, Race, Ethnicity, Dx_ICD9, Dx_ICD10 | Demographics + diagnosis |
| `meds.csv` | MRN, Med_ID, Name, Generic_name, Start_date, End_date | Medications |
| `waveform_summary.csv` | CSN, Type, Segments, Duration | Waveform availability per encounter |

### Preprocessed Waveforms (`MC_MED_ECG_PLETH_40hz_120hz_500hz_lab_vital/`)
- NPZ files: `{CSN}_P40_E120_E500_{n_segments}.npz`
- Already segmented at 30 seconds, already resampled:
  - PLETH40: [N_seg, 1200] float16 (40 Hz)
  - II120: [N_seg, 3600] float16 (120 Hz)
  - II500: [N_seg, 15000] float16 (500 Hz)
  - time: [N_seg] datetime64[ms]
  - emb_PLETH40_GPT19M: [N_seg, 512] float32 (pre-computed)
  - ehr_gt, ehr_mask, ehr_trend: [N_seg, 11] (old dense format)

## Variable Mapping (MC_MED name -> var_registry ID)

### Labs (from labs.csv Component_name)
| Component_name | var_id | var_registry name |
|----------------|--------|-------------------|
| POTASSIUM | 0 | Potassium |
| CALCIUM | 1 | Calcium |
| SODIUM | 2 | Sodium |
| GLUCOSE | 3 | Glucose |
| LACTIC ACID / LACTATE | 4 | Lactate |
| CREATININE | 5 | Creatinine |
| BILIRUBIN TOTAL | 6 | Bilirubin |
| PLATELET COUNT | 7 | Platelets |
| WBC | 8 | WBC |
| HEMOGLOBIN | 9 | Hemoglobin |

### Vitals (from numerics.csv Measure)
| Measure | var_id | var_registry name |
|---------|--------|-------------------|
| HR | 100 | HR |
| SpO2 | 101 | SpO2 |
| RR | 102 | RR |
| Temp | 103 | Temperature |
| SBP | 104 | NBPs |
| DBP | 105 | NBPd |
| MAP | 106 | NBPm |

## Patient ID
- MC_MED uses `CSN` (Clinical Service Number) as the encounter-level identifier
- `MRN` is the patient-level ID (one MRN can have multiple CSNs)
- Canonical output directories use CSN as the directory name

## Pipeline
```
python workzone/mcmed/stage1_scan_npz.py        # Inventory NPZ + cross-check CSVs
python workzone/mcmed/stage2_extract_ehr.py      # Extract labs + vitals -> parquet
python workzone/mcmed/stage3_convert_to_canonical.py  # NPZ -> per-patient dirs
python workzone/mcmed/stage4_manifest_splits.py  # Validate + create splits
```
