# MIMIC-III Dataset API

## Data Sources

### Waveform

| Field | Value |
|-------|-------|
| Format | WFDB (PhysioBank matched subset) |
| Location | `/labs/hulab/MIMIC_waveform_matched_subset/physionet.org/files/mimic3wdb-matched/1.0/` |
| Organization | `/{SUBJECT_ID}/` -> record directories with `.hea` + `.dat` files |
| Patient ID field | `SUBJECT_ID` (parsed from directory name) |
| Time reference | Record header `base_time` + `base_date` (UTC) |

### EHR - Clinical Tables

| Table | Location | Format | Key Columns |
|-------|----------|--------|-------------|
| Lab results | `/labs/hulab/MIMICIII-v1.4/LABEVENTS.csv.gz` | CSV.gz | SUBJECT_ID, HADM_ID, ITEMID, CHARTTIME, VALUENUM |
| Vitals (chart) | `/labs/hulab/MIMICIII-v1.4/CHARTEVENTS.csv.gz` | CSV.gz | SUBJECT_ID, HADM_ID, ITEMID, CHARTTIME, VALUENUM |
| Patients | `/labs/hulab/MIMICIII-v1.4/PATIENTS.csv` | CSV | SUBJECT_ID, GENDER, DOB, DOD |
| Admissions | `/labs/hulab/MIMICIII-v1.4/ADMISSIONS.csv` | CSV | SUBJECT_ID, HADM_ID, ADMITTIME, DISCHTIME, ETHNICITY |
| Diagnoses | `/labs/hulab/MIMICIII-v1.4/DIAGNOSES_ICD.csv` | CSV | SUBJECT_ID, HADM_ID, ICD9_CODE, SEQ_NUM |

### Patient ID Linkage

```
Waveform: SUBJECT_ID (directory name)
EHR:      SUBJECT_ID + HADM_ID (admission-level)
Linkage:  Direct match on SUBJECT_ID. One subject may have multiple HADM_IDs.
          Each admission is treated as a separate patient record.
Output patient_id format: {SUBJECT_ID}_{HADM_ID}
```

### Time Format

| Source | Format | Notes |
|--------|--------|-------|
| Waveform | UTC datetime from WFDB header | `base_time` + `base_date` fields |
| CHARTEVENTS | `CHARTTIME` string `YYYY-MM-DD HH:MM:SS` | Assumed local (same as waveform after MIMIC de-identification) |
| LABEVENTS | `CHARTTIME` string `YYYY-MM-DD HH:MM:SS` | Same |

No timezone offset needed (both sources use MIMIC's shifted time reference).

---

## Waveform Channels to Extract

| Source Channel | Source Rate (Hz) | Target Channel Name | Target Rate (Hz) | samples_per_seg (30s) | Notes |
|---------------|-----------------|--------------------|-----------------|-----------------------|-------|
| PLETH | 125 (varies) | PLETH40 | 40 | 1200 | PPG / plethysmography. Primary modality. |
| II | 125-500 (varies) | II120 | 120 | 3600 | ECG Lead II. Most common ECG lead. |
| II | 125-500 (varies) | II500 | 500 | 15000 | ECG Lead II at high resolution. |

**Channel selection logic**: Read WFDB header `sig_name` field. Match by exact name.
Fall back: if "II" not found, try "ECG", "MLII". If "PLETH" not found, try "SpO2", "PPG".

**Resampling**: `scipy.signal.resample_poly(signal, up, down)` where up/down are
computed from `gcd(source_rate, target_rate)`.

---

## EHR Variables to Extract

### Labs (from LABEVENTS)

| var_id | Variable | ITEMID(s) | Unit | Physiological Range | Notes |
|--------|----------|-----------|------|--------------------:|-------|
| 0 | Potassium | 50971, 50822 | mEq/L | 2.5 - 7.0 | 50971=serum, 50822=blood gas |
| 1 | Calcium | 50893 | mg/dL | 6.0 - 12.0 | Total calcium |
| 2 | Sodium | 50983, 50824 | mEq/L | 120 - 160 | 50983=serum, 50824=blood gas |
| 3 | Glucose | 50931, 50809 | mg/dL | 30 - 600 | 50931=serum, 50809=blood |
| 4 | Lactate | 50813 | mmol/L | 0.3 - 20.0 | Arterial blood gas |
| 5 | Creatinine | 50912 | mg/dL | 0.2 - 15.0 | Serum creatinine |

### Vitals (from CHARTEVENTS)

| var_id | Variable | ITEMID(s) | Unit | Physiological Range | Notes |
|--------|----------|-----------|------|--------------------:|-------|
| 6 | NBPs | 220179 | mmHg | 50 - 250 | Non-invasive BP Systolic (MetaVision) |
| 7 | NBPd | 220180 | mmHg | 20 - 150 | Non-invasive BP Diastolic |
| 8 | NBPm | 220181 | mmHg | 30 - 200 | Non-invasive BP Mean |

### Future variables (not yet extracted, reserved IDs)

| var_id | Variable | ITEMID(s) | Unit | Source Table | Notes |
|--------|----------|-----------|------|-------------|-------|
| 9 | HR | 220045 | bpm | CHARTEVENTS | Heart rate |
| 10 | SpO2 | 220277 | % | CHARTEVENTS | Pulse oximetry |
| 11 | RR | 220210 | /min | CHARTEVENTS | Respiratory rate |
| 12 | Bilirubin | 50885 | mg/dL | LABEVENTS | Total bilirubin |
| 13 | INR | 51237 | ratio | LABEVENTS | Coagulation |
| 14 | Temperature | 223761 | F | CHARTEVENTS | Fahrenheit, convert to C |

### Filtering Rules

```
- VALUENUM must not be null
- VALUENUM must be within Physiological Range (drop obvious errors)
- For labs with multiple ITEMIDs: combine, no priority (both are valid)
- Deduplicate: if same (SUBJECT_ID, HADM_ID, CHARTTIME, ITEMID), keep first
```

---

## Demographics to Extract (from PATIENTS + ADMISSIONS + DIAGNOSES_ICD)

| Field | Source | Column | Encoding |
|-------|--------|--------|----------|
| Age | PATIENTS + ADMISSIONS | DOB, ADMITTIME | years at admission (cap at 89 per MIMIC policy) |
| Gender | PATIENTS | GENDER | M=0, F=1 |
| Ethnicity | ADMISSIONS | ETHNICITY | categorical (group rare categories into "OTHER") |
| Insurance | ADMISSIONS | INSURANCE | categorical |
| Primary ICD-9 | DIAGNOSES_ICD | ICD9_CODE where SEQ_NUM=1 | string code |

---

## Processing Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Segment duration | 30 seconds | Matches existing pipeline, aligns with Physio_HNET row structure |
| Min segments per patient | 10 | Skip very short recordings (< 5 min) |
| Max NaN ratio per channel | 0.20 | Flag but don't drop (downstream handles NaN) |
| Normalization method | Robust quantile (p0 - p100) | Same as existing stp3_1_npz_prepare.py |
| Train/test split | 80/20, patient-level | Stratified by gender + age bucket + lab density |
| Split seed | 42 | Reproducibility |

---

## Output Specification

```
datasets/mimic3/
├── processed/
│   └── {SUBJECT_ID}_{HADM_ID}/
│       ├── PLETH40.npy          [N_seg, 1200]  float16
│       ├── II120.npy            [N_seg, 3600]  float16
│       ├── II500.npy            [N_seg, 15000] float16
│       ├── time_ms.npy          [N_seg]         int64
│       ├── ehr_events.npy       [N_events]      structured
│       └── meta.json
├── ehr/
│   ├── labs.parquet
│   ├── vitals.parquet
│   └── demographics.parquet
└── indices/
    ├── manifest.json
    ├── pretrain_splits.json
    └── downstream_splits.json
```

## References

- MIMIC-III Clinical Database: https://physionet.org/content/mimiciii/1.4/
- MIMIC-III Waveform Database Matched Subset: https://physionet.org/content/mimic3wdb-matched/1.0/
- MIMIC-III documentation: https://mimic.mit.edu/docs/iii/
- Existing preprocessing code: `/home/mxwan/workspace/MIMIC-III-preparation-for-UNIPHY_Plus/`
