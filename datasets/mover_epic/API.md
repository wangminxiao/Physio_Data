# MOVER/EPIC Data Source Specification

MOVER = UCI's **M**ultimodal **O**perating-room **V**ital and **E**HR **R**ecord
dataset. This spec covers the **EPIC subset** (~2018-2022 surgeries). SIS is a
peer dataset at `datasets/mover/` with its own API.

## Paths (on bedanalysis)

- **Raw dataset root**: `/opt/localdata100tb/UNIPHY_Plus/raw_datasets/MOVER/`
- **EPIC raw waveforms** (3 waves, **~926 k XMLs total, 1.26 TB**):
  - `epic_wave_1_v2/UCI_deidentified_part1_EPIC_07_22/Waveforms/{suffix}/{MRN}{CB|IP}-{YYYY-MM-DD-HH-MM-SS-mmm}Z.xml` (~302 k XMLs)
  - `epic_wave_2_v2/UCI_deidentified_part2_EPIC_08_10/Waveforms/...` (~304 k)
  - `epic_wave_3_v2/UCI_deidentified_part4_EPIC_11_28/Waveforms/...` (~320 k)
- **EPIC EHR tables**: `EPIC_EMR/EMR/patient_{information, visit, labs, medications, history, lda, coding, post_op_complications, procedure events}.csv`
- **Flowsheets**: `flowsheets_cleaned/flowsheet_part{1..19}.csv` (**142 GB, 1.44 B rows**, long format `FLO_NAME` / `FLO_DISPLAY_NAME` / `MEAS_VALUE`)
- **Crosswalk**: `EPIC_MRN_PAT_ID.csv` (65,729 rows: `LOG_ID, PAT_ID, MRN`, one row per LOG_ID)
- **Canonical output**: `/opt/localdata100tb/physio_data/mover_epic/{LOG_ID}/`

## Entity identifier

- `entity_id = str(LOG_ID)` (16-hex). One LOG_ID = one surgical encounter. 65,729 LOG_IDs total.
- `MRN` is patient-level (splits group by MRN to avoid leakage; one MRN can have multiple LOG_IDs).

## Waveform → LOG_ID attribution (non-trivial)

XML filename: `{16-hex-PAT_ID}{CB|IP}-{YYYY-MM-DD-HH-MM-SS-mmm}Z.xml`
- First 16 chars = **PAT_ID** (the patient-level key in `EPIC_MRN_PAT_ID.csv`, NOT the MRN). Verified empirically: 30/30 sample XML prefixes are in the crosswalk `PAT_ID` column, 0/30 match the `MRN` column. The raw files' directory naming calls these "MRN"-like but they're actually de-identified PAT_IDs.
- `CB` or `IP` = outpatient/inpatient class (informational).
- Timestamp is UTC (`Z` suffix) at 3-decimal millisecond precision.

Attribution algorithm (Stage A):
1. For each XML, parse filename → (PAT_ID, file_datetime_utc).
2. Look up PAT_ID in crosswalk → candidate LOG_IDs (one PAT_ID often has multiple encounters over time — 65k LOG_IDs / 40k unique PAT_IDs ≈ 1.66 encounters/patient).
3. Keep the (XML, LOG_ID) pair only when `file_datetime` falls in `[AN_START_DATETIME − 1h, AN_STOP_DATETIME + 1h]`.
4. XMLs outside every encounter's AN window are dropped (non-OR data — ICU, unrelated).

First full run: 926,147 XMLs → 2,628,786 (XML × LOG_ID) candidate pairs → 880,225 pairs in an AN window → **34,437 LOG_IDs** with at least 1 attributed XML (median 20 XMLs each).

## XML decoder

Same `<cpcArchive><cpc><mg>...</mg></cpc>` schema as SIS; same `decode_wave` function
(base64 → little-endian int16 → sentinel-mask → `* gain + offset`). **~40 % of EPIC XMLs are
DATADOWN placeholders** (empty `<measurements/>`) — these correctly yield 0 blocks.

## Waveform channels extracted

| File          | Canonical | Source (EPIC XML `<mg name="...">`) | Resample |
|---------------|-----------|-------------------------------------|----------|
| `PLETH40.npy` | 40 Hz     | `PLETH` @ 100 Hz                    | `resample_poly(2, 5)` |
| `II120.npy`   | 120 Hz    | `ECG1`  @ 300 Hz                    | `resample_poly(2, 5)` |

Ignored duplicates: `GE_ART`, `GE_ECG` (at 180 Hz), `INVP1` (100 Hz). Like SIS, only
LOG_IDs whose XMLs contain the `PLETH` channel pass the anchor.

## Time convention

- **XML `<cpc datetime="Z">`** = authoritative UTC.
- **`patient_information.csv`** (`HOSP_ADMSN_TIME`, `IN_OR_DTTM`, `AN_START_DATETIME`, etc.): naive `MM/DD/YY HH:MM` → **`America/Los_Angeles` → UTC**.
- **`flowsheet_part*.csv`** (`RECORDED_TIME`): naive `YYYY-MM-DD HH:MM:SS` → LA → UTC.
- **`patient_labs.csv`** (`Collection Datetime`): naive `YYYY-MM-DD HH:MM:SS` → LA → UTC.

## EHR variables extracted

### Vitals (from flowsheets, long format)

| `FLO_NAME` (stripped) | `FLO_DISPLAY_NAME` | var_id |
|-----------------------|---------------------|--------|
| Vital Signs           | Pulse               | 100    |
| Vital Signs           | SpO2                | 101    |
| Vital Signs           | Resp                | 102    |
| Vital Signs           | Temp                | 103    |
| Vital Signs           | MAP (mmHg)          | 106    |
| Devices Testing Template | Heart Rate       | 100    |
| Devices Testing Template | SpO2             | 101    |
| Devices Testing Template | Resp             | 102    |
| Devices Testing Template | ETCO2 (mmHg)     | 116    |
| ED Vitals             | Pulse / SpO2 / Resp / Temp | 100/101/102/103 |

**BP is NOT parsed** — the `"120/80"` string would need a separate splitting stage.
Only `MAP (mmHg)` is extracted as the BP-like vital in v1.

### Labs (from `patient_labs.csv`, long format)

| `Lab Name` (as EPIC writes it)           | var_id |
|------------------------------------------|--------|
| Potassium                                | 0      |
| Calcium / Calcium.ionized                | 1      |
| Sodium                                   | 2      |
| Glucose                                  | 3      |
| Creatinine                               | 5      |
| Bilirubin                                | 6      |
| Platelets                                | 7      |
| Hemoglobin                               | 9      |
| Coagulation tissue factor induced.INR    | 10     |
| Urea nitrogen                            | 11     |
| Albumin                                  | 12     |
| pH                                       | 13     |
| Oxygen                                   | 14 (paO2) |
| Carbon dioxide / Bicarbonate             | 16 (HCO3) |
| Aspartate aminotransferase               | 17     |
| Alanine aminotransferase                 | 18     |

Labs NOT in the registry (skipped in v1): Chloride, Hematocrit, Anion gap,
Magnesium, Phosphate, Lactate (if present under a different name — to be confirmed).

## Canonical output (per LOG_ID)

```
{LOG_ID}/
  PLETH40.npy         # [N_seg, 1200]  float16
  II120.npy           # [N_seg, 3600]  float16 (NaN where ECG absent)
  time_ms.npy         # [N_seg]         int64, 30s-aligned, monotonic
  ehr_baseline.npy    # pre-AN_START, capped at 30 d
  ehr_recent.npy      # pre-AN_START within 24 h
  ehr_events.npy      # during [AN_START, AN_STOP]
  ehr_future.npy      # post-AN_STOP within 7 d
  meta.json
```

Episode bounds: `AN_START_DATETIME` / `AN_STOP_DATETIME` from `patient_information.csv`.

## Pipeline

See `workzone/mover_epic/README.md` for stage commands + wall-time estimates.

## Known issues / deferred

- BP (systolic/diastolic from `"120/80"` strings) is not yet parsed.
- Lactate var_id 4 may have a different Lab Name in EPIC — to be added after the
  first Stage D run reveals coverage stats.
- Medications / inputs / vent settings / anesthesia-agent columns from flowsheets
  are not yet extracted (would add action var_ids 200+).
- Old `data_processing_ICML/` pipeline at `/labs/hulab/mxwang/data/MOVER/` NOT
  reused.
