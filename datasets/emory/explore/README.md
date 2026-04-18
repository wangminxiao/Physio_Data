# Emory — Step 0 exploration

**Host**: `bedanalysis.bmi.emory.edu` (user `mwang80`). Python env: `/labs/hulab/mxwang/anaconda3/envs/physio_data`.

## 0a. What this dataset is

Emory hospital system dataset combining:
- **Raw WFDB** waveforms from the `/labs/collab/Waveform_Data/Waveform_Data/{prefix}/{record}/` archive (Philips monitor export in WFDB-multisegment format).
- **EHR CSVs** (Siva-version CDW pull) under `/labs/hulab/Emory_EHR/CDW_Pull_ICU_Data_Siva_version/` — JGSEPSIS_* tables covering 995,598 encounters.
- **Sepsis cohort linkage** in `/labs/hulab/mxwang/data/sepsis/Wav/` linking encounters ↔ wfdb_records with quality-gated `valid_start/valid_end` windows and case/control/`sepsis_patient_but_not_sepsis_this_encounter` labels.

Onboarding starts with the sepsis task list `sepsis_cc_2025_06_13_all_collab_uniq_combine.csv` (11,715 encounters, case/control only). Pretraining expansion to the 64,750-encounter whole list can be appended later without re-extraction.

## 0b. Structural exploration

Demo entity: `encounter_nbr = 359559206` (patient `empi_nbr=1827183`, case, sepsis_time_zero = 2019-07-30 10:49 UTC).

### Raw WFDB layout
Record folder `/labs/collab/Waveform_Data/Waveform_Data/B035/B035-0564111269/` contains:

| Kind | File pattern | Notes |
|---|---|---|
| Multi-segment top header | `{rec}.hea` | declares `n_seg=80` segments; fs=240; base_datetime=1989-07-26 03:21:09 |
| Waveform segments | `{rec}_0XXX.hea` + `.mat` | each 8 h @ 240 Hz, 6 channels `I, II, III, V, SPO2, RR` (this record). ADC gain 409.836 for ECG, 200.0 for SPO2/RR. Units nominally "mV" for all (header quirk — SPO2 is % in practice, divide by adc_gain). |
| Dense numerics | `{rec}_0n.hea` + `.mat` | fs=**0.5 Hz** (one sample / 2 s), 26 channels. This record has: HR, ST-{II/I/V/III/AVF/AVL/AVR/V1/V6}, CUFF, APNEA, RESP, PVC, SPO2-%, SPO2-R, NBP-S/D/M, CO2-EX/IN/RR, AR2-S/D/M/R. Units include Bpm, mmHg, mm, BrMin, %. ADC gain 10.0 or 100.0 depending on channel. |
| Annotations | `.qrs*`, `.abpsqi`, `.ecgsqi*`, `.ppgsqi`, `.af`, `.afsvm`, `.vf`, `.wabp`, etc. | out-of-scope for canonical build; could be used later |

**Date shift verified**: WFDB `base_datetime = 1989-07-26 03:21:09` + `relativedelta(years=30)` = `2019-07-26 03:21:09`, which matches `valid_start` in the task list exactly. Treat shifted datetime as UTC.

**Sample rates confirmed from headers** (old pipeline constant `VITAL_FS=0.5` was correct; the updated demo notebook's `fs=2` is a typo):
- Waveform: 240 Hz, 6 channels `I, II, III, V, SPO2, RR` (for this record — channels vary by record)
- Dense numerics: 0.5 Hz, 26 channels

### Multi-segment structure
Top-level `.hea` lists 80 entries including the first (`_layout`, a virtual index) and one `~` gap marker between declared segments. Each real `_0XXX` segment is 28,800 s (8 h). Total length 2,158,506 s ≈ 25 days.

In `sepsis_cc_2025_06_13_all_collab.csv` (whole list) this one `wfdb_record` shows up as **76 rows** — the 8h sub-segments span 2019-07-26 03:21Z → 2019-08-20 02:56Z. All 76 rows type = `case`. Total `wfdb_len_seconds` sum = 2,147,644 ≈ same as header.

### EHR coverage for this encounter
All timestamps in CSV are **NY local naive**, parsed with `%m/%d/%Y %H:%M:%S`, then tz-replaced to `America/New_York` and converted to UTC.

| Table | Rows | Key time col | Local range | UTC range |
|---|---|---|---|---|
| `JGSEPSIS_VITALS2` | 937 | `RECORDED_TIME` | 2019-07-25 15:26 → 2019-08-22 17:20 | 2019-07-25 19:26 → 2019-08-22 21:20 |
| `JGSEPSIS_LABS` | 748 | `LAB_RESULT_TIME` (not `RESULT_TIME`) | 2019-07-25 16:25 → 2019-08-22 07:00 | 2019-07-25 20:25 → 2019-08-22 11:00 |
| `JGSEPSIS_ENCOUNTER` | 1 | `HOSPITAL_ADMISSION_DATE_TIME` | admit 2019-07-25 19:12 / discharge 2019-08-22 19:19 | admit 2019-07-25 23:12 / discharge 2019-08-22 23:19 |

EHR vitals start ~8 h before wave start (vitals recorded from admission; monitor attached later). 65 distinct lab `COMPONENT` values. Top components by count: Hematocrit/Platelet/Hemoglobin/RBC/RDW (30 each), Glucose (29), Potassium Level (27).

**Column-name discrepancy vs old code:**
- `stp2_matching_lab_vital_wav.py` treated labs time column as `RESULT_TIME`. Actual column is `LAB_RESULT_TIME`. The old code also used `Component_value` / `Component_name`; actual columns are `LAB_RESULT` / `COMPONENT`. Need to re-verify filter logic before reusing component_id whitelist.

### Encounter + demographics
- PAT_ID (demographics key, different from empi_nbr) = `19031078`, empi_nbr = `1827183`.
- Age at admit = 72; F; DOB 1946-08-31; Race African American or Black; Ethnicity Non-Hispanic or Latino.
- Primary Dx: G40.201 — focal epilepsy with status epilepticus. Not a sepsis admit diagnosis — sepsis was a downstream complication (sepsis_time_zero = 2019-07-30, 5 days after admission).
- Insurance lives in `JGSEPSIS_ENCOUNTER.INSURANCE_STATUS`, not in `JGSEPSIS_DEMOGRAPHICS`.

### Open items to verify on a second encounter
- Does every record in the task cohort have the same six waveform channels or do they vary? → scan `sig_name` across a random 50 records.
- Invasive ABP (channels `ART`, `ABP`, or `AR1-*`) — is it present on enough records to justify extracting ABP125 as a third canonical channel? This record only has AR2-* in `_0n` (0.5 Hz pressure number), not in the wave band.
- PLETH vs SPO2 naming — this record exposes `SPO2` as the pulse-ox waveform at 240 Hz. Need to check if other records use `Pleth` instead, and apply canonicalization before resample.

## Entity ID convention (decided 2026-04-17)

`{empi_nbr}_{encounter_nbr}/` — one directory per encounter, all wfdb_records within the encounter concatenated chronologically (gaps as time_ms jumps).

For cohort extraction: uniq_combine.csv has `encounter_nbr` but not `empi_nbr` — join back to the whole list (or JGSEPSIS_ENCOUNTER) to attach `empi_nbr`.

## Timezone + time-base rules

- **WFDB**: read `base_datetime`, apply `+ relativedelta(years=30)`, treat as **UTC**.
- **EHR CSVs**: naive strings `MM/DD/YYYY HH:MM:SS` → parse → `replace_time_zone("America/New_York", ambiguous="earliest")` → `convert_time_zone("UTC")` → `epoch_ms`.
- **Internal canonical time**: int64 ms UTC throughout.
- **Wave samples**: `t_ms[i] = wfdb_base_ms + (i / fs) * 1000`, where `wfdb_base_ms = (base_datetime + 30y).astype('datetime64[ms]').astype(int64)`.

## Files produced in this step

- `dataset_profile.json` — structured findings from 0b (channel lists, fs, sample counts, EHR overlap)
