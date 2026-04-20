# MC-MED Data Source Specification

MC-MED (Multimodal Clinical Monitoring in the Emergency Department, PhysioNet
v1.0.1) — 118,385 adult ED visits / 70,545 unique patients, 2020–2022. 83,623
visits have continuous waveforms (ECG II, Pleth/PPG, Resp).

## Paths (on bedanalysis)

- **Raw dataset root**: `/opt/localdata100tb/UNIPHY_Plus/raw_datasets/physionet.org/files/mc-med/1.0.1/data/`
- **Raw waveforms**: `{root}/waveforms/{CSN_suffix}/{CSN}/{II,Pleth,Resp}/{CSN}_{N}.{hea,dat}` (WFDB)
  - `CSN_suffix` = last 3 digits of CSN
- **EHR tables**: `{root}/{labs,numerics,visits,meds,orders,pmh,rads,waveform_summary}.csv`
- **Predefined splits**: `{root}/split_{random,chrono}_{train,val,test}.csv` (patient-safe per README)
- **Canonical output**: `/opt/localdata100tb/physio_data/mcmed/{CSN}/`

## Entity identifier

- `entity_id` = `str(CSN)` (visit-level; matches the filename stems in the raw waveforms)
- `MRN` is the patient-level key. Splits group by MRN; the MC_MED predefined splits already respect this, so we adopt them verbatim.

## Waveform channels

| File         | Canonical rate | Samples / 30 s | Source rate | Resample |
|--------------|---------------:|---------------:|-------------|----------|
| `PLETH40.npy`| 40 Hz          | 1200           | 125 Hz      | `resample_poly(up=8, down=25)` |
| `II120.npy`  | 120 Hz         | 3600           | 500 Hz      | `resample_poly(up=6, down=25)` |

- **Pleth-anchored**: drop windows with >20% NaN in PLETH. II NaN-filled when absent at that window start time.
- **Windowing**: 30 s **non-overlapping** (matches Emory; UCSF / MIMIC use 5 s overlap).
- Resp is available but NOT extracted in v1 (no canonical channel assigned).

## Time convention

All timestamps in MC_MED are ISO-8601 UTC with trailing `Z` (e.g. `2262-01-09T03:16:07Z`).
Times are "random-shifted per-patient, keeping season constant" — so absolute dates are
meaningless but intra-episode deltas are preserved. We treat them as UTC directly; no
timezone conversion. **No +30 y shift** (that is Emory-specific).

- WFDB segment `.hea` `base_datetime` is already UTC.
- Visit columns (`Arrival_time`, `Roomed_time`, `Dispo_time`, `Admit_time`, `Departure_time`) are UTC.
- Lab `Result_time` and numeric `Time` are UTC.

## EHR variables extracted

### Labs (from `labs.csv` `Component_name`)

| Component_name         | var_id | Canonical name |
|------------------------|-------:|----------------|
| POTASSIUM              | 0      | Potassium      |
| CALCIUM                | 1      | Calcium        |
| SODIUM                 | 2      | Sodium         |
| GLUCOSE                | 3      | Glucose        |
| LACTIC ACID / LACTATE  | 4      | Lactate        |
| CREATININE             | 5      | Creatinine     |
| BILIRUBIN TOTAL        | 6      | Bilirubin      |
| PLATELET COUNT         | 7      | Platelets      |
| WBC                    | 8      | WBC            |
| HEMOGLOBIN             | 9      | Hemoglobin     |
| INR                    | 10     | INR            |
| BUN                    | 11     | BUN            |
| ALBUMIN                | 12     | Albumin        |
| HCO3 / POC:HCO3        | 16     | HCO3           |
| AST                    | 17     | AST            |
| ALT                    | 18     | ALT            |

Event timestamp = `Result_time`. Non-numeric / out-of-physio-range values dropped.
Extensible: add new rows to the registry's `mcmed_lab_components` field and extend
the `COMPONENT_TO_VAR_ID` dict in `workzone/mcmed/stage_d_labs.py`.

### Vitals (from `numerics.csv` `Measure`)

| Measure | var_id | Canonical name |
|---------|-------:|----------------|
| HR      | 100    | HR             |
| SpO2    | 101    | SpO2           |
| RR      | 102    | RR             |
| Temp    | 103    | Temperature    |
| SBP     | 104    | NBPs           |
| DBP     | 105    | NBPd           |
| MAP     | 106    | NBPm           |

Event timestamp = `Time`. Extras (`Perf`, `Pain`, `LPM_O2`, `1min_HRV`, `5min_HRV`)
have no canonical IDs yet and are ignored in v1.

## Canonical output (per CSN)

```
{CSN}/
  PLETH40.npy         # [N_seg, 1200]  float16, C-contiguous
  II120.npy           # [N_seg, 3600]  float16, C-contiguous (NaN when II absent)
  time_ms.npy         # [N_seg]         int64, strictly monotonic
  ehr_baseline.npy    # [N_base]  EHR_EVENT_DTYPE  (pre-arrival, capped at 30 d)
  ehr_recent.npy      # [N_rec]   EHR_EVENT_DTYPE  (within 24 h of wave_start)
  ehr_events.npy      # [N_ev]    EHR_EVENT_DTYPE  (inside waveform, real seg_idx)
  ehr_future.npy      # [N_fut]   EHR_EVENT_DTYPE  (post-wave_end, capped at 7 d)
  meta.json
```

Episode bounds for 4-partition split: `arrival_ms` (episode_start), `departure_ms`
(episode_end). Since ED visits are short (median ~5 h), `ehr_baseline` / `ehr_future`
are usually very small.

## Pipeline (see `workzone/mcmed/README.md` for run commands)

```
A stage_a_cohort.py       -> valid_cohort.parquet
B stage_b_wave.py         -> PLETH40.npy, II120.npy, time_ms.npy, meta.json
C stage_c_vitals.py       -> vitals_events.npy (per CSN)
D stage_d_labs.py         -> labs_events.npy   (per CSN)
E stage_e_assemble.py     -> ehr_{baseline,recent,events,future}.npy
F stage_f_demographics.py -> demographics.csv
F stage_f_manifest.py     -> manifest.json, pretrain_splits{,_chrono}.json, downstream_splits.json
G workzone/common/build_estimation_task.py --spec lab_est_full.yaml / vital_est_full.yaml
```

Stage G reuses the shared estimation-task builder; no MC_MED–specific task code.

## Splits

MC_MED ships two patient-safe splits:
- `split_random_*.csv` (80/10/10) — adopted as **`pretrain_splits.json`** + **`downstream_splits.json`**.
- `split_chrono_*.csv` — adopted as **`pretrain_splits_chrono.json`** (time-ordered, same MRN constraint).

No new shuffling, so results remain comparable to the MC_MED ICML benchmark.

## Known issues / deferred

- Resp channel (62.5 Hz) present for 50,694 CSNs — not extracted in v1.
- Numerics extras (`Perf`, `Pain`, `LPM_O2`, `1min_HRV`, `5min_HRV`) not yet in var_registry.
- `labs.csv` has some Component_name strings with embedded commas / newlines that polars
  parses correctly but bare awk / head do not. Always use polars.
