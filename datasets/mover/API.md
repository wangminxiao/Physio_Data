# MOVER/SIS Data Source Specification

MOVER = UC Irvine's **M**ultimodal **O**perating-room **V**ital and **E**HR **R**ecord
dataset. Two waveform sources: SIS (legacy anesthesia info system, 2015–2017 ish)
and EPIC (newer, 2018–2020). This spec covers **SIS only** (v1); EPIC is deferred.

## Paths (on bedanalysis)

- **Raw dataset root**: `/opt/localdata100tb/UNIPHY_Plus/raw_datasets/MOVER/`
- **SIS raw waveforms**: `sis_wave_v2/UCI_deidentified_part3_SIS_11_07/Waveforms/{PID[:2]}/{PID}/{HH-MM-SS-mmm}Z.xml`
  - 18,828 PID dirs, 340,223 XMLs total (~18 XMLs per PID)
  - Each XML = 30 min of multi-channel data, `<cpcArchive duration="30">` containing ~2,697 `<cpc>` blocks (1 s each)
- **SIS EHR**: `EMR/patient_{information,vitals,labs,a_line,input_output,medication,observations,procedure_events,ventilator}.csv`
- **Canonical output**: `/opt/localdata100tb/physio_data/mover/{PID}/`

## Entity identifier

- `entity_id = str(PID)` (16-hex). One PID = one surgery = one entity per `patient_information.csv`.
- No LOG_ID in SIS (that's EPIC-only).

## Waveform channels extracted

| File          | Canonical rate | Samples / 30 s | Source rate | Resample | SIS channel |
|---------------|---------------:|---------------:|-------------|----------|-------------|
| `PLETH40.npy` | 40 Hz          | 1200           | 100 Hz      | `resample_poly(up=2, down=5)` | `PLETH` |
| `II120.npy`   | 120 Hz         | 3600           | 300 Hz      | `resample_poly(up=2, down=5)` | `ECG1`  |

- **Pleth-anchored**: emit windows only when PLETH has ≥24/30 seconds of data AND <20 % post-resample NaN.
- II is NaN-filled when ECG1 coverage <24/30 seconds.
- Other channels present in the XML (`GE_ART`, `GE_ECG`, `INVP1`) are ignored in v1 — they're duplicates of ECG/arterial pressure at different rates, with no canonical var_registry slot yet.

**XML decode (per UCI's `waveform_decode.py`)**: base64 → little-endian int16 → `sample * gain + offset`. Uses `np.frombuffer(raw, '<i2')` in our decoder. Gain / Offset come from `<m name="...">` children of each `<mg>` element. UCI hardcodes `gain=0.25` for `GE_ART` and `gain=0.01` for `INVP1` (XML gain incorrect for pressure channels) — we don't extract those in v1 so the override is noted but unused.

## Time convention

- **XML `<cpc datetime="...Z">`** attribute is **authoritative UTC** (already timezoned in Z). Parse directly.
- **`patient_information.csv`** has naive datetimes (`1/1/16 7:30`). MOVER is at UC Irvine (California), so these are interpreted as **America/Los_Angeles → UTC** with `ambiguous='earliest'`, `non_existent='null'`.
- **`patient_vitals.csv` / `patient_labs.csv`** `Obs_time` columns are naive `YYYY-MM-DD HH:MM:SS`. Same LA→UTC conversion.
- Filename `HH-MM-SS-mmmZ.xml` is **not** authoritative — use XML `datetime` attribute inside.

## EHR variables extracted

### Vitals (from `patient_vitals.csv`, wide format, melted)

Columns: `PID`, `Obs_time`, `HRe`, `HRp`, `nSBP`, `nMAP`, `nDBP`, `SP02`.

| Column | var_id | Canonical name |
|--------|-------:|----------------|
| `HRe`  | 100    | HR             |
| `SP02` | 101    | SpO2           |
| `nSBP` | 104    | NBPs           |
| `nDBP` | 105    | NBPd           |
| `nMAP` | 106    | NBPm           |

`HRp` (pulse rate from oximetry) skipped — redundant with `HRe`. No RR or Temperature columns in SIS vitals.

### Labs (from `patient_labs.csv`, wide format, melted)

Columns: `PID`, `Obs_time`, `Na`, `K`, `Ca`, `Gluc`, `Ph`, `PCO2`, `PO2`, `BE`, `HCO3`, `HgB`.

| Column | var_id | Canonical name |
|--------|-------:|----------------|
| `Na`   | 2      | Sodium         |
| `K`    | 0      | Potassium      |
| `Ca`   | 1      | Calcium        |
| `Gluc` | 3      | Glucose        |
| `Ph`   | 13     | Arterial_pH    |
| `PCO2` | 15     | paCO2          |
| `PO2`  | 14     | paO2           |
| `HCO3` | 16     | HCO3           |
| `HgB`  | 9      | Hemoglobin     |

`BE` (base excess) dropped — no canonical var_id.

## Canonical output (per PID)

```
{PID}/
  PLETH40.npy         # [N_seg, 1200]  float16, C-contiguous
  II120.npy           # [N_seg, 3600]  float16, C-contiguous (NaN where ECG1 missing)
  time_ms.npy         # [N_seg]         int64, 30s-aligned, monotonic
  ehr_baseline.npy    # EHR_EVENT_DTYPE  pre-OR, capped at 30 d
  ehr_recent.npy      # pre-OR within 24 h
  ehr_events.npy      # during-OR, real seg_idx
  ehr_future.npy      # post-OR within 7 d
  meta.json
```

Episode bounds: `or_start_ms` / `or_end_ms` from `patient_information.csv`.

## Pipeline

See `workzone/mover/README.md` for stage commands + wall-time estimates.

## Known issues / deferred

- EPIC waves (1/2/4) not yet extracted — would be a v2 effort with flat-XML + flowsheets EHR.
- `patient_a_line` / `patient_input_output` / `patient_medication` / `patient_observations` / `patient_procedure_events` / `patient_ventilator` CSVs not yet mined — would add actions (var_ids 200+) to the trajectory.
- SIS signals include invasive arterial pressure (INVP1 100 Hz, GE_ART 180 Hz) — if a future task needs ABP, we'd add `ABP125.npy` as a channel.
- Old `data_processing_ICML/` pipeline at `/labs/hulab/mxwang/data/MOVER/` is reference-only; we are NOT reusing its precomputed NPZ or split JSON.
