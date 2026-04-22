# VitalDB Data Source Specification

VitalDB = openvital.org's Asan Medical Center (Seoul) surgical monitoring dataset.
6,388 surgical cases, open access via `pip install vitaldb`.

## Paths (on bedanalysis)

- **Raw root**: `/opt/localdata100tb/UNIPHY_Plus/raw_datasets/vitalDB/`
- **Waveforms**: `vital_files/{caseid:04d}.vital` (6,388 files, ~96 GB total, ~15 MB each)
- **Clinical table**: `clinical_data.csv` — 6,388 rows × 75 cols (demographics, preop, intraop totals, surgery metadata)
- **Lab table**: `lab_data.csv` — 928k rows, long format `(caseid, dt, name, result)`
- **Schema dicts**: `clinical_parameters.csv`, `lab_parameters.csv`, `track_names.csv`
- **Canonical output**: `/opt/localdata100tb/physio_data/vitaldb/{caseid:04d}/`

## Entity identifier

- `entity_id = f'{caseid:04d}'` (4-digit zero-padded).
- `subjectid` = de-identified patient key in `clinical_data.csv`. One subjectid can have multiple caseids.
- Splits grouped by `subjectid` to prevent patient leakage.

## Waveform channels

| File          | Source track       | Source rate | Canonical rate | Resample |
|---------------|--------------------|-------------|----------------|----------|
| `PLETH40.npy` | `SNUADC/PLETH`     | 500 Hz      | 40 Hz          | `resample_poly(2, 25)` |
| `II120.npy`   | `SNUADC/ECG_II`    | 500 Hz      | 120 Hz         | `resample_poly(6, 25)` |

- **PLETH is always the anchor.** Windows require 30 s of 0 % NaN PLETH data (strict).
- **ECG_II is a proper Lead II track** (not a device-internal channel ID like MOVER's `ECG1`).
- Other SNUADC tracks (`ART`, `ECG_V5`, `FEM`, `CVP`) not extracted in v1.

## Time convention

- **All times inside VitalDB are relative to `casestart = 0` (in seconds)**.
- Each `.vital` file carries a `dtstart` (fake epoch seconds, intentionally shifted for deidentification). `vitaldb.VitalFile(path).dtstart` returns this.
- **Canonical `time_ms` = `(dtstart + relative_s) * 1000`** — each case lives in its own time band, but within a case the timeline is consistent.
- `anestart_s`, `aneend_s`, `opstart_s`, `opend_s`, `adm_s`, `dis_s` from `clinical_data.csv` are all relative; convert with the same formula.
- **No timezone conversions** — simplifies the pipeline vs MOVER / MC_MED.

## EHR variables extracted

### Vitals (from `.vital` Solar8000/* tracks, 1 Hz)

| Track | var_id | Canonical name |
|-------|-------:|----------------|
| `Solar8000/HR`         | 100 | HR      |
| `Solar8000/PLETH_SPO2` | 101 | SpO2    |
| `Solar8000/BT`         | 103 | Temperature |
| `Solar8000/NIBP_SBP`   | 104 | NBPs    |
| `Solar8000/NIBP_DBP`   | 105 | NBPd    |
| `Solar8000/NIBP_MBP`   | 106 | NBPm    |
| `Solar8000/CVP`        | 107 | CVP     |
| `Solar8000/ART_SBP`    | 110 | ABPs    |
| `Solar8000/ART_DBP`    | 111 | ABPd    |
| `Solar8000/ART_MBP`    | 112 | ABPm    |
| `Solar8000/ETCO2`      | 116 | EtCO2   |

RR is not extracted in v1 (no clean 1-Hz RR track in Solar8000). Consider adding later via Primus/ventilator.

### Labs (from `lab_data.csv`, long format)

| `name` | var_id | Description |
|--------|-------:|-------------|
| `k`     | 0  | Potassium |
| `ica`   | 1  | ionized Calcium (proxy for Calcium) |
| `na`    | 2  | Sodium |
| `gluc`  | 3  | Glucose |
| `lac`   | 4  | Lactate |
| `cr`    | 5  | Creatinine |
| `tbil`  | 6  | Total bilirubin |
| `plt`   | 7  | Platelets |
| `wbc`   | 8  | WBC |
| `hb`    | 9  | Hemoglobin |
| `ptinr` | 10 | INR |
| `bun`   | 11 | BUN |
| `alb`   | 12 | Albumin |
| `ph`    | 13 | Arterial pH |
| `po2`   | 14 | paO2 |
| `pco2`  | 15 | paCO2 |
| `hco3`  | 16 | HCO3 |
| `ast`   | 17 | AST |
| `alt`   | 18 | ALT |

Labs not mapped to canonical var_ids (deferred): `hct`, `gfr`, `ccr`, `esr`, `tprot`, `cl`, `ammo`, `crp`, `pt%`, `ptsec`, `aptt`, `fib`, `be`, `sao2`.

## Canonical output (per case)

```
{caseid:04d}/
  PLETH40.npy         # [N_seg, 1200]  float16
  II120.npy           # [N_seg, 3600]  float16 (NaN when ECG_II absent)
  time_ms.npy         # [N_seg]         int64
  ehr_baseline.npy    # EHR_EVENT_DTYPE  pre-anesthesia, capped 30 d
  ehr_recent.npy      # pre-anesthesia within 24 h
  ehr_events.npy      # during [anestart, aneend]
  ehr_future.npy      # post-aneend within 7 d
  meta.json           # includes: caseid, subjectid, dtstart_s, anestart_ms, aneend_ms
```

Episode bounds: `anestart_ms` / `aneend_ms` (anesthesia window) from clinical_data.

## Pipeline

See `workzone/vitaldb/README.md` for run commands.

## Known issues / deferred

- No BP-string parsing issue (unlike MOVER/EPIC flowsheets) — NIBP comes as three separate SBP/DBP/MBP tracks.
- `SNUADC/ART` (invasive arterial waveform at 500 Hz) not extracted; adds canonical `ABP125` channel later if needed.
- Medications / intraop drug doses in `clinical_data.csv` (intraop_*) are aggregate totals, not time-stamped events → not extractable as event stream without separate source.
- Preop labs in `clinical_data.csv` (preop_*) are single snapshots; `lab_data.csv` is the longitudinal source — we use lab_data exclusively for lab events.
