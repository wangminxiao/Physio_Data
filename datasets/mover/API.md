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

## Clock fix applied 2026-09-27 (SIS and EPIC)

* XML waveform `…Z` timestamps come from devices with a fixed UTC offset (no DST), so ≈40 % of cases were ±60 min off the
  EHR (winter −60, summer +60). `workzone/mover/stage_b2_clock.py` measured each case (charted HR vs PPG pulse rate, ECG
  when usable) and shifted `time_ms` by {−60, 0, +60} min: SIS 6,993 cases → −60×1,175 / 0×3,919 / +60×1,191, no vitals 708,
  unverified 1,023 (kept as is, `meta.clock_shift_confidence = "unverified"`); EPIC 1,820 → −60×357 / 0×1,092 / +60×311,
  unverified 331. Meta: `time_base = "utc_ms"`, `clock_shift_min`, `clock_shift_method`, `clock_shift_confidence`,
  `clock_shift_detail`, `clock_fix_version = 1`; per-case report `workzone/outputs/{mover,mover_epic}/clock_shift.parquet`.
* Gates (`verify_mover_clock.py`, all PASS): decided 84 % / 81 %; corrected store needs no residual shift (120/120 both);
  OR_start − wave_start inside the aligned envelope 98 % / 93 %; no case moved out of its OR window; structural OK; snapshots OK.
  Note the OR_start − wave_start distribution is bimodal even when aligned (waveform starts at OR entry ≈ −3 min or at
  induction ≈ +50 min).
* Stage E re-run (partitions, `seg_idx`), splits identical to before, tasks rebuilt (`lab_est_full` SIS 970→983 train,
  EPIC 309→275; `vital_est_full` unchanged). `mover_combine`: all 8,812 symlinks re-pointed from `/opt/localdata100tb` to
  `/mnt/localdata100tb` (they had been dangling since the 2026-08 migration); tasks rebuilt (`lab_est_full` 1,279→1,258 train).
  All `mover*/` scripts had hard-coded old server paths → translated; `mover`, `mover_epic`, `mover_combine` sections added
  to `workzone/configs/server_paths.yaml`.
* **v4 (season-constrained binary test, `stage_b2_clock.py --only-unverified`)** for the cases v3 left unverified (SIS 1,719
  incl. 708 without charted HR; EPIC 374 incl. 59): the device hypothesis fixes the sign of the shift by season (standard
  time → +60 only, DST → −60 only; 99.2 % / 98.7 % of the v3 decisions obey it), so each reference only has to separate
  "aligned" from "the one allowed hour" (MAE ≤ 10/8 bpm and MAE < 0.8× or r ≥ 0.5 with gain ≥ 0.15). `clock_fix_version = 2`,
  `clock_season_prior`, `clock_shift_v3_method` in meta; validated first on every v3-decided case (`--validate-decided`,
  chain stops below 98 % agreement). Cases still unverified stay in the store and in `pretrain_splits.json` but are
  **excluded from every `tasks/*/splits.json`** (`build_summary.json` spec `exclude_meta`, excluded ids listed in
  `splits.json.excluded_by_meta`). Validation (v4 replayed on the v3-decided cases from their pre-fix clock): SIS coverage
  97.9 % / agreement 99.4 %, EPIC 97.7 % / 99.4 %. Re-test of the unverified cases: SIS 1,719 → 938 decided (−60×234, +60×208,
  aligned 496; 74 high / 864 medium), EPIC 374 → 172 (−60×39, +60×38, aligned 95). **Final store: SIS −60×1,409 / 0×4,148 /
  +60×1,399, decided 6,200 (88.7 %), unverified 793 (744 undecided/conflict + 37 with < 20 charted-HR points, the old "no_vitals");
  EPIC −60×396 / 0×1,028 / +60×349, decided 1,601 (88.0 %), unverified 218.** Tasks after exclusion (train): SIS `lab_est_full`
  970→924, `vital_est_full` 4,771→4,351; EPIC 309→242, 1,211→1,115; `mover_combine` `lab_est_full` 1,279→1,166, `vital_est_full`
  5,982→5,466, `lab6_any_min2` rebuilt from `task_specs/lab6_any_min2.yaml` (lost `lab_task.py`, no spec). Per-case rows: `clock_shift.parquet` (v3 rows kept
  in `clock_shift_v3.parquet`), summaries `clock_shift_summary_{validate_season,season}.json`.
* BeeGFS: `/projects/xhu40-cdsfm/physio_data/mover_combine` holds real (dereferenced) copies incl. waveforms and FM sidecars
  but was **not** refreshed by the first copy job (symlinked entity dirs vs `rsync -rlt`); the fixed `copy` step uses
  `rsync -rLt`, and creates `/projects/xhu40-cdsfm/physio_data/{mover,mover_epic}` as per-entity symlinks into
  `../mover_combine/<id>` plus their own manifest / splits / demographics / `tasks/` (plan §2.6).
* **Downstream rule**: `tasks/*` already exclude `clock_shift_confidence == "unverified"`; anyone reading the store directly
  for sub-hour EHR↔waveform analyses must apply the same filter (`meta.clock_shift_confidence`). Plan and evidence:
  `datasets/CLOCK_FIX_PLAN_MIMIC_MOVER.md`.
