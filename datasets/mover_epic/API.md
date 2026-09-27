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
  `splits.json.excluded_by_meta`). Counts: `workzone/outputs/{mover,mover_epic}/clock_shift_summary_season.json` (jobs
  pending submission, see `datasets/CLOCK_FIX_PLAN_MIMIC_MOVER.md` §7).
* BeeGFS: `/projects/xhu40-cdsfm/physio_data/mover_combine` holds real (dereferenced) copies incl. waveforms and FM sidecars
  but was **not** refreshed by the first copy job (symlinked entity dirs vs `rsync -rlt`); the fixed `copy` step uses
  `rsync -rLt`, and creates `/projects/xhu40-cdsfm/physio_data/{mover,mover_epic}` as per-entity symlinks into
  `../mover_combine/<id>` plus their own manifest / splits / demographics / `tasks/` (plan §2.6).
* **Downstream rule**: `tasks/*` already exclude `clock_shift_confidence == "unverified"`; anyone reading the store directly
  for sub-hour EHR↔waveform analyses must apply the same filter (`meta.clock_shift_confidence`). Plan and evidence:
  `datasets/CLOCK_FIX_PLAN_MIMIC_MOVER.md`.
