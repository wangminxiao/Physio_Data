# UCSF Institutional Dataset API

De-identified UCSF ICU dataset hosted at Emory under `/labs/hulab/UCSF/` on
`bedanalysis.bmi.emory.edu`. Span: 2012-03 → 2018-06. See
`datasets/ucsf/explore/README.md` for the underlying scan that motivated every
parameter below.

## Data Sources

### Waveform

| Field | Value |
|-------|-------|
| Format | `.adibin` (binfilepy, 240 Hz) + `.vital` (vitalfilepy, 0.5 Hz monitor streams) |
| Location | `/labs/hulab/UCSF/{Wynton_folder}/DE{Patient_ID_GE}/{bed_subdir}/` |
| Organization | `{YYYY-MM}-deid/DE{Patient_ID_GE}/{session}/...` (70 cohort folders, 26,021 patient dirs) |
| Patient ID field | `Patient_ID_GE` (parsed from `DE{...}` directory name) |
| Time reference | `.adibin` header `{Year, Month, Day, Hour, Minute, Second}` (naive local, GE-shifted) |
| Vendored readers | `/labs/hulab/mxwang/data/ucsf_EHR/bedanalysis_waveformExtraction/{binfilepy,vitalfilepy}/` (not on PyPI; copy into `workzone/ucsf/`) |

### EHR — Clinical Tables

All under `/labs/hulab/UCSF/rdb_new/`, latin-1 `.txt` shards, **dirty-comma CSV
(unquoted commas in free-text fields)** — must be repaired before
`pl.read_csv` (vendor `remove_bad_commas` / `remove_bad_commas_quotes` from
`/home/mxwan/workspace/ucsf_ehr_code/EHR_encounter_polars.py:12-63`).

| Table | Location | Shards | Selected columns |
|-------|----------|--------|------------------|
| Labs | `Filtered_Lab_New/*.txt` | 4,054 | `Patient_ID, Lab_Encounter_ID, Lab_Collection_Date, Lab_Collection_Time, Lab_Value, Lab_Unit, Lab_Common_Name, Lab_Procedure_Code, LOINC_Code, cohort` |
| Medications | `Filtered_Medication_Orders_New/*.txt` | 782 | `Patient_ID, Medication_Order_Start_Date/Time, ..._End_Date/Time, Medication_Generic_Name, Medication_Therapeutic_Class, Medication_Order_Maximum_Dose, Medication_Order_Quantity, cohort` |
| Diagnoses | `Filtered_Diagnoses_New/*.txt` | 1,018 | `Patient_ID, Diagnosis_Start_Date, ICD9_Code, ICD10_Code, Diagnosis_Name, Primary_Coded_Diagnosis, cohort` |
| Encounters | `Filtered_Encounters_New/*.txt` | 447 | `Encounter_ID, Patient_ID, Encounter_Admission_Time, Encounter_Discharge_Time, Encounter_Age, Encounter_HospitalService, cohort` |
| Billing (CPT) | `Filtered_Billing_New/*.txt` | 6,107 | `Patient_ID, Billing_Service_Date, Billing_Procedure_Code, Billing_CPT_Level_*, cohort` |
| Procedures | `Filtered_Procedure_Orders_New/*.txt` | 3,508 | `Patient_ID, Procedure_Order_Date/Time, Procedure_CPT_Code, Procedure_Name, cohort` |
| Flowsheets | `FLOWSHEETVALUEFACT/*.txt` | 49,725 | `FlowsheetRowKey, Value, FlowDate, FlowTime, encounter_ID, patient_ID, cohort` — **skipped first pass** (see below) |

**FLOWSHEETVALUEFACT is deferred.** Lookup table at
`/labs/hulab/UCSF/rdb_database/FLOWSHEETROWDIM_New/FLOWSHEETROWDIM_New.csv`
(101k rows) maps rowkeys to variable names, but each clinical variable maps to
many rowkeys (SpO2 alone has 10+). Curated many-to-one mapping is follow-up
work. `.vital` monitor streams at 0.5 Hz cover the same vitals densely.

### Patient ID Linkage

```
Waveform:  Patient_ID_GE          (DE{...} directory name)
EHR:       Patient_ID + Encounter_ID
Linkage:   /labs/hulab/UCSF/encounter_date_offset_table_ver_Apr2024.xlsx
           bridges Encounter_ID ↔ Patient_ID ↔ Patient_ID_GE ↔ Wynton_folder.

Output entity_id: {Patient_ID_GE}_{WaveCycleUID}
```

`WaveCycleUID` is the per-bed-cycle key in `MRN-Mapping.csv` (one physical ICU
stay on one bed). One `Patient_ID_GE` can own multiple wave cycles
(re-admissions, bed transfers); each is a separate entity. **Splits group by
`Patient_ID_GE`** to prevent leakage.

### Time Format and Offset Alignment

| Source | Format | Notes |
|--------|--------|-------|
| `.adibin` waveform | header struct (naive local, **shifted by `offset_GE` days**) | de-identification |
| `.vital` waveform | offset_sec from file start (**file start shifted by `offset_GE`**) | zero-timestamp files: anchor offset_sec to wave-cycle `valid_start` |
| EHR tables | `MM/DD/YYYY HH:MM:SS` or `YYYY-MM-DD HH:MM:SS` (naive local, **shifted by `offset` days**) | two formats co-exist — try both |
| `MRN-Mapping.csv` | same two datetime formats | shifted by `offset_GE` (waveform side) |

**Critical alignment rule** (per-encounter, looked up from offset xlsx):

```
GE_time_ms = EHR_time_ms − (offset_GE − offset) × 86_400_000
```

Subtract the day-difference from EHR timestamps to bring them onto the GE
(waveform) calendar. Per-encounter deltas — the difference is non-zero for
every one of the 27,903 encounter rows. **Sign matters and is the trap that
required visual verification in Step 0c.**

No timezone correction needed once the day-shift is applied (both sides are
naive local in the same shifted reference).

---

## Waveform Channels to Extract

| Source Channel | Source Rate (Hz) | Target Channel Name | Target Rate (Hz) | samples_per_seg (30s) | Notes |
|---|---|---|---|---|---|
| `SPO2` (`.adibin`) | 240 | `PLETH40` | 40 | 1200 | PPG / plethysmography. Primary modality (PLETH-anchored segments). `resample_poly(1, 6)`. |
| `II` (`.adibin`) | 240 | `II120` | 120 | 3600 | ECG Lead II. `resample_poly(1, 2)`. |

**Channel selection logic**: `.adibin` headers consistently expose
`I, II, III, V, AVR, AVL, AVF, SPO2, RR` (= 7 ECG leads + PPG + respiration);
some files add `CVP1` / `AR2` invasive waveforms. Select by exact name from
`BinFile.channels`. Both `II` and `SPO2` present in every probed file → no
fallback channel needed for the core pair.

**Optional extensions** (out-of-scope for first pass; can be added without
touching existing arrays — drop new `.npy` files per entity):
`I120, III120, V120, AVR120, AVL120, AVF120` (other ECG leads),
`RR240` (respiration waveform), `CVP_wave120`, `AR_wave120` (invasive
pressures, sparse). Multi-arterial-line `.vital` AR{1,2,3} are folded into a
single ABP id pool — see EHR vitals section.

**Resampling**: `scipy.signal.resample_poly(signal, up, down)` where `up/down`
come from `gcd(source_rate, target_rate)`.

---

## EHR Variables to Extract

All categories share the structured dtype
`(time_ms: int64, seg_idx: int32, var_id: uint16, value: float32)` and are
partitioned across `ehr_baseline.npy` / `ehr_recent.npy` / `ehr_events.npy` /
`ehr_future.npy` by relative time to the wave window.

**Mapping principle**: same clinical semantic → same `var_id` regardless of
source cadence. Different clinical semantic (invasive vs non-invasive,
different vasculature, derived quantity) → distinct `var_id`.

### Labs (var_id 0–99, from `Filtered_Lab_New`)

Filter by `Patient_ID`; parse `Lab_Value` after stripping `%`; map by
`LOINC_Code` (preferred, robust) with fallback to `Lab_Common_Name` /
`Lab_Procedure_Code`. Reuse the schema-override + null-value list from
`/home/mxwan/workspace/ucsf_ehr_code/EHR_labtest_polars.py:55-86`.

All 17 registry labs (var_id 0–16) confirmed present via `Lab_Common_Name`
scan (Glucose, HEMATOCRIT, HEMOGLOBIN, PLATELETCOUNT, WBCCOUNT, Magnesium,
Phosphorus, PT, INR, PCO2, FIO2, PO2, ALT, AST, Potassium, Sodium, Creatinine,
Bilirubin, BICARBONATE, etc.). LOINC scan deferred until extraction time
(needs the dirty-comma repair to land first).

### Vitals (var_id 100–199, from `.vital` files)

`vitalfilepy.VitalFile.readVitalDataBuf` returns 4-tuples
`(value, offset_sec, sentinel_missing=-999999, constant=32768)`. Use only the
first two fields. NBP-{S,D,M} are intermittent (~15 min cuff cycles); all
other suffixes are 0.5 Hz continuous.

| var_id | Variable | UCSF `.vital` suffix(es) | Unit | Range | Notes |
|---|---|---|---|---|---|
| 100 | HR | `HR` | bpm | 10–300 | ECG-derived |
| 101 | SpO2 | `SPO2-%` | % | 20–100 | |
| 102 | RR | `RESP` | /min | 1–70 | Respiration rate (numeric, not waveform) |
| 103 | Temperature | `TMP-1`, `TMP-2` | °C | 25–45 | Two probes (rectal/skin); UCSF stores Celsius |
| 104 | NBPs | `NBP-S` | mmHg | 30–300 | **Intermittent** ~15 min cuff cycles |
| 105 | NBPd | `NBP-D` | mmHg | 10–200 | Intermittent |
| 106 | NBPm | `NBP-M` | mmHg | 20–250 | Intermittent |
| 107 | CVP | `CVP1`, `CVP2`, `CVP3` | mmHg | -10–40 | Up to 3 simultaneous lines, folded into one id |
| 110 | ABPs | `AR1-S`, `AR2-S`, `AR3-S` | mmHg | 40–300 | Invasive arterial systolic; multiple lines folded |
| 111 | ABPd | `AR1-D`, `AR2-D`, `AR3-D` | mmHg | 20–200 | Invasive arterial diastolic |
| 112 | ABPm | `AR1-M`, `AR2-M`, `AR3-M` | mmHg | 30–250 | Invasive arterial mean |
| 113 | PR_art | `AR1-R`, `AR2-R`, `AR3-R` | bpm | 20–300 | Pulse rate from arterial waveform (distinct from HR) |
| 114 | PVC_rate | `PVC` | /min | 0–60 | PVC count from monitor |
| 115 | SPO2_pulse_rate | `SPO2-R` | bpm | 20–300 | Pulse rate from oximeter |

**Multi-line folding (AR{1,2,3}, CVP{1,2,3})**: physiologically equivalent
pressure measurements; the radial-vs-femoral distinction is rarely
task-relevant. A patient with 2 art lines yields 2× the event density on each
ABP id. If a downstream task ever needs per-line provenance, extra channels
can be added in a post-stage.

**Deferred (reserve namespace, not in first pass)**: `PA2-{D,M,S}` (pulmonary
artery pressure), `ICP1`, `CPP1`, `ST-{I,II,III,V1,V2,V3}` (per-lead
ST-segment deviation), `SP{2,3}`. All rare enough to postpone.

### Actions (var_id 200–299, from `Filtered_Medication_Orders_New`)

Use `Medication_Order_Start_Date/Time` as event time; `value` stores rate/dose
(prefer `Medication_Order_Maximum_Dose`, fall back to
`Medication_Order_Quantity`). Mapping strategy: substring match on
`Medication_Generic_Name` + category via `Medication_Therapeutic_Class` →
existing registry ids 200 (`vasopressor_rate`) and 201 (`fluid_rate`).

Top generic names include drugs we already need (norepinephrine /
phenylephrine for 200; 0.9% NaCl + lactated Ringer's for 201) and several
sedatives / analgesics not yet in the registry (fentanyl / hydromorphone /
propofol / midazolam) which can be added when a task requires them.

### Scores (var_id 300–399)

None populated by the main pipeline. `SOFA_*` and `sepsis_onset` are
post-stage outputs if needed (UCSF tasks below don't currently require them).

### Filtering Rules

```
- Apply offset xlsx day-shift (GE_time = EHR_time - (offset_GE - offset) days) BEFORE writing events.
- Drop rows where parsed value is null.
- Drop rows whose value falls outside [physio_min, physio_max] for the mapped var_id.
- For .vital streams: drop the sentinel -999999 (missing marker from readVitalDataBuf).
- For labs with multiple LOINC codes mapping to same var_id: combine, no priority.
- Deduplicate: if same (entity_id, time_ms, var_id, value), keep first.
```

---

## Demographics to Extract

One row per `entity_id = {Patient_ID_GE}_{WaveCycleUID}`. Static across the
wave cycle.

| Field | Source | Column | Encoding |
|---|---|---|---|
| `entity_id` | derived | — | string index |
| `patient_id_ge` | dir name | `DE{...}` | int |
| `wave_cycle_uid` | `MRN-Mapping.csv` | `WaveCycleUID` | string |
| `encounter_id` | offset xlsx | `Encounter_ID` | string (joined via `Patient_ID_GE` + `Wynton_folder`) |
| `gender` | `Filtered_Encounters_New` | (TBD: confirm column name at extraction) | "M" / "F" / "" |
| `age_years` | `Filtered_Encounters_New` | `Encounter_Age` | float |
| `hospital_service` | `Filtered_Encounters_New` | `Encounter_HospitalService` | str |
| `wynton_folder` | offset xlsx | `Wynton_folder` | str (`{YYYY-MM}-deid`) |
| `cohort` | offset xlsx | derived | str (year/quarter bucket if needed) |

Categorical columns stored as raw strings; consumers encode at load time.

---

## Processing Parameters

| Parameter | Value | Rationale |
|---|---|---|
| Segment duration | 30 seconds | Standard across all datasets |
| Min segments per entity | 10 | Skip < 5 min recordings |
| Max NaN ratio per channel | 0.20 | Flag, don't drop |
| Anchor channel | `PLETH40` | Segments require PLETH40 present; II120 NaN-fillable |
| Cohort filter (extraction-time) | `vital_flag == 1` in offset xlsx | Upper bound 8,261 encounters with `.vital` files |
| Normalization | Robust quantile (p0–p100) | Same as MIMIC-III pipeline |
| Train/test split | 70/15/15 patient-level (by `Patient_ID_GE`) | All wave cycles of one patient stay together |
| Split seed | 42 | Reproducibility |

---

## Known Issues / Quirks

```
- Dirty-comma CSV: unquoted commas inside lab/drug names. Vendor remove_bad_commas
  before pl.read_csv (EHR_encounter_polars.py:12-63).
- Corrupt WaveStopTime: literal "2/17/69" sentinel. Fall back to BedTransfer_Out
  (mapValidWaveTime_polars.py:115-120).
- BP grid: .vital files give piecewise (range, len) chunks. We emit one event
  per raw (time, value) — no dense reconstruction needed for the sparse format.
- Per-session suffix grouping: D/M/S must co-exist for a BP session to be
  usable; R is optional (matching_vital_pipeline_BP.py:305).
- Two MRN-Mapping.csv datetime formats co-exist:
  "%m/%d/%Y %I:%M:%S %p" and "%Y-%m-%d %H:%M:%S". Try both.
- Zero-timestamp .vital files (common 2016+): anchor offset_sec to wave-cycle
  valid_start. Approximate but sufficient.
- 2018-06 cutoff: legitimate dataset end (cohort tail < 15/month afterward).
- FLOWSHEETVALUEFACT skipped first pass (rowkey curation deferred).
```

---

## Output Specification

```
datasets/ucsf/
├── processed/
│   └── {Patient_ID_GE}_{WaveCycleUID}/
│       ├── PLETH40.npy          [N_seg, 1200]   float16
│       ├── II120.npy            [N_seg, 3600]   float16
│       ├── time_ms.npy          [N_seg]         int64
│       ├── ehr_baseline.npy     [N_baseline]    structured   far history
│       ├── ehr_recent.npy       [N_recent]      structured   close history
│       ├── ehr_events.npy       [N_events]      structured   waveform-aligned
│       ├── ehr_future.npy       [N_future]      structured   post-waveform
│       └── meta.json
├── demographics.csv             one row per entity_id
├── manifest.json
├── pretrain_splits.json
├── downstream_splits.json
└── tasks/                       post-stage outputs
    ├── lab_estimation/          (cohort = any entity with ≥1 lab event in wave window)
    ├── vital_estimation/        (cohort = any entity with ≥1 vital event in wave window)
    └── ca_prediction/
        ├── cohort.json
        ├── splits.json
        └── extra_events/
            ├── {pid}.npy
            ├── {pid}.baseline.npy
            ├── {pid}.recent.npy
            └── {pid}.future.npy   forecasting labels — LEAKAGE if used as input
```

## EHR Trajectory Files

All four files share `EHR_EVENT_DTYPE`, sorted by `time_ms` ascending.
`seg_idx` is a real segment index only in `ehr_events.npy`; the other three
use sentinel values so accidental `signal[seg_idx]` fails loudly.

| File | Time window | `seg_idx` value |
|---|---|---|
| `ehr_baseline.npy` | `[max(episode_start, wave_start − baseline_cap), wave_start − context_window)` | `INT32_MIN` (-2147483648) |
| `ehr_recent.npy`   | `[wave_start − context_window, wave_start)` | `INT32_MIN + 1` |
| `ehr_events.npy`   | `[wave_start, wave_end]` | searchsorted index in `[0, N_seg)` |
| `ehr_future.npy`   | `(wave_end, min(episode_end, wave_end + future_cap)]` | `INT32_MIN + 2` |

Episode boundaries (`mapValidWaveTime_polars.py:103-120`):

```
episode_start_ms = ValidStartTime = max(BedTransfer_In, WaveStartTime)
episode_end_ms   = ValidStopTime  = min(BedTransfer_Out, WaveStopTime)
                                     # WaveStopTime falls back to BedTransfer_Out
                                     # if it equals the 2/17/69 corruption sentinel
```

Defaults (per `physio_data/ehr_trajectory.py`, overridable):
- `context_window_ms` = 24 h
- `baseline_cap_ms`   = 30 d
- `future_cap_ms`     = 7 d

`meta.json` includes: `n_events`, `n_baseline`, `n_recent`, `n_future`,
`n_baseline_vars`, `n_recent_vars`, `n_future_vars`,
`context_window_ms`, `baseline_cap_ms`, `future_cap_ms`,
`has_future_actions`, `has_future_sofa`, `has_future_sepsis_onset`,
`episode_start_ms`, `episode_end_ms`, `ehr_layout_version`.

**Actions (var_id 200–299)** are populated in `ehr_events.npy` only first pass
(`has_future_actions == false`). Extending to baseline/future is a follow-up
post-stage that re-queries `Filtered_Medication_Orders_New` for the full
encounter window.

---

## Downstream Tasks (UCSF scope)

Canonical pipeline stays task-agnostic. Task-specific cohorts and labels live
under `processed/tasks/` as post-stages — extract broadly, filter narrowly.

1. **Lab estimation** — predict lab values from waveforms. Uses canonical
   `ehr_events.npy` filtered to `var_id ∈ [0, 99]`. No cohort restriction;
   any entity with ≥1 lab event in the wave window is usable.

2. **Vital estimation** — predict vitals from waveforms. Uses canonical events
   filtered to `var_id ∈ [100, 199]`, with emphasis on invasive ABP
   (110–113) since UCSF provides them densely.

3. **Cardiac arrest (CA) prediction** — binary outcome.
   Cohort source: `/labs/hulab/mxwang/data/ucsf_EHR/bedanalysis_waveformExtraction/Output_new/ValidWaveTime_allEnc_eventtime.csv`
   (8,272 rows, 3,712 patients; 530 CA-positive rows from 192 patients;
   datetimes are GE-shifted, no further offset correction needed).
   Post-stage at `processed/tasks/ca_prediction/` writes:
   - `cohort.json` joined to canonical entities by `{Patient_ID_GE}_{WaveCycleUID}`
   - `splits.json` patient-level 70/15/15 stratified by CA label
   - optional `extra_events/{pid}.npy` if derived scores are needed

---

## Demographics CSV

`{output_dir}/demographics.csv`, one row per `entity_id`. Categorical columns
stored as raw strings; consumers encode to integer IDs at load time (0
reserved for unknown/pad). Schema as in the Demographics section above.

---

## References

- Local exploration notes: `datasets/ucsf/explore/README.md` (full Step 0a/0b/0c findings)
- Old extraction code (reference, will be partly vendored): `/home/mxwan/workspace/ucsf_ehr_code/`
- Remote vendored binary readers: `/labs/hulab/mxwang/data/ucsf_EHR/bedanalysis_waveformExtraction/{binfilepy,vitalfilepy}/`
- Offset linkage: `/labs/hulab/UCSF/encounter_date_offset_table_ver_Apr2024.xlsx`
- CA cohort labels: `/labs/hulab/mxwang/data/ucsf_EHR/bedanalysis_waveformExtraction/Output_new/ValidWaveTime_allEnc_eventtime.csv`
- Step 0c demo: `datasets/ucsf/explore/demo_214688354794344_38286/` (visually verified alignment)
