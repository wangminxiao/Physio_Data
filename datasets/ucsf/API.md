# UCSF Institutional Dataset API

De-identified UCSF ICU dataset hosted at Emory under `/mnt/localdata/storage/UCSF/`
(the former `/mnt/localdata/storage/UCSF/`) on DREAM lab node `xhu40-n01`, the only node that mounts it. Span: 2012-03 → 2018-06. See
`datasets/ucsf/explore/README.md` for the underlying scan that motivated every
parameter below.

## Data Sources

### Waveform

| Field | Value |
|-------|-------|
| Format | `.adibin` (binfilepy, 240 Hz) + `.vital` (vitalfilepy, 0.5 Hz monitor streams) |
| Location | `/mnt/localdata/storage/UCSF/{Wynton_folder}/DE{Patient_ID_GE}/{bed_subdir}/` |
| Organization | `{YYYY-MM}-deid/DE{Patient_ID_GE}/{session}/...` (70 cohort folders, 26,021 patient dirs) |
| Patient ID field | `Patient_ID_GE` (parsed from `DE{...}` directory name) |
| Time reference | `.adibin` header `{Year, Month, Day, Hour, Minute, Second}` (naive local, GE-shifted) |
| Vendored readers | `/mnt/localdata/storage/mxwang/data/ucsf_EHR/bedanalysis_waveformExtraction/{binfilepy,vitalfilepy}/` (not on PyPI; copy into `workzone/ucsf/`) |

### EHR — Clinical Tables

All under `/mnt/localdata/storage/UCSF/rdb_new/`, latin-1 `.txt` shards, **dirty-comma CSV
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
`/mnt/localdata/storage/UCSF/rdb_database/FLOWSHEETROWDIM_New/FLOWSHEETROWDIM_New.csv`
(101k rows) maps rowkeys to variable names, but each clinical variable maps to
many rowkeys (SpO2 alone has 10+). Curated many-to-one mapping is follow-up
work. `.vital` monitor streams at 0.5 Hz cover the same vitals densely.

### Patient ID Linkage

```
Waveform:  Patient_ID_GE          (DE{...} directory name)
EHR:       Patient_ID + Encounter_ID
Linkage:   /mnt/localdata/storage/UCSF/encounter_date_offset_table_ver_Apr2024.xlsx
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
| `.adibin` waveform | header struct (wall clock on the GE calendar, **absolute shift by `offset_GE` days**) | placed on the UTC-continuous grid by `clock.ge_wall_to_grid_ms` |
| `.vital` waveform | offset_sec from file start (**file start shifted by `offset_GE`**) | zero-timestamp files: anchor offset_sec to wave-cycle `valid_start` |
| EHR tables | `MM/DD/YYYY HH:MM:SS` or `YYYY-MM-DD HH:MM:SS` (naive local, **shifted by `offset` days**) | two formats co-exist — try both |
| `MRN-Mapping.csv` | same two datetime formats | wall-clock shift by `offset_GE` (ADT side): ±60 min vs the monitor headers when the real and GE dates differ in DST |

**Critical alignment rule** — see [`ALIGNMENT.md`](ALIGNMENT.md) (verified 2026-09-26; all conversions live in
`workzone/ucsf/clock.py`). The three de-identified clocks were shifted with different arithmetic: monitor files
(`.adibin`/`.vital`) in absolute (UTC) time, EHR and ADT tables on the wall clock. The former rule
`GE_time = EHR_time − (offset_GE − offset) days` is exact only when the real date and the GE-shifted date share a
DST state and is off by ±60 min for 45 % of encounters. The entity grid is **UTC-continuous** (`time_base =
"utc_continuous"`, Stage B v2): origin wall clock + real elapsed time. Convert with

```python
from clock import ehr_wall_to_grid_ms, real_wall_to_grid_ms, adt_wall_to_grid_ms
grid = ehr_wall_to_grid_ms(t_ehr_wall_ms, offset, offset_GE, episode_start_ms)     # labs, flowsheet, orders
grid = real_wall_to_grid_ms(t_real_wall_ms, offset_GE, episode_start_ms)           # Code Blue CodeTime
grid = adt_wall_to_grid_ms(t_adt_wall_ms, offset_GE, episode_start_ms)             # MRN-Mapping / ValidWaveTime
```

Never subtract days by hand, and never use the ValidWaveTime `EventTime` column (it also carries a spurious
+7/+8 h UTC conversion). `meta.json` records `time_base`, `dst_switch_in_cycle`, `grid_utc_offset_min`;
`meta.labs.time_rule = "utc_continuous_v2"` marks EHR files converted with the corrected rule.

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

## Clock fix applied 2026-09-27 (both stores)

* `ucsf_all`: Stage B v2 / C v2 / G rerun on the 663 cycles that straddle a DST switch on the GE calendar; gates B, C, F pass;
  manifest refreshed (`time_base`, `dst_switch_in_cycle`, `grid_utc_offset_min`, `stage_b_version`), splits unchanged; BeeGFS
  copy re-synced. All other entities are byte-identical to the 2026-09-10 build (their grid is the same).
* `ucsf`: Stage A/D/E rerun for every entity with `clock.ehr_wall_to_grid_ms` (labs shifted by ±60 min in ~49 % of entities),
  straddling cycles' waveform/vitals rebuilt, `tasks/ca_prediction` rebuilt from Code Blue times, manifest re-validated (4,913
  valid after the all-NaN rule; split membership kept). See `ALIGNMENT.md` §6 for the full table.

## `ucsf_all/tasks/ca_risk` — cardiac-arrest risk prediction (2026-09-27)

Built by `workzone/ucsf/stage_f_ca_risk.py` (`slurm/ucsf_all_ca_risk.sbatch`); design notes in the task's `README.md`.

* Universe: every valid wave cycle of the 3,754 patients held out of pretrain train (ValidWaveTime study cohort ∪ 235
  Code Blue CPA patients) — the FM pretrained on `ucsf_all` never saw them.
* Label `has_ca` (patient level) = Code Blue TypeCode CPA; `event_grid_ms` = CodeTime on the entity's UTC-continuous
  grid (may lie outside the cycle: see `event_in_cycle`, `event_offset_from_wave_end_min`). Auxiliary t0′ detector
  markers (`ecg_collapse_offset_min`, `t0_prime_offset_min`, `t0_prime_quality`) for QA only.
* Pre-event coverage per positive entity: `ppg_cov_{1,6,12,24}h`, `ecg_cov_{6,24}h`, `last_ppg_gap_min`,
  `ppg_hours_before_event`, `usable_pre_event` (gap ≤ 30 min & 6-h PPG ≥ 50 %); `self_control_end_ms` = event − 24 h.
* Controls: held-out non-CPA patients (`has_ca = 0`) + self-control windows before `self_control_end_ms`; exclude
  everything after the event.
* Split: train / test only (no val), grouped by patient, stratified by `has_ca`, 30 % test drawn from pretrain TEST
  patients only; `splits.json` has `train`, `test` (and an empty `val`) entity lists plus per-split counts.

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

- `.adibin` native rates differ per channel although every file is stored at 240 Hz:
  ECG 240 Hz native; PPG (`SPO2`) 60 Hz native, upsampled by 3 linear steps + 1 repeat;
  `RR` 60 Hz sample-and-hold; arterial/venous pressures 120 Hz linear ×2 (RAW_FORMAT.md §1).
- 37 % of `.vital` files have a zero header time (filename timestamp also zero); 25 % of
  all files additionally switch to absolute seconds-since-0001 on a different calendar
  part-way through (RAW_FORMAT.md §2).
- `NBP-S/D/M` are 2 s hold streams of the last cuff reading, not intermittent events.

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

## All-raw paired store (`ucsf_all`, 2026-09)

Second store built from the same raw tree without any EHR linkage: **every wave
cycle** (49,501 `(Patient_ID_GE, WaveCycleUID)` pairs across 26,021 patient dirs),
paired PPG + ECG + machine-sampled vitals. Raw-format facts behind every choice:
`datasets/ucsf/explore/RAW_FORMAT.md`. Config section `ucsf_all` in
`workzone/configs/server_paths.yaml`; output `/mnt/localdata100tb/physio_data/ucsf_all/`.
Entity ids are the same `{Patient_ID_GE}_{WaveCycleUID}` as the CA store, on purpose.

| Stage | Script | Output |
|---|---|---|
| A | `workzone/ucsf/stage_a_all_raw.py` | `valid_wave_window.parquet` (one row per wave cycle: window from the `.adibin` headers, `adibin_files` / `vital_files` lists, zero-time flags, MRN-Mapping extras); per-folder shards for resume |
| B | `workzone/ucsf/stage_b_adibin.py --dataset ucsf_all` | `PLETH40.npy`, `II120.npy`, `time_ms.npy`, `meta.json` (unchanged canonical channels; grid anchored at the first `.adibin` sample, whole seconds) |
| C | `workzone/ucsf/stage_c_vitals_hf.py` | `vitals_hf.npy`, `vitals_hf_abp_src.npy`, `nbp_events.npy`, `meta.vitals_hf` / `meta.nbp_events` |
| G | `workzone/ucsf/stage_g_ehr_hf.py --cadence pair --include-nbp` | `ehr_hf.npy` per entity — **MIMIC-III-compatible** monitor-numerics events (`EHR_EVENT_DTYPE`, var_ids 150–156 from `vitals_hf`, 157–159 from `nbp_events`), pair-end convention of `mimic3/stage3b_extract_numerics.py` (seg_idx = 2k+1, t = time_ms[2k+1] + 30 s, nearest reading within 30 s); `meta.ehr_hf` section; 755 M events over 37,985 entities |
| tasks | `workzone/common/build_estimation_task.py --spec task_specs/vital_est_hf.yaml` then `workzone/mimic3/build_abp_hf_task.py --src tasks/vital_est_hf/splits.json --min-abp 30 --write` | `tasks/vital_est_hf/{cohort,splits}.json` (targets 150–156; 37,916 entities = 20,731 / 8,679 / 8,506) and `tasks/abp_hf/{cohort,splits}.json` (ABPm_hf ≥ 30 ticks; 22,303 = 12,116 / 5,118 / 5,069) — same files and keys phase-4 consumes for MIMIC-III |
| F | `workzone/ucsf/stage_f_manifest.py --dataset ucsf_all` | `manifest.json` (+ `in_ca_cohort`, `has_ca` — **patient-level**: the ValidWaveTime CSV repeats a patient's single `EventTime` on every cycle row, so `has_ca=1` marks all cycles of the 192 CA patients; onset-anchored labels need the unique (patient, time) event mapped by time onto the patient's cycles), `pretrain_splits.json`, `downstream_splits.json` — grouped by `patient_id_ge`; **the cardiac-arrest study cohort (ValidWaveTime CSV: cases and study controls, ~3.7 k patients) is never in `train`** and is split 50/50 between `val` and `test` (`--holdout-val-frac`); `--splits-only` rebuilds splits from an existing manifest in seconds |

Jobs run only on `xhu40-n01` via `sbatch -p xhu40-b.q` (`workzone/ucsf/slurm/ucsf_all_*.sbatch`,
submitted from the BeeGFS mirror `/projects/mwang80/staging/Physio_Data/workzone/ucsf/logs`).

**Verification gates.** Every stage is followed by `workzone/ucsf/verify_stage_{a,b,c,f}.py`
(run as `ucsf_all_verify.sbatch <stage>`), which exits non-zero on any failed check and writes
`verify_stage_<s>.json` to the intermediate dir. The stages are chained with Slurm
`--dependency=afterok:` (B → verify B → C → verify C → F → verify F), so a failed gate stops the
pipeline by itself. Gate contents:

| Gate | Checks |
|---|---|
| A | shards = folders, entity ids unique, positive windows, `adibin_files` non-empty, coverage ≤ 1.05, duration median 5–100 h, ≥ 90 % of wave entities have `HR.vital`, header spot-check (30), Stage B disk need ≤ 60 % of free space |
| B | errors + killed ≤ 1 %, ok ≥ 90 %; 300-entity sample: shapes/dtypes/contiguity, 30 s `time_ms` steps, `n_seg` = ⌊min(dur, 14 d)/30⌋, no inf, NaN medians < 0.3, II |p99| 50–5000 µV, PLETH p50 50–4095 and non-constant |
| C | ok ≥ 95 %, no `no_meta`; 300-entity sample: sidecar shapes/dtypes, NBP events sorted and in range, HR valid median ≥ 0.8, ≥ 90 % of class-C non-NBP files re-anchored, NBP events/h 0.3–8; 40-entity physiology: ECG-derived HR vs `vitals_hf` HR median corr ≥ 0.6, lag −2…8 s, zero-time entities corr ≥ 0.5 |
| F | manifest/splits present, failed ≤ 0.5 %, valid ≥ 99 % of Stage B ok, all entities carry `vitals_hf`, splits disjoint by entity and patient, downstream lists consistent |

### Result of the first full build (2026-09-10, gates A–F all PASS)

| Quantity | Value |
|---|---|
| Wave cycles enumerated (Stage A) | 41,448 entities / 21,677 patients; 39,086 with II + SPO2 |
| Stage B | ok 38,744 (38,661 + 83 re-run after a `_place` bug fix), skip_short 2,670 (< 5 min or no valid file) |
| Stage C | 38,778 entities with `vitals_hf`; 8,638,158 NBP change events; `.vital` timing classes A / B / C = 69 % / 14 % / 18 % of files; all class-C non-NBP tails re-anchored |
| Stage F valid entities | **37,985** (793 rejected for an all-NaN channel); 21,536 patients (8,724 with several wave cycles) |
| Waveform volume | 298,583,061 segments = **2,488,192 h**; store ≈ 2.8 TB + sidecars |
| Cardiac-arrest study cohort in the store | 8,216 entities / 3,711 patients (394 CA-positive entities) — **none in `train`**: val 4,208 (226 pos), test 4,008 (168 pos) |
| Pretrain splits (patient-grouped) | train 20,773 entities / 12,478 patients · val 8,696 / 4,530 · test 8,516 / 4,528 (non-cohort patients 70/15/15, cohort patients val/test 50/50) |
| Physiological check (gate C) | ECG-derived HR vs `vitals_hf` HR: median corr 0.82 (header-time files 0.84, zero-time 0.64), numeric lags waveform by 4 s |
| ICU units (all adult; "NICU" = UCSF Neuro ICU) | 11NICU 9,406 · 13ICU 8,084 · 9ICU 7,857 · 10ICC 6,818 · 8NICU 5,820 entities; per-unit median HR 75–88 bpm — `unit` field in `manifest.json` |

Location: `/mnt/localdata100tb/physio_data/ucsf_all/` (xhu40-n01 only; not backed up) and a full copy at
`/projects/xhu40-cdsfm/physio_data/ucsf_all/` (BeeGFS, every DREAM node; 3.02 TB, 271,449 files, made 2026-09-11). Gate reports:
`workzone/outputs/ucsf_all/verify_stage_{a,b,c,f}.json` in the clone. Optional copy to BeeGFS:
`workzone/ucsf/slurm/ucsf_all_copy_to_projects.sbatch`.

### `vitals_hf.npy` — dense monitor numerics

`float32 [N_seg, 15, n_var]`, C-contiguous, NaN = no reading. Slot `k` of segment `i`
covers `[time_ms[i] + 2k s, +2 s)`; the `.vital` streams are a 2 s integer grid on the
same clock as the waveforms, so one reading lands in one slot (last wins if two).
Values are raw monitor readings filtered only by the registry `physio_min/max`.

| Axis 2 index | var_id | Name | `.vital` source | Notes |
|---|---|---|---|---|
| 0 | 150 | HR_hf | `HR` | ECG heart rate; lags the waveform by 2–4 s (monitor averaging) |
| 1 | 151 | SpO2_hf | `SPO2-%` | |
| 2 | 152 | RR_hf | `RESP` | impedance respiration rate |
| 3 | 153 | ABPs_hf | `AR1-S` → `AR2-S` → `AR3-S` → `FE1-S` → `FE3-S` | one line per slot, first available in that order |
| 4 | 154 | ABPd_hf | same line as the slot's `-S` | |
| 5 | 155 | ABPm_hf | same line | |
| 6 | 156 | PULSE_hf | `SPO2-R` | oximeter pulse rate |
| 7 | 113 | PR_art | same line, `-R` | arterial pulse rate (no MIMIC counterpart) |

`vitals_hf_abp_src.npy` (`uint8 [N_seg, 15]`) says which line filled each ABP slot
(0 none, 1 AR1, 2 AR2, 3 AR3, 4 FE1, 5 FE3); two lines are never mixed in one slot.

`nbp_events.npy` (`EHR_EVENT_DTYPE`, var_id 157/158/159 = NBPs/d/m_hf): the NBP streams
repeat the last cuff reading every 2 s, so only value changes are kept as events; the
event time is moved to the end of the matching `CUFF` inflation burst when one exists
within the preceding 5 min.

### `.vital` timing classes and policies (Stage C flags)

| Class | Share | Handling |
|---|---|---|
| A: header time + relative offsets | ~65 % | `t = header + offset` |
| B: zero header time, relative offsets | ~10 % | `--zero-anchor adibin_first` (default): anchor = entity `episode_start_ms` = first `.adibin` of the wave cycle (median error ≈ 5 s); `skip` drops them |
| C: B whose offsets switch to absolute seconds-since-0001 part-way | ~20 % of HR files, cohorts 2016–2018 only | relative part as B; absolute tail per `--abs-tail drop` (default) or `reanchor` (`abs_first = rel_last + 2 s`; validated on 248 files — re-anchored tails end at the waveform end, median +12 s; degenerate NBP tails are always dropped). Recommended: `reanchor` |

`meta.vitals_hf.files[suffix]` records class, policy, `n_rel`, `n_abs`, `n_in_window`.

### Caveats carried over to the CA store

- `II120` values are µV (header scale 2.44 is µV/LSB although the header unit says mV).
- Saturated samples (|x| > 65,000 µV) used to overflow float16 to ±inf (≈5 ppm of
  `II120` in the CA store); Stage B now stores them as NaN.
- The CA store's Stage C skipped every zero-time `.vital` file (classes B and C), so
  its `vitals_events.npy` covers only ~2/3 of the files.

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
   Cohort source: `/mnt/localdata/storage/mxwang/data/ucsf_EHR/bedanalysis_waveformExtraction/Output_new/ValidWaveTime_allEnc_eventtime.csv`
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
- Remote vendored binary readers: `/mnt/localdata/storage/mxwang/data/ucsf_EHR/bedanalysis_waveformExtraction/{binfilepy,vitalfilepy}/`
- Offset linkage: `/mnt/localdata/storage/UCSF/encounter_date_offset_table_ver_Apr2024.xlsx`
- CA cohort labels: `/mnt/localdata/storage/mxwang/data/ucsf_EHR/bedanalysis_waveformExtraction/Output_new/ValidWaveTime_allEnc_eventtime.csv`
- Step 0c demo: `datasets/ucsf/explore/demo_214688354794344_38286/` (visually verified alignment)
