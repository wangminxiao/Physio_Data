# UCSF Institutional Dataset — Exploration Notes

This is Step 0a research, extended in-place with Step 0b findings from the structural
scan on `bedanalysis.bmi.emory.edu` (see
`/labs/hulab/mxwang/Physio_Data/workzone/ucsf/explore/dataset_profile.json` for the
raw scan output).

## Step 0b headline numbers

- **70 Wynton cohort folders** `{YYYY-MM}-deid/` spanning 2013-03 → 2018-12. Early
  folders average ~400 patient dirs; 2018-06 onward drops sharply (186, then ~8/month).
- **26,021 `DE*` patient dirs** on filesystem; **22,089 unique `Patient_ID_GE`** in the
  offset xlsx; **27,841 unique `Encounter_ID`**. Offset xlsx covers 71 Wynton folders.
- **~981,462 `.adibin` files** total (cached in `/labs/hulab/UCSF/adibin_file_count.json`).
- **`.adibin` channels are RICHER than the old code extracted**: every probe shows
  `I, II, III, V, AVR, AVL, AVF, SPO2, RR` (= 7 ECG leads + PPG + respiration) at
  240 Hz. Some files add invasive waveforms (`CVP1`, `AR2`) also at 240 Hz.
  **Old pipeline only extracted II + SPO2.**
- **`.vital` suffix variants** (39 observed in 10-patient sample, sorted by frequency):
  `HR, NBP-D/M/S, CUFF, RESP, PVC, SPO2-%, SPO2-R, AR{1,2,3}-D/M/S/R, CVP{1,2,3},
  TMP-{1,2}, SP{2,3}, PA2-D/M/S, CPP1, ICP1, ST-I/II/III/V1/V2/V3`. Median cadence
  confirmed at **2.0 sec (0.5 Hz)**.
- **EHR tables** (all under `/labs/hulab/UCSF/rdb_new/`, latin-1 `.txt` shards):
  `Filtered_Lab_New` (4,054 shards, now with **LOINC_Code** column for robust mapping),
  `Filtered_Medication_Orders_New` (782), `Filtered_Diagnoses_New` (1,018, with
  ICD9+ICD10), `Filtered_Encounters_New` (447, admission/discharge times),
  `Filtered_Billing_New` (6,107, CPT), `Filtered_Procedure_Orders_New` (3,508),
  **`FLOWSHEETVALUEFACT`** (49,725 — Epic flowsheet, the UCSF equivalent of MIMIC
  CHARTEVENTS, keyed by `FlowsheetRowKey`; 223 distinct rowkeys seen in just 3
  shards).
- Per-patient extras: `MRN-Mapping.csv` (bed/wave cycle linkage) and `Alarms.csv`
  (alarm events with start/stop and `WavecycleuID`).

## Provenance

De-identified UCSF ICU data hosted at Emory under `/labs/hulab/UCSF/`. Date columns
are shifted per patient via `encounter_date_offset_table_ver_Apr2024.xlsx`; all
timestamps we consume are the already-shifted ones, so no re-offsetting is required
inside the pipeline.

## Data sources

### Waveforms (`/labs/hulab/UCSF/{Wynton_folder}/DE{Patient_ID_GE}/{bed_subdir}/`)

| Format | Reader | Rate | Channels (observed) |
|---|---|---|---|
| `.adibin` | `binfilepy.BinFile` (vendored at `/labs/hulab/mxwang/data/ucsf_EHR/bedanalysis_waveformExtraction/binfilepy/`) | **240 Hz uniformly** | `I, II, III, V, AVR, AVL, AVF` (ECG 7-lead) + `SPO2` (PPG) + `RR` (respiration waveform); occasional `CVP1`, `AR2` invasive pressure waveforms |
| `.vital` | `vitalfilepy.VitalFile` (same vendor path) | **suffix-dependent** — see table below | numeric monitor streams |

**`.vital` sample rate by suffix** (verified across 95 files / 6 patients, 2013–2017 cohorts):

| Suffix | Cadence | Notes |
|---|---|---|
| `HR`, `RESP`, `PVC`, `SPO2-%`, `SPO2-R`, `CUFF`, `TMP-1/2`, `AR{1,2,3}-{D,M,S,R}`, `CVP{1,2,3}`, `ST-{I,II,III,V1,V2,V3}`, `ICP1`, `CPP1`, `PA2-*` | **0.5 Hz** (exactly 2.000 s dt) | Continuous device stream |
| **`NBP-S` / `NBP-D` / `NBP-M`** | **median ~15 min, range 6 min – 1 h** | **Intermittent cuff cycling** — each event = one cuff reading |

Implication: NBP rows have the same **clinical semantic** as MIMIC nurse-charted
NBPs/NBPd/NBPm (cuff readings, just denser in UCSF than in MIMIC). They go to
var_ids **104/105/106** (existing). Invasive `AR*-*` streams at 0.5 Hz are a
different modality entirely → new ids **110/111/112/113** (ABPs/d/m/PR_art).

**`.vital` tuple layout**: `readVitalDataBuf` returns 4-tuples
`(value, offset_sec, sentinel_missing=-999999, constant=32768)`. Use only the first
two fields.

**Full `.vital` suffix taxonomy** (clinical grouping):

| Group | Suffixes | Notes |
|---|---|---|
| Standard monitor vitals | `HR, RESP, SPO2-%, SPO2-R, PVC` | HR from ECG, respiration rate, SpO2 %, pulse rate from ox, PVC count |
| Non-invasive BP | `NBP-D, NBP-M, NBP-S, CUFF` | Cuff readings; `CUFF` is a status channel, often sparse/bursty |
| Temperature | `TMP-1, TMP-2` | Rectal/skin probes (two sites) |
| Invasive arterial | `AR{1,2,3}-{D,M,S,R}` | Up to 3 simultaneous lines per patient |
| Invasive venous/pulmonary | `CVP{1,2,3}`, `PA2-{D,M,S}`, `SP{2,3}` | CVP, pulmonary artery pressure, misc. |
| Neuro ICU | `ICP1, CPP1` | Intracranial pressure / cerebral perfusion pressure |
| ECG ST analysis | `ST-{I,II,III,V1,V2,V3}` | ST-segment deviation per lead |

Filename patterns (from `matching_vital_pipeline_BP.py:304-309` and
`extract_wf_fast.py:206-211`):

```
.adibin:  DE{Patient_ID_GE}_{YYYYMMDDHHMMSS}_{session5}_{WaveCycleUID}.adibin
.vital :  DE{Patient_ID_GE}_{YYYYMMDDHHMMSS}_{session5}_AR{N}-{D|M|S|R}.vital
```

Each patient folder also contains `MRN-Mapping.csv` with per-row
`MRN_ADT, UnitBed, BedTransfer_In, BedTransfer_Out, WaveCycleUID, WaveStartTime,
WaveStopTime` — used to compute the valid wave window per wave cycle
(`mapValidWaveTime_polars.py`).

### EHR tables (`/labs/hulab/UCSF/rdb_new/`)

All `.txt`, latin-1, dirty-comma CSV (see quirks). Full header lists captured in
`dataset_profile.json::ehr_tables`.

| Subdir | Shards | Selected columns |
|---|---|---|
| `Filtered_Lab_New/` | 4,054 | `Patient_ID, Lab_Encounter_ID, Lab_Collection_Date, Lab_Collection_Time, Lab_Value, Lab_Unit, Lab_Name, Lab_Common_Name, Lab_Procedure_Code, Lab_Procedure_Name, LOINC_Code, LOINC_Name, Lab_Component_ID, cohort` |
| `Filtered_Medication_Orders_New/` | 782 | `Patient_ID, Medication_Order_Start_Date/Time, ..._End_Date/Time, Medication_Name, Medication_Generic_Name, Medication_Therapeutic_Class, Medication_Order_Quantity, Medication_Order_Minimum_Dose, Medication_Order_Maximum_Dose, Medication_Strength, Medication_Form, Medication_Route, Medication_Orders_Encounter_ID, cohort` |
| `Filtered_Diagnoses_New/` | 1,018 | `Patient_ID, Diagnosis_Start_Date, ICD9_Code, ICD10_Code, Diagnosis_Name, Diagnosis_Type, Diagnoses_Encounter_ID, Primary_Coded_Diagnosis, Principal_Problem_Diagnosis, cohort` |
| `Filtered_Encounters_New/` | 447 | `Encounter_ID, Patient_ID, _EncounterDate, Encounter_Admission_Time, _EncounterDischargeDate, Encounter_Discharge_Time, Encounter_LOS_in_Hours, Encounter_Admission_Type, Encounter_HospitalService, Encounter_DRG_Code, Encounter_Discharge_Disposition, Encounter_Is_Inpatient, Encounter_Age, cohort` |
| `Filtered_Billing_New/` | 6,107 | `Patient_ID, Billing_Encounter_ID, Billing_Service_Date, Billing_Procedure_Code, Billing_CPT_Level_*, cohort` |
| `Filtered_Procedure_Orders_New/` | 3,508 | `Patient_ID, Procedure_Orders_Encounter_ID, Procedure_Order_Date/Time, Procedure_CPT_Code, Procedure_HCPCS_Code, Procedure_Name, cohort` |
| `FLOWSHEETVALUEFACT/` | **49,725** | `FlowsheetRowKey, Value, Occurrence, Count, FlowDate, FlowTime, encounter_ID, patient_ID, cohort, U_ID` |

**`FLOWSHEETVALUEFACT` row-key mapping unknown.** The 223 distinct rowkeys seen so
far need a definition lookup (row 2 alone = 44% of rows). Possible sources: SQL
scripts in `/labs/hulab/UCSF/rdb_new/sql_scripts/`, a separate Epic metadata dump,
or empirical inference by correlating ranges against known vital definitions. **To
be resolved in 0b follow-up before we can map FLOWSHEETVALUEFACT events.**

**LOINC codes in labs** — robust mapping key, supersedes name-based matching. Use
`LOINC_Code` as primary lookup for `var_registry.json::variables[*].loinc` (to be
added alongside `mimic_itemids`).

Encoding is **latin-1**, and many rows contain **unquoted commas inside text fields**
(free-text lab notes, drug names, etc.). The old code repairs this with a regex
cleaner before `pl.read_csv`; we will vendor the same utility.
See `EHR_encounter_polars.py:12-63` (`remove_bad_commas`, `remove_bad_commas_quotes`).

### Linkage table

`/labs/hulab/UCSF/encounter_date_offset_table_ver_Apr2024.xlsx` columns used:
`Encounter_ID, Patient_ID, Patient_ID_GE, Wynton_folder`. This is the bridge
between the EHR side (keyed on `Patient_ID` and `Encounter_ID`) and the
waveform side (keyed on `Patient_ID_GE` + `Wynton_folder`).

### Optional event-time table (task-specific)

`/labs/hulab/mxwang/data/comet_UCSF_cardiac/SSM_FM_proj/CA_label_eventT_table.json`:
cardiac-arrest onset timestamps keyed by `Patient_ID_GE`. Used by the old pipeline
to restrict the wave window to `[event−14d, event+7d]`. For the canonical
pretraining dataset we will NOT apply this filter — it belongs in a `tasks/ca/`
post-stage (per skill: "extract EHR broadly, filter narrowly").

## Entity model

One output directory per **wave cycle within a bed episode**:

```
entity_id = {Patient_ID_GE}_{WaveCycleUID}
```

Rationale: the `MRN-Mapping.csv` aggregation in `mapValidWaveTime_polars.py:103-120`
collapses multiple `.adibin` / `.vital` files under the same `WaveCycleUID` into a
single valid `[ValidStartTime, ValidStopTime]` window — one physical ICU stay on
one bed. A single `Patient_ID_GE` can own multiple wave cycles (re-admissions,
bed transfers); treat them as separate entities for pretraining.

Episode boundaries (for 4-partition EHR layout):
- `episode_start_ms = ValidStartTime = max(BedTransfer_In, WaveStartTime)`
- `episode_end_ms   = ValidStopTime  = min(BedTransfer_Out, WaveStopTime)`
- `WaveStopTime` is sometimes corrupt (date in 1969, e.g., `2/17/69`) — fall back to
  `BedTransfer_Out` (`mapValidWaveTime_polars.py:115-120`).

Patient-level grouping for splits: all wave cycles sharing a `Patient_ID_GE` go into
the same split (no leakage).

## Signal plan

Core (matches skill canonical, minimum for pretraining):

| Target channel | Source | Source rate | Target rate | samples_per_seg |
|---|---|---|---|---|
| `PLETH40` | `.adibin` `SPO2` channel | 240 Hz | 40 Hz (resample_poly 1/6) | 1200 |
| `II120`   | `.adibin` `II`   channel | 240 Hz | 120 Hz (resample_poly 1/2) | 3600 |

PLETH-anchored per the skill: segments require `PLETH40` present; `II120` NaN-filled
if the ECG channel is absent. Since UCSF `.adibin` consistently has both, missingness
should be rare.

Optional extensions (out-of-scope for first pass, but the extra leads exist):
`I120, III120, V120, AVR120, AVL120, AVF120` — can be added without touching
existing arrays (new `.npy` files per entity). Respiration (`RR` channel at 240 Hz)
also available if a task needs it.

## Event plan

All events go into the shared `EHR_EVENT_DTYPE` (`time_ms, seg_idx, var_id, value`),
partitioned into `ehr_baseline` / `ehr_recent` / `ehr_events` / `ehr_future` by
relative time to the wave window.

### Labs (var_id 0–99, reuse registry)

Filter `Filtered_Lab_New` by `Patient_ID`; parse `Lab_Value` after stripping `%`;
map `Lab_Procedure_Code` / `Lab_Common_Name` to existing registry ids. The old
`process_lab_file` at `EHR_labtest_polars.py:55-86` gives the schema-override +
null-value list to reuse.

### Vitals (var_id 100–199, additions needed)

**Mapping principle after Step 0b:** same clinical semantic → same `var_id` regardless
of measurement cadence. Different clinical semantic (invasive vs non-invasive,
different vasculature, derived quantity) → new `var_id`.

**Reuse existing ids** — UCSF populates from BOTH FLOWSHEETVALUEFACT (nurse-charted,
coarse) AND `.vital` monitor streams (0.5 Hz, dense). Both are the same variable:

| existing id | variable | UCSF `.vital` suffix | UCSF FLOWSHEETVALUEFACT rowkey |
|---|---|---|---|
| 100 | HR          | `HR`     | TBD (awaiting rowkey lookup) |
| 101 | SpO2        | `SPO2-%` | TBD |
| 102 | RR          | `RESP`   | TBD |
| 103 | Temperature | `TMP-1` (primary), `TMP-2` (secondary) | TBD |
| 104 | NBPs        | `NBP-S`  | TBD |
| 105 | NBPd        | `NBP-D`  | TBD |
| 106 | NBPm        | `NBP-M`  | TBD |
| 107 | CVP         | `CVP1` (primary); `CVP2`, `CVP3` if present | TBD |

**New ids (clinically distinct from anything in MIMIC registry today):**

| proposed id | variable | unit | UCSF source | physio_min/max |
|---|---|---|---|---|
| 110 | ABPs   | mmHg | `.vital AR{1,2,3}-S` | 40 / 300 |
| 111 | ABPd   | mmHg | `.vital AR{1,2,3}-D` | 20 / 200 |
| 112 | ABPm   | mmHg | `.vital AR{1,2,3}-M` | 30 / 250 |
| 113 | PR_art | bpm  | `.vital AR{1,2,3}-R` | 20 / 300 |
| 114 | PVC_rate | /min | `.vital PVC` | 0 / 60 |
| 115 | SPO2_pulse_rate | bpm | `.vital SPO2-R` | 20 / 300 |

Multi-arterial-line handling: AR1 / AR2 / AR3 values are squashed into the single
ABPs/d/m/PR_art ids (a patient with 2 art lines yields 2× the event density for
those ids). Rationale: physiologically equivalent pressure measurements; the
distinction (radial vs femoral) is rarely task-relevant and would double schema
complexity.

**Deferred (reserve ids but not in first pass):** `PA_{S,D,M}` (pulmonary artery
pressure), `ICP`, `CPP`, `ST-{I,II,III,V1,V2,V3}` (ST-segment deviation per lead).
All rare enough to postpone until a task needs them.

## Downstream tasks (UCSF scope)

UCSF is used for three supervised tasks, not general pretraining. Canonical pipeline
stays task-agnostic; task-specific labels live under `processed/tasks/` as
post-stages:

1. **Lab estimation** — predict lab values from waveforms. Uses canonical
   `ehr_events.npy` filtered to var_ids 0–99. No task-specific cohort needed; any
   entity with ≥1 lab event in the wave window is usable.

2. **Vital estimation** — predict vitals from waveforms. Uses canonical events
   filtered to 100–199 (especially invasive ABP 110–113 which UCSF provides).

3. **Cardiac arrest (CA) prediction** — binary outcome. Cohort and labels come from
   `/labs/hulab/mxwang/data/ucsf_EHR/bedanalysis_waveformExtraction/Output_new/ValidWaveTime_allEnc_eventtime.csv`
   (8,272 rows, 3,712 patients; 530 CA-positive rows from 192 patients; datetimes
   GE-shifted). Post-stage at `processed/tasks/ca_prediction/` will write
   `cohort.json` (joined to canonical entities by `{Patient_ID_GE}_{WaveCycleUID}`),
   `splits.json` (patient-level 70/15/15 stratified by CA label), and optional
   `extra_events/{pid}.npy` if derived scores are needed.

### Actions (var_id 200–299)

From `Filtered_Medication_Orders_New`. Use `Medication_Order_Start_Date/Time` as
event time; `value` stores rate/dose (prefer `Medication_Order_Maximum_Dose` if
present, else `Medication_Order_Quantity`). Registry has `vasopressor_rate` (200)
and `fluid_rate` (201) today; we need to map UCSF med names to those. Need a
generic-name → var_id lookup. Step 0b will surface the top generic names observed
to decide what else to add.

### Scores (var_id 300–399)

None in the main pipeline. `sepsis_onset`, `SOFA_*` would be post-stage outputs.

## Non-trivial quirks (from the old code)

1. **Dirty-comma CSV repair** (`EHR_encounter_polars.py:12-63`): must run before
   `pl.read_csv` on the EHR `.txt` tables. Vendor this utility into
   `physio_data/` or into `workzone/ucsf/` if we decide it's UCSF-only.
2. **BP grid reconstruction** (`matching_vital_pipeline_BP.py:34-46`): `.vital`
   files give piecewise `(range, len)` chunks. Need to rebuild the timestamp grid
   per chunk before producing events. We DO NOT need to reconstruct a dense
   series — for the canonical sparse format we emit one event per raw (time, value)
   sample and let the adapter handle density at training time.
3. **Corrupt `WaveStopTime`**: `2/17/69` sentinel for missing stop. Fall back to
   `BedTransfer_Out`.
4. **Per-session suffix grouping** (regex at `matching_vital_pipeline_BP.py:305`):
   `D/M/S` must co-exist for a BP session to be usable; `R` is optional.
5. **Two datetime formats** in `MRN-Mapping.csv`
   (`mapValidWaveTime_polars.py:82-95`): `%m/%d/%Y %I:%M:%S %p` and
   `%Y-%m-%d %H:%M:%S`. Try both.
6. **Vendored readers**: `binfilepy/` and `vitalfilepy/` live in the old code tree
   and are not on PyPI. We'll copy them into `workzone/ucsf/` on the remote.
7. **Fahrenheit / unit conversion**: unclear yet whether UCSF Temperature is °F or
   °C. Confirm in 0b.

## Step 0b follow-up resolutions

Raw output: `workzone/ucsf/explore/dataset_profile_v2.json` on the server.

**1. FLOWSHEETVALUEFACT row-key lookup** — RESOLVED. Lookup lives at
`/labs/hulab/UCSF/rdb_database/FLOWSHEETROWDIM_New/FLOWSHEETROWDIM_New.csv`
(101,275 rows with columns `FlowsheetRowKey, IdType, Id, Name, Abbreviation,
ValueType, Unit, Description, DeidType`). However, one clinical variable maps
to **many** rowkeys — SpO2 alone has 10+ (`2=PULSE OXIMETRY`, `2281=UCSF R ANE SPO2 -1`,
`984/1007/1018=age-bucketed compiled`, etc.). Curated many-to-one var_id mapping
is substantial manual work. **Decision for first pass: skip FLOWSHEETVALUEFACT.**
`.vital` monitor streams at 0.5 Hz already cover HR/SpO2/RR/Temp/NBP/ABP/CVP
densely. Flowsheet integration becomes a later per-variable enhancement (add
rowkey list to a var_registry field).

**2. Offset xlsx semantics** — RESOLVED. Columns `offset` and `offset_GE` are
**integer day-shifts applied during de-identification**, different per encounter
(0 of 27,903 pairs equal):

- `offset` shifts **EHR timestamps** (`Encounter_Start_time`, lab/med dates).
- `offset_GE` shifts **waveform timestamps** (`bin_file_time`, `bed_transferin_time`,
  `alarm_time`, `MRN-Mapping.csv` times, `.adibin` header dates).

**To align EHR events to waveforms, add `(offset_GE - offset)` days to every EHR
timestamp** (or subtract from waveform side). Per-encounter deltas must be looked
up from the xlsx. This is *the* critical correction the old pipeline did not
explicitly document.

**3. `vital_flag`** — RESOLVED. `0` for 19,642 encounters, `1` for **8,261**
encounters. Very likely flags presence of `.vital` monitor-stream files. Filter
on `vital_flag == 1` yields the candidate pretraining cohort (upper bound ~8k
entities before requiring `.adibin` + EHR overlap).

**4. LOINC / lab coverage** — PARTIAL. `Lab_Common_Name` confirms all 17 registry
labs are present (Glucose, HEMATOCRIT, HEMOGLOBIN, PLATELETCOUNT, WBCCOUNT,
Magnesium, Phosphorus, PT, INR, PCO2, FIO2, PO2, ALT, AST, Potassium, Sodium,
Creatinine, Bilirubin, BICARBONATE, etc.). The proper LOINC scan was broken by
dirty-comma parsing (same issue the old code addressed with `remove_bad_commas_quotes`);
the real extractor must reuse that repair. **No blocker for 0c.**

**5. Medication mapping** — PARTIAL. Top generic names include the drugs we need
for existing action ids: fentanyl/hydromorphone/propofol/midazolam (analgesia /
sedation — not yet mapped), 0.9% NaCl / lactated Ringer's (for id 201 fluid_rate),
KCl/MgSO4 piggybacks, furosemide (diuresis). Vasopressors (norepi / phenylephrine)
will appear in the tail; confirm at extraction time. Mapping strategy: substring
match on `Medication_Generic_Name` + category via `Medication_Therapeutic_Class`.

**6. Timezone** — `.adibin` header stores `{Year, Month, Day, Hour, Minute, Second}`
fields directly (naive local). EHR tables are `MM/DD/YYYY HH:MM:SS` or
`YYYY-MM-DD HH:MM:SS` (two variants, both naive). After applying the offset-xlsx
day-shift difference, both are in the same shifted local reference — no
additional timezone correction needed.

**7. 2018-06 cutoff** — RESOLVED. The offset xlsx confirms this is a legitimate
dataset end: 2,450 encounters in 2018, with monthly counts dropping from 468
(Jan) → 373 (May) → 195+151 (Jun, two sub-folders `2018-06-deid` + `2018-06-deid-1`)
→ <15/month thereafter. Effective data span: **2012-03 → 2018-06**.

## Entity cohort — estimated upper bound

| Filter | Approx. count |
|---|---|
| Unique `Patient_ID_GE` in offset xlsx | 22,089 |
| Unique `Encounter_ID` in offset xlsx | 27,841 |
| Encounters with `vital_flag == 1` | **8,261** |
| Intersection with `.adibin` + PLETH channel + ≥1 EHR event | TBD (needs Stage 3 cross-check) |

Output dir convention: `{Patient_ID_GE}_{WaveCycleUID}`. One encounter can map to
multiple wave cycles (multiple bed stays). Split seed by `Patient_ID_GE`.

## References (local paths)

- `/home/mxwan/workspace/ucsf_ehr_code/EHR_encounter_polars.py` — EHR comma repair,
  per-patient grouping
- `/home/mxwan/workspace/ucsf_ehr_code/EHR_labtest_polars.py` — lab table parsing
- `/home/mxwan/workspace/ucsf_ehr_code/bedanalysis_waveformExtraction/mapValidWaveTime_polars.py`
  — wave-cycle valid-window computation
- `/home/mxwan/workspace/ucsf_ehr_code/bedanalysis_waveformExtraction/extract_wf_fast.py`
  — `.adibin` extraction, 240 Hz, II+SPO2
- `/home/mxwan/workspace/ucsf_ehr_code/bedanalysis_waveformExtraction/matching_vital_pipeline_BP.py`
  — `.vital` session grouping, BP alignment (to be rewritten as sparse events)
- `/home/mxwan/workspace/ucsf_ehr_code/bedanalysis_waveformExtraction/binfilepy/`,
  `vitalfilepy/` — binary-format readers to vendor
