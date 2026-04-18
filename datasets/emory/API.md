# Emory Institutional Dataset API

De-identified Emory ICU dataset on `bedanalysis.bmi.emory.edu` (user `mwang80`).
Waveforms under `/labs/collab/Waveform_Data/Waveform_Data/`, EHR under
`/labs/hulab/Emory_EHR/CDW_Pull_ICU_Data_Siva_version/`, sepsis cohort linkage
under `/labs/hulab/mxwang/data/sepsis/Wav/`. Span: ~2010–2023 (GE-shifted −30 y
on waveform side, see time format below). See `datasets/emory/explore/README.md`
and `alignment_report.json` for the verifications that motivated every parameter
below. First-pass cohort = sepsis task list (11,715 encounters); pretraining
expansion to the 64,750-encounter whole list is a drop-in extension.

## Data Sources

### Waveform

| Field | Value |
|-------|-------|
| Format | WFDB multisegment (Philips monitor export): `.hea` top header, `.{hea,mat}` per-segment waveforms, `.{hea,mat}` per-record dense numerics (`_0n` suffix). Dozens of derived annotations (`.qrs*`, `.abpsqi`, `.ecgsqi*`, `.ppgsqi`, `.af`, `.vf`, `.wabp`) — **out of scope for first pass**. |
| Location | `/labs/collab/Waveform_Data/Waveform_Data/{cohort_prefix}/{wfdb_record}/` where `cohort_prefix = wfdb_record.split("-")[0]` (e.g. `A063`, `B035`). |
| Organization | One `wfdb_record` = one physical ICU wave session, broken into fixed 8 h segments `{rec}_0000.{hea,mat}` … `{rec}_NNNN.{hea,mat}`, plus a single `{rec}_0n.{hea,mat}` that spans the entire record at 0.5 Hz. Top-level `{rec}.hea` lists segments with occasional `~` gap markers between them. |
| Patient ID field | `ENCOUNTER_NBR` in the cohort CSV; waveform folder names embed the record ID, not the patient. |
| Time reference | WFDB header `base_date/base_time` are naive, de-identified by `−30 years`. Apply `base_datetime + relativedelta(years=30)` and **treat the result as UTC** (empirically verified on encounter 359559206 — see below). |
| Vendored readers | `wfdb` Python package (v4.3.1 on remote); channel-name canonicalizer at `/labs/hulab/mxwang/data/vitals/emory_vital_lib.py:canonicalize_sig_names`. Strip the module-level `from IPython.display import display` before importing in a headless env. |

### EHR — Clinical Tables

All under `/labs/hulab/Emory_EHR/CDW_Pull_ICU_Data_Siva_version/`, standard UTF-8
CSV (no dirty-comma repair needed, unlike UCSF). Column names are UPPERCASE with
underscores. Dictionary at `data_dictionary.json` (98 KB) in the same folder.
JGSEPSIS_ENCOUNTER covers 995,598 encounters (superset of the 64,750 waveform
encounters; 99.2% overlap).

| Table | Size | Key columns | Notes |
|-------|------|-------------|-------|
| `JGSEPSIS_ENCOUNTER.csv` | 183 M | `ENCOUNTER_NBR, PAT_ID, AGE, HOSPITAL_ADMISSION_DATE_TIME, HOSPITAL_DISCHARGE_DATE_TIME, DIAGNOSIS_ICD10_CD, DIAGNOSIS_ICD10_DESC, INSURANCE_STATUS, ENCOUNTER_TYPE, ENTITY_HEALTHCARE_NM, PRE_ADMIT_LOCATION, DISCHARGE_TO` | Per-encounter admit/discharge bounds and primary Dx. |
| `JGSEPSIS_DEMOGRAPHICS.csv` | 279 M | `PAT_ID, DOB, GENDER, GENDER_CODE, RACE, RACE_CODE, ETHNICITY, ETHNICITY_CODE, DEATH_DATE` | **INSURANCE is in ENCOUNTER, not here.** |
| `JGSEPSIS_VITALS2.csv` | 5.0 G | `ENCOUNTER_NBR, PAT_ID, RECORDED_TIME, TEMPERATURE, TEMPROUTE, SBP_LINE, DBP_LINE, MAP_LINE, SBP_CUFF, DBP_CUFF, MAP_CUFF, PULSE, UNASSISTED_RESP_RATE, SPO2, CVP, END_TIDAL_CO2, O2_FLOW_RATE, DAILY_WEIGHT_KG, HEIGHT_CM` | Chart vitals, not waveform-rate. |
| `JGSEPSIS_LABS.csv` | 17 G | `ENCOUNTER_NBR, PAT_ID, PROC_CAT_ID, PROC_CAT_NAME, PROC_CODE, PROC_DESC, COMPONENT_ID, COMPONENT, LOINC_CODE, COLLECTION_TIME, RESULT_STATUS, LAB_RESULT_TIME, LAB_RESULT` | **`LAB_RESULT_TIME` / `LAB_RESULT` / `COMPONENT`** (not `RESULT_TIME / Component_value / Component_name` as the old `stp2` code assumed). |
| `JGSEPSIS_MEDS.csv` | 44 G | — | Deferred first pass; tackle as Stage H (like UCSF meds). |
| `JGSEPSIS_VENT.csv` | 256 M | `ENCOUNTER_NBR, RECORDED_TIME, VENT_MODE, FIO2, PEEP, VT, RR_SET, …` | Ventilator settings. |
| `JGSEPSIS_VENTDATA.txt` | 7.8 G | — | High-rate vent data; treat like vitals in Stage D. |
| `JGSEPSIS_FIO2.csv` | 17 M | `ENCOUNTER_NBR, RECORDED_TIME, FIO2` | Separate FiO2 stream. |
| `JGSEPSIS_OXYGEN.txt` | 3.4 G | — | O2-device deliveries (supplementary). |
| `JGSEPSIS_INOUTS_ALL.txt` | 2.2 G | — | Fluid I/O; actions (var_id 200+) candidate. |
| `JGSEPSIS_OUTPUT.txt` | 1.9 G | — | Urine/other output. |
| `JGSEPSIS_BEDLOCATION.csv` | 818 M | `ENCOUNTER_NBR, BED_IN_TIME, BED_OUT_TIME, BED_LOCATION` | Bed-transfer history → for ICU-presence filter. |
| `JGSEPSIS_DIAGNOSIS.csv` | 7.7 G | `ENCOUNTER_NBR, DIAGNOSIS_ICD10_CD, DIAGNOSIS_TYPE, DIAGNOSIS_DATE` | Diagnoses (no precise timestamps — baseline partition). |
| `JGSEPSIS_CULTURES.csv` | 180 M | `ENCOUNTER_NBR, COLLECTION_TIME, SPECIMEN, ORGANISM, …` | Culture events — for sepsis task post-stage. |
| `SBSEPSIS_RX.csv` | 1.6 G | — | Prescriptions. |

**Dense-ish chart-minute time series**:
`/labs/hulab/Emory_EHR/SBSEPSIS_TimeSeries.csv` (21.3 M rows, 129k encounters;
overlaps only 21,035 / 64,750 waveform encounters). Supplementary, not the
primary source.

### Cohort and label sources

`/labs/hulab/mxwang/data/sepsis/Wav/`:

| File | Rows | Unique enc. / pt. | Contents |
|------|------|-------------------|----------|
| `sepsis_cc_2025_06_13_all_collab.csv` | 1,042,708 | 64,750 / 51,904 | **whole list** — one row per 8h wfdb sub-segment. Columns: `empi_nbr, encounter_nbr, wfdb_record, sepsis_time_zero_dttm, wfdb_start, wfdb_end, bed_location_start, bed_location_end, wfdb_len_seconds, type ∈ {case, control, sepsis_patient_but_not_sepsis_this_encounter}, wfdb_dir`. `wfdb_start/end` carry an explicit `Z` UTC suffix. |
| `sepsis_cc_2025_06_13_all_collab_uniq_combine.csv` | 235,612 | 11,715 | **sepsis task list (first-pass cohort)** — one row per quality-gated 8h wave window. Columns: `row_index, encounter_nbr, wfdb_dir, wfdb_record, sepsis_time_zero_dttm, valid_start, valid_end, valid_duration_hour, valid_duration_sum, valid_ratio, type ∈ {case, control}`. `valid_start/end` naive UTC. **Does NOT contain `empi_nbr`** — join back to the whole list or `JGSEPSIS_ENCOUNTER.PAT_ID`→`empi_nbr` to attach the patient key. |

### Patient ID Linkage

```
Waveform (cohort CSV):  encounter_nbr + wfdb_record
EHR tables:             ENCOUNTER_NBR  (same as encounter_nbr after Int64 cast)
Patient ID:             whole_list.empi_nbr ↔ JGSEPSIS_*.PAT_ID
                        (JGSEPSIS.PAT_ID ≠ empi_nbr — distinct encoding; use
                         whole_list or JGSEPSIS_ENCOUNTER to bridge)

Output entity_id:       {empi_nbr}_{encounter_nbr}
```

One `encounter_nbr` may have multiple `wfdb_record`s (e.g. different beds);
one `wfdb_record` may be broken into multiple 8-h rows in the whole list. All
wfdb_records within the same encounter are **concatenated chronologically**
into one canonical entity. Gaps between records/segments surface as time_ms
jumps (not NaN padding). **Splits group by `empi_nbr`** to prevent patient
leakage across encounters.

### Time Format and Alignment

| Source | Raw format | Conversion to canonical UTC ms |
|--------|------------|--------------------------------|
| WFDB `base_datetime` (from `{rec}.hea` or any `{rec}_0XXX.hea` or `{rec}_0n.hea`) | naive, de-ID'd −30 y | `dt = base_datetime + relativedelta(years=30); ms = int(dt.replace(tzinfo=UTC).timestamp() * 1000)` |
| WFDB sample time | `base_ms + i / fs * 1000` | same (per-channel fs: 240 for waveforms, 0.5 for `_0n`) |
| List CSV `wfdb_start/end` | ISO8601 with `Z` | already UTC |
| List CSV `valid_start/end`, `sepsis_time_zero_dttm` | naive ISO8601 (no suffix) | **treat as UTC** (verified to match `wfdb_start` exactly) |
| EHR CSV `*_TIME`, `*_DATE_TIME`, `RECORDED_TIME`, `LAB_RESULT_TIME`, `COLLECTION_TIME`, `HOSPITAL_ADMISSION_DATE_TIME`, `HOSPITAL_DISCHARGE_DATE_TIME`, `DOB` | naive `MM/DD/YYYY HH:MM:SS` (DOB is `YYYY-MM-DD`) | `str.strptime → dt.replace_time_zone("America/New_York", ambiguous="earliest") → dt.convert_time_zone("UTC") → dt.epoch("ms")` |

**Alignment verification** (encounter 359559206, 680 EHR SBP_CUFF events vs
1,864 `_0n` NBP-S step events, wave window = 25 days):

- H1 hypothesis (WFDB+30y = UTC): median |Δvalue at next `_0n` step| = **0.0 mmHg**
- H2 hypothesis (WFDB+30y = NY local): median |Δvalue| = 16 mmHg
- → H1 confirmed; EHR uses `America/New_York` local.

**Cuff-cycle physics** (not a bug — do not "correct"): EHR charts at
cuff-start; `_0n` NBP-S steps at cuff-finish (p50 = 107 s later, p95 = 31 min
when cycles differ). Value at the post-step `_0n` equals EHR value exactly
(median |Δ| = 0 mmHg). Therefore **use EHR `SBP/DBP/MAP_CUFF` for cuff BP
events, never `_0n` NBP-S/D/M** — the latter is a HOLD signal, not an event
stream.

---

## Waveform Channels to Extract

Channel selection is per-segment: read `{rec}_0XXX.hea` `sig_name` list and
canonicalize with `emory_vital_lib.canonicalize_sig_names`. The example record
B035-0564111269 has `I, II, III, V, SPO2, RR` — consistent with typical Philips
ICU exports. Other records may additionally expose `Pleth`, `ABP`, `ART`,
`AR1–AR4`, `CVP*`. Channels that are absent in a segment become NaN-filled
after segmentation (PLETH-anchored rule: segments without PLETH are dropped).

| Source Channel | Source Rate (Hz) | Target Channel Name | Target Rate (Hz) | samples_per_seg (30 s) | Notes |
|---|---|---|---|---|---|
| `SPO2` (or `Pleth` after canonicalization) | 240 | `PLETH40` | 40 | 1200 | Primary modality. Anchor channel. `resample_poly(up=1, down=6)`. Scale: ADC gain 200.0 → divide by 200 (or use `wfdb.rdrecord(physical=True)`). |
| `II` | 240 | `II120` | 120 | 3600 | ECG Lead II. `resample_poly(up=1, down=2)`. Units mV. ADC gain 409.836. |

**Optional extensions** (first pass: skip; drop-in later as extra `.npy` files
without changing existing arrays):

- `I120, III120, V120`, `AVR/AVL/AVF120` — additional ECG leads.
- `ABP125` from `ABP` or `ART` waveform channels when present (invasive arterial
  line BP; candidate for a third canonical rate since the vanilla Philips
  `.hea` sample rate is 125 Hz for some sites, 240 Hz for others — verify).
- `RR240` — respiration waveform (lower clinical value; `RESP` in `_0n` at
  0.5 Hz is usually sufficient).

**Resample logic**: `resample_poly(signal, up, down)` with `gcd(source_rate,
target_rate)`. All stored float16, C-contiguous.

**Segment length / overlap**: 30 s, non-overlapping (same as MC_MED; unlike
UCSF / MIMIC-III which use 5 s overlap). Rationale: Emory sepsis task windows
are pre-gated to 8 h at `valid_ratio ≥ 0.9` in the uniq_combine CSV; overlap
does not buy more examples in this setting, and keeps N_seg cleanly divisible
by typical batch sizes. Revisit for pretraining expansion if training needs it.

---

## EHR Variables to Extract

All four partitions share the structured dtype
`(time_ms: int64, seg_idx: int32, var_id: uint16, value: float32)`.

### Labs (var_id 0-99) — from `JGSEPSIS_LABS.csv`

Filter by `COMPONENT_ID` (unambiguous) with `LOINC_CODE` as secondary
cross-check. 65 distinct `COMPONENT` strings seen on the demo encounter alone —
extract broadly and rely on the `var_registry` mapping to narrow to modelled
variables. First-pass mapping (extend as data permits):

| var_id | Variable | `COMPONENT` (LOINC where stable) | Unit | Physio range | Notes |
|--------|----------|----------------------------------|------|--------------|-------|
| 0 | Potassium | `Potassium Level` (2823-3) | mEq/L | 1.5 – 9 | |
| 1 | Calcium | `Calcium` (17861-6) | mg/dL | 5 – 15 | Total; ionized separate |
| 2 | Sodium | `Sodium` (2951-2) | mEq/L | 110 – 170 | |
| 3 | Glucose | `Glucose` (2345-7) | mg/dL | 20 – 800 | POC and lab distinguished at registry if needed |
| 4 | Lactate | `Lactate` / `Lactic Acid` (2524-7) | mmol/L | 0.3 – 20 | Arterial vs venous → same var_id |
| 5 | Creatinine | `Creatinine` (2160-0) | mg/dL | 0.1 – 20 | |
| 6 | Bilirubin | `Bilirubin Total` (1975-2) | mg/dL | 0.1 – 40 | Total only |
| 7 | Platelets | `Platelet Count` (777-3) | K/uL | 1 – 1000 | |
| 8 | WBC | `White Blood Cell Count` (6690-2) | K/uL | 0.1 – 100 | |
| 9 | Hemoglobin | `Hemoglobin` (718-7) | g/dL | 2 – 25 | |
| 10 | INR | `INR` (6301-6) | ratio | 0.5 – 15 | |
| 11 | BUN | `BUN` / `Urea Nitrogen` (3094-0) | mg/dL | 1 – 200 | |
| 12 | Albumin | `Albumin` (1751-7) | g/dL | 0.5 – 7 | |
| 13 | Arterial pH | `pH (Arterial)` (2744-1) | pH | 6.5 – 8.0 | ABG only |
| 14 | paO2 | `pO2 (Arterial)` (2703-7) | mmHg | 20 – 600 | |
| 15 | paCO2 | `pCO2 (Arterial)` (2019-8) | mmHg | 10 – 150 | |
| 16 | HCO3 | `Bicarbonate` / `CO2` (1963-8) | mmol/L | 2 – 60 | |
| 17 | AST | `AST` (1920-8) | U/L | 1 – 50000 | |
| 18 | ALT | `ALT` (1742-6) | U/L | 1 – 50000 | |

### Vitals (var_id 100-199)

Split by source: chart vitals from `JGSEPSIS_VITALS2` (per-measurement events),
dense monitor numerics from WFDB `_0n.mat` at 0.5 Hz. **Same clinical semantic
→ same var_id** — the `source` field in `meta.json` (per partition) notes
origin for downstream filters.

From `JGSEPSIS_VITALS2.csv` (chart events, `RECORDED_TIME`):

| var_id | Variable | Column | Unit | Physio range | Notes |
|--------|----------|--------|------|--------------|-------|
| 100 | HR | `PULSE` | bpm | 20 – 300 | Chart entry; WFDB `_0n.HR` populates the same var_id from monitor |
| 101 | SpO2 | `SPO2` | % | 20 – 100 | |
| 102 | RR | `UNASSISTED_RESP_RATE` | /min | 2 – 80 | `_0n.RESP` also maps here |
| 103 | Temperature | `TEMPERATURE` | °C | 25 – 45 | `TEMPROUTE` kept as metadata (oral / axillary / core) |
| 104 | NBPs | `SBP_CUFF` | mmHg | 30 – 280 | **Event source, not `_0n` NBP-S** |
| 105 | NBPd | `DBP_CUFF` | mmHg | 10 – 200 | |
| 106 | NBPm | `MAP_CUFF` | mmHg | 20 – 250 | |
| 110 | ABPs | `SBP_LINE` | mmHg | 30 – 300 | Invasive arterial line BP |
| 111 | ABPd | `DBP_LINE` | mmHg | 10 – 250 | |
| 112 | ABPm | `MAP_LINE` | mmHg | 20 – 280 | |
| 107 | CVP | `CVP` | mmHg | −10 – 40 | matches var_registry id 107 (UCSF CVP1-3) |
| 116 | EtCO2 | `END_TIDAL_CO2` | mmHg | 0 – 120 | |
| 117 | O2 flow | `O2_FLOW_RATE` | L/min | 0 – 60 | Action-adjacent but charted as a vital here |

From WFDB `_0n.mat` (dense monitor, fs=0.5 Hz, one sample / 2 s):

| var_id | Variable | `_0n` channel | Unit (after `/adc_gain`) | Notes |
|--------|----------|---------------|--------------------------|-------|
| 100 | HR | `HR` | bpm | Same id as EHR `PULSE`; distinguish via `source=wfdb_numerics` in meta |
| 101 | SpO2 | `SPO2-%` | % | |
| 102 | RR | `RESP` | /min | |
| 116 | EtCO2 | `CO2-EX` | mmHg | End-expiratory |
| 118 | inspiratory CO2 | `CO2-IN` | mmHg | Inspired partial pressure |
| 119 | capnometer RR | `CO2-RR` | /min | Distinguish from chart RR |
| 120 | ST-II | `ST-II` | mm | ECG ST segment |
| 121 | ST-I | `ST-I` | mm | |
| 122 | ST-V | `ST-V` | mm | |
| — | invasive ABP-S/M/D | `AR1-S/M/D` / `AR2-S/M/D` / `ART-S/M/D` (canonicalized) | mmHg | Map to var_id 110–112 (same semantic as chart LINE) |

**Skipped from `_0n`** (hold signals / low clinical value in first pass):
`NBP-S`, `NBP-D`, `NBP-M` (use EHR chart instead — see cuff-cycle note),
`CUFF`, `APNEA`, `PVC`, `ST-V1/V2/V3/V4/V5/V6/AVR/AVF/AVL/III`.

### Actions / Interventions (var_id 200-299)

Deferred in first pass. Candidates:

| Source | Columns | Proposed var_id range |
|--------|---------|-----------------------|
| `JGSEPSIS_MEDS.csv` | `MEDICATION_NAME, MEDICATION_GENERIC_NAME, THERAPEUTIC_CLASS, START_TIME, END_TIME, DOSE, DOSE_UNIT, ROUTE` | 210–250 — one id per therapeutic class (vasopressor_ordered, sedative_ordered, …) |
| `JGSEPSIS_INOUTS_ALL.txt` | fluid in/out | 260–270 |
| `JGSEPSIS_VENT.csv` / `VENTDATA.txt` | ventilator mode + settings | 280–299 |

Wire these in Stage H (post-Stage G) once Stages A–G are validated, following
UCSF precedent.

### Scores (var_id 300-399)

Sepsis-specific scores (SOFA, qSOFA, SIRS) are downstream-task concerns →
`tasks/sepsis/extra_events/`, not the canonical `ehr_*.npy`.

### New variables to add to `var_registry.json`

Any id above whose row does not already exist in `indices/var_registry.json`
must be appended (extending pattern used for UCSF: fill `name`, `unit`,
`physio_min/max`, `mimic_itemids` if applicable, `emory_sources` with list of
column names / `_0n` channels / MC_MED names as cross-dataset aliases).

### Filtering Rules

```
- DROP rows with null LAB_RESULT_TIME / RECORDED_TIME / value.
- DROP non-numeric LAB_RESULT (cast with strict=False, then filter is_not_null).
- DROP rows where value < 0 or outside physio_min/max (dataset-specific first
  pass; later: robust quantile clipping).
- DEDUP on (encounter_nbr, var_id, time_ms, value) — same lab repeated as separate
  PROC_CAT in JGSEPSIS_LABS.
- TEMPERATURE: VITALS2 values appear to already be in Celsius (range 35–42 on
  verified encounter). Do not re-scale. Record TEMPROUTE as categorical metadata.
- SPO2 (VITALS2): occasionally 0 — drop.
- _0n cuff channels (NBP-S/D/M): DO NOT EXTRACT as events; value is held.
- _0n all channels: `val / adc_gain` gives physical units. Zero values = missing.
  Negative values = missing.
```

---

## Demographics to Extract

`{output_dir}/demographics.csv`, one row per `entity_id`. Join path:
`entity_id.empi_nbr` → `sepsis_cc_...all_collab.csv` → `encounter_nbr` →
`JGSEPSIS_ENCOUNTER.PAT_ID` → `JGSEPSIS_DEMOGRAPHICS`.

| Column | Source | Column | Encoding |
|-------|--------|--------|----------|
| `entity_id` | derived | — | `{empi_nbr}_{encounter_nbr}` |
| `empi_nbr` | whole list | `empi_nbr` | int64 |
| `encounter_nbr` | task list | `encounter_nbr` | int64 |
| `pat_id` | ENCOUNTER | `PAT_ID` | string |
| `gender` | DEMOGRAPHICS | `GENDER` (+ `GENDER_CODE`) | raw string ("Female", "Male", "Unknown") |
| `dob_iso` | DEMOGRAPHICS | `DOB` | ISO date |
| `age_years` | derived | `HOSPITAL_ADMISSION_DATE_TIME − DOB` | float, clip to [0, 120] |
| `race` | DEMOGRAPHICS | `RACE` (+ `RACE_CODE`) | raw string |
| `ethnicity` | DEMOGRAPHICS | `ETHNICITY` (+ `ETHNICITY_CODE`) | raw string |
| `insurance` | ENCOUNTER | `INSURANCE_STATUS` | raw string |
| `admit_utc_ms` | ENCOUNTER | `HOSPITAL_ADMISSION_DATE_TIME` NY→UTC | int64 |
| `discharge_utc_ms` | ENCOUNTER | `HOSPITAL_DISCHARGE_DATE_TIME` NY→UTC | int64 |
| `los_days` | derived | (discharge − admit) / 86400000 | float |
| `admit_dx_icd10` | ENCOUNTER | `DIAGNOSIS_ICD10_CD` | string |
| `admit_dx_desc` | ENCOUNTER | `DIAGNOSIS_ICD10_DESC` | string |
| `entity_healthcare_nm` | ENCOUNTER | `ENTITY_HEALTHCARE_NM` | site/hospital name |
| `death_date_utc_ms` | DEMOGRAPHICS | `DEATH_DATE` | nullable int64 |

---

## Processing Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Cohort | `sepsis_cc_2025_06_13_all_collab_uniq_combine.csv` (11,715 encounters) first pass | Case/control-labelled quality-gated windows. Whole-list (64,750 encounters) expansion is drop-in. |
| Entity id | `{empi_nbr}_{encounter_nbr}` | Admission level; concatenate all wfdb_records per encounter |
| Segment duration | 30 seconds, non-overlapping | 30 s = 1200 @ 40 Hz, 3600 @ 120 Hz |
| Min segments per entity | 10 (5 min of wave) | Same as UCSF / MC_MED |
| PLETH-anchor | Drop segments without PLETH | Match canonical convention |
| Max NaN ratio per channel | 0.20 | Drop segment if exceeded |
| WFDB worker count | ≤ 24 (shared cluster = half of 48 cores) | Shared cluster cap |
| ProcessPool context | `spawn` (not `fork`) | Polars in child workers deadlocks on fork |
| EHR context_window_ms | 86_400_000 (24 h) | Canonical default |
| EHR baseline_cap_ms | 2_592_000_000 (30 d) | Canonical default |
| EHR future_cap_ms | 604_800_000 (7 d) | Canonical default |
| Episode bounds | `HOSPITAL_ADMISSION_DATE_TIME` → `HOSPITAL_DISCHARGE_DATE_TIME` | From JGSEPSIS_ENCOUNTER |
| Train / val / test | 70 / 15 / 15 stratified by `type`, grouped by `empi_nbr` | Prevent patient leakage across splits |
| Split seed | 42 | Reproducibility |

---

## Known Issues

```
- Old `stp2_matching_lab_vital_wav.py` uses wrong LABS column names
  (RESULT_TIME / Component_value / Component_name).
  Actual columns are LAB_RESULT_TIME / LAB_RESULT / COMPONENT.

- Newer emory_vital_lib.py (`/labs/hulab/mxwang/data/vitals/`) imports
  `IPython.display.display` at module top — breaks in headless env.
  Strip this import before use, or import only canonicalize_sig_names and
  VITAL_LIST.

- `_0n.mat` sample rate: old pipeline constant VITAL_FS=0.5 Hz is correct;
  demo notebook's fs=2 Hz claim is wrong. Trust `_0n.hea:fs`.

- Top-level `{rec}.hea` has `sig_name=None` (layout row). Read per-segment
  `{rec}_0XXX.hea` for channels. Skip `~` entries in the top-level
  `seg_name` list (gap markers).

- `_0n` NBP-S/D/M is a HOLD signal, not an event stream (~96% non-null
  samples). Do NOT extract as sparse events; use EHR SBP/DBP/MAP_CUFF.

- Waveform channel set varies by record. Must canonicalize via
  `emory_vital_lib.canonicalize_sig_names` (handles Pleth / SPO2 /
  aVR vs AVR / etc.). Before canonicalization, 'SPO2' and 'Pleth' can
  both appear for the same PPG semantic.

- uniq_combine.csv lacks `empi_nbr` — must join back to whole-list for
  patient-level splits.

- 494 / 64,750 whole-list encounters not found in JGSEPSIS_ENCOUNTER.
  These drop out at Stage 3 cross-check (expected, not an error).

- Valid_start/end in uniq_combine.csv are naive UTC — treat as
  replace_tzinfo(UTC), do NOT pass through `America/New_York`.

- EHR chart SBP_CUFF charts at cuff-start; _0n NBP-S step lands ~107 s
  later (p95 = 31 min when next cycle differs). Value diff at the post-step
  is 0 mmHg median. This is physics, not alignment error.
```

---

## Output Specification

```
datasets/emory/processed/     → canonical per-entity dirs
                                (actual storage will live under
                                /opt/localdata100tb/physio_data/emory/
                                or configured in server_paths.yaml)
├── {empi_nbr}_{encounter_nbr}/
│   ├── PLETH40.npy            [N_seg, 1200]  float16, C-contiguous
│   ├── II120.npy              [N_seg, 3600]  float16, C-contiguous
│   ├── time_ms.npy            [N_seg]        int64 UTC ms, monotonic
│   ├── ehr_baseline.npy       structured    pre-(wave − 24 h), up to 30 d
│   ├── ehr_recent.npy         structured    pre-wave within 24 h
│   ├── ehr_events.npy         structured    in-wave, seg_idx ∈ [0, N_seg)
│   ├── ehr_future.npy         structured    post-wave, up to 7 d
│   └── meta.json
├── demographics.csv           one row per entity_id
├── manifest.json
├── pretrain_splits.json
├── downstream_splits.json
└── tasks/
    └── sepsis/
        ├── cohort.json        11,715 entities + case/control + sepsis_time_zero
        ├── splits.json        70/15/15 stratified by type, grouped by empi_nbr
        └── extra_events/      (optional) SOFA / sepsis onset markers
```

## Downstream Task: Sepsis Prediction

`tasks/sepsis/cohort.json` columns:

| Field | Source | Notes |
|-------|--------|-------|
| `entity_id` | derived | `{empi}_{enc}` |
| `type` | uniq_combine.csv | `case` or `control` |
| `sepsis_time_zero_utc_ms` | uniq_combine.csv `sepsis_time_zero_dttm` (treat as UTC) | null for controls |
| `valid_start_utc_ms` | uniq_combine.csv | |
| `valid_end_utc_ms` | uniq_combine.csv | |
| `valid_duration_hour` | uniq_combine.csv | |
| `valid_ratio` | uniq_combine.csv | quality gate |
| `min_event_to_wave_end_minutes` | derived | null for controls; used for prediction-horizon filters |

Sepsis_time_zero distribution vs wave window (verify at post-stage): on the
demo encounter, `sepsis_time_zero = 2019-07-30 10:49 UTC` lands ~4.3 days
AFTER the 8-h task window ended. Expect a mix of before / during / after-wave
positions across the cohort — same as UCSF CA task (23% during, 53% after,
24% before wave).

## References

- Siva Bhavani et al., Emory CDW ICU pull (data dictionary at
  `/labs/hulab/Emory_EHR/CDW_Pull_ICU_Data_Siva_version/data_dictionary.json`).
- Old iterative pipelines (for reference only; do NOT port dense-format logic):
  - `/labs/hulab/mxwang/data/Emory/UNIPHY_pipeline/stp{1,2,2_1,3,4,5}_*.py`
  - `/labs/hulab/mxwang/data/Emory/format_for_uniphy_v{3..6}_tables.py`
  - `/labs/hulab/mxwang/data/sepsis/{EHR,Wav,pretraining_data_prepare}/`
- Shared vitals lib (channel-name canonicalizer):
  `/labs/hulab/mxwang/data/vitals/emory_vital_lib.py`
- Step 0 artifacts:
  - Local: `datasets/emory/explore/{README.md, dataset_profile.json, alignment_plot.png, alignment_report.json, demo_plot_30min.png, demo_meta.json, demo_visualize.ipynb}`
  - Remote: `/labs/hulab/mxwang/Physio_Data/workzone/emory/explore/{alignment_check.py, step_alignment.py, single_entity_demo.py, demo_visualize.ipynb, demo_entity/1827183_359559206/}`
