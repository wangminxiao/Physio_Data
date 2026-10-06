# MLADI Dataset API

Pitt MLADI: Philips IntelliVue bedside-monitor recordings exported by Data Warehouse Connect (DWC),
one HDF5 per encounter with that encounter's EHR tables inside the same file. Lives on PSC Bridges-2
only (`/ocean/projects/med250003p/shared/`); nothing leaves PSC except aggregate statistics and
explicitly approved figures. Findings below were measured on PSC 2026-09-28 .. 2026-10-05
(`workzone/mladi/explore/*.py`, research notes in `explore/README.md`).

**Status: reviewed 2026-10-05.** User decisions: (1) vitals_hf in **1-s slots**; (2) add the perfusion
index (`165 PERF_hf`); (3) lab mapping by the Physio_Data conventions of the other cohorts (registry
lists); (4) actions are their own data type -> `ehr_actions.npy` sidecar as MIMIC v2 / MOVER / MC-MED;
(5) waveform-only entities included; (6) blood pressure from the 1-Hz numerics only (no ART/ABP waveform).

## Data Sources

### Waveform

| Field | Value |
|-------|-------|
| Format | HDF5, **audata 1.1** (root `.meta` = `{audata_version, time_origin}`) |
| Location | `/ocean/projects/med250003p/shared/mladi_extract_2023_waves/<base>.h5` (20,161 files, 26.7 TB) |
| Channels | `/data/waveforms/<label>`: compound `(time f8, value f4)`; `.meta.dwc_meta` = {label, samplePeriod (ms), unitLabel, clipLow/High, minTime, maxTime, ECG low/highEdgeFrequency, …} |
| Present | II 19,760 files · V 19,613 · Resp 19,522 · aVR 16,828 · **Pleth 16,707** · III 12,541 · I 9,044 · **ART 6,692** · MCL 4,054 · **ABP 1,899** · CVP 758 · ICP 669 · … |
| Rates | Pleth / ART / ABP 125 Hz; II 500 Hz (250 Hz in ~0.7 %) |
| Monitor numerics | `/data/numerics/<label.sub>` `(time, value)` at **1.024 s**: HR.HR, RR.RR, SpO₂.SpO₂, SpO₂.Pulse, Perf.Perf, PVC.PVC, HR.BeatToBeat, ST (ECG.I/II/III/V/aVx), NBP.NBPs/d/m/Pulse (per cuff reading), **ART.Systolic/Diastolic/Mean/Pulse (6,117 files)**, ABP.ABPs/d/m/Pulse (1,593), Temp.Temp (3,356), CVP.CVPm, ICP.ICPm, QT/QTc, CO₂.etCO₂, awRR, … |
| Earlier derived set | `/ocean/projects/med250003p/shared/pretrain_wav_v2/`: `<base>_Pleth_40Hz_<n>_1200_mmap.npy`, `<base>_II_120Hz_<n>_3600_mmap.npy` (band-passed, float16), `<base>__meta.json` (`seg_list`), `<base>__numerics.npz`. 16,422 encounters, **183,074,986 thirty-second rows (1.5 M h)**. |

### EHR — Clinical Tables (inside each HDF5, `/ehr/<table>`)

Structured arrays; **integer columns are audata factors**: the per-file `levels` list sits in the
table's `.meta` attribute (`columns.<col>.levels`), code = index, negative = missing. Decode per file.
`resultVal` can itself be a factor (text such as `<0.5`); such rows are dropped from numeric variables.

| Table | Files | Key columns | Notes |
|---|---|---|---|
| `lab_results` | ~11.8 k | time, orderedAs (panel), **eventDisp (analyte)**, resultVal, resultUnit, normalcy, validDate | 2,563 analytes; CBC/BMP in 11.7 k encounters |
| `low_rate` | ~11.0 k | date, **eventName**, resultVal, resultUnit, resultStat | charted vitals & nursing items, 335 names |
| `medications` | ~11.8 k | time, orderedAs, **catalogDisp**, dose, doseUnit, route, volumeDose | 1,084 drugs |
| `infusions_and_outputs` | ~11.7 k | time, name, detail, volume, unit | Urine Output, Continuous Infusions, Oral Intake, … |
| `diagnostic_codes` | ~11.8 k | time, type, codeType (ICD-10-CM; ICD-9-CM rare), code, seq | |
| `demographic` | 19.9 k | race, age, sex, facility, unit, **regDate, dischDate**, ethnicity, encntrType, dischDisp | one row |
| `patient` | 20.0 k | time, category, pacedMode, resuscitationStatus, clinicalUnit, gender, admitState | |
| `location` | ~11.9 k | beginDate, endDate, facility, unit | 109 unit codes |
| `csce` / `culture_sensitivity` | ~7.8 k | code status / orders | |

EHR tables exist for ~57 % of the encounters that have Pleth (1,853 / 3,259 in a 4,000-file sample).

### Patient ID Linkage

```
base       = YYYYMMDD_<encounterID>_<patientID>          (one HDF5 = one encounter)
patient_id = last field of base                          (Physio_HNET build_mladi_labels.extract_patient_id)
entity_id  = base
```
20,143 patients for 20,161 files; at most 2 encounters per patient. EHR is inside the encounter's own
file, so no join is needed.

### Time Format and Clock Rule (`workzone/mladi/clock.py`)

Every audata time column is **seconds from the root `time_origin`** (`"2018-10-22 04:00:00.000000 EDT"`;
zone label EDT / EST, or LMT on the year-1800 origins; origin years 1800 (772 files), 1990 (341),
2016–2023 (the rest), visibly date-shifted).

Verified on data (`clock_survey.py`, `clock_verify.py`, `clock_pairs.py`, `clock_model.py`): charted
Systolic/Diastolic BP in `low_rate` equal the monitor's cuff NBP to the mmHg (nurses validate it), which
pins the offset between the two clocks without physiology.

1. **Both streams count local wall-clock seconds.** The raw Pleth clock jumps +3600 s at spring-forward
   (18/18 files spanning one); at fall-back the repeated hour was dropped (no backward step, no duplicate
   time, 21/21), i.e. one real hour is missing there.
2. **Different origin walls.** Monitor (DWC waveforms, numerics): the stamped wall `W`. EHR: the same
   origin instant rendered in the **DST state at discharge** (`demographic.dischDate`), so `W ± 1 h` when
   origin and discharge lie on different sides of a DST change. Charted-minus-monitor NBP offset observed
   0 / −60 / +60 min in 89 / 5 / 4 % of 2,361 encounters; the discharge rule predicts it in **96 %**
   (≈95 % of the ±60 cases; the last EHR row is an equally good fallback, waveform start/end are not).
   The offset is constant within an encounter, also across a DST change (98 crossing encounters).
3. **Year-1800 (LMT) origins**: EHR seconds are elapsed from the origin read as UTC (offset +240 / +300
   min against the wall-clock monitor stream) in ~75 % of the checkable files. Year-1990 origins follow
   rule 2 (reading them as UTC was off by −300 min in 58 of 60).
4. **Per-entity correction (Stage A, all 16,422 entities)**: with ≥ 3 exact NBP matches (7,580 entities)
   the residual after rules 1–3 is 0 (±5 min) in 7,462 → `verified`; within 5 min of a zone offset
   (±60 DST, ±240 / ±300 UTC vs EDT / EST, ±296 UTC vs LMT) in 112 → `corrected` with the measured shift
   (`ehr_extra_shift_min`); anything else in 6 (10–18 min, charting delay) → `conflict`, no shift.
   Entities that cannot be checked (8,842; 7,157 of them have no EHR) are `inferred` and carry
   `clock_risk` = the rule's error rate in their origin class (2016–19 0.5 %, 2020–23 4 %, 1800 25 %).
   **Stage D re-measures the residual on the final grid** (after rule 5, against the written
   `nbp_events`) and applies it exactly (`meta.ehr_clock`; `clock_confidence` / `ehr_extra_shift_min` updated,
   Stage A's kept as `*_stage_a`): |residual| ≤ 1 min → verified; ≥ 3 matches explaining ≥ 30 % of charted
   SBP → corrected by the measured minutes; else conflict, no shift. Snapping to a known zone offset was
   wrong for a 2019 group whose residual is −56 min (and +4 min in some "verified" ones): the monitor
   clock there runs 4 min off, which the exact shift absorbs. 2026-10-06 pilot: corrected entities matched
   at 0.05 with the snapped shift, the mode moves to 0 with the exact one.
6. **Rule 5 extension (2026-10-06):** a run that the wall rule would place before the previous run ended (the
   raw clock kept counting through a DST change and was not re-anchored after a gap) continues from the
   previous run in elapsed time (`meta.runs_continued_after_dst`).

Conversion: monitor `utc = NewYork(W + t)`; EHR `utc = NewYork(W_ehr + t)` (rule 3: `W as UTC + t`).

5. **Runs, not rows.** DWC times are synthesized from sample counts, so inside a contiguous run of grid
   rows (30 s apart, one block) raw elapsed time is real. The monitor grid places each run's FIRST row by
   the wall rule and adds raw elapsed seconds (`clock.Grid.dwc_rows`). 253 entities (1.5 %) have a run
   crossing a DST change: 66 at spring-forward with continuous raw time (a per-row wall conversion would go
   backwards), 187 at fall-back (no backward step, no dropped data). Elsewhere the run rule equals the
   per-row rule. These entities carry `meta.dst_crossing_runs` and `ehr_dst_note`: EHR events after the
   change may be 1 h off. The NBP check cannot arbitrate this case, because charted vitals are monitor
   validations and carry the monitor's own labels.
**Entity grid (as `ucsf_all`)**: UTC-continuous, anchored at the wall clock of the first segment:
`time_ms = wall_ms(first segment) + real elapsed ms`. No 1-h gap at spring-forward; the hour the device
dropped at fall-back stays a real gap. Demo encounter after the rule: charted SBP == monitor NBP at lag 0
(was −60 min before).

**Per-entity validation** (Stage A): where charted SBP/DBP match monitor NBP exactly (≈ 60 % of EHR
encounters), the residual offset after conversion is measured. `meta.json`:
`clock_rule` ("dwc_wall/ehr_wall_disch" | "dwc_wall/ehr_utc_origin"), `ehr_origin_shift_min`
(0/±60), `clock_check` {n_matches, n_charted, residual_min}, `clock_confidence` (verified / corrected /
conflict / inferred, rule 4), `ehr_extra_shift_min`, `clock_risk`.

---

## Waveform Channels to Extract

| Source | Source rate | Target | Rate | samples/seg | Notes |
|---|---|---|---|---|---|
| Pleth | 125 Hz | **PLETH40** | 40 | 1200 | base channel; raw DWC units (≈ 0.25–0.75, autoscaled) |
| II | 500 (250) Hz | **II120** | 120 | 3600 | mV; NaN-filled where II is absent |

- **Grid = the `pretrain_wav_v2` rows** (`__meta.json` `seg_list`): canonical segment `i` == mmap row `i`
  (30 s, no overlap; blocks split where both channels are absent > 5 s; rows kept only where Pleth is
  non-flat). Existing label caches, PAT targets and the e1 split therefore index the same seconds.
  Consecutive rows are processed as one stretch on the block's own sample grid (`start + n/fs`), then
  `resample_poly` — **no band-pass** (canonical = raw). Check (demo): band-passing the canonical rows as
  `data_preparing_v2` did reproduces the mmap rows, r p50 0.995 (Pleth) / 0.997 (II).
- Invalid values: non-finite or |v| > 1e3 (Pleth sentinels ~ −2.7e8, −1.7e7) → NaN on the samples they
  cover; interpolation never bridges them.
- Not in v1: ART/ABP waveforms (ABP125 for 41 % of encounters ≈ 3.6 TB; storage, see below). ART/ABP
  enter as 1-Hz numerics.
- Physio_HNET readers of the current MLADI encoders add the `data_preparing_v2` band-pass at runtime
  (Butterworth order 4, `filtfilt`: Pleth 0.5–12 Hz, II 0.5–50 Hz).

## Dense sidecars (monitor numerics)

`vitals_hf.npy` float32 `[N_seg, 30, n_var]`, **1-s slots** (slot `k` of segment `i` covers
`[time_ms[i] + k s, +1 s)`; the 1.024-s readings fill ~98 % of slots, one reading per slot, last wins if
two), NaN = none. `meta.vitals_hf` records `slot_sec = 1`, `slots_per_seg = 30` -- `ucsf_all` uses 2-s
slots, so readers take the slot size from meta. Values filtered by registry `physio_min/max`.

| idx | var_id | name | source key(s), first available |
|---|---|---|---|
| 0 | 150 | HR_hf | HR.HR |
| 1 | 151 | SpO2_hf | SpO₂.SpO₂ |
| 2 | 152 | RR_hf | RR.RR |
| 3 | 153 | ABPs_hf | ART.Systolic → ABP.ABPs |
| 4 | 154 | ABPd_hf | same line as the slot's systolic |
| 5 | 155 | ABPm_hf | ART.Mean → ABP.ABPm |
| 6 | 156 | PULSE_hf | SpO₂.Pulse |
| 7 | 113 | PR_art | ART.Pulse → ABP.Pulse |
| 8 | 160 | CVP_hf | CVP.CVPm |
| 9 | 164 | PVCrate_hf | PVC.PVC |
| 10 | 165 | PERF_hf (new, approved) | Perf.Perf (perfusion index) |

`vitals_hf_abp_src.npy` uint8 `[N_seg, 30]`: 0 none, 1 ART, 2 ABP (lines never mixed in a slot).
`nbp_events.npy` (EHR_EVENT_DTYPE, var 157/158/159 = NBPs/d/m_hf): one event per cuff reading (value
change), at the reading time. Stage G writes `ehr_hf.npy` (MIMIC-compatible pair-end convention of
`mimic3/stage3b_extract_numerics.py`, var 150–159) for the phase-4 readers, as `ucsf_all` does.

---

## EHR Variables to Extract

Times through `clock.py` (EHR rule); values numeric only; units checked per row against the decoded
`resultUnit` (rows in another unit are converted when the rule is listed, else dropped and counted).

### Labs (var_id 0–18, `lab_results.eventDisp`)

| var | registry | MLADI analytes (encounters) |
|---|---|---|
Mapping follows what the registry already does for the other cohorts:
- K, Na, glucose, lactate: central lab **plus** whole-blood / blood-gas / point-of-care results (MIMIC
  50822/50824/50809/50813, UCSF POTASSIUMBLOOD / SODIUMWB / GLUCOSE METERDOWNLOAD, MC-MED POC lactate).
- Calcium: **total** calcium only (mg/dL, range 4–15); ionized calcium (mmol/L) is not this variable.
- Creatinine: central lab only (MIMIC 50912, UCSF, MC-MED) -- iSTAT creatinine not mapped.
- Hemoglobin: laboratory plus calculated (UCSF HGB(CALCULATED)).
- HCO3: chemistry total CO2 plus blood-gas bicarbonate (MC-MED CO2/POC:HCO3/HCO3, MOVER-Epic Carbon dioxide/Bicarbonate).
- Blood gases: the registry variables are ARTERIAL; MLADI separates arterial and venous, so only arterial.

| var | registry | MLADI analytes (encounters) |
|---|---|---|
| 0 | Potassium | K (11,748) · Potassium(K) Whole Blood (2,168) · Potassium iSTAT (1,684) · Potassium Level (877) |
| 1 | Calcium (total) | Ca (11,670) · Calcium Level (1,088) |
| 2 | Sodium | Na (11,748) · Sodium(Na) Whole Blood (1,546) · Sodium Istat (1,684) · Sodium (Na) Level (591) |
| 3 | Glucose | Glucose (11,698) · Glucose (bedside) (6,287) · Glucose POC (2,068) · Glucose iSTAT (1,684) · Glucose Level Whole Blood (1,586) · Glucose Whole Blood (1,564) · Glucose Level (1,525) |
| 4 | Lactate | Lactate, Whole Blood (6,959) · Lactate (5,986) · Lactate Whole Blood (4,196) · Lactate Whole Blood (Syringe) (2,384) |
| 5 | Creatinine | Cr (11,743) |
| 6 | Bilirubin | Bili, Total (10,182) |
| 7 | Platelets | Platelets (11,757) |
| 8 | WBC | WBC (11,753) |
| 9 | Hemoglobin | Hgb (11,757) · Calc. Hemoglobin iSTAT (1,684) · Hemoglobin-Arterial (573) |
| 10 | INR | INR (10,213) |
| 11 | BUN | BUN (11,747) |
| 12 | Albumin | Albumin (10,247) |
| 13 | Arterial_pH | pHa (4,912) |
| 14 | paO2 | PaO2 (4,912) · Arterial pO2 (POC) (341) |
| 15 | paCO2 | PaCO2 (4,912) · Arterial pCO2 (POC) (341) |
| 16 | HCO3 | CO2 (11,748) · HCO3a (4,912) · HCO3 (2,046) |
| 17 | AST | AST/SGOT (10,174) |
| 18 | ALT | ALT/SGPT (10,174) |

Not mapped (no registry variable): venous gases (pHv, HCO3v, Venous pO2/pCO2), ionized calcium,
iSTAT creatinine, urinalysis. Each row's decoded `resultUnit` is checked against the registry unit
(mMol/L == mEq/L for K/Na; X10E+09/L == K/uL); any other unit is dropped and counted.

### Vitals, charted (var_id 100–199, `low_rate.eventName`)

| var | registry | MLADI names |
|---|---|---|
| 100 | HR | Pulse |
| 101 | SpO2 | O2 Saturation |
| 102 | RR | Respiratory Rate |
| 103 | Temperature (°C) | Temperature Metric (°C); Temperature (unit-checked, F→C) |
| 104/105/106 | NBPs/d/m | Systolic BP / Diastolic BP / Mean blood pressure |
| 108 | GCS_total | Glasgow Coma Score · Glascow Coma Score |
| 110/111/112 | ABPs/d/m | Arterial Systolic Pr(essure) / Arterial Diastolic P(ressure) / Mean arterial pressu(re) |
| 107 | CVP | Central Venous Press · Central Venous Pressure |
| 117 | O2_flow | Oxygen per liter |

Temperature rows are unit-checked (°C kept, °F converted, "Temperature Conversi(on)" rows taken only with
a temperature unit). End-tidal CO2 is not charted in `low_rate`; it exists only as monitor numerics
(CO₂.etCO₂, ~1.6 % of files) and is not mapped in v1.

### Actions (var_id 200–299) -> `ehr_actions.npy` sidecar

Actions are their own data type in Physio_Data: a per-entity **`ehr_actions.npy`** sidecar (same dtype
as `ehr_events`, same four-way time split by `seg_idx` sentinels), never written into `ehr_events.npy`
-- the convention of `mimic3/stage3b_actions_v2.py`, `mover_combine/stage3b_actions.py`,
`mcmed/stage3b_actions.py`. Value semantics shared with them: native dose / rate where the source gives
one, `0.0` = stopped (when a stop is charted), `NaN` = "occurred, magnitude unknown"; var 200 (NE-
equivalent aggregate) = NaN presence at each vasopressor administration, as MOVER / MC-MED, until a
continuous rate exists. Drug -> var_id with the same matcher rules (systemic routes only; ophthalmic /
nasal / inhaled / topical / flush excluded; lidocaine / bupivacaine and drug-in-vehicle piggybacks are not
fluid actions).

| var | registry | MLADI source |
|---|---|---|
| 200 | vasopressor_rate (NE-eq) | NaN presence at any 207–213 administration (v1) |
| 201 / 202 | fluid rate / bolus | medications: Sodium Chloride 0.9 %, Lactated Ringers, Plasma-Lyte (IV) -- volumeDose mL |
| 203 | FiO2 (fraction) | low_rate "Oxygen % (FiO2)", "FiO2 - vent", "FIO2" (÷ 100; "FIO2" is mostly text) |
| 204 | PEEP | low_rate "Positive end expiratory pressure (PEEP)", "Positive end expirat", "RRT PEEP or CPAP" (cmH2O) |
| 205 | mechvent | low_rate "RRT Mode" / "Ventilator mode" with an invasive mode (AC, PRVC, SIMV, VC, PC, APRV, pressure support; not BiPAP / AVAPS / CPAP), or "RRT Tidal Volume Set" / "Tidal volume - set" > 0 -> 1. ("RRT Vent Status" has no value; "RRT Ventilator Type" includes non-invasive machines such as V60 / Trilogy; value census 2026-10-06) |
| 206 | urine_output (mL) | infusions_and_outputs name "Urine Output", volume |
| 207–213 | per-drug vasopressors | medications catalogDisp norepinephrine / epinephrine / phenylephrine / dopamine / vasopressin / dobutamine / ePHEDrine. 207–212: NaN = given (the registry units are rates, MLADI charts amounts); 213 ephedrine: mg (registry unit) |
| 214 | prbc_transfusion | infusions_and_outputs "Blood Products/Colloids" with a red-cell `detail` |
| 215 | insulin | catalogDisp insulin regular / lispro / glargine / … (Unit(s)) |
| 216 | dextrose_hi | dextrose 50 % / 10 % |
| 217 / 218 / 219 / 220 | K / Ca / bicarbonate replacement, hypertonic saline | potassium chloride / calcium chloride (and gluconate) / sodium bicarbonate / NaCl 3 % |

Rows whose `resultStat` is "In Error" are dropped (labs, charted vitals, actions).

MLADI medication `doseUnit` is an amount (mg, mcg, mL, Unit(s); rates only in ~0.5 % of encounters), so
v1 has no continuous vasopressor rate. A later version can derive it from `infusions_and_outputs`
"Continuous Infusions" (volume per charting interval x concentration parsed from `detail` / weight from
low_rate "Dosing Weight (kg)") -- documented, not guessed in v1.

### Scores (300–399): none in v1.

### New variables (registry additions — need approval)
- `165 PERF_hf` (perfusion index, unitless, source Perf.Perf, `source: dwc_numerics`) -- **approved**.
- Registry fields `mladi_*` (`mladi_lab_names`, `mladi_low_rate_names`, `mladi_med_names`,
  `mladi_numerics_keys`) carrying the mapping above, as for the other cohorts.

### Filtering Rules
Registry `physio_min/max`; non-numeric values dropped (counted per variable in `meta.json`); exact
duplicate (time, var, value) rows removed; events sorted by `time_ms`.

## Demographics to Extract (`demographics.csv`, one row per entity)
entity_id, patient_id, age, sex, race, ethnicity, encntrType, dischDisp (decoded strings), stay_days,
admit-to-waveform days, clinicalUnit (first), has_ehr, has_art, e1_split. Outcomes (e.g. in-hospital
death from dischDisp) are task post-stages, not canonical fields.

## Processing Parameters

| Parameter | Value |
|---|---|
| Segment | 30 s, **no overlap** (as UCSF / MC-MED; MIMIC uses 25-s stride) |
| PLETH40 / II120 | 1200 / 3600 samples, float16, C-contiguous |
| vitals_hf | 1-s slots, 30 per segment |
| EHR trajectory | context 24 h, baseline cap 30 d, future cap 7 d (`physio_data/ehr_trajectory.py`) |
| Entities | every HDF5 with Pleth and ≥ 1 `pretrain_wav_v2` row (≈ 16.4 k); EHR optional (`has_ehr`) — phase-2 pretraining uses waveform-only entities too |
| Splits | `pretrain_splits.json` = `downstream_splits.json`: **e1 split kept** (patient-level 70/15/15, seed 42, 9,218 / 1,976 / 1,974 encounters); entities outside e1 take their patient's e1 split if it has one, else a patient-hash 70/15/15 (seed 42); no patient on two sides |

## Pipeline (`workzone/mladi/`, PSC, `sbatch` on RM-shared, ≤ half a node, chained `afterok`)

| Stage | Script | Output |
|---|---|---|
| A | `stage_a_inventory.py` | per-entity inventory (channels, rows, numerics keys, EHR tables, clock class, NBP-match clock check) → `inventory.parquet` |
| B | `stage_b_wave.py` | PLETH40, II120, time_ms, meta.json (grid = mmap rows; raw; NaN for invalid) |
| C | `stage_c_vitals_hf.py` | vitals_hf.npy, vitals_hf_abp_src.npy, nbp_events.npy |
| C2 | `stage_c2_nbp_twins.py` | twin NBP copies (known issue 8): nbp_events emptied, nbp_events_twins.npy, meta.nbp_twins |
| D | `stage_d_ehr.py` | ehr_baseline / recent / events / future (labs, charted vitals) |
| D2 | `stage_d2_actions.py` | ehr_actions.npy (actions, var 200–220) |
| E | `stage_e_meta.py` | meta.json completed (clock, coverage, counts) |
| F | `stage_f_manifest.py` | manifest.json, pretrain_splits.json, downstream_splits.json, demographics.csv |
| G | `stage_g_ehr_hf.py` | ehr_hf.npy (MIMIC-compatible numerics events) |
| tasks | `workzone/common/build_estimation_task.py --spec task_specs/vital_est_hf.yaml`, then the `abp_hf` builder | `tasks/vital_est_hf`, `tasks/abp_hf` (ABPm_hf ≥ 30 ticks) — the files phase-4 reads |

Every stage: `--limit 5` first, resumable per entity, a `verify_stage_<x>.py` gate.

### Verification gates
| Gate | Checks |
|---|---|
| A | entity count ≈ 16.4 k; clock classes; NBP-match residual 0 or ±60 corrected on ≥ 95 % of checkable entities |
| B | errors ≤ 1 %; shapes/dtypes/contiguity; N_seg == mmap rows; `time_ms` strictly increasing, 30-s steps inside blocks; 300-entity sample: band-passed rows vs mmap r ≥ 0.98; NaN fraction < 20 % |
| C | sidecar shapes; HR_hf coverage median ≥ 0.8; NBP events in range; ECG-derived HR vs HR_hf corr ≥ 0.6 on 40 entities |
| D | event dtype, sorted, `seg_idx` bounds / sentinels, var_id in registry, no event in two partitions; NBP residual check re-run on the written events |
| D2 | `ehr_actions` dtype / sentinels / var_id ∈ 200–220; vasopressor share among ART encounters plausible; no action var in `ehr_events` |
| F | splits disjoint by entity and patient; e1 assignments preserved; manifests consistent |

## Known Issues / Quirks
1. Clock rule above (wall-clock seconds; EHR origin at the discharge DST state; 1800/1990 origins UTC);
   at fall-back the monitor dropped one real hour.
2. Pleth invalid-value sentinels; II absent in some Pleth files (NaN-filled).
3. Labs lying > 7 days from the waveform while charted vitals cover it: 0.3 % (long stays, waveform
   starting > 2 weeks after admission). Kept; per-entity lab coverage in `meta.json`.
4. Hidden gaps: rows missing inside a block (Pleth flat) ~0.4 % of 5-min windows — the grid keeps them
   out, `time_ms` shows the jump.
5. ECG filter settings vary (highEdge 150 vs 40 Hz, lowEdge 0.05 vs 0.5 Hz, ~2.6 %): recorded per
   entity (`meta.ecg_filter`).
6. ECG→Pleth delay ~1.2 s with a Pleth re-sync sawtooth (Physio_HNET `model/hnet_wav/sawtooth.py`):
   a property of the signals, not corrected in the store.
7. Facility masked to a single value; `location.unit` coded (109 units).
8. **Twin NBP copies (2020+ files, ~3-4 % of entities).** Every cuff reading (s, d, pulse) of the monitor NBP
   stream reappears 240 or 300 min apart; waveforms and the other numerics are not duplicated
   (`explore/dup_check.py`, `nbp_twins_raw.py`). The real copy cannot be told reliably (cuff pulse vs HR
   leans to the earlier one, 425:250 with 1,110 undecided; the EHR clock from charted pulse vs monitor HR
   is 0 in 25 of 30, which favours the copy at charting time). Stage C2 removes ALL monitor NBP events of a
   flagged entity (kept in `nbp_events_twins.npy`, `meta.nbp_twins`), and Stage D does not use Stage A's
   shift for it (Stage A measured the copy lag). Charted SBP/DBP of these entities may include
   validations of a copy. An arterial-line arbiter (NBP vs ART systolic at each copy) can recover them later.

## Output Specification
`/ocean/projects/med250003p/shared/physio_data/mladi/{entity_id}/`: PLETH40.npy, II120.npy, time_ms.npy,
ehr_baseline.npy, ehr_recent.npy, ehr_events.npy, ehr_future.npy, ehr_actions.npy, vitals_hf.npy,
vitals_hf_abp_src.npy, nbp_events.npy, ehr_hf.npy, meta.json; plus manifest.json, pretrain_splits.json,
downstream_splits.json, demographics.csv, tasks/.

**Storage**: PLETH40 + II120 ≈ 183 M segments × 9.6 kB ≈ **1.76 TB**; vitals_hf (1-s, 11 vars) ≈
183 M × 30 × 11 × 4 B ≈ 0.24 TB; total ≈ 2.0 TB.
Project free space 2.49 TiB → ≈ 0.6 TiB left after ≈ 2.0 TB. Retiring `pretrain_wav_v2` after the canonical store is
verified (user decision) frees ≈ 1.76 TB.

## References
- Ruffolo et al. 2025, Physiol. Meas. 46:115006 (IntelliVue ECG–PPG delay and re-sync sawtooth).
