# `ucsf_all/tasks/ca_risk` — in-hospital cardiac-arrest risk prediction from bedside monitor waveforms

Task definition, file layout, field dictionary, cohort statistics and recommended experimental protocol.
Built 2026-09-27 by `workzone/ucsf/stage_f_ca_risk.py` (job 50249) on the `ucsf_all` store; files live in
`/mnt/localdata100tb/physio_data/ucsf_all/tasks/ca_risk/` and in the BeeGFS copy
`/projects/xhu40-cdsfm/physio_data/ucsf_all/tasks/ca_risk/`. Time conventions: `ALIGNMENT.md`. Store layout: `API.md`.

## 1. Purpose

Learn, from continuous PPG (40 Hz) and ECG lead II (120 Hz) — optionally the 0.5 Hz monitor numerics in
`vitals_hf.npy` — whether an ICU patient is heading towards an in-hospital cardiac arrest, using a foundation
model (FM) pretrained on `ucsf_all`. Three question types are supported by the same files:

1. **Risk curve / early warning**: a score every few minutes that rises in the hours before the arrest and stays
   low for patients who never arrest.
2. **Horizon classification**: does an arrest occur within the next H hours (H = 1, 6, 12, 24)?
3. **Biomarker discovery**: which pre-arrest changes (rhythm, PPG morphology, HR/SpO2 trajectories) separate the
   case windows from the same patient's earlier windows and from matched controls?

Everything in the task is defined on the FM's own entity grid, so embeddings computed for pretraining can be
reused without re-alignment.

## 2. Population and label

| Item | Definition | Count |
|---|---|---|
| Universe | every valid wave cycle (entity) of a patient **held out of pretrain train** (`held_out_of_train` in the manifest) | 8,344 entities / 3,754 patients |
| Positive patient | has a Code Blue event with TypeCode **CPA** (cardiopulmonary arrest) in `SAUCSFCodeBlue_FirstEvent_2013_2018_final.xlsx`, mapped to `Patient_ID_GE` through the encounter offset table | 235 patients / 633 entities |
| of which | ValidWaveTime study cases (`in_ca_cohort`) | 192 patients |
| of which | Code Blue only (`codeblue_only`: never in the ValidWaveTime CSV) | 43 patients |
| Negative patient | held-out patient without a CPA event = the ValidWaveTime study **controls** | 3,519 patients / 7,711 entities |
| Units (entities) | 10ICC / 13ICU / 9ICU / 11NICU / 8NICU (all adult units; NICU = Neuro ICU) | pos 217 / 179 / 169 / 37 / 31 — neg 2,044 / 2,095 / 2,189 / 851 / 532 |

Why this population: the FM was pretrained on the other 12,448 patients only (gate F enforces zero overlap),
so fine-tuning and evaluation here are free of pretraining leakage. Code Blue "ME" (medical emergency) and
"ARC" (acute respiratory compromise) events are **not** positives; those patients, if they are in the CSV cohort,
appear as negatives — see §7.

`has_ca` is a **patient-level** label (one event per patient, the first). Every cycle of a positive patient
carries `has_ca = 1`, but only the cycle containing the event has pre-arrest signal; the other cycles are either
long before the event (usable as within-patient negatives) or after it (post-arrest physiology — exclude).

## 3. Time zero

`event_grid_ms` is the Code Blue `CodeTime` (true clinical time of the code call) placed on **this entity's**
time grid (`clock.real_wall_to_grid_ms`: real wall clock → UTC → minus the patient's `offset_GE` day shift →
UTC-continuous grid anchored at the cycle origin). It is validated against the ECG collapse seen in the
waveform (median |offset| a few minutes, see `ALIGNMENT.md` §4) and against charted vitals.

Per positive entity the event can be:

| Position | Field values | Count | Use |
|---|---|---|---|
| inside the cycle | `event_in_cycle = true` | 194 entities (all 194 distinct patients) | case windows before the event, post-event exclusion |
| after the cycle ended | `event_offset_from_wave_end_min > 0` | 254 | whole cycle is pre-event; far-before data / self-controls |
| before the cycle started | `event_offset_from_wave_start_min < 0` | 185 | post-arrest recording: exclude from negatives and from case windows |

Auxiliary physiological markers from the t0′ detector (`workzone/ucsf/explore/ca_t0_detect.py`, run on the fixed
grid) are copied for the 207 events with a covering cycle: `ecg_collapse_offset_min` (marker B, present for
125 events; median +2 min after the code call), `t0_prime_offset_min`, `t0_prime_quality`
(good 12 / fair 81 / weak 27 / none 87) and `t0_prime_method`. Use them for QA and sensitivity analyses, not
as the primary time zero: the code time is the clinically defined onset and the detector still has false
negatives (no visible collapse in ~40 % of events) and threshold-dependent delays.

## 4. Pre-event coverage

Computed on the segment grid (30 s segments; a segment is "valid" when ≥ 50 % of its PPG or ECG samples are finite):

| Field | Meaning |
|---|---|
| `ppg_cov_1h`, `ppg_cov_6h`, `ppg_cov_12h`, `ppg_cov_24h` | fraction of valid PPG segments in `[event − H, event]` |
| `ecg_cov_6h`, `ecg_cov_24h` | same for ECG II |
| `last_ppg_gap_min` | minutes from the end of the last valid PPG segment before the event to the event (negative = PPG runs through the event) |
| `ppg_hours_before_event` | valid PPG hours in this entity before the event |
| `pre_event_segments` | segments before the event in this entity |
| `usable_pre_event` | `last_ppg_gap_min ≤ 30` **and** `ppg_cov_6h ≥ 0.5` |
| `self_control_end_ms`, `self_control_hours` | event − 24 h, and how many hours of this cycle end before that boundary |
| `ppg_valid_frac`, `ecg_valid_frac`, `hours` | every entity: sampled validity fractions (one segment in 5 / in 10) and recording length |

Among the 194 event-in-cycle entities: PPG coverage ≥ 0.8 in the last 1 h for 181, in the last 6 h for 174,
in the last 24 h for 177; ECG ≥ 0.8 in the last 6 h for 173; the last valid PPG segment is ≤ 5 min before the
event in 189. **`usable_pre_event` holds for 187 entities of 185 patients** (train 135 / test 50 patients).
Valid PPG hours before the event among usable entities: median 15.8 h (p25 4.7, p75 70); ≥ 12 h in 105,
≥ 24 h in 83, ≥ 48 h in 61. Recording length per entity: positives median 60 h (p10 4, p90 327), negatives
median 43 h (p10 7, p90 158). Sampled PPG/ECG validity is ≥ 0.9 / ≥ 0.8 for 90 % of entities in both groups.

## 5. Split

`splits.json` → `train` (5,906 entities, 2,628 patients) and `test` (2,438 entities, 1,126 patients);
`val` is present but empty (user decision: too few positives). Grouped by `patient_id_ge`, stratified by
`has_ca` with 30 % of positive and 30 % of negative patients in test; seed 42.

| | entities | patients | positive patients | positive entities | event in cycle | usable pre-event (entities / patients) | negative entities |
|---|---|---|---|---|---|---|---|
| train | 5,906 | 2,628 | 165 | 462 | 139 | 137 / 135 | 5,444 |
| test | 2,438 | 1,126 | 70 | 171 | 55 | 50 / 50 | 2,267 |

Leakage control: every **test** patient belongs to the pretrain **test** split (their entities never
contributed to pretraining or to pretraining checkpoint selection on pretrain val); train patients come from
pretrain val (4,261 entities) plus the remaining pretrain-test patients (1,645 entities). For model selection
inside this task use patient-grouped cross-validation on `train` (e.g. 5 folds, stratified by `has_ca`); never
touch `test` until the protocol is frozen.

## 6. Files

```
tasks/ca_risk/
├── cohort.json     task metadata, counts, `patients` table (3,754 rows), `entities` table (8,344 rows)
├── splits.json     task, source, seed, test_frac, group_by, counts, train / val (empty) / test entity lists
└── README.md       short version of this document, written by the builder
```

`patients` rows: `patient_id_ge`, `has_ca`, `split`, `pretrain_split`, `n_entities`, `codeblue_only`,
`usable_pre_event` (any of the patient's entities). `entities` rows: `entity_id`, `patient_id_ge`, `unit`,
`n_seg`, `wave_start_ms`, `wave_end_ms`, `has_ca`, `has_ca_csv` (ValidWaveTime label, cohort members only),
`in_ca_cohort`, `codeblue_only`, `pretrain_split`, `dst_switch_in_cycle`, `time_base`, the coverage fields of
§4, the event fields of §3. `time_base = "wall_clock"` marks entities not rebuilt by the 2026-09-27 fix; their
grid is identical to the UTC-continuous one because no DST switch falls inside the cycle.

Waveforms and numerics are the canonical entity files (`PLETH40.npy`, `II120.npy`, `time_ms.npy`,
`vitals_hf.npy`, `nbp_events.npy`, `ehr_hf.npy`, `meta.json`) under `ucsf_all/{entity_id}/`; the task adds no
new signal files. Segment `i` covers `[time_ms[i], time_ms[i] + 30 s)`; the index of the event is
`np.searchsorted(time_ms, event_grid_ms, side="right") − 1`.

## 7. Recommended protocol

**Windows and labels.**
* Case windows: windows of an `event_in_cycle` entity whose end lies in `[event − H, event]`. Recommended
  horizons 1 / 6 / 12 / 24 h (coverage fields exist for these). Keep the label horizon and the input window
  length separate (e.g. 30-min input, 6-h horizon).
* Post-event exclusion: drop every window that starts after `event_grid_ms` in any positive entity, and drop the
  185 "event before cycle start" entities entirely (or keep them only for post-arrest studies).
* Self-controls (within-patient negatives): windows of a positive patient ending before `self_control_end_ms`
  (event − 24 h), including whole earlier cycles that ended before the event (`event_offset_from_wave_end_min >
  24 × 60`). Within the event cycle itself, 63 usable entities have ≥ 24 h of self-control time; counting the earlier
  cycles of the same patients, 234 positive entities of 135 patients offer ≥ 24 h of self-control data. Self-controls remove patient-level confounders (baseline rhythm, unit, severity)
  and are the right comparison for "what changed in the last hours".
* Cohort controls: windows from `has_ca = 0` entities. Match at training time by `unit` and recording length,
  and sample the control window positions to mimic the distribution of time-since-admission of the case windows
  (arrests cluster in the first days of the ICU stay; controls have median 43 h of recording per cycle). Report
  results both with and without matching.
* Class balance: ~190 positive windows-of-interest per horizon versus millions of control windows. Sample
  controls (e.g. 20–50 control windows per case window, patient-balanced) or use patient-level bagging; report
  AUROC, AUPRC, sensitivity at a fixed alarm rate per patient-day, and lead time.

**Evaluation.**
* Primary unit is the patient: aggregate window scores per patient-hour, evaluate risk curves aligned at
  `event_grid_ms` (e.g. score trajectories from −24 h to 0), and compute per-horizon AUROC/AUPRC with
  patient-level bootstrap confidence intervals (only ~70 test positives: expect wide intervals; 5-fold
  patient-grouped CV on train for model development).
* Report a **strict** subset (`usable_pre_event` and `ppg_cov_6h ≥ 0.8`, 174 entities) and the full set.
* Sensitivity analyses: (a) shift the time zero to `ecg_collapse_offset_min` where available; (b) blank the
  last 15 min before the event to check the model is not just detecting the arrest itself; (c) exclude the 20
  positive / 135 negative entities with `dst_switch_in_cycle` (their vitals/waveform grid was rebuilt; the
  result should not change).

**Inputs.**
* PPG + ECG embeddings from the `ucsf_all` FM (frozen or fine-tuned); add `vitals_hf` HR/SpO2/RR/ABP and
  `nbp_events` as covariates when moving beyond the waveform-only question — they sit on the same grid.
* Static covariates available without EHR linkage: `unit`, recording length, time since cycle start. Age,
  diagnoses and labs require the encounter linkage documented in `ALIGNMENT.md` §5 (rule 3) — not included
  in this task by design.

**Do not.**
* Do not use the ValidWaveTime `EventTime` column (7–8 h late and DST-shifted) or the legacy `ucsf`
  `tasks/ca_prediction` event times built before 2026-09-27.
* Do not mix the 43 `codeblue_only` patients into the "ValidWaveTime study" analyses without noting it; they
  were outside the original study cohort (27 in train, 16 in test).
* Do not treat post-event cycles as negatives, and do not evaluate on `test` during development.

## 8. Known limitations

* Code Blue time is the time of the code **call**, typically 0–10 min after the physiological collapse and
  occasionally much later (a 12-h pre-event window is unaffected; a 15-min window should use the collapse
  marker as a sensitivity check).
* Some CPA events are respiratory or PEA arrests with preserved PPG pulsatility until late; the detector finds no
  collapse for ~40 % of events.
* Controls are the ValidWaveTime study controls (selection criteria of the original study); Code Blue "ME"/"ARC"
  patients may sit among them.
* 17 cycles of the `2016-01/02-deid` batches have an unexplained ±60 min residual clock anomaly (2 % of probed
  cycles; `ALIGNMENT.md` §4). None of them contains a CA event.

## 9. Rebuilding

```bash
cd /projects/mwang80/staging/Physio_Data/workzone/ucsf/logs
sbatch ../slurm/ucsf_all_ca_risk.sbatch            # optional: --test-frac 0.3 --seed 42 --self-control-gap-h 24
```

The builder reads `manifest.json`, `pretrain_splits.json`, the Code Blue parquet and the offset table, computes
coverage from the entity files (≈ 16 min with 8 workers) and syncs `tasks/ca_risk/` to the BeeGFS copy.
