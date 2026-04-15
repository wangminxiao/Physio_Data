---
name: physio-data
description: Onboard and preprocess physiological/biomedical time-series datasets (waveform + clinical events) into a canonical mmap-ready format for deep learning. Covers any dataset with continuous signals (ECG, PPG, ABP, EEG, EMG, respiratory, etc.) and sparse clinical data (labs, vitals, medications, diagnoses, annotations). Use when exploring a new dataset, writing extraction scripts, verifying pipeline output, or asking about the canonical format.
argument-hint: [dataset-name or command like "explore","verify","format"]
---

# Physio_Data: Physiological Dataset Preprocessing Skill

Preprocess any physiological/biomedical time-series dataset into a canonical format
optimized for deep learning training. Works with any combination of:

- **Continuous signals**: ECG, PPG/SpO2, ABP, EEG, EMG, respiratory, accelerometer, etc.
- **Clinical events**: labs, vitals, medications, procedures, diagnoses, annotations, etc.
- **Sources**: PhysioNet (MIMIC-III, eICU, MOVER), hospital systems (MC_MED), wearable devices, clinical trials, etc.

## Core Design Principles

1. **Signals and events are stored differently because they ARE different.**
   - Signals: dense, regular, high-rate -> mmap .npy, row = fixed-length segment
   - Events: sparse, irregular -> structured .npy, row = one measurement/annotation

2. **Alignment is an index, not a copy.**
   - Events carry `seg_idx` pointing into the signal segment array. No dense padding.

3. **Zero CPU overhead on the training hot path.**
   - No compression. True mmap. GPU server CPUs are weak.
   - `arr[a:b].reshape(-1)` is zero-copy (C-contiguous guarantee).

4. **Schema is extensible without breaking changes.**
   - New event variable = add row to registry. New signal channel = add .npy file.
   - No existing array shapes change.

5. **Raw values in storage. Processing at runtime.**
   - No normalization, no interpolation at storage time. These are task-specific runtime decisions.

## Canonical Output Format

```
{entity_id}/                           # patient, subject, session, encounter, ...
  {CHANNEL}.npy       # [N_seg, samples_per_seg]  float16, C-contiguous
  time_ms.npy         # [N_seg]                    int64, monotonically increasing
  ehr_baseline.npy    # [N_baseline]  structured   far history (pre-waveform, outside context window)
  ehr_recent.npy      # [N_recent]    structured   close history (pre-waveform, within context window)
  ehr_events.npy      # [N_events]    structured   waveform-overlapping events (seg_idx ∈ [0, N_seg))
  ehr_future.npy      # [N_future]    structured   post-waveform events
  meta.json           # metadata + array manifest
```

The four EHR files share a single dtype. Readers pick which subset they need; no
reader ever has to re-do time-based splitting. See "EHR Trajectory Structure" below.

**Signal arrays**: `[N_seg, rate_hz * seg_duration_sec]` float16.
All channels share dim 0. Segment index `i` = same time window across all channels.
Channel naming: `{SIGNAL}{RATE}` (e.g. PLETH40, II120, ABP125, EEG256).

**Windowing**: 30s windows with 5s overlap (25s stride). `time_ms[i+1] - time_ms[i]` = 25s
within a contiguous block; larger jumps indicate recording gaps.

**PLETH-anchored alignment**: PLETH (PPG) is the base channel. Only WFDB segments where
PLETH exists are included. ECG II is NaN-filled when absent in a PLETH-present segment.
Recording gaps produce time_ms jumps, not NaN padding. Windows never span gaps.

**Standard channels for physiological waveform pretraining:**
- **PLETH40**: PPG at 40 Hz (1200 samples/seg) -- base channel, always has real data
- **II120** (or **I120**): ECG Lead II (or I) at 120 Hz (3600 samples/seg) -- NaN when absent

These two are the minimum required. Higher-rate or additional channels (II500, ABP125)
are optional and dataset-specific. Keep storage lean -- only extract what training needs.

**Event array** (structured dtype, shared across all four EHR files):
```python
np.dtype([
    ('time_ms', 'int64'),     # actual event timestamp (absolute ms)
    ('seg_idx', 'int32'),     # aligned signal segment index or sentinel (see below)
    ('var_id',  'uint16'),    # variable ID (lookup in var_registry.json)
    ('value',   'float32'),   # raw measured value
])
# Each file sorted by time_ms ascending. Only actual measurements. No padding.
```

## EHR Trajectory Structure

Events inside an entity's full clinical episode are split into four files by their
time relation to the waveform window. The split is done **once, at storage time**,
so downstream adapters never need to re-filter by time.

| File | Time range | `seg_idx` value | Typical consumer |
|------|-----------|-----------------|------------------|
| `ehr_baseline.npy` | `[episode_start, wave_start − context_window)` capped at `baseline_cap_ms` | `INT32_MIN` | Chronic/trend features, long-history priors |
| `ehr_recent.npy`   | `[wave_start − context_window, wave_start)` | `INT32_MIN + 1` | PCS initial-state seed, immediate clinical context |
| `ehr_events.npy`   | `[wave_start, wave_end]` | `[0, N_seg)` (real index) | Waveform-aligned training / concurrent EHR |
| `ehr_future.npy`   | `(wave_end, episode_end]` capped at `future_cap_ms` | `INT32_MIN + 2` | Forecasting / outcome labels — **label-leakage risk** |

**Constants** (defined in `physio_data/ehr_trajectory.py`, overridable per dataset):
```python
SEG_IDX_BASELINE   = np.iinfo(np.int32).min      # -2147483648
SEG_IDX_RECENT     = SEG_IDX_BASELINE + 1
SEG_IDX_FUTURE     = SEG_IDX_BASELINE + 2

CONTEXT_WINDOW_MS  = 24 * 3600 * 1000            # recent vs baseline cutoff
BASELINE_CAP_MS    = 30 * 24 * 3600 * 1000       # don't dump lifelong history
FUTURE_CAP_MS      =  7 * 24 * 3600 * 1000
```

**Why sentinel = `INT32_MIN` not `-1`:** code paths that assume `seg_idx >= 0`
(e.g. `signal[seg_idx]`) fail loudly with IndexError instead of silently wrapping
or indexing `signal[-1]`.

**Why four files, not one:** readers can mmap only what they need. The common
case (waveform-aligned training) reads `ehr_events.npy` alone and gets the exact
same bytes it gets today. No new cost.

**`meta.json` additions** (per entity):
```json
{
  "n_events":    1873,
  "n_baseline":  120,
  "n_recent":    412,
  "n_future":     85,
  "n_baseline_vars": 14,
  "n_recent_vars":   11,
  "n_future_vars":    3,
  "context_window_ms": 86400000,
  "baseline_cap_ms":   2592000000,
  "future_cap_ms":     604800000,
  "has_future_actions":      true,
  "has_future_sofa":         false,
  "has_future_sepsis_onset": false
}
```

`has_future_*` flags let outcome-prediction tasks assert no treatment leakage
before using `ehr_future.npy` as labels.

## Consumer Recipes

Any of these is a few lines on top of the canonical files; none require a
preprocessing pass.

| Purpose | Files used | Recipe |
|---------|-----------|--------|
| Waveform-only pretraining | `{CHANNEL}.npy` | standard mmap read |
| Waveform + concurrent EHR | `{CHANNEL}.npy` + `ehr_events.npy` | index events by `seg_idx` |
| PCS / patient-state prior | `ehr_recent.npy` | group by `var_id`, take last value |
| Chronic / baseline conditioning | `ehr_baseline.npy` | aggregate per `var_id` (mean/min/max) |
| EHR-only pretraining (no waveform) | `ehr_baseline + recent + events + future` | concat, re-sort by `time_ms` |
| Forecasting / outcome prediction | `ehr_future.npy` | labels; assert `has_future_actions==False` |
| Demographics conditioning | `demographics.csv` (dataset-level) | join by `entity_id` |

**Runtime rule:** adapters consume these files; they do not modify them. Any
downstream task-specific data (cohort membership, task labels, derived scores)
belongs in `tasks/{task_name}/` — never in the canonical entity directory.

**Variable registry** (`var_registry.json`): global ID -> name, unit, category mapping.
Shared across datasets. Stable IDs. Extensible by appending.
Includes `physio_min`/`physio_max` for outlier filtering and `mimic_itemids` for MIMIC-III
ITEMID mapping. Unit conversions (e.g. Fahrenheit->Celsius) via `mimic_convert` field.

**EHR variable categories** (encoded in var_id ranges):
```
var_id 0-99:     Labs       Point measurements from blood draws (Potassium, Creatinine, ...)
var_id 100-199:  Vitals     Semi-regular bedside monitor readings (HR, SpO2, RR, BP, ...)
var_id 200-299:  Actions    Interventions charted at rate changes (vasopressors, fluids, vent, ...)
var_id 300-399:  Scores     Derived values computed from above (SOFA, sepsis onset, ...)
```

All categories share the same sparse event dtype `(time_ms, seg_idx, var_id, value)`.
Actions store the rate/dose at each charting point (value=0.0 means stopped).
Category filtering at runtime: `events[events['var_id'] < 100]` = labs only.

## Workflow: Onboarding a New Dataset

Follow these steps IN ORDER. Each has verification. Do not skip ahead.

### Step 0a: Research

Before touching any data, understand the dataset:
- Search online for documentation, papers, PhysioNet/data repository pages
- What raw formats? (WFDB, EDF, CSV, Parquet, NPZ, HDF5, TDMS, ...)
- What signals? What sample rates?
- What clinical/event data? What variables?
- How are entities (patients/subjects) identified?
- How are signals and events linked? (shared ID? join table?)
- Known issues? (time zones, missing channels, encoding, de-identification artifacts)

Write to `datasets/{dataset}/explore/README.md`.

### Step 0b: Structural Exploration (code)

Write code to inspect the raw data:

```python
# 1. Scan directory structure
#    What files exist? How are they organized? File counts and sizes.

# 2. Read a sample signal file
#    Channels found, sample rates, duration, dtype, value ranges, NaN %

# 3. Read a sample clinical data file
#    Columns, dtypes, entity IDs, time range, variable names

# 4. Check entity overlap
#    How many entities have both signals AND events?
#    What's the matching key?
```

Save structured findings to `datasets/{dataset}/explore/dataset_profile.json`:
```json
{
  "name": "dataset_name",
  "raw_format": {"signals": {"type": "...", "path": "..."}, "events": {"type": "...", "path": "..."}},
  "entity_id_key": {"signals": "field_name", "events": "field_name", "linkage": "how they match"},
  "channels_found": [{"name": "...", "source_rate_hz": N, "target": "...", "target_rate_hz": N}],
  "event_variables_found": [{"source_name": "...", "maps_to_var_id": N, "unit": "..."}],
  "n_entities_overlap": N,
  "time_format": "...",
  "known_issues": ["..."]
}
```

### Step 0c: Single-Entity Demo Alignment

**Most important step.** Pick one entity with both signals and events. Run full pipeline:

1. Load raw signal -> resample to target rate -> segment into fixed windows
2. Load raw events -> compute `seg_idx` via `np.searchsorted(time_ms_array, event_time)`
3. Save in canonical format
4. **Plot signals with event markers on shared time axis**
5. **Ask user to visually verify alignment before proceeding**

Save to `datasets/{dataset}/explore/demo_{entity_id}/`.

### Step 0d: Write API.md

Fill in `datasets/{dataset}/API.md` specifying:
- **Where**: exact local file paths to raw signal and event data
- **Signals**: which channels, source rates, target rates, channel naming
- **Events**: which variables, source column names/IDs, units, physiological/valid ranges
- **Linkage**: how entity IDs map between signal and event sources
- **Time**: timestamp formats, timezone handling
- **Parameters**: segment duration, NaN thresholds, split ratios
- **Known issues**: from exploration

Use `datasets/TEMPLATE_API.md` as starting point. See `datasets/mimic3/API.md` for example.

**API.md must be reviewed before writing extraction scripts.**

### Pre-flight: Check server resources

Before running any heavy processing, check what you're working with:
- CPU cores and current load (to set `--workers`)
- Available memory (each wfdb worker uses ~1-2 GB)
- Free disk on the output directory
- GPU availability (for later training, not preprocessing)
- Job scheduler (SLURM/PBS) -- may need `sbatch` instead of direct `python`

**Shared clusters: never use more than half of total resources.** Research servers are
shared. Cap `--workers` at 50% of total cores, even if the machine looks idle. Other
users' jobs may start at any time.

### Steps 1-5: Extraction Pipeline

The pipeline has a critical ordering constraint: **check EHR availability BEFORE
extracting waveforms.** Waveform-only patients are not useful for training. Don't
waste hours extracting signals for patients with no labs/vitals.

```
Stage 1: Scan signals     -> which entities have the required channels?
Stage 2: Extract events   -> filter EHR tables to target variables
Stage 3: Cross-check      -> keep ONLY entities with BOTH signals AND events
Stage 4: Extract signals  -> read raw, resample, segment, align events, save .npy
Stage 5: Manifest + splits -> validate, build index, generate train/test
```

**Stage 3 (cross-check) is the gate.** It joins the signal inventory with EHR event
counts per entity and drops any entity with insufficient overlap. Report the numbers:
how many entities have signals only, events only, both, neither. This catches data
linkage problems early (e.g., ID mismatch between signal and event sources).

For each stage, use shared utilities:
- Resampling: `scipy.signal.resample_poly`
- Segmenting: split 1D signal into `[N_seg, samples_per_seg]` float16 C-contiguous
- Event alignment: `np.searchsorted` on segment timestamps
- Saving: enforce dtype, contiguity, meta.json schema
- Verification: run after each stage

### Testing: Always `--limit` first

Signal extraction is the slow stage (hours for large datasets). Always test
with `--limit 5` first to verify the output format before committing to a full run.
Check the output directory: each entity should have all expected .npy files + meta.json.

### Parallelization

Signal extraction is per-patient independent -- parallelize with multiprocessing:
- Pre-group EHR data by patient ID before distributing to workers (avoid copying full tables)
- Use `mp.Pool` with `imap_unordered` for best throughput
- Cap `--workers` at 50% of total cores (shared cluster)
- Use `tqdm` for progress bar with live OK/SKIP/ERR counts
- Run in tmux/screen so SSH disconnect doesn't kill the job

### Verification: Run After EVERY Stage

| After | Check | Abort if |
|-------|-------|----------|
| Signal scan | Target channels found, sample rates consistent | No entities with required channels |
| Event extraction | No duplicates, values in physiological range, timestamps sane | Null IDs, timestamps out of range |
| **Cross-check** | **Any EHR overlap (>=1 event). Report variable coverage for diagnostics.** | **Zero overlap** |
| Signal extraction (.npy) | float16, C-contiguous, N_seg consistent, NaN < 20%, time_ms monotonic | Shape mismatch, all NaN |
| Event building (ehr_baseline / recent / events / future) | Correct dtype, each file sorted by time_ms, `ehr_events.seg_idx ∈ [0, N_seg)`, sentinels match spec for others, var_id in registry, no event straddles two files | seg_idx out of bounds in `ehr_events.npy`, or an event appears in two partitions |
| Manifest + splits | All dirs valid, no **subject**-level overlap in splits, ratio within 5% | Missing dirs, same subject in both sets |

**Critical: split by subject, not by admission.** One subject may have multiple
admissions/encounters. All admissions from the same subject must be in the same
train/val/test set. Splitting by directory (admission-level) causes data leakage.
Use train/val/test (70/15/15) not just train/test.

**Inline assertions in save function:**
```python
assert arr.dtype == np.float16
assert arr.flags['C_CONTIGUOUS']
assert arr.shape[0] == len(time_ms)       # signal-time consistency
assert np.all(np.diff(time_ms) > 0)       # monotonicity
assert np.all(events['seg_idx'] >= 0)
assert np.all(events['seg_idx'] < n_seg)  # bounds for ehr_events.npy only
assert np.all(np.diff(events['time_ms']) >= 0)  # sorted
# For baseline/recent/future: assert seg_idx equals the file's sentinel value
# and that time_ms respects the partition boundary.
```

## Post-Stages: Downstream Task Adaptation

The main pipeline produces a **task-agnostic** canonical dataset. Downstream tasks
(sepsis prediction, AKI detection, mortality prediction, etc.) need additional steps:

1. **Cohort filtering** -- select patients matching task criteria (e.g., Sepsis-3 cohort)
2. **Add task-specific EHR** -- SOFA scores, onset times, mortality labels, treatment actions
3. **Add task-specific labels** -- binary labels, time-to-event, severity scores
4. **Generate task-specific splits** -- filtered version of downstream_splits.json

**Critical: post-stages NEVER modify canonical files.** Canonical `.npy` files are
immutable after the main pipeline. Post-stages write task-specific data ALONGSIDE them.
This ensures:
- Pretraining sees only base EHR events (no task-specific var_ids leaking in)
- Different downstream tasks don't contaminate each other
- Post-stages are idempotent (safe to re-run)

```
{output_dir}/
├── {patient_id}/              # Canonical (untouched by post-stages)
│   ├── PLETH40.npy
│   ├── II120.npy
│   ├── time_ms.npy
│   ├── ehr_baseline.npy
│   ├── ehr_recent.npy
│   ├── ehr_events.npy
│   ├── ehr_future.npy
│   └── meta.json
├── demographics.csv           # One row per entity, static attributes
├── manifest.json              # All patients
├── pretrain_splits.json       # All patients
├── tasks/                     # Task-specific (added by post-stages)
│   ├── sepsis/
│   │   ├── cohort.json        # Patient list + onset times + labels
│   │   ├── splits.json        # Task-specific train/val/test
│   │   └── extra_events/      # Task-specific EHR events (SOFA, onset markers)
│   │       └── {patient_id}.npy  # Same dtype as ehr_events.npy
│   └── aki/
│       └── ...

At training time, the adapter loads whichever canonical EHR files the task needs
and optionally merges `tasks/{task}/extra_events/{patient_id}.npy`. Task-specific
events follow the same four-file split convention (`seg_idx` sentinel rules apply)
so forecasting tasks still get clean label/feature separation.
```

**Important design principle:** The main pipeline should extract ALL available EHR
variables broadly, not filter to a narrow set. Task-specific variable requirements
are a post-stage concern. If the main pipeline filters too aggressively (e.g., requiring
70% of 9 specific variables), patients valid for other tasks may be excluded.

## Adding EHR Variables to Existing Data

The sparse event format means new variables can be added without re-extracting waveforms.
Waveform extraction is the slow step (hours). EHR event building is fast (minutes).

When the user wants to add new variables (e.g., HR, SpO2, Bilirubin) to already-processed data:

1. Add the new variable to `var_registry.json` (assign next ID)
2. Write a **stage 3b** script that:
   - Reads the new variable's events from raw EHR tables
   - For each existing patient directory, loads `time_ms.npy` and episode bounds from `meta.json`
   - Splits new events into baseline / recent / events / future using the same
     rules as the main pipeline (see `physio_data/ehr_trajectory.split_events`)
   - **Merges** each new partition with the matching existing file
     (load + append + re-sort by `time_ms`)
   - Re-saves all four `ehr_*.npy` files and updates `meta.json` counts
3. Do NOT re-extract waveforms -- `.npy` waveform files stay untouched

This is fast because:
- Only reads small EHR tables, not raw WFDB
- Only writes the small `ehr_events.npy` per patient, not the large waveform arrays
- Can be parallelized same as stage 3

## Key Constraints

- **No compression.** Disk is not a constraint. CPU on GPU servers is weak.
- **No dense event arrays.** Events are sparse; store them sparse. Dense = adapter runtime concern.
- **No normalization in storage.** Raw values. Normalization is task-specific.
- **No interpolation in storage.** Interpolation strategy is a downstream decision.
- **Zero new runtime dependencies.** Training/eval code only needs numpy.
- **C-contiguous float16 for signals.** This guarantees zero-copy `reshape(-1)`.
- **Signal + event co-existence.** Don't process signals without events or vice versa.
- **Extract EHR broadly, filter narrowly.** The main pipeline should extract ALL available
  EHR variables, not just a narrow set. Task-specific variable requirements belong in
  post-stages. If the main pipeline filtered too aggressively, the post-stage can
  **extract missing patients** on demand -- run waveform extraction only for patients
  in the task cohort that were dropped by the main pipeline. This avoids re-running
  the full pipeline just to recover a few hundred patients.
- **Admission-level linkage.** One patient may have multiple hospital visits. Match signal
  recording time to the correct admission/encounter, then filter events by that admission ID.
  Output directory should be `{patient_id}_{admission_id}/`, not just `{patient_id}/`.

## Project Structure

```
Physio_Data/
├── physio_data/              # Reusable package (schema, explore, resample, segment, ehr, io, verify)
├── datasets/
│   ├── TEMPLATE_API.md       # Blank template
│   └── {dataset}/
│       ├── API.md            # Dataset specification (paths, channels, variables)
│       ├── explore/          # Step 0 output (README, profile, demo entity)
│       ├── processed/        # Canonical per-entity dirs
│       ├── ehr/              # Source event tables (parquet)
│       └── indices/          # manifest.json, splits, var_registry.json
├── scripts/{dataset}/        # Dataset-specific extraction scripts
├── adapters/                 # Consumer-specific adapters (dataset-agnostic)
└── indices/
    ├── var_registry.json     # Global variable registry
    └── normalization.json    # Per-dataset normalization stats
```
