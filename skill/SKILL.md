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
  {CHANNEL}.npy     # [N_seg, samples_per_seg]  float16, C-contiguous
  time_ms.npy       # [N_seg]                    int64, monotonically increasing
  ehr_events.npy    # [N_events]                 structured array (sparse)
  meta.json         # metadata + array manifest
```

**Signal arrays**: `[N_seg, rate_hz * seg_duration_sec]` float16.
All channels share dim 0. Segment index `i` = same time window across all channels.
Channel naming: `{SIGNAL}{RATE}` (e.g. PLETH40, II120, ABP125, EEG256).

**Event array** (structured dtype):
```python
np.dtype([
    ('time_ms', 'int64'),     # actual event timestamp (absolute ms)
    ('seg_idx', 'int32'),     # aligned signal segment index
    ('var_id',  'uint16'),    # variable ID (lookup in var_registry.json)
    ('value',   'float32'),   # raw measured value
])
# Sorted by time_ms. Only actual measurements. No padding.
```

**Variable registry** (`var_registry.json`): global ID -> name, unit, type mapping.
Shared across datasets. Stable IDs. Extensible by appending.

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

Signal extraction (stage 4) is the slow stage (hours for large datasets). Always test
with `--limit 5` first to verify the output format before committing to a full run.
Check the output directory: each entity should have all expected .npy files + meta.json.

### Verification: Run After EVERY Stage

| After | Check | Abort if |
|-------|-------|----------|
| Signal scan | Target channels found, sample rates consistent | No entities with required channels |
| Event extraction | No duplicates, values in physiological range, timestamps sane | Null IDs, timestamps out of range |
| **Cross-check** | **>=70% of target variables present, each with >=2 data points** | **Zero overlap or too few variables covered** |
| Signal extraction (.npy) | float16, C-contiguous, N_seg consistent, NaN < 20%, time_ms monotonic | Shape mismatch, all NaN |
| Event building (ehr_events) | Correct dtype, sorted by time_ms, seg_idx in bounds, var_id in registry | seg_idx out of bounds |
| Manifest + splits | All dirs valid, no entity overlap in splits, ratio within 5% | Missing dirs, split overlap |

**Inline assertions in save function:**
```python
assert arr.dtype == np.float16
assert arr.flags['C_CONTIGUOUS']
assert arr.shape[0] == len(time_ms)       # signal-time consistency
assert np.all(np.diff(time_ms) > 0)       # monotonicity
assert np.all(events['seg_idx'] >= 0)
assert np.all(events['seg_idx'] < n_seg)  # bounds
assert np.all(np.diff(events['time_ms']) >= 0)  # sorted
```

## Key Constraints

- **No compression.** Disk is not a constraint. CPU on GPU servers is weak.
- **No dense event arrays.** Events are sparse; store them sparse. Dense = adapter runtime concern.
- **No normalization in storage.** Raw values. Normalization is task-specific.
- **No interpolation in storage.** Interpolation strategy is a downstream decision.
- **Zero new runtime dependencies.** Training/eval code only needs numpy.
- **C-contiguous float16 for signals.** This guarantees zero-copy `reshape(-1)`.
- **Signal + event co-existence.** Don't process signals without events or vice versa.

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
