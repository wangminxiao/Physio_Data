# Physio_Data: Unified Physiological Data Hub

---

# Part 1: High-Level Design

## Design Principles

1. **Waveform and EHR are stored differently because they ARE different.**
   - Waveform: dense, regular, high-rate -> mmap .npy, row = fixed-length segment
   - EHR: sparse, irregular, event-driven -> structured .npy, row = one measurement event

2. **Alignment is an index, not a copy.**
   - EHR events carry a `seg_idx` that points into the waveform segment array.
   - No dense `[N_seg, N_vars]` padding. The sparse event IS the alignment.

3. **Zero CPU overhead on the training hot path.**
   - All .npy files are uncompressed, C-contiguous, directly mmap-ready.
   - GPU server nodes have weak CPUs. Every cycle spent on decompression
     is a cycle not spent on data preprocessing (NaN handling, clipping, tokenization).

4. **Storage format is dataset-agnostic.**
   - MIMIC-III, MC_MED, MOVER, etc. -- different raw sources, same output layout.
   - Dataset-specific logic lives only in extraction scripts, not in the stored format.

5. **Schema is extensible without breaking changes.**
   - Adding a new EHR variable = add a row to `var_registry.json`, re-run EHR extraction.
   - Adding a new waveform channel = add a .npy file to the patient directory.
   - No existing array shapes change.

## Canonical Output Format

### Per-patient directory

```
{PATIENT_ID}/
├── {CHANNEL}.npy          # [N_seg, samples_per_seg]  float16, C-contiguous
│                          #   e.g. PLETH40.npy  [1440, 1200]
│                          #        II120.npy    [1440, 3600]
│                          #        II500.npy    [1440, 15000]
├── time_ms.npy            # [N_seg]  int64, absolute ms timestamp per segment
├── ehr_events.npy         # [N_events]  structured array (sparse)
└── meta.json              # patient metadata + array manifest
```

All waveform .npy files share dimension 0 (N_seg). Segment index `i` across all files
corresponds to the same time window `[time_ms[i], time_ms[i] + seg_dur)`.

### Windowing: overlapping segments

- **Window**: 30 seconds (`seg_dur`)
- **Overlap**: 5 seconds
- **Stride**: 25 seconds (`seg_dur - overlap`)
- `time_ms[i]` is the absolute start time of window `i`
- Within a contiguous recording block: `time_ms[i+1] - time_ms[i] = stride (25s)`
- Across a recording gap: `time_ms[i+1] - time_ms[i] > stride` (the jump IS the gap)
- ~20% more segments than non-overlapping, smoother boundaries for models

### Channel alignment: PLETH-anchored

PLETH (PPG) is the **base channel**. Only WFDB segments where PLETH exists are included.
Other channels (ECG II) are NaN-filled when absent in a PLETH-present segment.
If PLETH is missing, the segment is skipped for ALL channels (even if ECG exists).
Recording gaps (null segments or PLETH-absent segments) are NOT NaN-filled —
they produce jumps in `time_ms`. Windows never span gap boundaries.

### Waveform arrays

- **Shape**: `[N_seg, samples_per_seg]` where `samples_per_seg = sample_rate * seg_dur`
- **Dtype**: float16 (sufficient for physiological signals, halves storage vs float32)
- **Layout**: C-contiguous (row-major), guaranteed by `np.ascontiguousarray()` at save time
- **Access**: `np.load(path, mmap_mode='r')` -> OS-backed memory mapping
- **NaN**: segments where a non-base channel was absent are filled with NaN (float16 supports NaN)

**Why this shape enables zero-copy concatenation:**
```python
mmap = np.load('PLETH40.npy', mmap_mode='r')  # [N_seg, 1200]
window = mmap[a:b, :]     # contiguous row slice -> numpy VIEW (no copy)
flat = window.reshape(-1)  # contiguous reshape -> numpy VIEW (no copy)
# Only copy: torch.from_numpy(flat.copy()) at the very end
```

### EHR events (sparse)

```python
ehr_event_dtype = np.dtype([
    ('time_ms', 'int64'),     # actual measurement timestamp (absolute ms)
    ('seg_idx', 'int32'),     # index into waveform segment array (alignment key)
    ('var_id',  'uint16'),    # variable ID -> lookup in var_registry.json
    ('value',   'float32'),   # raw measured value (not normalized)
])
```

- **Sorted by `time_ms`** -> `seg_idx` is non-decreasing -> binary search for range queries
- **Only actual measurements** -> no zeros, no interpolation, no padding
- **Raw values** -> normalization/interpolation are runtime decisions, not storage decisions

**Range query** (find EHR events for waveform segments [start, end)):
```python
events = np.load('ehr_events.npy', mmap_mode='r')
lo = np.searchsorted(events['seg_idx'], start, side='left')
hi = np.searchsorted(events['seg_idx'], end, side='left')
window_events = events[lo:hi]   # O(log N) lookup, typically ~5-15 events per hour
```

### Variable registry (`var_registry.json`)

Global, shared across all patients and datasets. Maps variable IDs to metadata:

```json
{
  "variables": [
    {"id": 0,  "name": "Potassium",  "unit": "mEq/L",  "type": "lab"},
    {"id": 1,  "name": "Calcium",    "unit": "mg/dL",   "type": "lab"},
    {"id": 2,  "name": "Sodium",     "unit": "mEq/L",   "type": "lab"},
    {"id": 3,  "name": "Glucose",    "unit": "mg/dL",   "type": "lab"},
    {"id": 4,  "name": "Lactate",    "unit": "mmol/L",  "type": "lab"},
    {"id": 5,  "name": "Creatinine", "unit": "mg/dL",   "type": "lab"},
    {"id": 6,  "name": "NBPs",       "unit": "mmHg",    "type": "vital"},
    {"id": 7,  "name": "NBPd",       "unit": "mmHg",    "type": "vital"},
    {"id": 8,  "name": "NBPm",       "unit": "mmHg",    "type": "vital"},
    {"id": 9,  "name": "HR",         "unit": "bpm",     "type": "vital"},
    {"id": 10, "name": "SpO2",       "unit": "%",       "type": "vital"},
    {"id": 11, "name": "RR",         "unit": "/min",    "type": "vital"}
  ]
}
```

IDs are stable across datasets. A dataset may only populate a subset of variables.
Adding a variable = append to this list. No array shape changes anywhere.

### Normalization stats (`normalization.json`)

Per-variable, per-dataset normalization parameters. Computed during extraction, applied at runtime:

```json
{
  "mimic3": {
    "0": {"min": 2.5, "max": 6.5, "p01": 2.8, "p99": 6.2, "mean": 4.1, "std": 0.5},
    "1": {"min": 6.0, "max": 12.0, ...},
    ...
  },
  "mcmed": { ... }
}
```

### Per-patient `meta.json`

```json
{
  "patient_id": "10032_174162",
  "source_dataset": "mimic3",
  "n_segments": 1727,
  "segment_duration_sec": 30,
  "overlap_sec": 5,
  "stride_sec": 25,
  "total_duration_hours": 12.0,
  "recording_start_ms": 1234567890000,
  "n_blocks": 2,
  "n_gaps": 1,
  "channels": {
    "PLETH40": {"sample_rate_hz": 40,  "shape": [1727, 1200],  "dtype": "float16"},
    "II120":   {"sample_rate_hz": 120, "shape": [1727, 3600],  "dtype": "float16"}
  },
  "per_channel": {
    "PLETH40": {"nan_ratio": 0.0,  "valid_seg_ratio": 1.0},
    "II120":   {"nan_ratio": 0.03, "valid_seg_ratio": 0.97}
  },
  "n_ehr_events": 187
}
```

### Global `manifest.json`

```json
[
  {
    "dir": "10032_174162",
    "source_dataset": "mimic3",
    "n_segments": 1440,
    "duration_hours": 12.0,
    "channels": ["PLETH40", "II120", "II500"],
    "n_ehr_events": 187
  }
]
```

Read once at Dataset init. Enables building sample index and TBTT streams without
opening individual patient directories.

### I/O comparison: mmap .npy vs alternatives

Reading 1-hour waveform window (120 segments x 1200 samples x float16 = 281 KB):

| Format | Mechanism | Per-access overhead | Sequential prefetch |
|--------|-----------|--------------------|--------------------|
| mmap .npy | page fault -> OS loads pages | 0 syscalls, 0 memcpy | OS prefetcher auto-optimizes |
| Zarr (no compress) | open + read + close per chunk | 3 syscalls + 1 memcpy | None (separate chunk files) |
| NPZ | decompress full array on access | 1 zlib decompress + 1 memcpy | None |

---

# Part 2: Generalization Across Datasets

## The problem

Different source datasets are organized differently:

| Dataset | Raw waveform format | Raw EHR format | Patient ID | Channels |
|---------|-------------------|----------------|------------|----------|
| MIMIC-III | WFDB records (PhysioBank matched subset) | CSV.gz (CHARTEVENTS, LABEVENTS) | SUBJECT_ID + HADM_ID | PLETH, II, V, ABP, ... |
| MC_MED | Pre-extracted NPZ per encounter | CSV (labs.csv, numerics.csv, meds.csv) | Encounter number (CSN) | ECG, PLETH, variable channels |
| MOVER | TBD | TBD | TBD | ABP, ECG, PLETH |

But the output format is always the same per-patient directory with .npy + ehr_events.

## Architecture: Step 0 exploration -> extraction -> canonical output

```
┌─────────────────────────┐
│  Step 0: Explore         │   datasets/{dataset}/explore/
│  (new dataset)           │   Search docs, read raw files, align 1 patient as demo.
│                          │   Output: exploration notebook + dataset_profile.json
└────────────┬────────────┘
             │ "now I understand the raw format"
             ▼
┌─────────────────────────┐
│  Dataset-specific        │   scripts/mimic3/*, scripts/mcmed/*, ...
│  extraction scripts      │   Each dataset gets its own extraction pipeline
│  (read raw, resample,    │   that handles its unique raw format.
│   segment, align, save)  │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Canonical output format │   datasets/{dataset}/processed/{patient_id}/
│  (same for all datasets) │     {CHANNEL}.npy + time_ms.npy + ehr_events.npy + meta.json
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Dataset-agnostic        │   adapters/hnet_npy_dataset.py
│  adapters                │   adapters/uniphy_npy_adapter.py
│  (read canonical format) │   Don't know or care about the source dataset.
└─────────────────────────┘
```

## Step 0: Dataset Exploration (before writing any extraction code)

Every new dataset starts here. The goal is to answer these questions before writing
a single line of pipeline code:

### Phase 0a: Research

- What is this dataset? (paper, documentation, PhysioNet page, etc.)
- What raw file formats? (WFDB, EDF, CSV, Parquet, NPZ, proprietary)
- What signals are recorded? (ECG leads, PPG, ABP, EEG, etc.)
- What sample rates?
- What clinical data is available? (labs, vitals, medications, diagnoses)
- How are patients identified? (subject_id, encounter_id, CSN, MRN)
- How are waveform and EHR linked? (shared patient ID? separate linkage table?)
- Known issues or gotchas? (missing channels, time zone offsets, duplicate records)

Output: `datasets/{dataset}/explore/README.md` documenting findings.

### Phase 0b: Structural exploration (code)

Use `physio_data.explore` utilities to answer:

```python
from physio_data import explore

# 1. Scan raw data directory -- what files exist, how are they organized?
report = explore.scan_directory("/path/to/raw/dataset")
# -> prints tree structure, file type counts, total sizes

# 2. Sample a few waveform files -- what's inside?
explore.inspect_waveform("/path/to/sample_record")
# -> prints: channels found, sample rates, duration, dtype, value ranges, NaN %

# 3. Sample EHR files -- what columns, what patient IDs?
explore.inspect_ehr("/path/to/clinical_data.csv")
# -> prints: columns, dtypes, unique patient count, time range, variable names

# 4. Find the linkage -- how do waveform patients map to EHR patients?
explore.find_patient_overlap(waveform_ids, ehr_ids)
# -> prints: N matched, N waveform-only, N ehr-only, matching key type
```

Output: `datasets/{dataset}/explore/dataset_profile.json`:

```json
{
  "name": "mcmed",
  "raw_format": {
    "waveform": {"type": "npz", "path": "/home/mxwan/workspace/MC_MED_ECG_PLETH_40hz_120hz_500hz_lab_vital/"},
    "ehr": {"type": "csv", "path": "/home/mxwan/workspace/mc_med_csv/"}
  },
  "patient_id_key": {
    "waveform": "encounter_nbr (parsed from NPZ filename)",
    "ehr": "CSN (in labs.csv, numerics.csv)",
    "linkage": "encounter_nbr == CSN"
  },
  "channels_found": [
    {"name": "PLETH40", "source_rate_hz": 40, "target": "PLETH40", "target_rate_hz": 40},
    {"name": "II120",   "source_rate_hz": 120, "target": "II120",  "target_rate_hz": 120},
    {"name": "II500",   "source_rate_hz": 500, "target": "II500",  "target_rate_hz": 500}
  ],
  "ehr_variables_found": [
    {"source_name": "POTASSIUM",  "maps_to_var_id": 0,  "source_unit": "mEq/L"},
    {"source_name": "CALCIUM",    "maps_to_var_id": 1,  "source_unit": "mg/dL"},
    {"source_name": "HR",         "maps_to_var_id": 9,  "source_unit": "bpm"},
    {"source_name": "SpO2",       "maps_to_var_id": 10, "source_unit": "%"}
  ],
  "n_patients_waveform": 600,
  "n_patients_ehr": 600,
  "n_patients_overlap": 600,
  "time_format": "unix_ms",
  "known_issues": [
    "NPZ files already contain pre-extracted waveforms -- no raw WFDB to read",
    "Waveforms already at target rates -- no resampling needed, just reformat"
  ]
}
```

### Phase 0c: Single-patient demo alignment

The most important step. Pick one patient, load waveform + EHR from raw sources, align
them, and verify visually. This becomes the template for the full pipeline.

```python
from physio_data import explore

# Build an end-to-end demo for 1 patient
demo = explore.build_demo_patient(
    dataset_profile="datasets/mcmed/explore/dataset_profile.json",
    patient_id="98972716",
    waveform_path="/home/mxwan/workspace/MC_MED_ECG_PLETH_40hz_120hz_500hz_lab_vital/98972716_P40_E120_E500_577.npz",
    ehr_path="/home/mxwan/workspace/mc_med_csv/labs.csv",
)

# demo contains:
#   demo.waveform_raw     - dict of raw signals before resampling
#   demo.waveform_resampled - dict of resampled signals
#   demo.waveform_segmented - dict of [N_seg, samples_per_seg] arrays
#   demo.time_ms          - segment timestamps
#   demo.ehr_events_raw   - raw EHR measurements with original timestamps
#   demo.ehr_events       - structured array with seg_idx alignment
#   demo.alignment_report - which events landed on which segments, any misses

# Visualize to verify alignment makes sense
explore.plot_demo(demo, time_range_hours=(2, 4))
# -> Shows waveform + EHR event markers on shared time axis

# If everything looks right, save as the reference for this dataset
explore.save_demo(demo, "datasets/mcmed/explore/demo_patient_98972716/")
```

The demo patient directory serves dual purpose:
1. **Verification**: visual check that alignment logic is correct before running full pipeline
2. **Template**: the demo code becomes the skeleton of the extraction scripts

### Phase 0 output summary

```
datasets/{dataset}/explore/
├── README.md                    # Research findings, documentation links
├── dataset_profile.json         # Structured profile (channels, IDs, EHR vars, issues)
├── exploration.ipynb            # Interactive notebook with all inspection code
└── demo_patient_{id}/           # One fully processed patient for verification
    ├── waveform_raw/            # Raw signals before processing
    ├── PLETH40.npy              # Processed output (canonical format)
    ├── II120.npy
    ├── time_ms.npy
    ├── ehr_events.npy
    ├── meta.json
    └── alignment_plot.png       # Visual verification
```

### Phase 0d: Write the Dataset API document

After exploration, codify all decisions into `datasets/{dataset}/API.md` using the
template at `datasets/TEMPLATE_API.md`. This document specifies:

- **Where**: exact file paths to raw waveform and EHR data on the local filesystem
- **What**: which waveform channels to extract, at what target rates
- **What**: which EHR variables to extract, source column names / ITEMIDs, physiological ranges
- **How**: patient ID linkage between waveform and EHR, time format handling
- **Parameters**: segment duration, NaN thresholds, split ratios

See `datasets/mimic3/API.md` for a filled-in example.

The API.md serves as:
1. **Input spec for extraction scripts** -- scripts read this to know what to extract
2. **Review artifact** -- can be reviewed by collaborators before running full pipeline
3. **Documentation** -- permanent record of dataset-specific decisions

Only after Step 0 (exploration + API.md) is complete do you write the extraction scripts.

## Directory layout

```
Physio_Data/
├── physio_data/                   # THE PACKAGE (reusable across all datasets)
│   ├── __init__.py
│   ├── schema.py                  # Format contract: dtypes, ehr_event_dtype, meta schema
│   ├── explore.py                 # Step 0: scan dirs, inspect files, build demo patient
│   ├── resample.py                # Waveform resampling (scipy wrappers)
│   ├── segment.py                 # Waveform windowing into [N_seg, samples_per_seg]
│   ├── ehr.py                     # Build ehr_events.npy from (time, var, value)
│   ├── registry.py                # var_registry.json management (load, lookup, extend)
│   ├── io.py                      # save_patient(), load_patient(), mmap helpers
│   ├── manifest.py                # Build manifest.json, generate splits
│   ├── normalize.py               # Compute & apply per-variable normalization
│   └── validate.py                # Verify a patient dir meets the contract
│
├── datasets/
│   ├── TEMPLATE_API.md            # Blank template for new datasets
│   │
│   ├── mimic3/
│   │   ├── API.md                # Data locations, channels, EHR vars, parameters
│   │   ├── explore/              # Step 0 output: README, profile, demo patient
│   │   ├── processed/            # Canonical per-patient dirs
│   │   ├── ehr/                  # Source EHR tables (parquet)
│   │   └── indices/              # manifest, splits, alignment_index
│   │
│   ├── mcmed/
│   │   ├── API.md
│   │   ├── explore/
│   │   ├── processed/
│   │   ├── ehr/
│   │   └── indices/
│   │
│   └── (future datasets follow same structure)
│
├── indices/
│   ├── var_registry.json          # GLOBAL variable registry (shared across datasets)
│   └── normalization.json         # Per-dataset, per-variable normalization stats
│
├── scripts/                       # Dataset-SPECIFIC extraction (one dir per source)
│   ├── mimic3/
│   │   ├── stage0_extract_ehr.py
│   │   ├── stage1_align_waveform_ehr.py
│   │   ├── stage2_extract_waveforms.py
│   │   ├── stage3_build_ehr_events.py
│   │   └── stage4_quality_and_splits.py
│   │
│   └── mcmed/
│       ├── convert_npz.py
│       └── build_ehr_events.py
│
├── configs/
│   ├── mimic3.yaml
│   └── mcmed.yaml
│
└── pyproject.toml
```

## What each dataset-specific extractor must produce

Regardless of raw format, the extraction pipeline for dataset X must output:

```
datasets/X/processed/{patient_id}/
  ├── {CHANNEL}.npy       # [N_seg, samples_per_seg] float16, C-contiguous
  ├── time_ms.npy         # [N_seg] int64
  ├── ehr_events.npy      # [N_events] structured (time_ms, seg_idx, var_id, value)
  └── meta.json           # standard schema

datasets/X/indices/
  ├── manifest.json       # all patients
  └── (split files)       # dataset-specific train/test splits
```

The **contract** is the file layout + dtypes + meta.json schema. How the raw data gets
there is the extractor's business.

## The `physio_data` package

All reusable logic lives here. Dataset-specific scripts import from this package.

```python
# physio_data/schema.py -- the contract
EHR_EVENT_DTYPE = np.dtype([
    ('time_ms', 'int64'), ('seg_idx', 'int32'), ('var_id', 'uint16'), ('value', 'float32')
])
WAVEFORM_DTYPE = np.float16
TIME_DTYPE = np.int64
SEGMENT_DUR_SEC = 30

def validate_patient_dir(dir_path, var_registry): ...
def validate_meta_json(meta): ...


# physio_data/verify.py -- per-stage verification (fail fast, clear errors)
#
# Design: every function returns a VerifyReport with pass/fail/warnings.
# Pipeline scripts call verify after each stage and abort on failure.

class VerifyReport:
    passed: bool
    errors: list[str]       # hard failures -- must fix before proceeding
    warnings: list[str]     # suspicious but not blocking
    stats: dict             # summary numbers for logging

def verify_ehr_tables(labs_path, vitals_path, demo_path) -> VerifyReport:
    """After Stage 0: EHR extraction.
    Checks:
    - No duplicate (patient, time, variable) rows
    - Values within physiological ranges (K: 1-10, Na: 100-180, HR: 20-300, ...)
    - Timestamps not in future, not before 1990
    - Patient count matches expected (from dataset docs)
    - No null patient IDs
    Stats: n_patients, n_lab_events, n_vital_events, per-variable value distributions
    """

def verify_alignment_index(index_path, ehr_path) -> VerifyReport:
    """After Stage 1: waveform-EHR alignment.
    Checks:
    - Every entry has start_time < end_time
    - Extraction windows are reasonable duration (minutes to days, not seconds or years)
    - Patient IDs in alignment index exist in EHR tables
    - At least 1 overlapping lab or vital per entry (otherwise why extract?)
    - No duplicate (subject, hadm, record) entries
    Stats: n_records, median_window_hours, n_patients_with_labs, channel_availability
    """

def verify_waveforms(patient_dir) -> VerifyReport:
    """After Stage 2: waveform extraction. Run per patient.
    Checks:
    - All channel .npy files: dtype == float16, C-contiguous, 2D [N_seg, expected_cols]
    - time_ms.npy: dtype == int64, monotonically increasing, len == N_seg
    - N_seg consistent across all channels
    - NaN ratio per channel < threshold (flag if > 20%)
    - Not flat-line: std > 1e-6 for at least 80% of segments
    - Value range plausible (no inf, no extreme outliers beyond 5 sigma from channel median)
    - Zero-copy reshape works: arr[0:120,:].reshape(-1).base is arr
    Stats: n_segments, nan_ratio, value_range, flat_segments_pct per channel
    """

def verify_ehr_events(patient_dir, var_registry) -> VerifyReport:
    """After Stage 3: EHR event building. Run per patient.
    Checks:
    - ehr_events.npy dtype matches EHR_EVENT_DTYPE exactly
    - Sorted by time_ms (non-decreasing)
    - seg_idx in [0, N_seg) where N_seg from waveform arrays
    - var_id values exist in var_registry
    - No duplicate (time_ms, var_id) pairs
    - Values within physiological ranges (per var_id)
    - At least 1 event exists (warn if 0 -- patient has waveform but no labs?)
    Stats: n_events, events_per_variable, time_span_hours, avg_inter_event_gap
    """

def verify_manifest_and_splits(manifest_path, splits_path, processed_dir) -> VerifyReport:
    """After Stage 4: manifest and splits.
    Checks:
    - Every dir in processed/ has an entry in manifest
    - Every manifest entry has a dir that exists
    - n_segments in manifest matches actual .npy shape
    - Train/test splits have zero patient overlap
    - Split proportions within 5% of target
    - No empty splits
    Stats: n_train, n_test, split_ratio, total_hours, total_ehr_events
    """

def verify_full_pipeline(dataset_dir, var_registry) -> VerifyReport:
    """End-to-end: run all per-patient verifications + manifest check.
    Prints a summary table:
      patients processed: 2674
      patients passed:    2671
      patients failed:    3    (lists patient IDs + failure reasons)
      warnings:           12   (lists details)
    """


# physio_data/explore.py -- step 0 toolbox
def scan_directory(path) -> dict:
    """Walk raw data dir. Report: tree structure, file types, sizes, sample counts."""

def inspect_waveform(path, format='auto') -> dict:
    """Read one waveform file (wfdb/edf/npz/npy). Report channels, rates, duration, NaN%."""

def inspect_ehr(path) -> dict:
    """Read one EHR file (csv/parquet). Report columns, dtypes, patient IDs, time range."""

def find_patient_overlap(wav_ids, ehr_ids) -> dict:
    """Compare patient ID sets. Report matched/unmatched/overlap count."""

def build_demo_patient(profile, patient_id, wav_path, ehr_path) -> DemoPatient:
    """End-to-end: read raw -> resample -> segment -> align EHR -> one canonical patient."""

def plot_demo(demo, time_range_hours=None):
    """Waveform + EHR event markers on shared time axis."""

def save_demo(demo, out_dir):
    """Save demo patient in canonical format + raw + alignment plot."""


# physio_data/resample.py
def to_rate(signal_1d, src_fs, target_fs) -> np.ndarray:
    """Resample via scipy.signal.resample_poly. Returns float64 (segmenter converts dtype)."""


# physio_data/segment.py
def window(signal_1d, rate_hz, seg_sec=30) -> np.ndarray:
    """Split 1D signal into [N_seg, samples_per_seg] float16 C-contiguous array."""

def compute_segment_times(start_ms, n_seg, seg_sec=30) -> np.ndarray:
    """Generate [N_seg] int64 timestamp array."""


# physio_data/ehr.py
def build_events(measurements_df, segment_times_ms, var_registry) -> np.ndarray:
    """DataFrame of (time, variable_name, value) -> structured ehr_events array.
    Handles: var_name -> var_id lookup, searchsorted alignment, sorting."""

def query_events(events, seg_start, seg_end) -> np.ndarray:
    """O(log N) range query on sorted events via searchsorted on seg_idx."""

def events_to_dense(events, n_seg, target_var_ids, norm_stats, interp='linear'):
    """Sparse events -> dense (mask, gt, trend) arrays. Runtime reconstruction."""


# physio_data/registry.py
def load(path) -> dict:            # Load var_registry.json
def lookup_id(name) -> int:        # Variable name -> ID
def lookup_name(var_id) -> str:    # ID -> variable name
def extend(new_vars) -> None:      # Add new variables, assign next IDs


# physio_data/io.py
def save_patient(out_dir, channels, time_ms, ehr_events=None, meta_extra=None):
    """Save one patient in canonical format. Handles: dtype, contiguity, meta.json."""

def load_patient(dir_path, mmap=True) -> dict:
    """Load patient dir. Returns {channel_name: mmap_array, time_ms: ..., ehr_events: ...}."""

def mmap_open(path) -> np.memmap:
    """np.load with mmap_mode='r', with assertion on C-contiguity."""


# physio_data/manifest.py
def build_manifest(processed_dir) -> list:
    """Scan all patient dirs, build manifest.json entries."""

def generate_splits(manifest, test_fraction, seed, stratify_by=None) -> dict:
    """Patient-level stratified split."""


# physio_data/normalize.py
def compute_stats(events_across_patients, var_id) -> dict:
    """Robust quantile-based stats: min, max, p01, p99, mean, std."""

def apply_norm(values, stats) -> np.ndarray:
    """(value - min) / (max - min), clipped to [0, 1]."""
```

## Adding a new dataset (full workflow)

### Step 0: Explore (use `physio_data.explore`)

```bash
# 0a: Research -- read docs, internal wiki, data dictionary
#     Write findings to datasets/mcmed/explore/README.md

# 0b: Structural exploration
python -c "
from physio_data import explore
explore.scan_directory('/home/mxwan/workspace/MC_MED_ECG_PLETH_40hz_120hz_500hz_lab_vital/')
explore.inspect_waveform('/home/mxwan/workspace/MC_MED_ECG_PLETH_40hz_120hz_500hz_lab_vital/98972716_P40_E120_E500_577.npz')
explore.inspect_ehr('/home/mxwan/workspace/mc_med_csv/labs.csv')
"
# Fill in datasets/mcmed/explore/dataset_profile.json

# 0c: Demo one patient
python -c "
from physio_data import explore
demo = explore.build_demo_patient(
    profile='datasets/mcmed/explore/dataset_profile.json',
    patient_id='98972716',
    wav_path='/home/mxwan/workspace/MC_MED_ECG_PLETH_40hz_120hz_500hz_lab_vital/98972716_P40_E120_E500_577.npz',
    ehr_path='/home/mxwan/workspace/mc_med_csv/labs.csv',
)
explore.plot_demo(demo, time_range_hours=(2, 4))
explore.save_demo(demo, 'datasets/mcmed/explore/demo_patient_98972716/')
"
# Visually verify alignment. If wrong, fix dataset_profile.json and re-run.
```

### Steps 1-4: Extract (write thin scripts using `physio_data.*`)

```python
# scripts/mcmed/convert_npz.py  -- MC_MED already has NPZ, just reformat
from physio_data import segment, ehr, io, registry
import numpy as np

var_reg = registry.load("indices/var_registry.json")

for npz_path in npz_files:
    data = np.load(npz_path)
    encounter_nbr = parse_encounter_from_filename(npz_path)

    # MC_MED NPZ already has segmented waveforms at target rates
    channels = {
        "PLETH40": np.ascontiguousarray(data["PLETH40"], dtype=np.float16),
        "II120":   np.ascontiguousarray(data["II120"],   dtype=np.float16),
        "II500":   np.ascontiguousarray(data["II500"],   dtype=np.float16),
    }
    time_ms = data["time"].astype(np.int64)

    labs_df = load_patient_labs(encounter_nbr)  # from mc_med_csv/labs.csv
    events = ehr.build_events(labs_df, time_ms, var_reg)

    io.save_patient(f"datasets/mcmed/processed/{encounter_nbr}",
                    channels=channels, time_ms=time_ms,
                    ehr_events=events)
```

### Step 5: Validate + manifest

```python
from physio_data import manifest, validate, registry
var_reg = registry.load("indices/var_registry.json")

# Validate every patient dir
for d in glob("datasets/mcmed/processed/*/"):
    validate.validate_patient_dir(d, var_reg)

# Build manifest + splits
m = manifest.build_manifest("datasets/mcmed/processed")
splits = manifest.generate_splits(m, test_fraction=0.2, seed=42)
```

No changes needed to adapters. Physio_HNET/UNIPHY_Plus configs just point `root` to
`datasets/mcmed/processed/` and everything works.

## Channel naming convention

Consistent naming across datasets:

| Channel | Meaning | Typical rate | samples_per_seg (30s) |
|---------|---------|-------------|----------------------|
| PLETH40 | PPG/SpO2 pleth @ 40Hz | 40 Hz | 1200 |
| II120 | ECG lead II @ 120Hz | 120 Hz | 3600 |
| II500 | ECG lead II @ 500Hz | 500 Hz | 15000 |
| ABP125 | Arterial BP @ 125Hz | 125 Hz | 3750 |

Format: `{SIGNAL_TYPE}{RATE}`. The rate suffix disambiguates when the same signal type
exists at multiple rates. `meta.json` always records the actual sample rate.

---

# Part 3: Adapting to Current Projects

## Physio_HNET Adapter: `hnet_npy_dataset.py`

### What Physio_HNET expects

From `NpyMmapDataset` in `Physio_HNET/data/formats/npy_mmap.py`:
- `__getitem__` returns `(waveform_tensor, info_dict)`
- Waveform: mmap .npy `[N_rows, L_row]`, read `concat_rows` consecutive rows, flatten
- EHR: optional, currently via separate E1 dataset classes
- TBTT: `get_tbtt_streams()` returns stream definitions for `TBTTBatchSampler`
- Registration: `@register("dataset_type_name")` in dataset registry

### Adapter design

```python
from data.registry import register, WaveformDataset

@register("npy_mimic3")   # or "npy_mcmed", "npy_mover" -- one per dataset
class NpyMimic3Dataset(WaveformDataset):
    """
    Reads canonical per-patient .npy directories.
    Same interface as NpyMmapDataset: mmap-backed, zero-copy, TBTT-compatible.
    """
    def __init__(self, cfg):
        root = cfg["root"]              # datasets/mimic3/processed
        indices_dir = os.path.join(root, "..", "indices")

        # 1. Load manifest + splits
        manifest = json.load(open(os.path.join(indices_dir, "manifest.json")))
        splits = json.load(open(os.path.join(indices_dir, "pretrain_splits.json")))
        patients = [e for e in manifest if e["dir"] in splits[cfg["split"]]]

        # 2. mmap all files at init (cached for lifetime of dataset)
        self._mmaps = {}
        for entry in patients:
            d = os.path.join(root, entry["dir"])
            self._mmaps[entry["dir"]] = {
                wave: np.load(os.path.join(d, f"{wave}.npy"), mmap_mode='r')
                for wave in cfg["waves"] if wave in entry["channels"]
            }
            self._mmaps[entry["dir"]]["ehr_events"] = \
                np.load(os.path.join(d, "ehr_events.npy"), mmap_mode='r')

        # 3. Build sample index: [(patient_dir, start_row), ...]
        #    Same logic as NpyMmapDataset._build_sample_index()
        self._index = []
        for entry in patients:
            n = entry["n_segments"]
            for start in range(0, n - cfg["concat_rows"] + 1, cfg["start_row_stride"]):
                self._index.append((entry["dir"], start))

    def __getitem__(self, idx):
        patient_dir, start_row = self._index[idx]
        end_row = start_row + self.concat_rows
        wave_mmap = self._mmaps[patient_dir][self.wave]

        # WAVEFORM: zero-copy until final .copy()
        raw = wave_mmap[start_row:end_row, :self.row_limit]  # view
        flat = raw.reshape(-1)                                 # view
        arr = self._sanitize_1d(flat)
        arr = self._clip_1d(arr)

        info = {
            "sample_key": f"{patient_dir}_{start_row}",
            "start_row": start_row,
            "concat_rows": self.concat_rows,
            "stream_id": ...,    # from TBTT stream assignment
            "stream_pos": ...,
            "is_stream_start": start_row == 0,
        }

        # EHR (optional, for E1-style tasks):
        if self.load_ehr:
            events = self._mmaps[patient_dir]["ehr_events"]
            lo = np.searchsorted(events['seg_idx'], start_row, side='left')
            hi = np.searchsorted(events['seg_idx'], end_row, side='left')
            info["ehr_events"] = events[lo:hi]  # sparse events in this window

        return torch.from_numpy(arr.copy()), info

    def get_tbtt_streams(self):
        # One stream per patient, same structure as NpyMmapDataset
        streams = []
        for entry in self._patients:
            n = entry["n_segments"]
            n_items = (n - self.concat_rows) // self.start_row_stride + 1
            streams.append({
                "stream_id": len(streams),
                "length": n_items,
                "source_name": entry["dir"],
                "start_row_first": 0,
                "start_row_last": (n_items - 1) * self.start_row_stride,
                "row_stride": self.start_row_stride,
                "concat_rows": self.concat_rows,
            })
        return streams
```

### Integration with Physio_HNET

1. Copy `hnet_npy_dataset.py` to `Physio_HNET/data/formats/`
2. Add import in `Physio_HNET/data/formats/__init__.py`
3. Use in config:

```json
{
  "dataset": {
    "type": "npy_mimic3",
    "root": "/home/mxwan/workspace/Physio_Data/datasets/mimic3/processed",
    "waves": ["PLETH40"],
    "concat_rows": 120,
    "row_limit": 1200,
    "clip_min": 1000.0,
    "clip_max": 3076.0,
    "start_row_stride": 120,
    "split": "train"
  }
}
```

No changes to model, trainer, collate, or sampler code.

---

## UNIPHY_Plus_ICML Adapter: `uniphy_npy_adapter.py`

### What UNIPHY_Plus expects

From `UNIPHY_dataset_v2.py`:
- Loads data via `np.load(file_path, mmap_mode="r", allow_pickle=True)`
- Accesses keys: `"time"` or `"emb_time"`, waveform channel, `"ehr_mask"`, `"ehr_gt"`, `"ehr_trend"`
- `ehr_mask`: `[N_seg, N_target_vars]` uint8 (0=none, 1=interpolated, 2=measured)
- `ehr_gt`: `[N_seg, N_target_vars]` float32 (normalized values)
- `ehr_trend`: `[N_seg, N_target_vars]` float32 (interpolated trend)
- Split file: `[filename, pid, start_idx, end_idx, evt_idx, label]`

### Adapter design

```python
class NpyDirAsNpz:
    """
    Makes a per-patient .npy directory look like np.load(npz, mmap_mode='r').
    Waveform/time: true mmap (zero-copy).
    EHR: reconstructs dense ehr_mask/ehr_gt/ehr_trend from sparse events on demand.
    """
    def __init__(self, dir_path, var_registry, normalization, target_var_ids):
        self._dir = dir_path
        self._var_reg = var_registry
        self._norm = normalization
        self._target_var_ids = target_var_ids  # e.g. [0,1,2,3,4,5,6,7,8]
        self._cache = {}

    def __getitem__(self, key):
        if key in ('ehr_mask', 'ehr_gt', 'ehr_trend'):
            if 'ehr_mask' not in self._cache:
                self._build_dense_ehr()
            return self._cache[key]
        if key == 'time':
            key = 'time_ms'
        if key not in self._cache:
            self._cache[key] = np.load(
                os.path.join(self._dir, f"{key}.npy"), mmap_mode='r')
        return self._cache[key]

    def _build_dense_ehr(self):
        """One-time reconstruction: sparse events -> dense aligned arrays."""
        events = self['ehr_events']
        meta = json.load(open(os.path.join(self._dir, 'meta.json')))
        n_seg = meta['n_segments']
        n_vars = len(self._target_var_ids)

        mask  = np.zeros((n_seg, n_vars), dtype=np.uint8)
        gt    = np.zeros((n_seg, n_vars), dtype=np.float32)
        trend = np.zeros((n_seg, n_vars), dtype=np.float32)

        for col, var_id in enumerate(self._target_var_ids):
            var_ev = events[events['var_id'] == var_id]
            if len(var_ev) == 0:
                continue
            idxs = var_ev['seg_idx'].astype(np.intp)
            vals = var_ev['value'].astype(np.float32)

            # Normalize
            stats = self._norm[str(var_id)]
            normed = np.clip((vals - stats['min']) / (stats['max'] - stats['min'] + 1e-8), 0, 1)

            # Measured points
            mask[idxs, col] = 2
            gt[idxs, col] = normed

            # Linear interpolation between consecutive measurements
            for i in range(len(idxs) - 1):
                s, e = idxs[i], idxs[i + 1]
                if e - s > 1:
                    interp = np.linspace(normed[i], normed[i + 1], e - s + 1)
                    trend[s:e + 1, col] = interp
                    mask[s + 1:e, col] = 1

        self._cache['ehr_mask'] = mask
        self._cache['ehr_gt'] = gt
        self._cache['ehr_trend'] = trend

    def __contains__(self, key):
        if key in ('ehr_mask', 'ehr_gt', 'ehr_trend', 'time'):
            return True
        return os.path.exists(os.path.join(self._dir, f"{key}.npy"))

    @property
    def files(self):
        base = [p.stem for p in Path(self._dir).glob("*.npy")]
        return base + ['ehr_mask', 'ehr_gt', 'ehr_trend']
```

### Integration with UNIPHY_Plus_ICML

**Change in `UNIPHY_dataset_v2.py`** (~5 lines):

```python
# In read_sample_emb() or wherever np.load is called:
if os.path.isdir(file_path):
    data = NpyDirAsNpz(file_path, self.var_registry, self.normalization,
                        self.target_var_ids)
else:
    data = np.load(file_path, mmap_mode="r", allow_pickle=True)
```

**Change in split file generation** (stage 4):

`downstream_splits.json` uses directory path instead of .npz file path:

```json
{
  "train_control_list": [
    ["datasets/mimic3/processed/10032_174162", 10032, 0, 1440, -1, 0],
    ...
  ],
  "test_control_list": [...]
}
```

**Benefits over current NPZ approach:**
- Waveform access becomes true mmap (was fake mmap via NPZ decompression)
- Dense EHR reconstruction from sparse events costs microseconds per patient (cached)
- Same adapter handles MIMIC-III, MC_MED, any dataset in canonical format

No changes to model, runner, sampler, collate, or evaluation code.

---

## MIMIC-III Extraction Pipeline

The dataset-specific scripts for MIMIC-III, adapted from existing
`MIMIC-III-preparation-for-UNIPHY_Plus/`:

### Stage 0: EHR Extraction

**Adapted from**: `stp0_0_vital_filter.py`, `stp0_1_lab_filter.py`, `stp1_1_demographics.py`

- Input: Raw MIMIC-III CSVs
- Output: `datasets/mimic3/ehr/labs.parquet`, `vitals.parquet`, `demographics.parquet`
- Uses Polars for fast filtering
- Computes per-variable normalization stats -> `normalization.json`

**Verify** (`verify.verify_ehr_tables`):
```
[PASS] 46,520 patients in labs table
[PASS] No duplicate (patient, time, variable) rows
[PASS] Potassium range [1.2, 9.8] within physiological bounds [1, 10]
[PASS] Sodium range [108, 178] within bounds [100, 180]
[WARN] 23 Glucose values > 800 mg/dL -- extreme but possible (DKA patients)
[PASS] All timestamps in [2001-06-25, 2012-10-03] -- consistent with MIMIC-III
[PASS] No null patient IDs
```

### Stage 1: Waveform-EHR Alignment

**Adapted from**: `stp1_0_mapping_wav_lab_vital.py`

- Input: WFDB headers + EHR parquets
- Output: `datasets/mimic3/indices/alignment_index.parquet`

**Verify** (`verify.verify_alignment_index`):
```
[PASS] 8,342 alignment entries
[PASS] All windows have start_time < end_time
[PASS] Median window duration: 14.2 hours (range: 0.5 - 168 hours)
[PASS] 2,674 unique patients with both waveform and >=1 lab/vital overlap
[WARN] 312 entries have 0 lab overlap (vital-only) -- expected for some patients
[FAIL] Entry subject=10445 has window spanning 8,760 hours -- likely timestamp error
       -> Must fix before proceeding
```

### Stage 2: Waveform Extraction

**Adapted from**: `stp2_wav_reader_extraction_mp.py`

- Input: Raw WFDB files + alignment index
- Output: Per-patient .npy waveforms + time_ms.npy + meta.json

**Verify** (`verify.verify_waveforms`, run per patient):
```
[PASS] PLETH40.npy: float16, C-contiguous, shape [1440, 1200]
[PASS] II120.npy:   float16, C-contiguous, shape [1440, 3600]
[PASS] time_ms.npy: int64, monotonically increasing, len=1440
[PASS] N_seg=1440 consistent across all channels
[PASS] NaN ratio: PLETH40=0.3%, II120=1.2% (both < 20% threshold)
[PASS] Non-flat: 98.5% of PLETH40 segments have std > 1e-6
[PASS] Zero-copy: arr[0:120,:].reshape(-1).base is arr -> True
[WARN] II120 segments 812-819: std < 1e-6 (flat-line, likely lead disconnect)
```

After processing all patients, print summary:
```
Stage 2 complete: 2,674 patients
  2,668 passed all checks
  6 passed with warnings (flat-line segments, high NaN regions)
  0 failed
```

### Stage 3: Build EHR Events

**Adapted from**: `stp3_1_npz_prepare.py`

- Input: Patient directories + EHR parquets + var_registry.json
- Output: `ehr_events.npy` added to each patient directory

**Verify** (`verify.verify_ehr_events`, run per patient):
```
[PASS] ehr_events.npy dtype matches EHR_EVENT_DTYPE
[PASS] 187 events, sorted by time_ms
[PASS] All seg_idx in [0, 1440) -- within waveform range
[PASS] All var_id in var_registry: {0,1,2,3,4,5,6,7,8}
[PASS] No duplicate (time_ms, var_id) pairs
[PASS] Value ranges: Potassium [3.1, 5.8], Calcium [7.2, 9.8] -- plausible
[PASS] Events span 11.8 of 12.0 recording hours
[WARN] NBPm only has 2 measurements in 12 hours -- sparse but valid
```

Cross-check with waveform:
```
[PASS] First event seg_idx=3 -> time_ms matches waveform time_ms[3] within ±30s
[PASS] Last event seg_idx=1437 -> within recording bounds
```

### Stage 4: Quality & Splits

**Adapted from**: `stp3_0_quality_analysis_train_test_split.ipynb`

- Input: All patient directories
- Output: `manifest.json`, `pretrain_splits.json`, `downstream_splits.json`

**Verify** (`verify.verify_manifest_and_splits`):
```
[PASS] manifest.json: 2,674 entries
[PASS] Every processed/ dir has manifest entry and vice versa
[PASS] n_segments in manifest matches actual .npy shapes for all patients
[PASS] Train: 2,139 patients, Test: 535 patients (80.0% / 20.0%)
[PASS] Zero patient overlap between train and test
[PASS] Total: 28,450 recording hours, 412,380 EHR events
```

### Pipeline verification summary

Every stage gate-checks before proceeding:

```
Step 0 (explore)  -> visual check of demo patient alignment plot
                     -> human approval required before proceeding

Stage 0 (EHR)     -> verify.verify_ehr_tables()
                     -> ABORT if: duplicates, null IDs, timestamps out of range
                     -> WARN if: extreme values (log for review)

Stage 1 (align)   -> verify.verify_alignment_index()
                     -> ABORT if: impossible windows, missing patient linkage
                     -> WARN if: zero-overlap entries

Stage 2 (waveform) -> verify.verify_waveforms() per patient
                      -> SKIP patient if: all-NaN, shape mismatch, not C-contiguous
                      -> WARN if: high NaN ratio, flat-line segments

Stage 3 (EHR)      -> verify.verify_ehr_events() per patient
                      -> SKIP patient if: seg_idx out of bounds, unknown var_id
                      -> WARN if: 0 events, very sparse

Stage 4 (splits)   -> verify.verify_manifest_and_splits()
                      -> ABORT if: patient overlap in splits, missing dirs
                      -> WARN if: split ratio off by >5%

End-to-end         -> verify.verify_full_pipeline()
                      -> prints summary table of all patients with pass/fail/warn
```

**Inline assertions in `physio_data/io.py`** (catch problems at write time):

```python
def save_patient(out_dir, channels, time_ms, ehr_events=None, ...):
    for name, arr in channels.items():
        assert arr.dtype == np.float16, f"{name}: expected float16, got {arr.dtype}"
        assert arr.ndim == 2, f"{name}: expected 2D, got {arr.ndim}D"
        assert arr.flags['C_CONTIGUOUS'], f"{name}: not C-contiguous"
        assert not np.all(np.isnan(arr)), f"{name}: all NaN"

    assert time_ms.dtype == np.int64
    assert np.all(np.diff(time_ms) > 0), "time_ms not monotonically increasing"

    n_seg = len(time_ms)
    for name, arr in channels.items():
        assert arr.shape[0] == n_seg, f"{name}: {arr.shape[0]} rows != {n_seg} time_ms entries"

    if ehr_events is not None:
        assert ehr_events.dtype == schema.EHR_EVENT_DTYPE
        assert np.all(ehr_events['seg_idx'] >= 0)
        assert np.all(ehr_events['seg_idx'] < n_seg)
        assert np.all(np.diff(ehr_events['time_ms']) >= 0), "ehr_events not sorted"
```

These fire immediately when something is wrong, not hours later during training.

---

## MC_MED Conversion Pipeline

Convert existing NPZ files in `MC_MED_ECG_PLETH_40hz_120hz_500hz_lab_vital/`:

### convert_npz.py

- Read each existing NPZ: extract PLETH40, II120, II500, time arrays
- Save as per-patient .npy directory in `datasets/mcmed/processed/`
- **Verify per patient**: `verify.verify_waveforms(patient_dir)`
- **Verify round-trip**: `assert np.array_equal(original_npz[key], loaded_npy[key])`

### build_ehr_events.py

- Read `mc_med_csv/labs.csv`, `numerics.csv` -> build ehr_events.npy per patient
- Map MC_MED column names to global var_registry IDs
- **Verify per patient**: `verify.verify_ehr_events(patient_dir, var_registry)`

---

## Implementation Order

1. `physio_data/schema.py` + `physio_data/verify.py` -- contract and verification first
2. `physio_data/explore.py` -- step 0 toolbox
3. `physio_data/resample.py`, `segment.py`, `ehr.py`, `io.py` -- core utilities
4. `physio_data/registry.py`, `manifest.py`, `normalize.py` -- metadata utilities
5. `configs/mimic3.yaml` + `indices/var_registry.json`
6. `scripts/mimic3/stage0` through `stage4` -- MIMIC-III extraction (verify after each)
7. `adapters/hnet_npy_dataset.py` -- Physio_HNET adapter
8. `adapters/uniphy_npy_adapter.py` -- UNIPHY_Plus adapter
9. `scripts/mcmed/` -- MC_MED conversion (optional, when needed)
