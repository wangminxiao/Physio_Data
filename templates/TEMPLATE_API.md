# {DATASET_NAME} Dataset API

## Data Sources

### Waveform

| Field | Value |
|-------|-------|
| Format | {WFDB / EDF / NPZ / CSV / ...} |
| Location | `{/path/to/raw/waveforms/}` |
| Organization | {describe directory structure} |
| Patient ID field | {field name, e.g. SUBJECT_ID, encounter_nbr, CSN} |
| Time reference | {how timestamps are stored: UTC, local, unix epoch, ...} |

### EHR - Clinical Tables

| Table | Location | Format | Key Columns |
|-------|----------|--------|-------------|
| Labs | `{path}` | {format} | {patient_id, time, variable, value columns} |
| Vitals | `{path}` | {format} | {columns} |
| Demographics | `{path}` | {format} | {columns} |
| {other tables} | `{path}` | {format} | {columns} |

### Patient ID Linkage

```
Waveform: {ID field}
EHR:      {ID field}
Linkage:  {direct match / join table / ...}
Output patient_id format: {how the canonical patient_id is constructed}
```

### Time Format

| Source | Format | Notes |
|--------|--------|-------|
| Waveform | {format} | {timezone, offset, de-identification notes} |
| EHR | {format} | {any offset vs waveform?} |

---

## Waveform Channels to Extract

| Source Channel | Source Rate (Hz) | Target Channel Name | Target Rate (Hz) | samples_per_seg (30s) | Notes |
|---------------|-----------------|--------------------|-----------------|-----------------------|-------|
| {name} | {rate} | {SIGNAL}{RATE} | {target} | {rate * 30} | {notes} |

**Channel selection logic**: {how to identify channels in the source format}

---

## EHR Variables to Extract

All categories share the single structured dtype
`(time_ms: int64, seg_idx: int32, var_id: uint16, value: float32)` and are stored
together in `ehr_baseline.npy` / `ehr_recent.npy` / `ehr_events.npy` / `ehr_future.npy`.
Category is encoded in the `var_id` range.

### Labs (var_id 0-99)

| var_id | Variable | Source Column / ID | Unit | Physiological Range | Notes |
|--------|----------|--------------------|------|--------------------:|-------|
| {id} | {name} | {source identifier} | {unit} | {min - max} | {notes} |

### Vitals (var_id 100-199)

| var_id | Variable | Source Column / ID | Unit | Physiological Range | Notes |
|--------|----------|--------------------|------|--------------------:|-------|
| {id} | {name} | {source identifier} | {unit} | {min - max} | {notes} |

### Actions / Interventions (var_id 200-299)

Medications, fluids, ventilator settings, etc. `value` stores the rate/dose at
each charting point; `value=0.0` means stopped. Note `has_future_actions` in
`meta.json` — forecasting tasks must assert it before using `ehr_future.npy`.

| var_id | Variable | Source Column / ID | Unit | Physiological Range | Notes |
|--------|----------|--------------------|------|--------------------:|-------|
| {id} | {name} | {source identifier} | {unit} | {min - max} | {notes} |

### Scores (var_id 300-399)

Derived values (SOFA, sepsis onset, ...). Usually populated by a post-stage, not the main pipeline.

| var_id | Variable | Source Column / ID | Unit | Physiological Range | Notes |
|--------|----------|--------------------|------|--------------------:|-------|
| {id} | {name} | {source identifier} | {unit} | {min - max} | {notes} |

### New variables (not in global var_registry yet)

| Proposed var_id | Variable | Source Column / ID | Unit | Physiological Range | Notes |
|----------------|----------|--------------------|------|--------------------:|-------|
| {next id} | {name} | {source identifier} | {unit} | {min - max} | {must add to var_registry.json} |

### Filtering Rules

```
- {describe null handling}
- {describe range filtering}
- {describe deduplication}
- {any dataset-specific cleaning}
```

---

## Demographics to Extract

| Field | Source | Column | Encoding |
|-------|--------|--------|----------|
| {field} | {table} | {column} | {how to encode} |

---

## Processing Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Segment duration | 30 seconds | Standard |
| Min segments per patient | {N} | {why} |
| Max NaN ratio per channel | 0.20 | {adjust if needed} |
| Normalization method | Robust quantile | {or specify alternative} |
| Train/test split | 80/20, patient-level | {stratification variables} |
| Split seed | 42 | Reproducibility |

---

## Known Issues

```
- {list any known data quality issues discovered during exploration}
- {timezone mismatches, missing channels, encoding errors, etc.}
```

---

## Output Specification

```
datasets/{dataset_name}/
├── processed/
│   └── {entity_id}/
│       ├── {CHANNEL}.npy            [N_seg, samples_per_seg]  float16
│       ├── time_ms.npy              [N_seg]                    int64
│       ├── ehr_baseline.npy         [N_baseline]  structured   far history
│       ├── ehr_recent.npy           [N_recent]    structured   close history
│       ├── ehr_events.npy           [N_events]    structured   waveform-aligned
│       ├── ehr_future.npy           [N_future]    structured   post-waveform
│       └── meta.json
├── demographics.csv                 one row per entity_id
├── manifest.json
├── pretrain_splits.json
├── downstream_splits.json
└── tasks/                           post-stage outputs (cohorts, labels, extra_events)
    └── {task}/
        ├── cohort.json
        ├── splits.json
        └── extra_events/
            ├── {pid}.npy            in-waveform extra events
            ├── {pid}.baseline.npy
            ├── {pid}.recent.npy
            └── {pid}.future.npy     forecasting labels — LEAKAGE if used as input
```

## EHR Trajectory Files

All four EHR files share `EHR_EVENT_DTYPE`, each sorted by `time_ms` ascending.
`seg_idx` is a real segment index only in `ehr_events.npy`; the other three use
sentinel values so accidental `signal[seg_idx]` fails loudly.

| File | Time window | `seg_idx` value |
|---|---|---|
| `ehr_baseline.npy` | `[max(episode_start, wave_start − baseline_cap), wave_start − context_window)` | `INT32_MIN` (-2147483648) |
| `ehr_recent.npy`   | `[wave_start − context_window, wave_start)` | `INT32_MIN + 1` |
| `ehr_events.npy`   | `[wave_start, wave_end]` | searchsorted index in `[0, N_seg)` |
| `ehr_future.npy`   | `(wave_end, min(episode_end, wave_end + future_cap)]` | `INT32_MIN + 2` |

Defaults (see `physio_data/ehr_trajectory.py`, overridable per dataset):
- `context_window_ms` = 24 h
- `baseline_cap_ms`   = 30 d
- `future_cap_ms`     = 7 d

`meta.json` must include: `n_events`, `n_baseline`, `n_recent`, `n_future`,
`n_baseline_vars`, `n_recent_vars`, `n_future_vars`,
`context_window_ms`, `baseline_cap_ms`, `future_cap_ms`,
`has_future_actions`, `has_future_sofa`, `has_future_sepsis_onset`,
`episode_start_ms`, `episode_end_ms`, `ehr_layout_version`.

## Demographics CSV

`{output_dir}/demographics.csv`, one row per `entity_id`. Categorical columns
are stored as raw strings; consumers encode to integer IDs at load time (0
reserved for unknown/pad).

| Column | Type | Source | Notes |
|---|---|---|---|
| `{entity_id}` | str | derived | Use as index |
| `gender` | str | {source} | |
| `age_years` | float | {source} | |
| {other static attributes} | | | |

## References

- {dataset paper / PhysioNet page / documentation links}
- {any related preprocessing code}
