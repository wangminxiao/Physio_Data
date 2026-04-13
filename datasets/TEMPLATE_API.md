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

### Labs

| var_id | Variable | Source Column / ID | Unit | Physiological Range | Notes |
|--------|----------|--------------------|------|--------------------:|-------|
| {id} | {name} | {source identifier} | {unit} | {min - max} | {notes} |

### Vitals

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
│   └── {patient_id}/
│       ├── {CHANNEL}.npy
│       ├── time_ms.npy
│       ├── ehr_events.npy
│       └── meta.json
├── ehr/
│   ├── labs.parquet
│   ├── vitals.parquet
│   └── demographics.parquet
└── indices/
    ├── manifest.json
    └── splits.json
```

## References

- {dataset paper / PhysioNet page / documentation links}
- {any related preprocessing code}
