# Physio_Data

Canonical format and preprocessing toolkit for physiological time-series datasets (waveform + clinical events), optimized for deep learning training.

## What this is

A standardized way to store and load physiological data that works across datasets (MIMIC-III, MC_MED, MOVER, etc.) and consumers (pretraining, downstream eval).

**Signals** (ECG, PPG, ABP, ...) are stored as mmap-ready `.npy` arrays -- zero-copy, zero CPU overhead.  
**Clinical events** (labs, vitals, meds) are stored as sparse structured arrays -- only actual measurements, no dense padding.  
**Alignment** between signals and events is an index (`seg_idx`), not a copy.

## Quick start

```bash
pip install -e .

# Install the Claude Code skill (for AI-assisted dataset onboarding)
ln -s "$(pwd)/skill" ~/.claude/skills/physio-data
```

Then invoke `/physio-data mimic3` in Claude Code to start onboarding a dataset.

## Canonical output format

```
{patient_id}/
  PLETH40.npy          # [N_seg, 1200]  float16, C-contiguous
  II120.npy            # [N_seg, 3600]  float16, C-contiguous
  time_ms.npy          # [N_seg]         int64
  ehr_events.npy       # [N_events]      structured (time_ms, seg_idx, var_id, value)
  meta.json
```

## Repo structure

```
physio_data/           # Python package (schema, utilities)
skill/                 # Claude Code skill (dataset onboarding workflow)
templates/             # API.md template for new datasets
examples/              # Filled examples (MIMIC-III, MC_MED)
indices/               # Global variable registry
PLAN.md                # Full design document
```

## Adding a new dataset

See [PLAN.md](PLAN.md) Part 2, or invoke `/physio-data` in Claude Code for guided onboarding.

## Setup on a new machine

```bash
git clone <this-repo>
cd Physio_Data
pip install -e .
ln -s "$(pwd)/skill" ~/.claude/skills/physio-data
```
