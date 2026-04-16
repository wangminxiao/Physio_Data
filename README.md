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
{entity_id}/
  PLETH40.npy          # [N_seg, 1200]  float16, C-contiguous  (base channel)
  II120.npy            # [N_seg, 3600]  float16, C-contiguous  (NaN when absent)
  time_ms.npy          # [N_seg]         int64,  monotonic
  ehr_baseline.npy     # [N_baseline]    structured  far history (pre-waveform)
  ehr_recent.npy       # [N_recent]      structured  close history (within context window)
  ehr_events.npy       # [N_events]      structured  waveform-overlapping (seg_idx ∈ [0, N_seg))
  ehr_future.npy       # [N_future]      structured  post-waveform (label-leakage risk)
  meta.json
```

All four EHR files share one structured dtype `(time_ms, seg_idx, var_id, value)`.
The time-based split is done once at storage time; readers mmap only the partition
they need. See `skill/SKILL.md` → "EHR Trajectory Structure" for the full spec
(sentinel `seg_idx` values, caps, `has_future_*` flags).

Task-specific cohorts, labels, and extra events live under `tasks/{task}/` and
never modify the canonical entity directories.

## Repo structure

```
physio_data/           # Python package (schema, ehr_trajectory, utilities)
skill/                 # Claude Code skill (dataset onboarding workflow)
templates/             # API.md template for new datasets
examples/              # Filled examples (MIMIC-III, MC_MED)
indices/               # Global variable registry
workzone/              # Active dataset extraction scripts
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
