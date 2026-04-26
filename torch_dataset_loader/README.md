# torch_dataset_loader

PyTorch dataset wrappers over the canonical `physio_data` layout. Builds
`(waveform_window, single_point_label)` samples for any lab/vital target,
across any preprocessed dataset (Emory, UCSF, MIMIC-III, MC_MED, MOVER, VitalDB,
…), and feeds them through a standard `DataLoader`.

## What it does

For a target variable (e.g. potassium = `var_id 0`, HR = `var_id 100`) the
loader:

1. scans `ehr_events.npy` for every entity in a processed dir,
2. keeps events whose `var_id` matches your target(s),
3. builds an **aligned waveform window** of the requested length around each
   label (window-end / window-middle / window-start at the label time),
4. rejects windows that fall off the recorded waveform or cross a non-contiguous
   segment boundary,
5. (optional) emits **multiple overlapping windows per label** by shifting the
   alignment anchor through a list of offsets,
6. returns one sample per (label, offset) — ready for `DataLoader` collation
   into batches keyed by canonical channel name.

## Canonical layout assumed

```
{processed_dir}/{entity_id}/
    {CHANNEL}.npy        [N_seg, samples_per_seg]  float16   e.g. PLETH40, II120, ABP125
    time_ms.npy          [N_seg]                    int64     UTC ms, segment-start, monotonic
    ehr_events.npy       structured (time_ms, seg_idx, var_id, value)  in-waveform events
    ehr_baseline.npy     structured                                    (pre-wave history)
    ehr_recent.npy       structured                                    (pre-wave 24h)
    ehr_future.npy       structured                                    (post-wave forecasting labels)
    meta.json
{processed_dir}/manifest.json
{processed_dir}/{pretrain,downstream}_splits.json
```

Segments are 30 s by default (`physio_data.schema.SEGMENT_DUR_SEC`); a
window may span any number of segments as long as they are contiguous in time.

## Install

```python
# from repo root
pip install -e .          # installs physio_data
pip install torch numpy   # runtime deps
```

## Quick start

```python
from torch.utils.data import DataLoader
from torch_dataset_loader import ChannelSpec, PointEstimationDataset, physio_collate

ds = PointEstimationDataset(
    processed_dir="/mnt/e/mcmed/mcmed",       # canonical processed dir
    target_var_ids=100,                       # 100 = HR (see indices/var_registry.json)
    channels=[
        ChannelSpec("PLETH40", 40),           # 40 Hz PPG
        ChannelSpec("II120",  120),           # 120 Hz ECG-II
    ],
    window_seconds=30.0,
    alignment="end",                          # label sits at window end
    physio_clip=(20.0, 300.0),                # drop labels outside HR range
    dataset_name="mcmed",
)
print(ds.stats())   # {'n_samples': ..., 'n_entities': ..., 'by_var_id': {100: ...}}

loader = DataLoader(ds, batch_size=64, shuffle=True,
                    num_workers=4, collate_fn=physio_collate)

for batch in loader:
    pleth = batch["waveforms"]["PLETH40"]     # [B, 1200] float32
    ecg   = batch["waveforms"]["II120"]       # [B, 3600] float32
    y     = batch["label"]                    # [B] float32
    # ... train your model
```

## Multi-dataset

```python
from torch_dataset_loader import MultiPhysioDataset
from torch_dataset_loader.multi import DatasetConfig

ds = MultiPhysioDataset(
    cohorts=[
        DatasetConfig(
            name="mcmed",
            processed_dir="/mnt/e/mcmed/mcmed",
            split_file="/mnt/e/mcmed/mcmed/downstream_splits.json",
            split_name="train",
        ),
        DatasetConfig(
            name="emory",
            processed_dir="/data/emory/processed",
            split_file="/data/emory/processed/downstream_splits.json",
            split_name="train",
        ),
    ],
    target_var_ids=0,                        # potassium
    channels=[ChannelSpec("PLETH40", 40), ChannelSpec("II120", 120)],
    window_seconds=300.0,                    # 5 min
    alignment="middle",
    physio_clip=(1.5, 9.0),
    label_offsets_ms=(-60_000, -30_000, 0, 30_000, 60_000),  # 5 windows per label
)
```

Channels must exist (with the same canonical name + sample rate) in every
cohort. The cohort name is attached to each sample as `batch["dataset"]`.

## Choosing a window + alignment

| Choice | When to use |
|---|---|
| `alignment="end"` | Label is the *outcome* of the window — model sees waveform leading up to the measurement. Standard for lab/vital estimation. |
| `alignment="middle"` | Label is a *current state*; you want context on both sides. Good for instantaneous vitals (HR, SpO2). |
| `alignment="start"` | Forecasting — predict a value from waveform that *precedes* it. Mostly for downstream forecasting tasks. |
| `window_seconds=30` | Single canonical segment; cheap to load, dense indexing. |
| `window_seconds=300` (5 min) | More waveform context; trades sample count (gap-rejected windows rise) for richer input. |
| `window_seconds=1800` (30 min) | Long-context models; expect substantial yield drop on datasets with frequent gaps (MOVER, MC_MED). |

## Overlap controls

Two independent overlap mechanisms:

**(a) Per-label dense windows.** `label_offsets_ms=(o1, o2, …)` emits one
sample per offset for *each* label. The window's alignment anchor becomes
`t_label + offset`, so the label drifts through the window. The stored label
value and `t_label_ms` are unchanged; `offset_ms` and `t_anchor_ms` are
returned per sample so you can condition on, or shuffle, the offset.

```python
# 5 overlapping 5-min windows per potassium draw, anchors stepped 30 s apart
PointEstimationDataset(..., window_seconds=300.0, alignment="middle",
    label_offsets_ms=(-60_000, -30_000, 0, 30_000, 60_000))
```

On real MC_MED with 200 entities this lifts potassium from 79 → 397 samples.

**(b) Inter-label window overlap.** Always allowed. When two different labels
fall within `window_seconds` of each other, both get their own (overlapping)
window. Throttle with `min_label_interval_ms` if you want to *suppress* it
for densely-charted vitals:

```python
# keep at most one HR label per 60 s (cuts _0n's every-2-s cadence to once-per-min)
PointEstimationDataset(..., target_var_ids=100, min_label_interval_ms=60_000)
```

## Where do I see what labels exist for each dataset?

Two places:

### Global variable registry
`indices/var_registry.json` — single source of truth for `var_id` → name,
unit, physiological range, and per-dataset source columns. Use this to pick
target ids that are mapped across cohorts (so multi-dataset training really
shares semantics).

```python
import json
reg = json.load(open("indices/var_registry.json"))
for v in reg["variables"]:
    if v["category"] == "lab":
        print(v["id"], v["name"], v.get("physio_min"), v.get("physio_max"))
```

### Per-dataset API spec
Every dataset's preprocessing contract lives at
`datasets/{name}/API.md`. The "EHR Variables to Extract" section enumerates
exactly which `var_id`s the pipeline writes into `ehr_events.npy` and from
which source columns:

| Dataset | API spec | Notes |
|---|---|---|
| Emory | `datasets/emory/API.md` | Labs (var 0–18), vitals (100–122), `_0n` numerics + chart vitals share var_ids; cuff BP comes from EHR (`SBP/DBP/MAP_CUFF`), `_0n` NBP-S/D/M skipped (hold signal). |
| UCSF | `datasets/ucsf/API.md` | Lab common-name → var_id mapping; vitals via channel suffixes (`HR`, `SPO2-%`, `NBP-S`, `AR1-S`, …). |
| MIMIC-III | `datasets/mimic3/API.md` | `mimic_itemids` field in `var_registry.json` lists the CHARTEVENTS / LABEVENTS itemids per variable. |
| MC_MED | `datasets/mcmed/API.md` | Labs from `Lab*.csv`; vitals from `Numeric*.csv` and waveform-derived. |

To inspect what's actually in your processed data (the ground truth):

```python
import numpy as np
from collections import Counter
from pathlib import Path

c = Counter()
for d in sorted(Path("/mnt/e/mcmed/mcmed").iterdir())[:500]:
    ev = np.load(d / "ehr_events.npy")
    c.update(ev["var_id"].tolist())
print("top var_ids in MC_MED first 500 entities:", c.most_common(20))
```

Cross-reference the `var_id`s you see against `indices/var_registry.json` to
get human names. Use `physio_clip=(physio_min, physio_max)` from the registry
to drop sentinel/garbage values.

## Sample dict

`__getitem__` returns:

| Key | Type | Notes |
|---|---|---|
| `waveforms` | `dict[str, Tensor]` | per-channel float32, length `window_seconds * fs` |
| `label` | scalar `Tensor` (float32) | the EHR `value` (after optional `label_transform`) |
| `var_id` | int | which target this label belongs to |
| `t_label_ms` | int | absolute UTC ms of the EHR measurement |
| `t_anchor_ms` | int | `t_label_ms + offset_ms`; the window-alignment anchor |
| `offset_ms` | int | per-sample offset (0 unless `label_offsets_ms` set) |
| `window_ms` | int | in case the trainer needs it |
| `alignment` | str | `"end"`/`"middle"`/`"start"` |
| `entity_id` | str | per-cohort entity directory name |
| `dataset` | str \| None | cohort tag (for `MultiPhysioDataset`) |

After `physio_collate(batch)` the same keys appear, with waveforms / label /
var_id / t_*_ms / offset_ms stacked into tensors and `entity_id` / `dataset`
returned as `list[str]`.

## API surface

| Symbol | Module |
|---|---|
| `ChannelSpec(name, fs)` | `torch_dataset_loader.dataset` |
| `PointEstimationDataset` | `torch_dataset_loader.dataset` |
| `MultiPhysioDataset`, `DatasetConfig` | `torch_dataset_loader.multi` |
| `physio_collate` | `torch_dataset_loader.multi` |
| `locate_window`, `slice_channel` | `torch_dataset_loader.windowing` (low-level) |
| `list_entities`, `EntityPaths` | `torch_dataset_loader.canonical` (low-level) |

## Files

```
torch_dataset_loader/
├── __init__.py          # public exports
├── canonical.py         # path resolution + lazy npy loaders (mmap)
├── windowing.py         # alignment + cross-segment slicing + gap rejection
├── dataset.py           # PointEstimationDataset
├── multi.py             # MultiPhysioDataset + physio_collate + DatasetConfig
└── README.md            # this file
```

Dev tests + real-data validators live separately under
`workzone/torch_dataset_loader_dev/`.
