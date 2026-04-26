"""Cross-dataset wrapper + collate.

Different cohorts (Emory, UCSF, MIMIC-III, MC_MED, MOVER, VitalDB) all live in
the same canonical layout, so a `MultiPhysioDataset` is just a `ConcatDataset`
of `PointEstimationDataset` instances — one per cohort — with a unified collate
that batches by canonical channel name. The user picks ONE set of channels +
window + alignment + targets; per-dataset roots and (optional) splits vary.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch.utils.data import ConcatDataset, Dataset

from .dataset import ChannelSpec, PointEstimationDataset
from .windowing import Alignment


@dataclass
class DatasetConfig:
    """One cohort's mounting parameters."""
    name: str                        # e.g. "emory"
    processed_dir: str | Path
    entities: list[str] | None = None
    split_file: str | Path | None = None
    split_name: str | None = None


class MultiPhysioDataset(Dataset):
    """Concatenate per-cohort datasets into one DataLoader-ready object.

    Every per-cohort dataset shares the same `target_var_ids`, `channels`,
    `window_seconds`, and `alignment`. Channel files (e.g. `PLETH40.npy`) must
    exist in every cohort's processed dir at the canonical sample rate.
    """

    def __init__(
        self,
        cohorts: Sequence[DatasetConfig],
        target_var_ids: int | Iterable[int],
        channels: Sequence[ChannelSpec],
        *,
        window_seconds: float = 30.0,
        alignment: Alignment = "end",
        physio_clip: tuple[float, float] | None = None,
        label_transform=None,
        waveform_transform=None,
        min_label_interval_ms: int | None = None,
        label_offsets_ms: Sequence[int] = (0,),
        require_channels: bool = True,
        mmap: bool = True,
    ):
        if not cohorts:
            raise ValueError("at least one DatasetConfig is required")
        self.subsets: list[PointEstimationDataset] = []
        for cfg in cohorts:
            sub = PointEstimationDataset(
                processed_dir=cfg.processed_dir,
                target_var_ids=target_var_ids,
                channels=channels,
                window_seconds=window_seconds,
                alignment=alignment,
                entities=cfg.entities,
                split_file=cfg.split_file,
                split_name=cfg.split_name,
                physio_clip=physio_clip,
                label_transform=label_transform,
                waveform_transform=waveform_transform,
                min_label_interval_ms=min_label_interval_ms,
                label_offsets_ms=label_offsets_ms,
                require_channels=require_channels,
                dataset_name=cfg.name,
                mmap=mmap,
            )
            self.subsets.append(sub)
        self._concat = ConcatDataset(self.subsets)

    def __len__(self) -> int:
        return len(self._concat)

    def __getitem__(self, idx: int) -> dict:
        return self._concat[idx]

    def stats(self) -> dict:
        return {sub.dataset_name: sub.stats() for sub in self.subsets}


# ---------------------------------------------------------------------- collate


def physio_collate(batch: list[dict]) -> dict:
    """Default collate. Stacks waveforms per channel; returns lists for the
    string fields (entity_id, dataset, alignment).

    Expects every sample to share the same channel keys and per-channel length —
    which holds when window + fs are fixed across the dataset.
    """
    if not batch:
        return {}
    channels = list(batch[0]["waveforms"].keys())
    waves: dict[str, torch.Tensor] = {}
    for c in channels:
        waves[c] = torch.stack([b["waveforms"][c] for b in batch], dim=0)

    return {
        "waveforms": waves,
        "label": torch.stack([b["label"] for b in batch], dim=0),
        "var_id": torch.tensor([b["var_id"] for b in batch], dtype=torch.long),
        "t_label_ms": torch.tensor([b["t_label_ms"] for b in batch], dtype=torch.long),
        "t_anchor_ms": torch.tensor([b["t_anchor_ms"] for b in batch], dtype=torch.long),
        "offset_ms": torch.tensor([b["offset_ms"] for b in batch], dtype=torch.long),
        "window_ms": batch[0]["window_ms"],
        "alignment": batch[0]["alignment"],
        "entity_id": [b["entity_id"] for b in batch],
        "dataset": [b["dataset"] for b in batch],
    }
