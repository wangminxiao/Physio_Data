"""Path resolution and lazy loaders for the canonical per-entity layout."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from physio_data.ehr_trajectory import FNAME_EVENTS
from physio_data.schema import EHR_EVENT_DTYPE, SEGMENT_DUR_SEC

SEGMENT_DUR_MS: int = SEGMENT_DUR_SEC * 1000


@dataclass(frozen=True)
class EntityPaths:
    root: Path
    entity_id: str

    @property
    def time_ms(self) -> Path:
        return self.root / "time_ms.npy"

    @property
    def ehr_events(self) -> Path:
        return self.root / FNAME_EVENTS

    @property
    def meta(self) -> Path:
        return self.root / "meta.json"

    def channel(self, name: str) -> Path:
        return self.root / f"{name}.npy"


def list_entities(
    processed_dir: Path,
    *,
    split_file: Path | str | None = None,
    split_name: str | None = None,
    explicit: Iterable[str] | None = None,
) -> list[str]:
    """Return entity_ids to load.

    Resolution order:
      1. explicit list, if given
      2. split_file[split_name], if given (e.g. downstream_splits.json["train"])
      3. all subdirectories of processed_dir that look like entities
    """
    processed_dir = Path(processed_dir)
    if explicit is not None:
        return list(explicit)
    if split_file is not None:
        if split_name is None:
            raise ValueError("split_name must be provided with split_file")
        with open(split_file) as f:
            splits = json.load(f)
        if split_name not in splits:
            raise KeyError(f"{split_name!r} not in {sorted(splits)}")
        return list(splits[split_name])
    ents = []
    for p in sorted(processed_dir.iterdir()):
        if p.is_dir() and (p / "time_ms.npy").exists():
            ents.append(p.name)
    return ents


def load_meta(paths: EntityPaths) -> dict:
    with open(paths.meta) as f:
        return json.load(f)


def load_time_ms(paths: EntityPaths, *, mmap: bool = True) -> np.ndarray:
    return np.load(paths.time_ms, mmap_mode="r" if mmap else None)


def load_channel(paths: EntityPaths, name: str, *, mmap: bool = True) -> np.ndarray:
    return np.load(paths.channel(name), mmap_mode="r" if mmap else None)


def load_ehr_events(paths: EntityPaths) -> np.ndarray:
    arr = np.load(paths.ehr_events)
    if arr.dtype != EHR_EVENT_DTYPE:
        raise TypeError(
            f"{paths.ehr_events}: dtype {arr.dtype} != EHR_EVENT_DTYPE"
        )
    return arr
