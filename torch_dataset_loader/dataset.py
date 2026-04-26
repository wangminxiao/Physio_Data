"""PointEstimationDataset: single-point label + aligned waveform window.

Sample shape (per `__getitem__`):

    {
      "waveforms": {channel_name: float32 tensor [n_samples_for_that_fs]},
      "label":    float32 tensor (scalar),
      "var_id":   int,
      "t_label_ms": int,
      "window_ms": int,
      "alignment": str,
      "entity_id": str,
      "dataset":  str,    # set by MultiPhysioDataset, else None
    }
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "torch_dataset_loader requires PyTorch. Install with `pip install torch`."
    ) from e

from .canonical import (
    EntityPaths,
    SEGMENT_DUR_MS,
    list_entities,
    load_channel,
    load_ehr_events,
    load_meta,
    load_time_ms,
)
from .windowing import Alignment, WindowSpan, locate_window, slice_channel


@dataclass(frozen=True)
class ChannelSpec:
    """A waveform channel to extract.

    `name` is the canonical file stem (e.g. "PLETH40", "II120", "ABP125").
    `fs` is the sampling rate in Hz; must match `samples_per_seg / 30s` for the
    target file.
    """
    name: str
    fs: int


@dataclass
class Sample:
    """One returned sample, before tensor conversion / collation."""
    waveforms: dict[str, np.ndarray]   # channel -> float32 [n]
    label: float
    var_id: int
    t_label_ms: int
    window_ms: int
    alignment: str
    entity_id: str
    dataset: str | None = None


@dataclass
class _IndexEntry:
    entity_idx: int
    t_label_ms: int       # original label timestamp (what the model predicts)
    t_anchor_ms: int      # window-alignment anchor = t_label + offset
    offset_ms: int        # offset used for this sample (0 by default)
    var_id: int
    value: float


class PointEstimationDataset(Dataset):
    """Single-point estimation dataset over one canonical processed dir.

    Parameters
    ----------
    processed_dir : path
        Folder containing per-entity subfolders.
    target_var_ids : int | iterable[int]
        var_id(s) to extract as labels. Each event becomes one sample.
    channels : list[ChannelSpec]
        Waveform channels to slice for every sample. Every entity must have
        each requested channel file (see `require_channels`).
    window_seconds : float
        Length of the waveform window aligned to the label.
    alignment : "end" | "middle" | "start"
        How the label timestamp is positioned relative to the window.
        - "end":    window ends at the label
        - "middle": window centred on the label
        - "start":  window starts at the label (forecasting style)
    entities : list[str] | None
        Explicit entity_id list. If None, falls back to split_file/split_name,
        else uses every entity dir under processed_dir.
    split_file, split_name : path, str
        Optional splits.json + key (e.g. downstream_splits.json["train"]).
    physio_clip : (lo, hi) | None
        If given, drop label values outside [lo, hi].
    label_transform : callable | None
        Maps raw float value -> float (e.g. log, z-score). Applied at sample time.
    waveform_transform : callable | None
        Maps {channel: ndarray} -> {channel: ndarray} (e.g. normalisation,
        NaN imputation). Applied at sample time.
    min_label_interval_ms : int | None
        If set, keep at most one label per `(entity, var_id)` per interval —
        useful for dense vitals (e.g. _0n HR every 2 s).
    label_offsets_ms : sequence[int]
        Per-label window offsets, in ms. Each label expands into one sample
        per offset. The window's *anchor time* (the point that `alignment`
        positions inside the window) is `t_label + offset`. The stored label
        value and `t_label_ms` are unchanged. Default `(0,)` = one window per
        label (no oversampling). Use e.g. `(-60_000, -30_000, 0, 30_000, 60_000)`
        to emit 5 overlapping windows per label, anchors spaced 30 s apart.
        Combined with the natural overlap between *different* labels' windows,
        this gives full control over both intra-label and inter-label overlap.
    require_channels : bool
        If True, every entity must have every requested channel; otherwise
        missing-channel entities are skipped at index time.
    dataset_name : str | None
        Tag attached to each sample (used by MultiPhysioDataset).
    mmap : bool
        Memory-map waveform files (default True; cheap to skip for tests).
    """

    def __init__(
        self,
        processed_dir: str | Path,
        target_var_ids: int | Iterable[int],
        channels: Sequence[ChannelSpec],
        *,
        window_seconds: float = 30.0,
        alignment: Alignment = "end",
        entities: Iterable[str] | None = None,
        split_file: str | Path | None = None,
        split_name: str | None = None,
        physio_clip: tuple[float, float] | None = None,
        label_transform=None,
        waveform_transform=None,
        min_label_interval_ms: int | None = None,
        label_offsets_ms: Sequence[int] = (0,),
        require_channels: bool = True,
        dataset_name: str | None = None,
        mmap: bool = True,
    ):
        self.processed_dir = Path(processed_dir)
        self.target_var_ids = (
            {int(target_var_ids)}
            if isinstance(target_var_ids, int)
            else {int(v) for v in target_var_ids}
        )
        self.channels = list(channels)
        if not self.channels:
            raise ValueError("at least one ChannelSpec is required")
        self.window_ms = int(round(window_seconds * 1000))
        if self.window_ms <= 0:
            raise ValueError("window_seconds must be positive")
        self.alignment: Alignment = alignment
        self.physio_clip = physio_clip
        self.label_transform = label_transform
        self.waveform_transform = waveform_transform
        self.min_label_interval_ms = min_label_interval_ms
        self.label_offsets_ms = tuple(int(o) for o in label_offsets_ms)
        if not self.label_offsets_ms:
            raise ValueError("label_offsets_ms must contain at least one offset")
        self.dataset_name = dataset_name
        self.mmap = mmap

        # Resolve entity list
        self.entity_ids: list[str] = list_entities(
            self.processed_dir,
            split_file=split_file,
            split_name=split_name,
            explicit=entities,
        )
        self.entity_paths: list[EntityPaths] = []
        kept: list[str] = []
        for eid in self.entity_ids:
            root = self.processed_dir / eid
            paths = EntityPaths(root=root, entity_id=eid)
            if not paths.time_ms.exists():
                continue
            if require_channels and not all(
                paths.channel(c.name).exists() for c in self.channels
            ):
                continue
            kept.append(eid)
            self.entity_paths.append(paths)
        self.entity_ids = kept

        self._index: list[_IndexEntry] = []
        self._build_index()

    # ------------------------------------------------------------------ index

    def _build_index(self) -> None:
        clip_lo, clip_hi = (self.physio_clip or (None, None))
        for ent_idx, paths in enumerate(self.entity_paths):
            time_ms = load_time_ms(paths, mmap=True)
            if len(time_ms) == 0:
                continue
            try:
                events = load_ehr_events(paths)
            except FileNotFoundError:
                continue
            if events.size == 0:
                continue

            mask = np.isin(events["var_id"], list(self.target_var_ids))
            if not np.any(mask):
                continue
            sub = events[mask]

            # Filter values
            vals = sub["value"].astype(np.float64)
            keep = np.isfinite(vals)
            if clip_lo is not None:
                keep &= vals >= clip_lo
            if clip_hi is not None:
                keep &= vals <= clip_hi
            sub = sub[keep]
            if sub.size == 0:
                continue

            # Verify each candidate has a valid aligned window before indexing
            for var_id in np.unique(sub["var_id"]):
                rows = sub[sub["var_id"] == var_id]
                last_kept_t: int | None = None
                for r in rows:
                    t = int(r["time_ms"])
                    if (
                        self.min_label_interval_ms is not None
                        and last_kept_t is not None
                        and t - last_kept_t < self.min_label_interval_ms
                    ):
                        continue
                    any_offset_valid = False
                    for off in self.label_offsets_ms:
                        anchor = t + off
                        span = locate_window(
                            time_ms, anchor, self.window_ms, self.alignment
                        )
                        if span is None:
                            continue
                        any_offset_valid = True
                        self._index.append(
                            _IndexEntry(
                                entity_idx=ent_idx,
                                t_label_ms=t,
                                t_anchor_ms=anchor,
                                offset_ms=off,
                                var_id=int(var_id),
                                value=float(r["value"]),
                            )
                        )
                    if any_offset_valid:
                        last_kept_t = t

    # ------------------------------------------------------------------ API

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        e = self._index[idx]
        paths = self.entity_paths[e.entity_idx]
        time_ms = load_time_ms(paths, mmap=self.mmap)
        span = locate_window(time_ms, e.t_anchor_ms, self.window_ms, self.alignment)
        if span is None:
            # Should not happen — _build_index already verified — but guard anyway.
            raise RuntimeError(
                f"window vanished for {paths.entity_id} at t={e.t_label_ms} "
                f"(var={e.var_id}); index may be stale"
            )

        wf: dict[str, np.ndarray] = {}
        for cs in self.channels:
            arr = load_channel(paths, cs.name, mmap=self.mmap)
            samples_per_seg = arr.shape[1]
            expected = cs.fs * (SEGMENT_DUR_MS // 1000)
            if samples_per_seg != expected:
                raise ValueError(
                    f"{paths.entity_id}/{cs.name}: samples_per_seg={samples_per_seg} "
                    f"!= fs*{SEGMENT_DUR_MS // 1000} = {expected}"
                )
            wf[cs.name] = slice_channel(arr, time_ms, span, cs.fs)

        if self.waveform_transform is not None:
            wf = self.waveform_transform(wf)

        label = e.value
        if self.label_transform is not None:
            label = float(self.label_transform(label))

        out = {
            "waveforms": {k: torch.from_numpy(v) for k, v in wf.items()},
            "label": torch.tensor(label, dtype=torch.float32),
            "var_id": e.var_id,
            "t_label_ms": e.t_label_ms,
            "t_anchor_ms": e.t_anchor_ms,
            "offset_ms": e.offset_ms,
            "window_ms": self.window_ms,
            "alignment": self.alignment,
            "entity_id": paths.entity_id,
            "dataset": self.dataset_name,
        }
        return out

    # ------------------------------------------------------------------ debug

    def stats(self) -> dict:
        """Per-var counts. Useful for sanity-checking before training."""
        from collections import Counter
        c = Counter(e.var_id for e in self._index)
        return {
            "n_samples": len(self._index),
            "n_entities": len(self.entity_paths),
            "by_var_id": dict(c),
        }
