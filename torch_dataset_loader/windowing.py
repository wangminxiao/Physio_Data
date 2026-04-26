"""Window alignment + cross-segment slicing.

The canonical layout stores waveforms as `[N_seg, samples_per_seg]` arrays where
each segment is `SEGMENT_DUR_SEC` seconds long and `time_ms[i]` is the absolute
UTC ms of segment `i`'s first sample. Segments are stored chronologically but
gaps are allowed (e.g. 8 h Philips boundaries, ICU disconnects). A window that
spans multiple segments is only valid when those segments are *contiguous in
time* — `time_ms[i+1] - time_ms[i] == SEGMENT_DUR_MS` within tolerance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .canonical import SEGMENT_DUR_MS

Alignment = Literal["end", "middle", "start"]

# segments are stored at integer ms, but allow 1 ms tolerance on contiguity
CONTIG_TOL_MS = 1


@dataclass(frozen=True)
class WindowSpan:
    t_start_ms: int     # absolute UTC ms, window start
    t_end_ms: int       # absolute UTC ms, window end (exclusive)
    seg_lo: int         # first segment index used
    seg_hi: int         # last segment index used (inclusive)


def _align(t_label_ms: int, window_ms: int, alignment: Alignment) -> tuple[int, int]:
    if alignment == "end":
        return t_label_ms - window_ms, t_label_ms
    if alignment == "middle":
        half = window_ms // 2
        return t_label_ms - half, t_label_ms - half + window_ms
    if alignment == "start":
        return t_label_ms, t_label_ms + window_ms
    raise ValueError(f"unknown alignment {alignment!r}; use end|middle|start")


def locate_window(
    time_ms: np.ndarray,
    t_label_ms: int,
    window_ms: int,
    alignment: Alignment,
    *,
    segment_dur_ms: int = SEGMENT_DUR_MS,
    contig_tol_ms: int = CONTIG_TOL_MS,
) -> WindowSpan | None:
    """Resolve `(t_start, t_end, seg_lo, seg_hi)` for a label-aligned window.

    Returns None if any of:
      - window starts before the first segment
      - window ends after the last segment finishes
      - covered segments are not contiguous (a gap inside the window)
    """
    if window_ms <= 0:
        raise ValueError("window_ms must be positive")
    n_seg = int(len(time_ms))
    if n_seg == 0:
        return None

    t_start, t_end = _align(int(t_label_ms), int(window_ms), alignment)

    # seg_lo: largest i with time_ms[i] <= t_start
    seg_lo = int(np.searchsorted(time_ms, t_start, side="right")) - 1
    # seg_hi: largest i with time_ms[i] < t_end (i.e. covers up to t_end)
    seg_hi = int(np.searchsorted(time_ms, t_end, side="left")) - 1

    if seg_lo < 0 or seg_hi < 0:
        return None
    if seg_hi >= n_seg or seg_lo > seg_hi:
        return None

    # last covered segment must extend through t_end
    if int(time_ms[seg_hi]) + segment_dur_ms < t_end:
        return None

    # contiguity check across covered segments
    if seg_hi > seg_lo:
        diffs = np.diff(time_ms[seg_lo : seg_hi + 1].astype(np.int64))
        if not np.all(np.abs(diffs - segment_dur_ms) <= contig_tol_ms):
            return None

    return WindowSpan(t_start_ms=t_start, t_end_ms=t_end, seg_lo=seg_lo, seg_hi=seg_hi)


def slice_channel(
    channel_arr: np.ndarray,
    time_ms: np.ndarray,
    span: WindowSpan,
    fs: int,
) -> np.ndarray:
    """Cross-segment slice for one channel at `fs` Hz.

    `channel_arr` is `[N_seg, samples_per_seg]` (mmap is fine). Returns a
    contiguous float32 ndarray of length `(t_end - t_start) * fs / 1000`.
    """
    samples_per_seg = channel_arr.shape[1]
    block = np.asarray(
        channel_arr[span.seg_lo : span.seg_hi + 1], dtype=np.float32
    ).reshape(-1)
    base_t = int(time_ms[span.seg_lo])
    sample_lo = int(round((span.t_start_ms - base_t) * fs / 1000.0))
    n_samples = int(round((span.t_end_ms - span.t_start_ms) * fs / 1000.0))
    sample_hi = sample_lo + n_samples
    if sample_lo < 0 or sample_hi > block.shape[0]:
        raise IndexError(
            f"slice [{sample_lo}:{sample_hi}] out of block of length {block.shape[0]} "
            f"(seg_lo={span.seg_lo}, seg_hi={span.seg_hi}, fs={fs}, samples_per_seg={samples_per_seg})"
        )
    return np.ascontiguousarray(block[sample_lo:sample_hi])
