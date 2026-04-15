"""Shared constants + helper for the 4-partition EHR trajectory layout.

Events for an entity are split into four files, all sharing EHR_EVENT_DTYPE:

    ehr_baseline.npy   far history   seg_idx = SEG_IDX_BASELINE
    ehr_recent.npy     close history seg_idx = SEG_IDX_RECENT
    ehr_events.npy     waveform-overlapping  seg_idx in [0, N_seg)
    ehr_future.npy     post-waveform  seg_idx = SEG_IDX_FUTURE

Partition boundaries are derived from the waveform's time_ms array and
optional admission/episode bounds. Caps limit how far baseline and future
reach so per-entity files stay small.
"""
from __future__ import annotations

import numpy as np

from .schema import EHR_EVENT_DTYPE, TIME_DTYPE  # noqa: F401


# --- Sentinel seg_idx values ----------------------------------------------
# INT32_MIN + offset so any code path that assumes seg_idx >= 0 (e.g.
# `signal[seg_idx]`) raises IndexError loudly instead of silently wrapping.
SEG_IDX_BASELINE: int = int(np.iinfo(np.int32).min)
SEG_IDX_RECENT:   int = SEG_IDX_BASELINE + 1
SEG_IDX_FUTURE:   int = SEG_IDX_BASELINE + 2

# --- Default partition windows (milliseconds) -----------------------------
CONTEXT_WINDOW_MS: int = 24 * 3600 * 1000           # 24 h: recent <-> baseline
BASELINE_CAP_MS:   int = 30 * 24 * 3600 * 1000      # 30 d: don't go lifelong
FUTURE_CAP_MS:     int = 7  * 24 * 3600 * 1000      # 7 d: bound forecast window

# --- File names -----------------------------------------------------------
FNAME_BASELINE = "ehr_baseline.npy"
FNAME_RECENT   = "ehr_recent.npy"
FNAME_EVENTS   = "ehr_events.npy"
FNAME_FUTURE   = "ehr_future.npy"
ALL_FNAMES = (FNAME_BASELINE, FNAME_RECENT, FNAME_EVENTS, FNAME_FUTURE)


def split_events(
    events: np.ndarray,
    time_ms: np.ndarray,
    *,
    episode_start_ms: int | None = None,
    episode_end_ms:   int | None = None,
    context_window_ms: int = CONTEXT_WINDOW_MS,
    baseline_cap_ms:   int = BASELINE_CAP_MS,
    future_cap_ms:     int = FUTURE_CAP_MS,
) -> dict[str, np.ndarray]:
    """Partition events into the four trajectory files.

    Parameters
    ----------
    events : np.ndarray of EHR_EVENT_DTYPE
        Event rows. Only time_ms and var_id/value/... are read. seg_idx is
        recomputed here.
    time_ms : np.ndarray of int64
        Segment start timestamps from the canonical waveform file.
    episode_start_ms, episode_end_ms : int or None
        Admission/episode bounds. If None, baseline_cap_ms / future_cap_ms
        are used to bound the outer partitions.
    context_window_ms : int
        Recent <-> baseline cutoff, measured backwards from wave_start.
    baseline_cap_ms, future_cap_ms : int
        Hard caps on how far back/forward to look if no episode bounds.

    Returns
    -------
    dict with keys "baseline", "recent", "events", "future", each an
    np.ndarray of EHR_EVENT_DTYPE, sorted by time_ms ascending. seg_idx is
    set to the correct sentinel for the outer partitions and to the real
    searchsorted index for the in-waveform partition.
    """
    if len(time_ms) == 0:
        raise ValueError("time_ms is empty; cannot split events without a waveform window")

    wave_start = int(time_ms[0])
    wave_end   = int(time_ms[-1])  # last segment start; the segment extends slightly beyond
    n_seg      = int(len(time_ms))

    # Compute partition boundaries
    recent_start = wave_start - int(context_window_ms)
    baseline_start = wave_start - int(baseline_cap_ms)
    if episode_start_ms is not None:
        baseline_start = max(baseline_start, int(episode_start_ms))
    future_end = wave_end + int(future_cap_ms)
    if episode_end_ms is not None:
        future_end = min(future_end, int(episode_end_ms))

    if events.dtype != EHR_EVENT_DTYPE:
        raise TypeError(f"events must be {EHR_EVENT_DTYPE}, got {events.dtype}")

    t = events["time_ms"]

    def _pack(mask: np.ndarray, seg_idx_val, sort_key: str = "time_ms") -> np.ndarray:
        sub = events[mask]
        if len(sub) == 0:
            return np.empty(0, dtype=EHR_EVENT_DTYPE)
        out = sub.copy()
        out["seg_idx"] = seg_idx_val
        out.sort(order=sort_key)
        return out

    # baseline: [baseline_start, recent_start)
    baseline = _pack(
        (t >= baseline_start) & (t < recent_start),
        SEG_IDX_BASELINE,
    )
    # recent: [recent_start, wave_start)
    recent = _pack(
        (t >= recent_start) & (t < wave_start),
        SEG_IDX_RECENT,
    )
    # future: (wave_end, future_end]
    future = _pack(
        (t > wave_end) & (t <= future_end),
        SEG_IDX_FUTURE,
    )
    # events: [wave_start, wave_end]  -- real seg_idx via searchsorted
    in_window_mask = (t >= wave_start) & (t <= wave_end)
    sub = events[in_window_mask]
    if len(sub) > 0:
        seg = np.searchsorted(time_ms, sub["time_ms"], side="right") - 1
        keep = (seg >= 0) & (seg < n_seg)
        sub = sub[keep]
        seg = seg[keep]
        in_window = sub.copy()
        in_window["seg_idx"] = seg.astype(np.int32)
        in_window.sort(order="time_ms")
    else:
        in_window = np.empty(0, dtype=EHR_EVENT_DTYPE)

    return {
        "baseline": baseline,
        "recent":   recent,
        "events":   in_window,
        "future":   future,
    }


def merge_partition(existing: np.ndarray, incoming: np.ndarray) -> np.ndarray:
    """Merge two arrays of EHR_EVENT_DTYPE, dedupe on (time_ms, var_id, value), sort."""
    if len(existing) == 0:
        out = incoming.copy() if len(incoming) else np.empty(0, dtype=EHR_EVENT_DTYPE)
    elif len(incoming) == 0:
        out = existing.copy()
    else:
        out = np.concatenate([existing, incoming]).copy()
    if len(out) == 0:
        return out
    # dedupe on (time_ms, var_id, value) -- same measurement written twice is one event
    key = np.array(
        list(zip(out["time_ms"], out["var_id"], out["value"])),
        dtype=[("t", "i8"), ("v", "u2"), ("x", "f4")],
    )
    _, idx = np.unique(key, return_index=True)
    out = out[np.sort(idx)]
    out.sort(order="time_ms")
    return out


def validate_partition(arr: np.ndarray, *, kind: str, n_seg: int) -> list[str]:
    """Validate one partition. Returns list of error strings (empty = ok)."""
    errs: list[str] = []
    if len(arr) == 0:
        return errs
    if arr.dtype != EHR_EVENT_DTYPE:
        errs.append(f"{kind}: dtype mismatch ({arr.dtype} != {EHR_EVENT_DTYPE})")
        return errs
    if not np.all(np.diff(arr["time_ms"]) >= 0):
        errs.append(f"{kind}: not sorted by time_ms")
    if kind == "events":
        if np.any(arr["seg_idx"] < 0):
            errs.append(f"{kind}: negative seg_idx found")
        if np.any(arr["seg_idx"] >= n_seg):
            errs.append(f"{kind}: seg_idx >= n_seg={n_seg}")
    else:
        expected = {
            "baseline": SEG_IDX_BASELINE,
            "recent":   SEG_IDX_RECENT,
            "future":   SEG_IDX_FUTURE,
        }[kind]
        if not np.all(arr["seg_idx"] == expected):
            bad = int(np.sum(arr["seg_idx"] != expected))
            errs.append(f"{kind}: {bad} events have seg_idx != {expected}")
    return errs
