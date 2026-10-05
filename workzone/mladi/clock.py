"""
MLADI clock conventions -- one place for every time conversion in the MLADI pipelines.

Background (verified on data 2026-10-05; workzone/mladi/explore/clock_*.py, write-up in
datasets/mladi/API.md "Time base"):

  * Each DWC HDF5 (audata 1.1) stamps a `time_origin` wall clock with a zone label (EDT / EST, or LMT
    for the year-1800 origins) and stores every time column as seconds from it.
  * Those seconds are LOCAL WALL-CLOCK seconds, for the monitor streams (waveforms, data/numerics) AND
    for /ehr: the raw Pleth clock jumps +3600 s at spring-forward (18/18 files), and at fall-back the
    repeated hour was dropped (no backward step, 21/21), so real elapsed time has a 1-h hole there.
  * The two streams use different origin walls.  Monitor (DWC): the stamped wall `W`.  EHR: the same
    origin instant rendered in the DST state at DISCHARGE (`ehr/demographic.dischDate`), i.e. `W` +- 1 h
    when the origin and the discharge lie on different sides of a DST change.  Checked against charted
    Systolic/Diastolic BP that equal the monitor's cuff reading to the mmHg: the rule predicts the
    observed charted-minus-monitor offset (0 / -60 / +60 min) in 96 % of 2,361 encounters.
  * LMT (year 1800) origins: EHR seconds are elapsed from the origin read as UTC (offset +240 / +300 min
    against the wall-clock monitor stream) in ~75 % of the checkable files; the rest follow local time.
    Year-1990 origins follow the general rule (checked: the UTC reading was off by -300 min in 58/60).
  * Per encounter, Stage A measures the residual against exact charted-vs-monitor NBP matches and later
    stages apply it when it is a known zone offset (+-60, +-240, +-296, +-300 min); see stage_a_inventory.

The entity grid follows the UCSF store: UTC-CONTINUOUS, anchored at the wall clock of the first
segment.  "wall ms" = a naive New York wall-clock datetime encoded as ms since 1970-01-01 as if UTC;
"utc ms" = true epoch ms; "grid ms" = wall ms of the grid origin + real elapsed ms (what `time_ms.npy`
and every event's `time_ms` hold).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np

NY = ZoneInfo("America/New_York")
MS_PER_HOUR = 3_600_000
_EPOCH = datetime(1970, 1, 1)
UTC_ORIGIN_YEARS = (1800,)               # EHR seconds measured from the origin read as UTC (LMT origins)


def parse_origin(time_origin: str):
    """'2018-10-22 04:00:00.000000 EDT' -> (naive wall datetime, zone label)."""
    stamp, label = time_origin.rsplit(" ", 1)
    return datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S.%f"), label


def _wall_ms(dt: datetime) -> int:
    return int(round((dt - _EPOCH).total_seconds() * 1000))


def _offset_ms_of_wall(wall_ms: np.ndarray, fold: int = 0) -> np.ndarray:
    """UTC offset (ms) of New York wall times, looked up per wall-clock hour."""
    wall_ms = np.atleast_1d(np.asarray(wall_ms, np.int64))
    hrs = wall_ms // MS_PER_HOUR
    out = np.empty(wall_ms.size, np.int64)
    for h in np.unique(hrs):
        dt = (_EPOCH + timedelta(hours=int(h), minutes=30)).replace(tzinfo=NY, fold=fold)
        out[hrs == h] = int(dt.utcoffset().total_seconds() * 1000)
    return out


def wall_to_utc_ms(wall_ms) -> np.ndarray:
    w = np.atleast_1d(np.asarray(wall_ms, np.int64))
    return w - _offset_ms_of_wall(w)


def origin_utc_ms(W: datetime, label: str) -> int:
    """The origin instant: its wall read in the zone it is stamped with (fold picks EDT vs EST in
    the repeated fall-back hour); LMT origins as UTC."""
    if label == "LMT":
        return _wall_ms(W)
    fold = 1 if label == "EST" else 0
    return _wall_ms(W) - int(W.replace(tzinfo=NY, fold=fold).utcoffset().total_seconds() * 1000)


def dwc_to_utc_ms(t_s, W: datetime) -> np.ndarray:
    """Monitor stream (waveforms, data/numerics) seconds -> utc ms: wall = W + t."""
    return wall_to_utc_ms(_wall_ms(W) + np.atleast_1d(np.round(np.asarray(t_s, float) * 1000).astype(np.int64)))


def ehr_origin_wall(W: datetime, label: str, disch_s: float | None) -> datetime:
    """The EHR stream's origin wall: the origin instant rendered in the DST state at discharge
    (disch_s = ehr/demographic.dischDate seconds; None -> the stamped wall)."""
    if disch_s is None or not np.isfinite(disch_s):
        return W
    o_utc = origin_utc_ms(W, label)
    disch_wall = _wall_ms(W) + int(round(disch_s * 1000))
    off_b = int(_offset_ms_of_wall(np.array([disch_wall]))[0])
    return _EPOCH + timedelta(milliseconds=o_utc + off_b)


def ehr_to_utc_ms(t_s, W: datetime, label: str, disch_s: float | None) -> np.ndarray:
    """EHR seconds -> utc ms.  UTC-origin files (LMT / 1990): origin read as UTC + elapsed seconds.
    Otherwise wall-clock seconds from the discharge-state origin wall."""
    t_ms = np.atleast_1d(np.round(np.asarray(t_s, float) * 1000).astype(np.int64))
    if label == "LMT" or W.year in UTC_ORIGIN_YEARS:
        return _wall_ms(W) + t_ms
    return wall_to_utc_ms(_wall_ms(ehr_origin_wall(W, label, disch_s)) + t_ms)


class Grid:
    """UTC-continuous entity grid anchored at the wall clock of the first segment."""

    def __init__(self, W: datetime, first_seg_t_s: float):
        self.wall0 = _wall_ms(W) + int(round(first_seg_t_s * 1000))
        self.utc0 = int(wall_to_utc_ms(self.wall0)[0])

    def from_utc(self, utc_ms) -> np.ndarray:
        return self.wall0 + (np.asarray(utc_ms, np.int64) - self.utc0)

    def dwc(self, t_s, W: datetime) -> np.ndarray:
        return self.from_utc(dwc_to_utc_ms(t_s, W))

    def ehr(self, t_s, W: datetime, label: str, disch_s: float | None) -> np.ndarray:
        return self.from_utc(ehr_to_utc_ms(t_s, W, label, disch_s))
