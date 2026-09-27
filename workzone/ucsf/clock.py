"""
UCSF clock conventions — one place for every time conversion in the UCSF pipelines.

Background (verified on data, 2026-09-26; full write-up in datasets/ucsf/ALIGNMENT.md):

  * Every UCSF timestamp is de-identified by a per-encounter day shift, but the shift arithmetic differs
    by source.  GE monitor files (`.adibin` headers, `.vital` headers) were shifted in ABSOLUTE time
    (UTC epoch minus N days, then rendered as America/Los_Angeles wall clock at the shifted date).  EHR
    tables (`Filtered_*`, `FLOWSHEETVALUEFACT`; shift `offset` days) and ADT tables (MRN-Mapping,
    ValidWaveTime; shift `offset_GE` days) were shifted on the WALL CLOCK (naive datetime minus N days).
    `offset_GE - offset == 12 days` for every encounter.
  * Consequently the naive rule `T_ge = T_ehr - 12 d` is exact only when the real date and the GE-shifted
    date share the same DST state; otherwise it is off by +-60 min (45 % of encounters).
  * Inside one wave cycle the monitor streams are continuous in real elapsed time, but each `.adibin` chunk
    carries its own wall-clock header, so a DST switch on the GE calendar inside a cycle makes wall-clock
    placement open a 1-h gap (spring) or overlap 1 h (fall) and desynchronises `.vital` streams from the
    waveform.  The entity grid is therefore defined as UTC-CONTINUOUS: `time_ms[0]` is the wall clock of
    the first `.adibin` header (GE calendar) and every later grid time is that origin plus REAL elapsed
    milliseconds (measured in UTC).

All "wall ms" values below are naive local wall-clock datetimes encoded as milliseconds since
1970-01-01 00:00 as if they were UTC (the convention used by `.adibin`/`.vital` headers, `time_ms.npy`
and every EHR timestamp in this repo).  "utc ms" values are true epoch milliseconds.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np

LA = ZoneInfo("America/Los_Angeles")
MS_PER_DAY = 86_400_000
MS_PER_HOUR = 3_600_000
_EPOCH = datetime(1970, 1, 1)
EHR_MINUS_GE_DAYS = 12          # offset_GE - offset, constant over the whole offset table


# ---------------------------------------------------------------- scalar primitives
def wall_ms_to_dt(wall_ms: int) -> datetime:
    """naive wall-clock datetime from wall ms."""
    return _EPOCH + timedelta(milliseconds=int(wall_ms))


def dt_to_wall_ms(dt: datetime) -> int:
    """wall ms from a naive datetime (tzinfo, if any, is ignored)."""
    return int((dt.replace(tzinfo=None) - _EPOCH).total_seconds() * 1000)


def utc_offset_ms(wall_ms: int, fold: int = 0) -> int:
    """America/Los_Angeles UTC offset (ms, negative) in force at this wall-clock instant.
    Ambiguous fall-back hour resolves to the first occurrence (PDT) with fold=0."""
    dt = wall_ms_to_dt(wall_ms).replace(tzinfo=LA, fold=fold)
    return int(dt.utcoffset().total_seconds() * 1000)


def is_dst(wall_ms: int) -> bool:
    return wall_ms_to_dt(wall_ms).replace(tzinfo=LA).dst().total_seconds() != 0


def wall_to_utc_ms(wall_ms: int) -> int:
    return int(wall_ms) - utc_offset_ms(wall_ms)


def utc_to_wall_ms(utc_ms: int) -> int:
    dt = datetime.fromtimestamp(int(utc_ms) / 1000.0, tz=timezone.utc).astimezone(LA).replace(tzinfo=None)
    return dt_to_wall_ms(dt) + (int(utc_ms) % 1 if False else 0)


def dst_switch_between(a_wall_ms: int, b_wall_ms: int) -> bool:
    """True when the LA UTC offset differs between the two wall-clock instants (a DST switch lies between)."""
    return utc_offset_ms(a_wall_ms) != utc_offset_ms(b_wall_ms)


# ---------------------------------------------------------------- vectorised primitives
def _bucket_map(ms: np.ndarray, fn, bucket_ms: int = MS_PER_HOUR) -> np.ndarray:
    """Apply a scalar ms -> ms function per hour bucket (DST offsets are piecewise constant on hours)."""
    ms = np.asarray(ms, dtype=np.int64)
    if ms.size == 0:
        return np.zeros(0, dtype=np.int64)
    b = np.floor_divide(ms, bucket_ms)
    ub, inv = np.unique(b, return_inverse=True)
    vals = np.array([fn(int(x) * bucket_ms) for x in ub], dtype=np.int64)
    return vals[inv]


def utc_offset_ms_array(wall_ms: np.ndarray) -> np.ndarray:
    return _bucket_map(wall_ms, utc_offset_ms)


def wall_to_utc_ms_array(wall_ms: np.ndarray) -> np.ndarray:
    wall_ms = np.asarray(wall_ms, dtype=np.int64)
    return wall_ms - utc_offset_ms_array(wall_ms)


def utc_to_wall_ms_array(utc_ms: np.ndarray) -> np.ndarray:
    utc_ms = np.asarray(utc_ms, dtype=np.int64)
    # offset as a function of the UTC instant is unambiguous
    def off_at_utc(u):
        return int(datetime.fromtimestamp(u / 1000.0, tz=timezone.utc).astimezone(LA).utcoffset().total_seconds() * 1000)
    return utc_ms + _bucket_map(utc_ms, off_at_utc)


# ---------------------------------------------------------------- grid conversions
def _as_array(x):
    return isinstance(x, (np.ndarray, list, tuple))


def ge_wall_to_grid_ms(wall_ms, episode_start_wall_ms: int):
    """GE-calendar wall clock (an `.adibin`/`.vital` header, MRN-Mapping-style time already on the monitor
    clock) -> UTC-continuous grid ms:  grid = origin + (UTC(wall) - UTC(origin)).
    Identity when no DST switch lies between the origin and `wall`."""
    o = int(episode_start_wall_ms)
    base = o - wall_to_utc_ms(o)
    if _as_array(wall_ms):
        return wall_to_utc_ms_array(np.asarray(wall_ms, dtype=np.int64)) + base
    return wall_to_utc_ms(int(wall_ms)) + base


def grid_to_ge_wall_ms(grid_ms, episode_start_wall_ms: int):
    """Inverse of ge_wall_to_grid_ms (wall clock the monitor would have shown)."""
    o = int(episode_start_wall_ms)
    base = wall_to_utc_ms(o) - o
    if _as_array(grid_ms):
        return utc_to_wall_ms_array(np.asarray(grid_ms, dtype=np.int64) + base)
    return utc_to_wall_ms(int(grid_ms) + base)


def real_wall_to_grid_ms(real_wall_ms, offset_ge_days: float, episode_start_wall_ms: int):
    """TRUE local wall clock (e.g. a Code Blue time from the EHR, not de-identified) -> grid ms.
    UTC(real) - offset_GE days is the de-identified UTC instant; the grid is continuous in UTC."""
    o = int(episode_start_wall_ms)
    shift = int(round(float(offset_ge_days) * MS_PER_DAY))
    base = o - wall_to_utc_ms(o)
    if _as_array(real_wall_ms):
        return wall_to_utc_ms_array(np.asarray(real_wall_ms, dtype=np.int64)) - shift + base
    return wall_to_utc_ms(int(real_wall_ms)) - shift + base


def grid_to_real_wall_ms(grid_ms, offset_ge_days: float, episode_start_wall_ms: int):
    """Inverse of real_wall_to_grid_ms (re-identifying; use only for internal validation)."""
    o = int(episode_start_wall_ms)
    shift = int(round(float(offset_ge_days) * MS_PER_DAY))
    base = wall_to_utc_ms(o) - o
    if _as_array(grid_ms):
        return utc_to_wall_ms_array(np.asarray(grid_ms, dtype=np.int64) + base + shift)
    return utc_to_wall_ms(int(grid_ms) + base + shift)


def ehr_wall_to_real_wall_ms(ehr_wall_ms, offset_days: float):
    """EHR-table time (wall clock shifted by `offset` days) -> true local wall clock (wall arithmetic)."""
    shift = int(round(float(offset_days) * MS_PER_DAY))
    if _as_array(ehr_wall_ms):
        return np.asarray(ehr_wall_ms, dtype=np.int64) + shift
    return int(ehr_wall_ms) + shift


def ehr_wall_to_grid_ms(ehr_wall_ms, offset_days: float, offset_ge_days: float, episode_start_wall_ms: int):
    """EHR-table time (labs, flowsheet, orders, encounter times; naive shift by `offset` days) -> grid ms.
    THIS replaces the legacy rule `T_ge = T_ehr - (offset_GE - offset) days`."""
    return real_wall_to_grid_ms(ehr_wall_to_real_wall_ms(ehr_wall_ms, offset_days), offset_ge_days, episode_start_wall_ms)


def adt_wall_to_grid_ms(adt_wall_ms, offset_ge_days: float, episode_start_wall_ms: int):
    """MRN-Mapping / ValidWaveTime ADT times (wall clock shifted by `offset_GE` days) -> grid ms."""
    return real_wall_to_grid_ms(ehr_wall_to_real_wall_ms(adt_wall_ms, offset_ge_days), offset_ge_days, episode_start_wall_ms)


def ehr_wall_to_ge_wall_naive_ms(ehr_wall_ms, offset_days: float, offset_ge_days: float):
    """LEGACY naive rule (kept only for comparisons / migration checks)."""
    shift = int(round((float(offset_ge_days) - float(offset_days)) * MS_PER_DAY))
    if _as_array(ehr_wall_ms):
        return np.asarray(ehr_wall_ms, dtype=np.int64) - shift
    return int(ehr_wall_ms) - shift


def matlab_datenum_to_wall_ms(datenum: float) -> int:
    """MATLAB datenum (days since 0000-01-00) -> wall ms."""
    d = float(datenum)
    dt = datetime.fromordinal(int(d)) + timedelta(days=d % 1) - timedelta(days=366)
    return dt_to_wall_ms(dt)
