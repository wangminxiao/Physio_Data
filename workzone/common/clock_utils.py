"""
Shared clock helpers — the ONLY place that turns a naive datetime into epoch milliseconds.

Conventions (see datasets/ucsf/ALIGNMENT.md and datasets/CLOCK_FIX_PLAN_MIMIC_MOVER.md):
  * "wall ms": a naive (tz-less) wall-clock datetime encoded as ms since 1970-01-01 00:00 *as if it were UTC*.
    This is the encoding of every EHR timestamp parsed with pandas/polars (`.timestamp()` of a naive Timestamp,
    `astype("int64") // 1e6`, `.dt.replace_time_zone("UTC").dt.timestamp("ms")`) and of `time_ms.npy` in the
    MIMIC-III and UCSF stores.
  * "utc ms": true epoch ms of an aware datetime (MOVER, MC_MED stores).

Never call `datetime.timestamp()` on a naive datetime in pipeline code: it applies the machine's local time
zone (America/New_York on the lab node), which produced the +4/+5 h MIMIC offset fixed in 2026-09.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np

_EPOCH = datetime(1970, 1, 1)
NY = ZoneInfo("America/New_York")
LA = ZoneInfo("America/Los_Angeles")

# old server -> new server path prefixes (meta.json `source_path` written before the 2026-08 migration)
PATH_MAP = (("/labs/hulab/", "/mnt/localdata/storage/"), ("/opt/localdata100tb/", "/mnt/localdata100tb/"),
            ("/labs/collab/", "/mnt/localdata/storage/collab/"))


def wall_ms(dt: datetime) -> int:
    """naive wall-clock datetime -> wall ms (tzinfo, if present, is ignored)."""
    return int((dt.replace(tzinfo=None) - _EPOCH).total_seconds() * 1000)


def wall_ms_to_dt(ms: int) -> datetime:
    return _EPOCH + timedelta(milliseconds=int(ms))


def utc_ms(dt: datetime) -> int:
    """aware datetime -> true epoch ms (raises on naive input)."""
    if dt.tzinfo is None:
        raise ValueError("utc_ms() needs an aware datetime; use wall_ms() for naive wall-clock times")
    return int(dt.timestamp() * 1000)


def wall_ms_array(values) -> np.ndarray:
    """numpy datetime64 / pandas datetime64[ns] series -> wall ms int64 (NaT -> min int64)."""
    a = np.asarray(values).astype("datetime64[ms]")
    return a.astype("int64")


def local_epoch_ms(dt: datetime, tz=NY, fold: int = 0) -> int:
    """What `datetime.timestamp()` returned for a NAIVE dt on a machine in `tz` (legacy behaviour, for migrations)."""
    return int(dt.replace(tzinfo=tz, fold=fold).timestamp() * 1000)


def legacy_local_epoch_to_wall_ms(local_epoch_ms_value: int, tz=NY) -> tuple[int, int]:
    """Invert the legacy conversion: (wall ms, delta_ms) with delta_ms = wall − legacy (−4 h EDT / −5 h EST for NY).
    Exact except in the ambiguous fall-back hour, where the earlier occurrence is chosen."""
    dt_local = datetime.fromtimestamp(local_epoch_ms_value / 1000.0, tz=tz).replace(tzinfo=None)
    # verify the round trip (fold 0); fall back to fold 1 if needed
    if local_epoch_ms(dt_local, tz, 0) != int(local_epoch_ms_value) and local_epoch_ms(dt_local, tz, 1) == int(local_epoch_ms_value):
        pass
    w = wall_ms(dt_local)
    return w, w - int(local_epoch_ms_value)


def translate_path(p: str | None) -> str | None:
    if not p:
        return p
    for old, new in PATH_MAP:
        if p.startswith(old):
            return new + p[len(old):]
    return p
