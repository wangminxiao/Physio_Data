"""Shared helpers for the MLADI stages: paths, audata decoding, value parsing, the NBP clock check."""
from __future__ import annotations

import collections
import json
import os
import re

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EHR_EVENT_DTYPE = np.dtype([("time_ms", "int64"), ("seg_idx", "int32"), ("var_id", "uint16"), ("value", "float32")])


def cfg() -> dict:
    """The `mladi` section of workzone/configs/server_paths.yaml (no PyYAML needed: flat key: value)."""
    path = os.path.join(_REPO, "workzone", "configs", "server_paths.yaml")
    out, inside = {}, False
    for line in open(path):
        if re.match(r"^mladi:\s*$", line):
            inside = True; continue
        if inside:
            if line.strip() and not line.startswith(" "):
                break
            m = re.match(r"^\s+([A-Za-z0-9_]+):\s*(\S+)", line)
            if m:
                out[m.group(1)] = m.group(2)
    return out


def factor(ds) -> dict:
    """Decode an audata table: factor columns -> object arrays of level strings (None = missing)."""
    a = ds[:]
    cm = json.loads(ds.attrs[".meta"]).get("columns", {}) if ".meta" in ds.attrs else {}
    out = {}
    for c in a.dtype.names:
        lev = cm.get(c, {}).get("levels")
        if lev:
            codes = a[c].astype(int)
            out[c] = np.array([lev[k] if 0 <= k < len(lev) else None for k in codes], dtype=object)
        else:
            out[c] = a[c]
    return out


def num(v) -> float:
    """A value as float, NaN when it is text (audata factors such as '<0.5') or non-finite."""
    try:
        x = float(v)
        return x if np.isfinite(x) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def nums(a) -> np.ndarray:
    return np.array([num(x) for x in a], float)


def nbp_offset(ehr_ms, sbp, dbp, mon_ms, mon_s, mon_d, window_min: float = 360.0):
    """Charted-minus-monitor offset (min) from exact cuff matches: every charted (SBP, DBP) against every
    monitor reading with s and d within 0.5 mmHg (d ignored where either side lacks it), time
    differences binned to 1 min, the mode. Returns (offset_min, n_explained, n_charted) or
    (None, 0, n) when nothing matches."""
    ehr_ms = np.asarray(ehr_ms, np.int64); mon_ms = np.asarray(mon_ms, np.int64)
    diffs = []
    for t, s, d in zip(ehr_ms, sbp, dbp):
        m = np.abs(mon_s - s) <= 0.5
        if np.isfinite(d):
            m &= (np.abs(mon_d - d) <= 0.5) | ~np.isfinite(mon_d)
        z = (t - mon_ms[m]) / 60000.0
        diffs.append(z[np.abs(z) <= window_min])
    if not any(len(z) for z in diffs):
        return None, 0, len(diffs)
    mode = collections.Counter(np.round(np.concatenate(diffs)).astype(int).tolist()).most_common(1)[0][0]
    n_ok = int(sum(np.any(np.abs(z - mode) <= 1.5) for z in diffs))
    return int(mode), n_ok, len(diffs)
