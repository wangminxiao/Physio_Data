#!/usr/bin/env python3
"""
Tighter alignment check: match EHR SBP_CUFF events to _0n NBP-S STEP CHANGES
(not nearest sample of the held signal).

A step change = consecutive samples where the value differs by > tol mmHg.
For each step-change instant, record the new value.

Then for each EHR SBP_CUFF event in the wave window, find the nearest
step-change instant under H1 (WFDB=UTC) and report both time delta and
value at that step.

If H1 alignment is right:
  - nearest step should be close in time (seconds to a minute)
  - new step value should match EHR value closely

If the hold-signal-lag explanation is correct:
  - delta_step - delta_ehr should be consistently POSITIVE (_0n steps after EHR)
  - value at _0n step should match EHR value (both describe the same cuff reading)
"""
import os, json
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta
import numpy as np
import polars as pl
import wfdb
from scipy.io import loadmat

ENC = 359559206
REC = 'B035-0564111269'
WFDB_ROOT = '/labs/collab/Waveform_Data/Waveform_Data'
CSV_DIR = '/labs/hulab/Emory_EHR/CDW_Pull_ICU_Data_Siva_version'
OUT = '/labs/hulab/mxwang/Physio_Data/workzone/emory/explore/step_alignment.json'
UTC = timezone.utc

# --- WFDB header + _0n ---
rec_dir = f'{WFDB_ROOT}/{REC.split("-")[0]}/{REC}'
hdr0n = wfdb.rdheader(f'{rec_dir}/{REC}_0n')
base_ms = int((hdr0n.base_datetime + relativedelta(years=30)).replace(tzinfo=UTC).timestamp() * 1000)
fs = float(hdr0n.fs)
step_ms = 1000.0 / fs
mat = loadmat(f'{rec_dir}/{REC}_0n.mat')
raw = mat['val'].astype(np.float32)
names = list(hdr0n.sig_name)
gains = np.asarray(hdr0n.adc_gain, dtype=np.float32)


def chan(name):
    i = names.index(name)
    s = raw[i] / gains[i]
    s[s == 0] = np.nan
    s[s < 0] = np.nan
    return s


nbp_s = chan('NBP-S')
nbp_m = chan('NBP-M')
nbp_d = chan('NBP-D')
t_0n = base_ms + np.arange(len(nbp_s)) * step_ms

# --- find step-changes in NBP-S ---
# step = sample where value differs from previous non-NaN sample by > 0.5 mmHg
def steps(arr, tol=0.5):
    out_idx = []
    prev = None
    for i, v in enumerate(arr):
        if np.isnan(v):
            continue
        if prev is None or abs(v - prev) > tol:
            out_idx.append(i)
            prev = v
        else:
            prev = v
    return np.array(out_idx, dtype=np.int64)


step_idx = steps(nbp_s, tol=0.5)
step_t_ms = t_0n[step_idx]
step_vals = nbp_s[step_idx]
print(f'NBP-S step events: {len(step_idx)}')

# --- EHR SBP_CUFF events ---
vitals = (
    pl.scan_csv(f'{CSV_DIR}/JGSEPSIS_VITALS2.csv', infer_schema_length=10000, ignore_errors=True)
    .filter(pl.col('ENCOUNTER_NBR').cast(pl.Int64, strict=False) == ENC)
    .with_columns(pl.col('RECORDED_TIME').str.strptime(pl.Datetime, '%m/%d/%Y %H:%M:%S', strict=False).alias('t_local'))
    .with_columns(pl.col('t_local').dt.replace_time_zone('America/New_York', ambiguous='earliest').dt.convert_time_zone('UTC').alias('t_utc'))
    .with_columns(pl.col('SBP_CUFF').cast(pl.Float64, strict=False))
    .filter(pl.col('SBP_CUFF').is_not_null() & (pl.col('SBP_CUFF') > 0))
    .collect()
)
ehr_t = vitals['t_utc'].dt.epoch('ms').to_numpy()
ehr_v = vitals['SBP_CUFF'].to_numpy()
print(f'EHR SBP_CUFF events: {len(ehr_t)}')

# --- match EHR events to nearest _0n STEP (restrict to wave window) ---
wave_end = base_ms + int(len(nbp_s) * step_ms)
in_window = (ehr_t >= base_ms) & (ehr_t <= wave_end)
ehr_t_w = ehr_t[in_window]
ehr_v_w = ehr_v[in_window]

rows = []
for t, v in zip(ehr_t_w, ehr_v_w):
    diffs = step_t_ms - t
    # nearest on each side
    abs_i = int(np.argmin(np.abs(diffs)))
    # step at-or-after (forward)
    fwd_mask = diffs >= 0
    if fwd_mask.any():
        fwd_i = int(np.argmin(np.where(fwd_mask, diffs, np.inf)))
        fwd_dt_s = (step_t_ms[fwd_i] - t) / 1000
        fwd_val = step_vals[fwd_i]
    else:
        fwd_dt_s = None
        fwd_val = None
    # step before (backward)
    bwd_mask = diffs < 0
    if bwd_mask.any():
        bwd_i = int(np.argmax(np.where(bwd_mask, diffs, -np.inf)))
        bwd_dt_s = (step_t_ms[bwd_i] - t) / 1000
        bwd_val = step_vals[bwd_i]
    else:
        bwd_dt_s = None
        bwd_val = None
    rows.append({
        'ehr_t_ms': int(t),
        'ehr_value': float(v),
        'nearest_step_dt_s': float((step_t_ms[abs_i] - t) / 1000),
        'nearest_step_value': float(step_vals[abs_i]),
        'next_step_after_ehr_dt_s': fwd_dt_s,
        'next_step_after_ehr_value': float(fwd_val) if fwd_val is not None else None,
        'prev_step_before_ehr_dt_s': bwd_dt_s,
        'prev_step_before_ehr_value': float(bwd_val) if bwd_val is not None else None,
        'value_diff_next_step': (float(fwd_val - v)) if fwd_val is not None else None,
    })

# --- summary stats ---
near_dt = np.array([r['nearest_step_dt_s'] for r in rows])
fwd_dts = [r['next_step_after_ehr_dt_s'] for r in rows if r['next_step_after_ehr_dt_s'] is not None]
fwd_dvs = [r['value_diff_next_step'] for r in rows if r['value_diff_next_step'] is not None]

summary = {
    'encounter': ENC,
    'n_ehr_in_window': int(len(ehr_t_w)),
    'n_0n_nbp_s_steps': int(len(step_idx)),
    'nearest_step_dt_s': {
        'median': float(np.median(near_dt)),
        'median_abs': float(np.median(np.abs(near_dt))),
        'p05': float(np.percentile(near_dt, 5)),
        'p50': float(np.percentile(near_dt, 50)),
        'p95': float(np.percentile(near_dt, 95)),
    },
    'next_step_after_ehr_dt_s': {
        'median': float(np.median(fwd_dts)) if fwd_dts else None,
        'p05': float(np.percentile(fwd_dts, 5)) if fwd_dts else None,
        'p50': float(np.percentile(fwd_dts, 50)) if fwd_dts else None,
        'p95': float(np.percentile(fwd_dts, 95)) if fwd_dts else None,
        'n': len(fwd_dts),
    },
    'value_diff_at_next_step': {
        'median': float(np.median(fwd_dvs)) if fwd_dvs else None,
        'median_abs': float(np.median(np.abs(fwd_dvs))) if fwd_dvs else None,
        'p05': float(np.percentile(fwd_dvs, 5)) if fwd_dvs else None,
        'p95': float(np.percentile(fwd_dvs, 95)) if fwd_dvs else None,
        'n': len(fwd_dvs),
    },
    'sample_first_10_rows': rows[:10],
}

with open(OUT, 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print(json.dumps(summary, indent=2, default=str))
