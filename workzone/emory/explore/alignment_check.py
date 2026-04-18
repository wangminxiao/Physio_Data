#!/usr/bin/env python3
"""
Step 0c — 4-source time alignment check for Emory.

Verifies the time frame of every source used in the canonical pipeline:
  1. wav        — WFDB waveform segments (top .hea + _0XXX.hea)
  2. vital      — WFDB dense numerics (_0n.mat @ 0.5 Hz, incl NBP-S/D/M)
  3. ehr        — JGSEPSIS_VITALS2.csv + JGSEPSIS_LABS.csv (NY local naive strings)
  4. list_csv   — whole-list (wfdb_start Z-suffixed) + task list (valid_start naive)
                  + sepsis_time_zero_dttm

Test encounter: 359559206 (empi=1827183, case), wfdb_record B035-0564111269.

Two timezone hypotheses for WFDB base_datetime-after-+30y:
  H1: already UTC (old pipeline's implicit assumption, matches Z-suffix)
  H2: NY local, needs convert to UTC

The empirical test joins EHR SBP_CUFF events to _0n NBP-S events by nearest
timestamp under each hypothesis. The correct hypothesis wins on two criteria:
  - small time delta between matched EHR/WFDB NBP events (should be < a minute)
  - small value diff (device and EHR record the same measurement)

Output:
  alignment_report.json — full numerical report
  alignment_plot.png    — shared-time visualization (both hypotheses stacked)
"""
import os
import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from dateutil.relativedelta import relativedelta

import numpy as np
import polars as pl
import wfdb
from scipy.io import loadmat
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ENC = 359559206
EMPI = 1827183
REC = "B035-0564111269"
WFDB_ROOT = "/labs/collab/Waveform_Data/Waveform_Data"
CSV_DIR = "/labs/hulab/Emory_EHR/CDW_Pull_ICU_Data_Siva_version"
WHOLE_CSV = "/labs/hulab/mxwang/data/sepsis/Wav/sepsis_cc_2025_06_13_all_collab.csv"
TASK_CSV = "/labs/hulab/mxwang/data/sepsis/Wav/sepsis_cc_2025_06_13_all_collab_uniq_combine.csv"
OUT_DIR = "/labs/hulab/mxwang/Physio_Data/workzone/emory/explore"
os.makedirs(OUT_DIR, exist_ok=True)

NY = ZoneInfo("America/New_York")
UTC = timezone.utc


def naive_to_utc_ms(dt_naive):
    return int(dt_naive.replace(tzinfo=UTC).timestamp() * 1000)


def naive_ny_to_utc_ms(dt_naive):
    return int(dt_naive.replace(tzinfo=NY).astimezone(UTC).timestamp() * 1000)


print(f"\n=== encounter {ENC} / record {REC} ===")

# --- 1. LIST CSV ---
whole = pl.read_csv(WHOLE_CSV).filter(pl.col("encounter_nbr") == ENC)
task = pl.read_csv(TASK_CSV).filter(pl.col("encounter_nbr") == ENC)
list_info = {
    "whole_wfdb_start_min_with_Z": whole["wfdb_start"].min(),
    "whole_wfdb_end_max_with_Z": whole["wfdb_end"].max(),
    "task_valid_start_naive": task["valid_start"][0],
    "task_valid_end_naive": task["valid_end"][0],
    "task_sepsis_time_zero_naive": task["sepsis_time_zero_dttm"][0],
    "type": task["type"][0],
    "valid_ratio": float(task["valid_ratio"][0]),
}
print(json.dumps(list_info, indent=2, default=str))

# --- 2. WFDB HEADERS (top + first segment + dense numerics) ---
rec_dir = f"{WFDB_ROOT}/{REC.split('-')[0]}/{REC}"
hdr_top = wfdb.rdheader(f"{rec_dir}/{REC}")
hdr_seg0 = wfdb.rdheader(f"{rec_dir}/{REC}_0000")
hdr_0n = wfdb.rdheader(f"{rec_dir}/{REC}_0n")

base_dt = hdr_top.base_datetime  # naive
shifted = base_dt + relativedelta(years=30)

shifted_as_UTC_ms = naive_to_utc_ms(shifted)
shifted_as_NY_then_UTC_ms = naive_ny_to_utc_ms(shifted)

wfdb_info = {
    "top_base_datetime_naive": str(base_dt),
    "top_base_plus30y_naive": str(shifted),
    "top_base_plus30y_as_UTC_iso": datetime.fromtimestamp(shifted_as_UTC_ms / 1000, UTC).isoformat(),
    "top_base_plus30y_as_NY_then_UTC_iso": datetime.fromtimestamp(shifted_as_NY_then_UTC_ms / 1000, UTC).isoformat(),
    "top_sig_len": hdr_top.sig_len,
    "top_fs": hdr_top.fs,
    "seg0_sig_name": list(hdr_seg0.sig_name),
    "seg0_sig_len": hdr_seg0.sig_len,
    "seg0_fs": hdr_seg0.fs,
    "seg0_adc_gain": list(hdr_seg0.adc_gain),
    "_0n_sig_name": list(hdr_0n.sig_name),
    "_0n_sig_len": hdr_0n.sig_len,
    "_0n_fs": hdr_0n.fs,
    "_0n_adc_gain": list(hdr_0n.adc_gain),
}
print(f"\nWFDB headers: base_dt={base_dt}, +30y={shifted}")
print(f"  as UTC ms: {shifted_as_UTC_ms}")
print(f"  as NY->UTC ms: {shifted_as_NY_then_UTC_ms}")
print(f"  diff (hours): {(shifted_as_NY_then_UTC_ms - shifted_as_UTC_ms)/3600000:.2f}")

# Build _0n time arrays under both hypotheses
T0 = hdr_0n.sig_len
fs_0n = float(hdr_0n.fs)
step_ms_0n = 1000.0 / fs_0n
t_0n_H1 = shifted_as_UTC_ms + np.arange(T0) * step_ms_0n
t_0n_H2 = shifted_as_NY_then_UTC_ms + np.arange(T0) * step_ms_0n

# --- 3. VITAL — read _0n.mat NBP-S/D/M ---
mat = loadmat(f"{rec_dir}/{REC}_0n.mat")
raw = mat["val"].astype(np.float32)  # [C, T]
names_0n = list(hdr_0n.sig_name)
gains_0n = np.asarray(hdr_0n.adc_gain, dtype=np.float32)


def get_0n(name):
    if name not in names_0n:
        return None
    i = names_0n.index(name)
    s = raw[i] / gains_0n[i]  # physical units
    s[s == 0] = np.nan
    s[s < 0] = np.nan
    return s


nbp_s_0n = get_0n("NBP-S")
nbp_d_0n = get_0n("NBP-D")
nbp_m_0n = get_0n("NBP-M")
hr_0n = get_0n("HR")
spo2_0n = get_0n("SPO2-%")

n_obs_nbp_s = int(np.sum(~np.isnan(nbp_s_0n))) if nbp_s_0n is not None else 0
print(f"\n_0n total samples: {T0}  NBP-S observed count: {n_obs_nbp_s}")

# --- 4. EHR — read VITALS2 and LABS for this encounter, NY local -> UTC ---
vitals = (
    pl.scan_csv(f"{CSV_DIR}/JGSEPSIS_VITALS2.csv", infer_schema_length=10000, ignore_errors=True)
    .filter(pl.col("ENCOUNTER_NBR").cast(pl.Int64, strict=False) == ENC)
    .with_columns(pl.col("RECORDED_TIME").str.strptime(pl.Datetime, "%m/%d/%Y %H:%M:%S", strict=False).alias("t_local"))
    .with_columns(pl.col("t_local").dt.replace_time_zone("America/New_York", ambiguous="earliest").dt.convert_time_zone("UTC").alias("t_utc"))
    .collect()
)
print(f"EHR VITALS2 rows for this enc: {vitals.height}")


def ehr_col_events(col):
    v = vitals.with_columns(pl.col(col).cast(pl.Float64, strict=False).alias(col))
    v = v.filter(pl.col(col).is_not_null() & (pl.col(col) > 0)).select(["t_utc", col]).sort("t_utc")
    ts = v["t_utc"].dt.epoch("ms").to_numpy()
    vs = v[col].to_numpy()
    return ts, vs


sbp_ehr_ms, sbp_ehr_vals = ehr_col_events("SBP_CUFF")
dbp_ehr_ms, dbp_ehr_vals = ehr_col_events("DBP_CUFF")
map_ehr_ms, map_ehr_vals = ehr_col_events("MAP_CUFF")
pulse_ehr_ms, pulse_ehr_vals = ehr_col_events("PULSE")
temp_ehr_ms, temp_ehr_vals = ehr_col_events("TEMPERATURE")
print(f"  EHR SBP_CUFF n={len(sbp_ehr_ms)}  MAP_CUFF n={len(map_ehr_ms)}  PULSE n={len(pulse_ehr_ms)}")


# --- 5. ALIGNMENT TEST ---
# For each EHR SBP_CUFF event, find nearest _0n NBP-S observed event under each H
def match_nearest(ehr_ms, ehr_vals, wfdb_ms, wfdb_vals, obs_mask):
    ws = wfdb_ms[obs_mask]
    wv = wfdb_vals[obs_mask]
    if len(ws) == 0 or len(ehr_ms) == 0:
        return []
    rows = []
    for t, v in zip(ehr_ms, ehr_vals):
        i = int(np.argmin(np.abs(ws - t)))
        rows.append({
            "ehr_t_ms": int(t),
            "wfdb_t_ms": int(ws[i]),
            "delta_ms": int(ws[i] - t),
            "ehr_value": float(v),
            "wfdb_value": float(wv[i]),
            "value_diff": float(wv[i] - v),
        })
    return rows


def summarize(rows):
    if not rows:
        return {"n": 0}
    d = np.array([r["delta_ms"] for r in rows]) / 60000.0  # minutes
    vd = np.array([r["value_diff"] for r in rows])
    return {
        "n": len(rows),
        "delta_min_median": float(np.median(d)),
        "delta_min_median_abs": float(np.median(np.abs(d))),
        "delta_min_p05": float(np.percentile(d, 5)),
        "delta_min_p50": float(np.percentile(d, 50)),
        "delta_min_p95": float(np.percentile(d, 95)),
        "value_diff_median": float(np.median(vd)),
        "value_diff_median_abs": float(np.median(np.abs(vd))),
    }


obs_mask_nbp_s = ~np.isnan(nbp_s_0n)

# Restrict EHR events to within the wave window (under each H)
wave_start_H1 = shifted_as_UTC_ms
wave_end_H1 = shifted_as_UTC_ms + int(T0 * step_ms_0n)
wave_start_H2 = shifted_as_NY_then_UTC_ms
wave_end_H2 = shifted_as_NY_then_UTC_ms + int(T0 * step_ms_0n)

in_window_H1 = (sbp_ehr_ms >= wave_start_H1) & (sbp_ehr_ms <= wave_end_H1)
in_window_H2 = (sbp_ehr_ms >= wave_start_H2) & (sbp_ehr_ms <= wave_end_H2)

matches_H1 = match_nearest(sbp_ehr_ms[in_window_H1], sbp_ehr_vals[in_window_H1], t_0n_H1, nbp_s_0n, obs_mask_nbp_s)
matches_H2 = match_nearest(sbp_ehr_ms[in_window_H2], sbp_ehr_vals[in_window_H2], t_0n_H2, nbp_s_0n, obs_mask_nbp_s)

sum_H1 = summarize(matches_H1)
sum_H2 = summarize(matches_H2)
print("\n=== alignment summary ===")
print(" H1 (WFDB+30y treated as UTC):", json.dumps(sum_H1, indent=2))
print(" H2 (WFDB+30y treated as NY local):", json.dumps(sum_H2, indent=2))

# --- 6. Save report ---
sepsis_t_naive = task["sepsis_time_zero_dttm"][0]
sepsis_dt_naive = datetime.fromisoformat(sepsis_t_naive)

report = {
    "encounter_nbr": ENC,
    "empi_nbr": EMPI,
    "wfdb_record": REC,
    "sources": {
        "list_csv": list_info,
        "wfdb_headers": wfdb_info,
        "wave_start_under_H1_utc_iso": datetime.fromtimestamp(wave_start_H1 / 1000, UTC).isoformat(),
        "wave_end_under_H1_utc_iso": datetime.fromtimestamp(wave_end_H1 / 1000, UTC).isoformat(),
        "wave_start_under_H2_utc_iso": datetime.fromtimestamp(wave_start_H2 / 1000, UTC).isoformat(),
        "wave_end_under_H2_utc_iso": datetime.fromtimestamp(wave_end_H2 / 1000, UTC).isoformat(),
        "sepsis_time_zero_naive": sepsis_t_naive,
        "sepsis_time_zero_as_UTC_iso": sepsis_dt_naive.replace(tzinfo=UTC).isoformat(),
        "sepsis_time_zero_as_NY_then_UTC_iso": sepsis_dt_naive.replace(tzinfo=NY).astimezone(UTC).isoformat(),
        "ehr_vitals_t_range_utc": [str(vitals["t_utc"].min()), str(vitals["t_utc"].max())] if vitals.height else None,
        "ehr_sbp_cuff_count": int(len(sbp_ehr_ms)),
        "ehr_dbp_cuff_count": int(len(dbp_ehr_ms)),
        "ehr_map_cuff_count": int(len(map_ehr_ms)),
        "0n_NBP_S_observed_count": n_obs_nbp_s,
    },
    "alignment_test_nbp_sbp": {
        "H1_wfdb_as_UTC": sum_H1,
        "H2_wfdb_as_NY_local": sum_H2,
        "winning_hypothesis": ("H1" if sum_H1.get("delta_min_median_abs", 9e9) < sum_H2.get("delta_min_median_abs", 9e9) else "H2"),
        "sample_matches_H1": matches_H1[:10],
        "sample_matches_H2": matches_H2[:10],
    },
}
with open(f"{OUT_DIR}/alignment_report.json", "w") as f:
    json.dump(report, f, indent=2, default=str)
print(f"\nReport saved: {OUT_DIR}/alignment_report.json")

# --- 7. Plot: overlay NBP-S (_0n) and SBP_CUFF (EHR) under each hypothesis ---
fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=False)


def plot_overlay(ax, wfdb_ms, wfdb_vals, ehr_ms, ehr_vals, title, wave_start_ms, wave_end_ms, sepsis_ms, annot_lines):
    obs = ~np.isnan(wfdb_vals)
    wt = np.array(wfdb_ms[obs], dtype="datetime64[ms]")
    ax.scatter(wt, wfdb_vals[obs], s=18, color="red", alpha=0.6, label=f"_0n NBP-S (n={int(obs.sum())})")
    et = np.array(ehr_ms, dtype="datetime64[ms]")
    ax.scatter(et, ehr_vals, s=24, color="blue", marker="x", alpha=0.8, label=f"EHR SBP_CUFF (n={len(ehr_ms)})")
    ax.axvline(np.datetime64(int(wave_start_ms), "ms"), color="green", ls="--", lw=1, label="wave_start")
    ax.axvline(np.datetime64(int(wave_end_ms), "ms"), color="olive", ls="--", lw=1, label="wave_end")
    ax.axvline(np.datetime64(int(sepsis_ms), "ms"), color="orange", ls="-.", lw=1.2, label="sepsis_t0")
    for nm, ms_val, color, ls in annot_lines:
        ax.axvline(np.datetime64(int(ms_val), "ms"), color=color, ls=ls, lw=0.8, label=nm)
    ax.set_ylabel("SBP (mmHg)")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    ax.grid(alpha=0.3)
    loc = mdates.AutoDateLocator(minticks=6, maxticks=12)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


# Parse list-csv timestamps to UTC ms
wfdb_start_Z_ms = int(datetime.fromisoformat(list_info["whole_wfdb_start_min_with_Z"].replace("Z", "+00:00")).timestamp() * 1000)
valid_start_as_UTC_ms = naive_to_utc_ms(datetime.fromisoformat(list_info["task_valid_start_naive"]))
sepsis_as_UTC_ms = naive_to_utc_ms(sepsis_dt_naive)
sepsis_as_NY_then_UTC_ms = naive_ny_to_utc_ms(sepsis_dt_naive)

# H1 panel — WFDB=UTC, sepsis=UTC
plot_overlay(
    axes[0],
    t_0n_H1,
    nbp_s_0n,
    sbp_ehr_ms,
    sbp_ehr_vals,
    f"H1: WFDB+30y treated as UTC | EHR=NY→UTC | sepsis treated as UTC",
    wave_start_H1,
    wave_end_H1,
    sepsis_as_UTC_ms,
    [
        ("list wfdb_start(Z)", wfdb_start_Z_ms, "gray", ":"),
        ("list valid_start", valid_start_as_UTC_ms, "purple", ":"),
    ],
)

# H2 panel — WFDB=NY→UTC, sepsis=NY→UTC
plot_overlay(
    axes[1],
    t_0n_H2,
    nbp_s_0n,
    sbp_ehr_ms,
    sbp_ehr_vals,
    f"H2: WFDB+30y treated as NY→UTC | EHR=NY→UTC | sepsis as NY→UTC",
    wave_start_H2,
    wave_end_H2,
    sepsis_as_NY_then_UTC_ms,
    [
        ("list wfdb_start(Z)", wfdb_start_Z_ms, "gray", ":"),
        ("list valid_start naive→UTC", valid_start_as_UTC_ms, "purple", ":"),
        ("list valid_start NY→UTC", naive_ny_to_utc_ms(datetime.fromisoformat(list_info["task_valid_start_naive"])), "magenta", ":"),
    ],
)

# Zoom: first 8h wave window under the winning hypothesis
best = "H1" if sum_H1.get("delta_min_median_abs", 9e9) < sum_H2.get("delta_min_median_abs", 9e9) else "H2"
if best == "H1":
    ws, we = wave_start_H1, wave_end_H1
    wfdb_ms = t_0n_H1
else:
    ws, we = wave_start_H2, wave_end_H2
    wfdb_ms = t_0n_H2
in_zoom = (sbp_ehr_ms >= ws) & (sbp_ehr_ms <= we)
ax = axes[2]
obs = ~np.isnan(nbp_s_0n)
wt = np.array(wfdb_ms[obs], dtype="datetime64[ms]")
wv = nbp_s_0n[obs]
# Only within zoom
mask_zoom = (wfdb_ms[obs] >= ws) & (wfdb_ms[obs] <= we)
ax.scatter(wt[mask_zoom], wv[mask_zoom], s=40, color="red", alpha=0.7, label="_0n NBP-S")
ax.scatter(np.array(sbp_ehr_ms[in_zoom], dtype="datetime64[ms]"), sbp_ehr_vals[in_zoom], s=60, color="blue", marker="x", alpha=0.9, label="EHR SBP_CUFF")
ax.axvline(np.datetime64(int(ws), "ms"), color="green", ls="--", lw=1, label="wave_start")
ax.axvline(np.datetime64(int(we), "ms"), color="olive", ls="--", lw=1, label="wave_end")
ax.set_ylabel("SBP (mmHg)")
ax.set_title(f"Zoom on wave window — winning hypothesis {best}")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
loc = mdates.AutoDateLocator(minticks=4, maxticks=8)
ax.xaxis.set_major_locator(loc)
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))

fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/alignment_plot.png", dpi=110, bbox_inches="tight")
print(f"Plot saved: {OUT_DIR}/alignment_plot.png")
