#!/usr/bin/env python3
"""
Step 0c demo — single-entity canonical extraction + visualization.

Encounter 359559206 (empi 1827183, case), record B035-0564111269, first 8h segment.

Does everything end-to-end on a small scale:
  - loads one WFDB waveform segment (240 Hz, I/II/III/V/SPO2/RR)
  - resamples SPO2 -> PLETH40 (40 Hz), II -> II120 (120 Hz)
  - 30s windowed, float16 C-contiguous, int64 UTC ms time axis
  - pulls EHR VITALS2 + LABS for this encounter, maps to var_ids
  - assigns seg_idx via searchsorted
  - writes canonical entity dir
  - plots a 30-minute zoom with waveform + overlaid events

Output:
  demo_entity/1827183_359559206/           — canonical dir
  demo_entity/1827183_359559206/meta.json
  demo_plot_30min.png                      — zoomed visualization
"""
import os
import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from dateutil.relativedelta import relativedelta

import numpy as np
import polars as pl
import wfdb
from scipy.signal import resample_poly
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ENC = 359559206
EMPI = 1827183
REC = "B035-0564111269"
WFDB_ROOT = "/labs/collab/Waveform_Data/Waveform_Data"
CSV_DIR = "/labs/hulab/Emory_EHR/CDW_Pull_ICU_Data_Siva_version"
OUT_ROOT = "/labs/hulab/mxwang/Physio_Data/workzone/emory/explore"
ENTITY_ID = f"{EMPI}_{ENC}"
ENTITY_DIR = f"{OUT_ROOT}/demo_entity/{ENTITY_ID}"
os.makedirs(ENTITY_DIR, exist_ok=True)

UTC = timezone.utc
NY = ZoneInfo("America/New_York")

SEG_SEC = 30
PLETH_FS = 40
II_FS = 120
WAVE_SRC_FS = 240
SEG_LEN_PLETH = SEG_SEC * PLETH_FS   # 1200
SEG_LEN_II = SEG_SEC * II_FS         # 3600

# placeholder var_ids for demo (final mapping defined later in the real pipeline)
VAR_IDS = {
    # labs 0-99
    "POTASSIUM":          0,
    "CALCIUM":            1,
    "SODIUM":             2,
    "GLUCOSE":            3,
    "LACTATE":            4,
    "CREATININE":         5,
    "PLATELET COUNT":     7,
    "HEMOGLOBIN":         9,
    # vitals 100-199
    "HR_EHR":           100,  # PULSE
    "SPO2_EHR":         101,
    "RR_EHR":           102,  # UNASSISTED_RESP_RATE
    "TEMP_EHR":         103,
    "SBP_CUFF":         104,
    "DBP_CUFF":         105,
    "MAP_CUFF":         106,
    # from _0n continuous
    "HR_MON":           150,
    "SPO2_MON":         151,
    "RESP_MON":         152,
}
EHR_EVENT_DTYPE = np.dtype([
    ("time_ms", "int64"),
    ("seg_idx", "int32"),
    ("var_id", "uint16"),
    ("value", "float32"),
])
SEG_IDX_BASELINE = np.iinfo(np.int32).min
SEG_IDX_RECENT = SEG_IDX_BASELINE + 1
SEG_IDX_FUTURE = SEG_IDX_BASELINE + 2
CONTEXT_WINDOW_MS = 24 * 3600 * 1000
BASELINE_CAP_MS = 30 * 24 * 3600 * 1000
FUTURE_CAP_MS = 7 * 24 * 3600 * 1000


def naive_to_utc_ms(dt):
    return int(dt.replace(tzinfo=UTC).timestamp() * 1000)


# =========================================================
# 1. Load waveform — first segment only (8h @ 240 Hz)
# =========================================================
print("Loading waveform segment _0000 ...")
rec_dir = f"{WFDB_ROOT}/{REC.split('-')[0]}/{REC}"
rec = wfdb.rdrecord(f"{rec_dir}/{REC}_0000", physical=True)
sig_names = list(rec.sig_name)
fs_src = rec.fs
p = rec.p_signal  # [T, C] float
print(f"  shape={p.shape}  fs={fs_src}  channels={sig_names}")

wave_base_dt = rec.base_datetime + relativedelta(years=30)
wave_start_ms = naive_to_utc_ms(wave_base_dt)
wave_end_ms = wave_start_ms + int(p.shape[0] / fs_src * 1000)
print(f"  wave window UTC: {datetime.fromtimestamp(wave_start_ms/1000, UTC)} -> {datetime.fromtimestamp(wave_end_ms/1000, UTC)}")

idx_spo2 = sig_names.index("SPO2")
idx_ii = sig_names.index("II")
spo2_raw = p[:, idx_spo2].astype(np.float32)
ii_raw = p[:, idx_ii].astype(np.float32)

# =========================================================
# 2. Resample -> 40 / 120 Hz
# =========================================================
print("Resampling SPO2 240 -> 40 Hz and II 240 -> 120 Hz ...")
pleth40 = resample_poly(spo2_raw, up=1, down=6).astype(np.float32)
ii120 = resample_poly(ii_raw, up=1, down=2).astype(np.float32)
print(f"  PLETH40 samples: {len(pleth40)}  II120 samples: {len(ii120)}")

# =========================================================
# 3. Segment to 30s, float16 C-contiguous
# =========================================================
n_seg = len(pleth40) // SEG_LEN_PLETH
pleth40 = pleth40[: n_seg * SEG_LEN_PLETH].reshape(n_seg, SEG_LEN_PLETH).astype(np.float16)
ii120 = ii120[: n_seg * SEG_LEN_II].reshape(n_seg, SEG_LEN_II).astype(np.float16)
pleth40 = np.ascontiguousarray(pleth40)
ii120 = np.ascontiguousarray(ii120)
time_ms = (wave_start_ms + np.arange(n_seg, dtype=np.int64) * SEG_SEC * 1000)
print(f"  n_seg={n_seg}  PLETH40.shape={pleth40.shape}  II120.shape={ii120.shape}")

assert pleth40.flags["C_CONTIGUOUS"] and ii120.flags["C_CONTIGUOUS"]
assert np.all(np.diff(time_ms) == SEG_SEC * 1000)

np.save(f"{ENTITY_DIR}/PLETH40.npy", pleth40)
np.save(f"{ENTITY_DIR}/II120.npy", ii120)
np.save(f"{ENTITY_DIR}/time_ms.npy", time_ms)

# =========================================================
# 4. EHR events — VITALS2 (chart) + LABS  + _0n continuous vitals
# =========================================================
print("Reading EHR for encounter ...")
vitals = (
    pl.scan_csv(f"{CSV_DIR}/JGSEPSIS_VITALS2.csv", infer_schema_length=10000, ignore_errors=True)
    .filter(pl.col("ENCOUNTER_NBR").cast(pl.Int64, strict=False) == ENC)
    .with_columns(pl.col("RECORDED_TIME").str.strptime(pl.Datetime, "%m/%d/%Y %H:%M:%S", strict=False).alias("t_local"))
    .with_columns(pl.col("t_local").dt.replace_time_zone("America/New_York", ambiguous="earliest").dt.convert_time_zone("UTC").alias("t_utc"))
    .collect()
)

ehr_rows = []


def push_col(df, col, var_id):
    v = df.with_columns(pl.col(col).cast(pl.Float64, strict=False).alias(col))
    v = v.filter(pl.col(col).is_not_null() & (pl.col(col) > 0))
    if v.height == 0:
        return
    ts = v["t_utc"].dt.epoch("ms").to_numpy()
    vs = v[col].to_numpy()
    for t, x in zip(ts, vs):
        ehr_rows.append((int(t), int(var_id), float(x)))


push_col(vitals, "SBP_CUFF", VAR_IDS["SBP_CUFF"])
push_col(vitals, "DBP_CUFF", VAR_IDS["DBP_CUFF"])
push_col(vitals, "MAP_CUFF", VAR_IDS["MAP_CUFF"])
push_col(vitals, "PULSE", VAR_IDS["HR_EHR"])
push_col(vitals, "TEMPERATURE", VAR_IDS["TEMP_EHR"])
push_col(vitals, "UNASSISTED_RESP_RATE", VAR_IDS["RR_EHR"])
push_col(vitals, "SPO2", VAR_IDS["SPO2_EHR"])

# Labs — a demo subset via COMPONENT substring match
print("Reading LABS ...")
labs = (
    pl.scan_csv(f"{CSV_DIR}/JGSEPSIS_LABS.csv", infer_schema_length=10000, ignore_errors=True)
    .filter(pl.col("ENCOUNTER_NBR").cast(pl.Int64, strict=False) == ENC)
    .with_columns(pl.col("LAB_RESULT_TIME").str.strptime(pl.Datetime, "%m/%d/%Y %H:%M:%S", strict=False).alias("t_local"))
    .with_columns(pl.col("t_local").dt.replace_time_zone("America/New_York", ambiguous="earliest").dt.convert_time_zone("UTC").alias("t_utc"))
    .with_columns(pl.col("LAB_RESULT").cast(pl.Float64, strict=False).alias("val"))
    .filter(pl.col("val").is_not_null())
    .collect()
)

LAB_MAP = {
    "Potassium Level": VAR_IDS["POTASSIUM"],
    "Calcium": VAR_IDS["CALCIUM"],
    "Sodium": VAR_IDS["SODIUM"],
    "Glucose": VAR_IDS["GLUCOSE"],
    "Lactic Acid": VAR_IDS["LACTATE"],
    "Lactate": VAR_IDS["LACTATE"],
    "Creatinine": VAR_IDS["CREATININE"],
    "Platelet Count": VAR_IDS["PLATELET COUNT"],
    "Hemoglobin": VAR_IDS["HEMOGLOBIN"],
}
for comp, var_id in LAB_MAP.items():
    sub = labs.filter(pl.col("COMPONENT") == comp)
    if sub.height == 0:
        continue
    ts = sub["t_utc"].dt.epoch("ms").to_numpy()
    vs = sub["val"].to_numpy()
    for t, x in zip(ts, vs):
        ehr_rows.append((int(t), int(var_id), float(x)))

# _0n continuous — HR, SPO2-%, RESP (sample once per minute for demo, not the hold signals)
print("Reading _0n.mat for continuous vitals (HR, SPO2-%, RESP) ...")
from scipy.io import loadmat

mat = loadmat(f"{rec_dir}/{REC}_0n.mat")
nhdr = wfdb.rdheader(f"{rec_dir}/{REC}_0n")
names_0n = list(nhdr.sig_name)
gains_0n = np.asarray(nhdr.adc_gain, dtype=np.float32)
fs_0n = float(nhdr.fs)
vals = mat["val"].astype(np.float32)


def get_0n(name):
    if name not in names_0n:
        return None
    i = names_0n.index(name)
    s = vals[i] / gains_0n[i]
    s[s == 0] = np.nan
    s[s < 0] = np.nan
    return s


hr = get_0n("HR")
spo2p = get_0n("SPO2-%")
resp = get_0n("RESP")

# _0n covers the full 25-day record; clip to wave window (first 8h = first segment only)
step_ms_0n = 1000.0 / fs_0n
t_0n = wave_start_ms + np.arange(len(hr)) * step_ms_0n
in_window = (t_0n >= wave_start_ms) & (t_0n <= wave_end_ms)

# For demo: subsample to 1 Hz (every 2nd sample since fs=0.5 Hz)
idxs_in_window = np.where(in_window)[0]


def push_0n(arr, var_id, step=1):
    if arr is None:
        return
    sel = idxs_in_window[::step]
    for i in sel:
        x = arr[i]
        if not np.isfinite(x):
            continue
        ehr_rows.append((int(t_0n[i]), int(var_id), float(x)))


push_0n(hr, VAR_IDS["HR_MON"], step=2)      # 1 Hz -> once per 2s * 2 = 4s
push_0n(spo2p, VAR_IDS["SPO2_MON"], step=2)
push_0n(resp, VAR_IDS["RESP_MON"], step=2)

print(f"Total EHR events: {len(ehr_rows)}")

# =========================================================
# 5. Split events into 4 partitions, compute seg_idx
# =========================================================
episode_start = wave_start_ms  # demo: treat as admission
episode_end = wave_end_ms
context_start = max(wave_start_ms - CONTEXT_WINDOW_MS, episode_start, wave_start_ms - BASELINE_CAP_MS)
future_end = min(episode_end, wave_end_ms + FUTURE_CAP_MS)

evs = np.array(ehr_rows, dtype=[("time_ms", "int64"), ("var_id", "uint16"), ("value", "float32")])
evs.sort(order="time_ms")

events_mask = (evs["time_ms"] >= wave_start_ms) & (evs["time_ms"] <= wave_end_ms)
recent_mask = (evs["time_ms"] >= max(wave_start_ms - CONTEXT_WINDOW_MS, episode_start)) & (evs["time_ms"] < wave_start_ms)
baseline_mask = (evs["time_ms"] >= episode_start) & (evs["time_ms"] < max(wave_start_ms - CONTEXT_WINDOW_MS, episode_start))
future_mask = (evs["time_ms"] > wave_end_ms) & (evs["time_ms"] <= min(episode_end, wave_end_ms + FUTURE_CAP_MS))


def pack(evs_subset, seg_idx_fill):
    out = np.zeros(len(evs_subset), dtype=EHR_EVENT_DTYPE)
    out["time_ms"] = evs_subset["time_ms"]
    out["var_id"] = evs_subset["var_id"]
    out["value"] = evs_subset["value"]
    if seg_idx_fill is None:
        out["seg_idx"] = np.searchsorted(time_ms, evs_subset["time_ms"], side="right") - 1
        out["seg_idx"] = np.clip(out["seg_idx"], 0, n_seg - 1)
    else:
        out["seg_idx"] = seg_idx_fill
    return out


baseline_pack = pack(evs[baseline_mask], SEG_IDX_BASELINE)
recent_pack = pack(evs[recent_mask], SEG_IDX_RECENT)
events_pack = pack(evs[events_mask], None)
future_pack = pack(evs[future_mask], SEG_IDX_FUTURE)

np.save(f"{ENTITY_DIR}/ehr_baseline.npy", baseline_pack)
np.save(f"{ENTITY_DIR}/ehr_recent.npy", recent_pack)
np.save(f"{ENTITY_DIR}/ehr_events.npy", events_pack)
np.save(f"{ENTITY_DIR}/ehr_future.npy", future_pack)
print(f"  partitions: baseline={len(baseline_pack)} recent={len(recent_pack)} events={len(events_pack)} future={len(future_pack)}")

assert events_pack["seg_idx"].min() >= 0 and events_pack["seg_idx"].max() < n_seg, "event seg_idx out of bounds"

# =========================================================
# 6. meta.json
# =========================================================
meta = {
    "entity_id": ENTITY_ID,
    "empi_nbr": EMPI,
    "encounter_nbr": ENC,
    "wfdb_record": REC,
    "source_dataset": "emory_sepsis",
    "n_segments": int(n_seg),
    "segment_duration_sec": SEG_SEC,
    "total_duration_hours": round(n_seg * SEG_SEC / 3600, 2),
    "wave_start_utc_ms": int(wave_start_ms),
    "wave_end_utc_ms": int(wave_end_ms),
    "episode_start_utc_ms": int(episode_start),
    "episode_end_utc_ms": int(episode_end),
    "channels": {
        "PLETH40": {"sample_rate_hz": PLETH_FS, "shape": list(pleth40.shape), "dtype": "float16", "source_channel": "SPO2", "source_fs_hz": WAVE_SRC_FS},
        "II120": {"sample_rate_hz": II_FS, "shape": list(ii120.shape), "dtype": "float16", "source_channel": "II", "source_fs_hz": WAVE_SRC_FS},
    },
    "ehr_layout_version": 2,
    "n_baseline": len(baseline_pack),
    "n_recent": len(recent_pack),
    "n_events": len(events_pack),
    "n_future": len(future_pack),
    "context_window_ms": CONTEXT_WINDOW_MS,
    "baseline_cap_ms": BASELINE_CAP_MS,
    "future_cap_ms": FUTURE_CAP_MS,
    "var_ids_used_in_demo": sorted(set(int(x) for x in evs["var_id"])),
    "note": "DEMO extraction — only first 8h segment (_0000) of full 25-day record used; var_id mapping is placeholder.",
}
with open(f"{ENTITY_DIR}/meta.json", "w") as f:
    json.dump(meta, f, indent=2, default=str)
print(f"\nWrote canonical entity dir: {ENTITY_DIR}")

# =========================================================
# 7. Plot — 30-min zoom from around the first cuff event
# =========================================================
print("\nRendering 30-min zoom plot ...")
# Pick zoom window: 30 min starting ~20 min into the wave
zoom_start_ms = wave_start_ms + 20 * 60 * 1000
zoom_end_ms = zoom_start_ms + 30 * 60 * 1000
zoom_start_dt = np.datetime64(zoom_start_ms, "ms")
zoom_end_dt = np.datetime64(zoom_end_ms, "ms")

# Find segment slice
seg_zoom_start = (zoom_start_ms - wave_start_ms) // (SEG_SEC * 1000)
seg_zoom_end = (zoom_end_ms - wave_start_ms) // (SEG_SEC * 1000)
pleth_slice = pleth40[seg_zoom_start:seg_zoom_end].reshape(-1)  # flatten
ii_slice = ii120[seg_zoom_start:seg_zoom_end].reshape(-1)

pleth_t = np.array(zoom_start_ms, dtype="datetime64[ms]") + np.arange(len(pleth_slice)) * np.timedelta64(int(1000 / PLETH_FS), "ms")
ii_t = np.array(zoom_start_ms, dtype="datetime64[ms]") + np.arange(len(ii_slice)) * np.timedelta64(int(1000 / II_FS), "ms")

# EHR events in zoom window
ev_in = events_pack[(events_pack["time_ms"] >= zoom_start_ms) & (events_pack["time_ms"] <= zoom_end_ms)]

fig, axes = plt.subplots(4, 1, figsize=(15, 11), sharex=True)

ax = axes[0]
ax.plot(pleth_t, pleth_slice, lw=0.5, color="steelblue")
ax.set_ylabel("PLETH40\n(a.u.)")
ax.grid(alpha=0.3)
ax.set_title(f"Encounter {ENTITY_ID} — 30 min zoom ({zoom_start_dt} to {zoom_end_dt} UTC)")

ax = axes[1]
ax.plot(ii_t, ii_slice, lw=0.4, color="darkgreen")
ax.set_ylabel("II120\n(mV)")
ax.grid(alpha=0.3)

# Panel 3: dense vitals (HR, SPO2-%, RESP)
ax = axes[2]
colors = {VAR_IDS["HR_MON"]: ("tab:red", "HR monitor"), VAR_IDS["SPO2_MON"]: ("tab:orange", "SPO2% monitor"), VAR_IDS["RESP_MON"]: ("tab:purple", "RESP monitor")}
for vid, (c, lbl) in colors.items():
    m = ev_in["var_id"] == vid
    if m.any():
        t = np.array(ev_in["time_ms"][m], dtype="datetime64[ms]")
        ax.scatter(t, ev_in["value"][m], s=6, color=c, label=lbl, alpha=0.6)
ax.set_ylabel("_0n dense\nvitals")
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.3)

# Panel 4: EHR chart events (cuff BP, pulse, temp, RR, SPO2)
ax = axes[3]
colors = {
    VAR_IDS["SBP_CUFF"]: ("tab:blue", "SBP_CUFF"),
    VAR_IDS["DBP_CUFF"]: ("tab:cyan", "DBP_CUFF"),
    VAR_IDS["MAP_CUFF"]: ("tab:olive", "MAP_CUFF"),
    VAR_IDS["HR_EHR"]: ("tab:red", "PULSE (EHR)"),
    VAR_IDS["TEMP_EHR"]: ("magenta", "TEMP"),
    VAR_IDS["SPO2_EHR"]: ("tab:orange", "SPO2 (EHR)"),
    VAR_IDS["RR_EHR"]: ("tab:purple", "RR (EHR)"),
}
for vid, (c, lbl) in colors.items():
    m = ev_in["var_id"] == vid
    if m.any():
        t = np.array(ev_in["time_ms"][m], dtype="datetime64[ms]")
        ax.scatter(t, ev_in["value"][m], s=40, color=c, marker="x", label=lbl)
ax.set_ylabel("EHR chart\nevents")
ax.set_xlabel("UTC time")
ax.legend(fontsize=8, loc="upper right", ncol=3)
ax.grid(alpha=0.3)

# Format x-axis as time
loc = mdates.AutoDateLocator(minticks=6, maxticks=10)
for ax in axes:
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
fig.autofmt_xdate(rotation=30)
fig.tight_layout()
out_plot = f"{OUT_ROOT}/demo_plot_30min.png"
fig.savefig(out_plot, dpi=110, bbox_inches="tight")
print(f"Saved plot: {out_plot}")

# Also a directory listing
print("\nEntity dir contents:")
for f in sorted(os.listdir(ENTITY_DIR)):
    path = f"{ENTITY_DIR}/{f}"
    size = os.path.getsize(path)
    print(f"  {f}  ({size:,} bytes)")
