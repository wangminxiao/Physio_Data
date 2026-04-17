"""UCSF Step 0c: single-entity demo alignment.

- Picks WaveCycleUID 38286 of DE214688354794344 (2017-03 cohort).
- Concatenates all .adibin files in the wave cycle, resamples II->120 Hz and SPO2->40 Hz.
- Segments into 30-s windows with 5-s overlap (25-s stride) per skill canonical.
- Reads .vital files for the same session, maps suffix -> var_id, builds sparse events.
- Reads labs + meds from EHR text shards (2016_*.txt only), applies offset delta shift.
- Saves canonical dir demo_{pid}_{wavecycle}/ + overview plot.

Run on bedanalysis. Outputs under workzone/ucsf/explore/demo_{pid}_{wavecycle}/.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timedelta
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

sys.path.insert(0, "/labs/hulab/mxwang/data/ucsf_EHR/bedanalysis_waveformExtraction")
from binfilepy import BinFile  # type: ignore  # noqa: E402
from vitalfilepy.vitalfile import VitalFile  # type: ignore  # noqa: E402

# ---- config ---------------------------------------------------------------

PID_GE = "214688354794344"
PAT_ID_EHR = "214688334614038"
WAVECYCLE_UID = 38286
WYNTON = "2017-03-deid"
PAT_DIR = f"/labs/hulab/UCSF/{WYNTON}/DE{PID_GE}"
OUT_ROOT = Path(f"/labs/hulab/mxwang/Physio_Data/workzone/ucsf/explore/demo_{PID_GE}_{WAVECYCLE_UID}")

SRC_FS = 240            # .adibin
PLETH_FS = 40
ECG_FS = 120
SEG_SEC = 30
OVERLAP_SEC = 5
STRIDE_SEC = SEG_SEC - OVERLAP_SEC  # 25
PLETH_LEN = PLETH_FS * SEG_SEC       # 1200
ECG_LEN = ECG_FS * SEG_SEC           # 3600

# suffix -> var_id map (as proposed in datasets/ucsf/explore/README.md)
VITAL_SUFFIX_TO_VARID: dict[str, int] = {
    "HR":      100,
    "SPO2-%":  101,
    "RESP":    102,
    "TMP-1":   103,
    "NBP-S":   104,
    "NBP-D":   105,
    "NBP-M":   106,
    # CVP handled via CVP1/CVP2/CVP3 -> 107
    "CVP1":    107,
    "AR1-S":   110, "AR2-S": 110, "AR3-S": 110,  # ABPs (merged across lines)
    "AR1-D":   111, "AR2-D": 111, "AR3-D": 111,  # ABPd
    "AR1-M":   112, "AR2-M": 112, "AR3-M": 112,  # ABPm
    "AR1-R":   113, "AR2-R": 113, "AR3-R": 113,  # PR_art
    "PVC":     114,
    "SPO2-R":  115,
}
# ignore: CUFF (status), TMP-2 (secondary), ST-* (reserve), SP*, ICP*, CPP*, PA*

# lab name -> var_id (minimal subset for demo; substring match on Lab_Common_Name)
LAB_NAME_TO_VARID: dict[str, int] = {
    "POTASSIUM": 0,
    "CALCIUM": 1,  # avoid "CALCIUM IONIZED" mapping caveat -> keep simple for demo
    "SODIUM": 2,
    "GLUCOSE": 3,
    "LACTATE": 4, "LACTIC ACID": 4,
    "CREATININE": 5,
    "BILIRUBIN": 6,
    "PLATELET": 7,
    "WBC": 8,
    "HEMOGLOBIN": 9,
    "INR": 10, "INT'LNORMALIZRATIO": 10,
    "BUN": 11,
    "ALBUMIN": 12,
    "PH": 13,
    "PO2": 14,
    "PCO2": 15,
    "BICARBONATE": 16, "HCO3": 16,
}

EHR_DTYPE = np.dtype([
    ('time_ms', 'int64'),
    ('seg_idx', 'int32'),
    ('var_id',  'uint16'),
    ('value',   'float32'),
])

# ---- time helpers ---------------------------------------------------------

def dt_to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def parse_mrn_dt(s: str) -> datetime:
    # two formats observed
    for fmt in ("%m/%d/%Y %I:%M:%S %p", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"unrecognised datetime: {s}")


# ---- wave cycle window ---------------------------------------------------

def load_wave_cycle_window() -> tuple[datetime, datetime, str]:
    mrn = pd.read_csv(os.path.join(PAT_DIR, "MRN-Mapping.csv"), on_bad_lines="skip")
    mrn.columns = [c.strip() for c in mrn.columns]
    g = mrn[mrn["WaveCycleUID"] == WAVECYCLE_UID]
    bed_in  = min(parse_mrn_dt(s) for s in g["BedTransfer_In"])
    bed_out = max(parse_mrn_dt(s) for s in g["BedTransfer_Out"])
    w_start_raw = [parse_mrn_dt(s) for s in g["WaveStartTime"] if s]
    w_stop_raw  = [parse_mrn_dt(s) for s in g["WaveStopTime"]  if s]
    w_start = min(w_start_raw)
    w_stop = max(w_stop_raw)
    if w_stop < w_start:
        w_stop = bed_out  # known 1969 sentinel fallback
    valid_start = max(bed_in, w_start)
    valid_stop = min(bed_out, w_stop)
    unit_bed = g["UnitBed"].iloc[0] if "UnitBed" in g.columns else ""
    return valid_start, valid_stop, unit_bed


# ---- .adibin loading ------------------------------------------------------

def list_adibin_files() -> list[str]:
    hits = []
    for bed in sorted(os.listdir(PAT_DIR)):
        bed_p = os.path.join(PAT_DIR, bed)
        if not os.path.isdir(bed_p):
            continue
        for f in sorted(os.listdir(bed_p)):
            if f.endswith(f"_{WAVECYCLE_UID}.adibin"):
                hits.append(os.path.join(bed_p, f))
    return hits


def load_one_adibin(path: str, valid_start: datetime, valid_stop: datetime) -> dict | None:
    """Return {'start', 'samples_II', 'samples_SPO2'} clipped to valid window, or None."""
    with BinFile(path, "r") as fh:
        fh.readHeader()
        h = fh.header
        chans = [c.Title if c.Title else f"Unnamed{i}" for i, c in enumerate(fh.channels)]
        if "II" not in chans or "SPO2" not in chans:
            return None
        fs = 1.0 / h.secsPerTick
        assert int(fs) == SRC_FS, f"unexpected fs {fs}"
        file_start = datetime(h.Year, h.Month, h.Day, h.Hour, h.Minute,
                              int(h.Second), int((h.Second % 1) * 1e6))
        file_dur = timedelta(seconds=h.SamplesPerChannel * h.secsPerTick)
        file_end = file_start + file_dur
        if file_end < valid_start or file_start > valid_stop:
            return None
        clip_start = max(file_start, valid_start)
        clip_end = min(file_end, valid_stop)
        offset_sec = (clip_start - file_start).total_seconds()
        length_sec = (clip_end - clip_start).total_seconds()
        if length_sec <= 0:
            return None
        wav = fh.readChannelData_new(
            offset=offset_sec, length=length_sec,
            useSecForOffset=True, useSecForLength=True
        )
        i_ii = chans.index("II")
        i_sp = chans.index("SPO2")
        return {
            "clip_start": clip_start,
            "clip_end":   clip_end,
            "II":   np.asarray(wav[i_ii], dtype=np.float32),
            "SPO2": np.asarray(wav[i_sp], dtype=np.float32),
            "n_src": len(wav[i_ii]),
        }


def segments_from_clips(clips: list[dict]) -> dict:
    """Build [N_seg, samples_per_seg] arrays + time_ms from contiguous clips.

    Each clip becomes its own block (no spanning across gaps). Within a block,
    we produce segments with STRIDE_SEC stride. time_ms is segment-start (ms).
    """
    pleth_segs: list[np.ndarray] = []
    ecg_segs:   list[np.ndarray] = []
    starts_ms:  list[int] = []
    n_dropped_nan = 0
    for clip in clips:
        # resample: II 240->120 (1:2), SPO2 240->40 (1:6)
        ii_120 = resample_poly(clip["II"], up=1, down=2).astype(np.float32)
        sp_40  = resample_poly(clip["SPO2"], up=1, down=6).astype(np.float32)
        dur_sec = clip["n_src"] / SRC_FS
        # walk windows
        t0 = clip["clip_start"]
        w = 0.0
        while w + SEG_SEC <= dur_sec + 1e-6:
            i_p0 = int(round(w * PLETH_FS)); i_p1 = i_p0 + PLETH_LEN
            i_e0 = int(round(w * ECG_FS));   i_e1 = i_e0 + ECG_LEN
            if i_p1 > len(sp_40) or i_e1 > len(ii_120):
                break
            p_win = sp_40[i_p0:i_p1]
            e_win = ii_120[i_e0:i_e1]
            # NaN guard on PLETH anchor
            if np.mean(np.isfinite(p_win)) < 0.8:
                n_dropped_nan += 1
                w += STRIDE_SEC
                continue
            pleth_segs.append(p_win.astype(np.float16))
            ecg_segs.append(e_win.astype(np.float16))
            starts_ms.append(dt_to_ms(t0 + timedelta(seconds=w)))
            w += STRIDE_SEC
    if not pleth_segs:
        return {"PLETH40": np.zeros((0, PLETH_LEN), np.float16),
                "II120":   np.zeros((0, ECG_LEN), np.float16),
                "time_ms": np.zeros((0,), np.int64),
                "n_dropped_nan": n_dropped_nan}
    return {
        "PLETH40": np.ascontiguousarray(np.stack(pleth_segs, axis=0)),
        "II120":   np.ascontiguousarray(np.stack(ecg_segs, axis=0)),
        "time_ms": np.array(starts_ms, dtype=np.int64),
        "n_dropped_nan": n_dropped_nan,
    }


# ---- .vital events --------------------------------------------------------

def vital_events_for_session(valid_start: datetime) -> list[tuple[int, int, float]]:
    """Parse .vital files that belong to WAVECYCLE_UID.

    Many 2016+ cohort files have zero-timestamped headers (filename
    `..._00000000000000_..._.vital`). For those, we anchor offset_sec to
    `valid_start` of the wave cycle -- approximate but workable for sparse
    events at 0.5 Hz. Files with real headers use the header timestamp.
    """
    events: list[tuple[int, int, float]] = []
    n_zero = n_header = 0
    for bed in sorted(os.listdir(PAT_DIR)):
        bed_p = os.path.join(PAT_DIR, bed)
        if not os.path.isdir(bed_p):
            continue
        for f in sorted(os.listdir(bed_p)):
            if not f.endswith(".vital"):
                continue
            m = re.match(r"DE\d+_(\d{14})_(\d{5})_(.+)\.vital$", f)
            if not m:
                continue
            ts14, sess, sfx = m.group(1), m.group(2), m.group(3)
            if int(sess) != WAVECYCLE_UID:
                continue  # restrict to the picked wave cycle
            var_id = VITAL_SUFFIX_TO_VARID.get(sfx)
            if var_id is None:
                continue
            path = os.path.join(bed_p, f)
            try:
                with VitalFile(path, "r") as fh:
                    fh.readHeader()
                    h = fh.header
                    if ts14 == "00000000000000" or h.Year == 0:
                        file_start = valid_start
                        n_zero += 1
                    else:
                        file_start = datetime(h.Year, h.Month, h.Day, h.Hour, h.Minute,
                                              int(h.Second), int((h.Second % 1) * 1e6))
                        n_header += 1
                    raw = fh.readVitalDataBuf(fh.numSamplesInFile)
                for t in raw:
                    val = float(t[0]); off_sec = float(t[1])
                    if val <= -999999.0:
                        continue
                    ts = dt_to_ms(file_start + timedelta(seconds=off_sec))
                    events.append((ts, var_id, val))
            except Exception as e:
                print(f"  vital fail {path}: {e}")
                continue
    print(f"  vital anchor: header={n_header}, valid_start={n_zero}")
    return events


# ---- EHR events -----------------------------------------------------------

# simple comma-repair (from EHR_encounter_polars.py)
_BAD_NUM = re.compile(r"(?<=[^\d,])(\d{1,3}),(?=\d{3}(?!\d))")

def _clean(line: str) -> str:
    return _BAD_NUM.sub(r"\1", line)


def read_lab_events(lab_shards: list[str], pat_id: str,
                    shift_days: int, valid_start: datetime,
                    valid_stop: datetime) -> list[tuple[int, int, float]]:
    out: list[tuple[int, int, float]] = []
    for shard in lab_shards:
        with open(shard, "r", encoding="latin-1") as fh:
            header = fh.readline().rstrip("\n").split(",")
            try:
                i_cd = header.index("Lab_Collection_Date")
                i_ct = header.index("Lab_Collection_Time")
                i_val = header.index("Lab_Value")
                i_name = header.index("Lab_Common_Name")
                i_pat = header.index("Patient_ID")
            except ValueError:
                continue
            for line in fh:
                if pat_id not in line:
                    continue  # fast pre-filter
                parts = _clean(line).rstrip("\n").split(",")
                if len(parts) < len(header):
                    continue
                if parts[i_pat].strip('"') != pat_id:
                    continue
                name = parts[i_name].strip('"').upper()
                var_id = None
                for key, vid in LAB_NAME_TO_VARID.items():
                    if key in name:
                        var_id = vid
                        break
                if var_id is None:
                    continue
                try:
                    val = float(parts[i_val].strip('"').strip("%"))
                except ValueError:
                    continue
                date_s = parts[i_cd].strip('"')
                time_s = parts[i_ct].strip('"') or "00:00:00"
                dt = None
                for fmt in ("%m/%d/%Y %H:%M:%S.%f", "%m/%d/%Y %H:%M:%S",
                            "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
                    try:
                        dt = datetime.strptime(f"{date_s} {time_s}", fmt)
                        break
                    except ValueError:
                        continue
                if dt is None:
                    continue
                dt_shifted = dt + timedelta(days=shift_days)
                if dt_shifted < valid_start or dt_shifted > valid_stop:
                    continue
                out.append((dt_to_ms(dt_shifted), var_id, val))
    return out


# ---- main -----------------------------------------------------------------

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    valid_start, valid_stop, unit_bed = load_wave_cycle_window()
    print(f"[window] {valid_start} -> {valid_stop} ({unit_bed})")

    adibin_files = list_adibin_files()
    print(f"[adibin] {len(adibin_files)} files for WCUID {WAVECYCLE_UID}")

    # load and clip each adibin file
    clips = []
    for p in adibin_files:
        try:
            c = load_one_adibin(p, valid_start, valid_stop)
            if c is not None:
                clips.append(c)
        except Exception as e:
            print(f"  ! {p}: {e}")
    print(f"[adibin] {len(clips)} usable clips, total src samples {sum(c['n_src'] for c in clips)}")

    # segment + build signal arrays
    segs = segments_from_clips(clips)
    print(f"[seg] N_seg={segs['PLETH40'].shape[0]}, dropped_nan={segs['n_dropped_nan']}")
    print(f"[seg] time span ms: {segs['time_ms'][0] if len(segs['time_ms']) else '-'} "
          f"-> {segs['time_ms'][-1] if len(segs['time_ms']) else '-'}")

    # .vital events
    vital_events = vital_events_for_session(valid_start)
    print(f"[vital] {len(vital_events)} raw monitor events (before window clip)")
    # clip to valid window
    vs_ms, ve_ms = dt_to_ms(valid_start), dt_to_ms(valid_stop)
    vital_events = [(t, v, x) for (t, v, x) in vital_events if vs_ms <= t <= ve_ms]
    print(f"[vital] {len(vital_events)} in-window events")

    # EHR labs: scan 2016_*.txt only (encounter in 2016-06 shifted-EHR = 2016-06 also)
    lab_shards = sorted(glob("/labs/hulab/UCSF/rdb_new/Filtered_Lab_New/2016_*.txt"))
    # To transform EHR-shifted time -> GE-shifted (waveform) time, SUBTRACT (offset_GE - offset).
    # Walkthrough (this patient): offset=284, offset_GE=296, delta=12.
    # real_time = EHR_time - offset = GE_time - offset_GE
    # => GE_time = EHR_time - (offset_GE - offset) = EHR_time - 12 days.
    shift_days = -(296 - 284)
    print(f"[labs] scanning {len(lab_shards)} shards, EHR->wave shift = {shift_days}d")
    t_lab = time.time()
    lab_events = read_lab_events(lab_shards, PAT_ID_EHR, shift_days, valid_start, valid_stop)
    print(f"[labs] {len(lab_events)} in-window lab events  ({time.time()-t_lab:.1f}s)")

    # merge events, compute seg_idx
    all_ev = [(t, v, x) for (t, v, x) in vital_events]
    all_ev.extend(lab_events)
    all_ev.sort(key=lambda r: r[0])
    time_ms_arr = segs["time_ms"]
    if len(all_ev) and len(time_ms_arr):
        event_ts = np.array([e[0] for e in all_ev], dtype=np.int64)
        seg_ends = time_ms_arr + SEG_SEC * 1000
        seg_idx_per_event = np.searchsorted(time_ms_arr, event_ts, side="right") - 1
        seg_idx_per_event = np.clip(seg_idx_per_event, -1, len(time_ms_arr) - 1)
        in_any_seg = (seg_idx_per_event >= 0) & (event_ts <= seg_ends[seg_idx_per_event])
        seg_idx_per_event = np.where(in_any_seg, seg_idx_per_event, -1)
    else:
        seg_idx_per_event = np.zeros(len(all_ev), dtype=np.int32)

    ehr_events = np.zeros(len(all_ev), dtype=EHR_DTYPE)
    for i, (ts, var, val) in enumerate(all_ev):
        ehr_events[i]["time_ms"] = ts
        ehr_events[i]["seg_idx"] = int(seg_idx_per_event[i]) if len(seg_idx_per_event) else -1
        ehr_events[i]["var_id"] = var
        ehr_events[i]["value"] = val
    print(f"[events] {len(ehr_events)} total; "
          f"{int((ehr_events['seg_idx']>=0).sum())} inside a signal segment")

    # save canonical
    np.save(OUT_ROOT / "PLETH40.npy", segs["PLETH40"])
    np.save(OUT_ROOT / "II120.npy",   segs["II120"])
    np.save(OUT_ROOT / "time_ms.npy", segs["time_ms"])
    np.save(OUT_ROOT / "ehr_events.npy", ehr_events)
    meta = {
        "dataset": "ucsf",
        "entity_id": f"{PID_GE}_{WAVECYCLE_UID}",
        "patient_id_ge": PID_GE,
        "patient_id_ehr": PAT_ID_EHR,
        "encounter_id": "703726390376687",
        "wavecycle_uid": WAVECYCLE_UID,
        "unit_bed": unit_bed,
        "wave_window_start_ms": dt_to_ms(valid_start),
        "wave_window_end_ms":   dt_to_ms(valid_stop),
        "n_seg": int(segs["PLETH40"].shape[0]),
        "seg_sec": SEG_SEC,
        "overlap_sec": OVERLAP_SEC,
        "channels": {
            "PLETH40": {"rate_hz": PLETH_FS, "samples_per_seg": PLETH_LEN, "src": "adibin.SPO2"},
            "II120":   {"rate_hz": ECG_FS,   "samples_per_seg": ECG_LEN,   "src": "adibin.II"},
        },
        "event_counts": {
            "total": len(ehr_events),
            "vital_monitor": len(vital_events),
            "labs": len(lab_events),
        },
        "offset_shift_days_ehr_to_wave": shift_days,
    }
    with open(OUT_ROOT / "meta.json", "w") as fh:
        json.dump(meta, fh, indent=2, default=str)

    # plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def make_plot(out_path: Path, window_hours: float, title_suffix: str):
        if not len(segs["time_ms"]):
            return
        t0_ms = segs["time_ms"][0]
        t_end_ms = min(segs["time_ms"][-1], t0_ms + int(window_hours * 3600 * 1000))
        seg_mask = (segs["time_ms"] >= t0_ms) & (segs["time_ms"] <= t_end_ms)
        idxs = np.where(seg_mask)[0]
        if not len(idxs):
            return

        # Decimate signals for display to keep plot size reasonable.
        # For 4-day view, full PLETH = 40 Hz * 4d * 86400 = 13.8M samples -> too many.
        # Downsample PLETH to ~0.1 Hz (1/400), ECG to ~0.1 Hz (1/1200), by taking every Nth.
        pleth = segs["PLETH40"][idxs].astype(np.float32).reshape(-1)
        ecg   = segs["II120"][idxs].astype(np.float32).reshape(-1)
        step_p = max(1, len(pleth) // 60000)  # cap ~60k pts
        step_e = max(1, len(ecg)   // 60000)
        t_pleth = np.linspace(0, (t_end_ms - t0_ms) / 1000.0, len(pleth), endpoint=False)[::step_p]
        t_ecg   = np.linspace(0, (t_end_ms - t0_ms) / 1000.0, len(ecg),   endpoint=False)[::step_e]

        # Robust y-limit (5th-95th percentile) for signals so outliers don't flatten.
        def robust_lim(x):
            lo, hi = np.nanpercentile(x, [5, 95])
            pad = 0.3 * (hi - lo)
            return lo - pad, hi + pad

        fig, axes = plt.subplots(5, 1, figsize=(16, 12), sharex=True)
        axes[0].plot(t_pleth / 3600.0, pleth[::step_p], lw=0.3, color="tab:blue")
        axes[0].set_ylim(*robust_lim(pleth[::step_p]))
        axes[0].set_ylabel("PLETH40\n(ADC units)")
        axes[0].set_title(f"UCSF demo: {PID_GE} WCUID={WAVECYCLE_UID} "
                          f"({valid_start.isoformat()} — {title_suffix})")
        axes[1].plot(t_ecg / 3600.0, ecg[::step_e], lw=0.3, color="tab:red")
        axes[1].set_ylim(*robust_lim(ecg[::step_e]))
        axes[1].set_ylabel("II120\n(ADC units)")

        def pick_events(var_ids_set):
            m = np.isin(ehr_events["var_id"], list(var_ids_set)) & (ehr_events["time_ms"] <= t_end_ms)
            t_ev = (ehr_events["time_ms"][m] - t0_ms) / 3600_000.0  # hours
            v_ev = ehr_events["value"][m]
            i_ev = ehr_events["var_id"][m]
            return t_ev, v_ev, i_ev

        colors_abp = {110: "tab:blue", 111: "tab:brown", 112: "tab:cyan"}
        t_abp, v_abp, i_abp = pick_events({110, 111, 112})
        for vid in (110, 111, 112):
            sel = i_abp == vid
            axes[2].scatter(t_abp[sel], v_abp[sel], s=2, c=colors_abp[vid],
                            alpha=0.5, label={110:"ABPs", 111:"ABPd", 112:"ABPm"}[vid])
        axes[2].set_ylabel("ABP (mmHg)")
        axes[2].set_ylim(0, 350)
        axes[2].legend(loc="upper right", markerscale=3)

        colors_v = {100: "tab:red", 101: "tab:green", 102: "tab:orange"}
        t_v, v_v, i_v = pick_events({100, 101, 102})
        for vid in (100, 101, 102):
            sel = i_v == vid
            axes[3].scatter(t_v[sel], v_v[sel], s=2, c=colors_v[vid],
                            alpha=0.5, label={100:"HR", 101:"SpO2", 102:"RR"}[vid])
        axes[3].set_ylabel("HR / SpO2 / RR")
        axes[3].set_ylim(0, 220)
        axes[3].legend(loc="upper right", markerscale=3)

        t_lab, v_lab, i_lab = pick_events(set(range(0, 100)))
        if len(t_lab):
            axes[4].scatter(t_lab, v_lab, s=30, c=i_lab, cmap="tab20",
                            alpha=0.9, edgecolor="black", linewidth=0.4)
            for tt, vv, ii in zip(t_lab, v_lab, i_lab):
                axes[4].annotate(f"id{ii}", (tt, vv), fontsize=7, alpha=0.8,
                                  xytext=(3, 3), textcoords="offset points")
        axes[4].set_ylabel(f"lab values\n(n={len(t_lab)})")
        axes[4].set_xlabel("hours from wave start")
        plt.tight_layout()
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        print(f"[plot] wrote {out_path}")

    make_plot(OUT_ROOT / "overview_first_2h.png", 2.0, "first 2 h")
    make_plot(OUT_ROOT / "overview_full.png",
              (segs["time_ms"][-1] - segs["time_ms"][0]) / 3600_000.0 + 0.1,
              "full wave cycle")

    print(f"[done] total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
