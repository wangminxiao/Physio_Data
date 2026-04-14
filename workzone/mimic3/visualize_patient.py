#!/usr/bin/env python3
"""
Visualize waveform + EHR events for one patient.

Generates a multi-panel figure:
  - Panel 1: PLETH40 waveform (30s detail window)
  - Panel 2: II120 waveform (same 30s window)
  - Panel 3: Full-recording EHR event timeline (all variables, color by category)
  - Panel 4: Waveform signal envelope (full recording overview)

Usage:
  python workzone/mimic3/visualize_patient.py <patient_dir>
  python workzone/mimic3/visualize_patient.py <patient_dir> --hour 3.5
  python workzone/mimic3/visualize_patient.py <patient_dir> --seg 100

Output: saves PNG to workzone/outputs/mimic3/vis_{patient_id}.png
"""
import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO_ROOT / "workzone" / "outputs" / "mimic3"

# Load var_registry for variable names
with open(REPO_ROOT / "indices" / "var_registry.json") as f:
    VAR_REGISTRY = json.load(f)

VAR_NAMES = {v["id"]: v["name"] for v in VAR_REGISTRY["variables"]}
VAR_CATS = {v["id"]: v["category"] for v in VAR_REGISTRY["variables"]}

# Colors by category
CAT_COLORS = {
    "lab":    "#2196F3",  # blue
    "vital":  "#4CAF50",  # green
    "action": "#FF9800",  # orange
    "score":  "#F44336",  # red
}
CAT_MARKERS = {
    "lab":    "o",
    "vital":  "s",
    "action": "D",
    "score":  "^",
}


def load_patient(patient_dir):
    """Load all patient data from canonical format."""
    meta = json.load(open(os.path.join(patient_dir, "meta.json")))
    time_ms = np.load(os.path.join(patient_dir, "time_ms.npy"))
    ehr_events = np.load(os.path.join(patient_dir, "ehr_events.npy"))

    channels = {}
    for ch_name in meta.get("channels", {}):
        path = os.path.join(patient_dir, f"{ch_name}.npy")
        if os.path.exists(path):
            channels[ch_name] = np.load(path, mmap_mode='r')

    return meta, time_ms, ehr_events, channels


def time_ms_to_hours(time_ms, ref_ms):
    """Convert absolute ms to hours relative to recording start."""
    return (time_ms - ref_ms) / 3600000.0


def plot_patient(patient_dir, detail_seg=None, detail_hour=None, out_path=None):
    meta, time_ms, ehr_events, channels = load_patient(patient_dir)
    patient_id = meta.get("patient_id", os.path.basename(patient_dir))
    n_seg = len(time_ms)
    ref_ms = int(time_ms[0])
    stride_sec = meta.get("stride_sec", 25)

    hours = time_ms_to_hours(time_ms, ref_ms)
    total_hours = hours[-1] + meta.get("segment_duration_sec", 30) / 3600

    # Determine detail segment
    if detail_hour is not None:
        detail_seg = int(detail_hour * 3600 / stride_sec)
    if detail_seg is None:
        # Pick a segment near the middle that has EHR events nearby
        if len(ehr_events) > 0:
            mid_idx = len(ehr_events) // 2
            detail_seg = int(ehr_events[mid_idx]['seg_idx'])
        else:
            detail_seg = n_seg // 2
    detail_seg = max(0, min(detail_seg, n_seg - 1))

    # ---------- Figure setup ----------
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f"Patient {patient_id}  |  {n_seg} segments  |  {total_hours:.1f} hours  |  "
                 f"{len(ehr_events)} EHR events", fontsize=13, fontweight='bold')

    gs = fig.add_gridspec(4, 1, height_ratios=[1.5, 1.5, 2, 1.2], hspace=0.35)

    # ---------- Panel 1: PLETH detail ----------
    ax1 = fig.add_subplot(gs[0])
    if "PLETH40" in channels:
        pleth = channels["PLETH40"]
        seg_data = pleth[detail_seg].astype(np.float32)
        seg_time = np.linspace(0, meta.get("segment_duration_sec", 30), len(seg_data))
        nan_pct = np.isnan(seg_data).mean() * 100

        if nan_pct < 100:
            ax1.plot(seg_time, seg_data, color='#1565C0', linewidth=0.5, alpha=0.9)
        ax1.set_ylabel("PLETH40\n(NU)", fontsize=10)
        ax1.set_title(f"Segment {detail_seg}  (t = {hours[detail_seg]:.2f} h)  |  "
                      f"NaN: {nan_pct:.0f}%", fontsize=10, loc='left')
    else:
        ax1.text(0.5, 0.5, "PLETH40 not found", transform=ax1.transAxes, ha='center')
    ax1.set_xlabel("Time within segment (s)", fontsize=9)

    # ---------- Panel 2: II detail ----------
    ax2 = fig.add_subplot(gs[1])
    if "II120" in channels:
        ii = channels["II120"]
        seg_data = ii[detail_seg].astype(np.float32)
        seg_time = np.linspace(0, meta.get("segment_duration_sec", 30), len(seg_data))
        nan_pct = np.isnan(seg_data).mean() * 100

        if nan_pct < 100:
            ax2.plot(seg_time, seg_data, color='#C62828', linewidth=0.3, alpha=0.9)
        ax2.set_ylabel("II120\n(mV)", fontsize=10)
        title_str = f"NaN: {nan_pct:.0f}%"
        if nan_pct == 100:
            title_str += "  (ECG absent in this segment)"
        ax2.set_title(title_str, fontsize=10, loc='left')
    else:
        ax2.text(0.5, 0.5, "II120 not found", transform=ax2.transAxes, ha='center')
    ax2.set_xlabel("Time within segment (s)", fontsize=9)

    # ---------- Panel 3: EHR event timeline ----------
    ax3 = fig.add_subplot(gs[2])

    if len(ehr_events) > 0:
        event_hours = time_ms_to_hours(ehr_events['time_ms'].astype(np.float64), ref_ms)
        var_ids = ehr_events['var_id']
        values = ehr_events['value']

        # Group by category for plotting
        plotted_cats = set()
        for cat in ["lab", "vital", "action", "score"]:
            cat_mask = np.array([VAR_CATS.get(int(vid), "") == cat for vid in var_ids])
            if not np.any(cat_mask):
                continue
            plotted_cats.add(cat)

            cat_hours = event_hours[cat_mask]
            cat_vids = var_ids[cat_mask]

            # Y-axis: variable ID (gives natural grouping)
            ax3.scatter(cat_hours, cat_vids.astype(np.float32),
                       c=CAT_COLORS[cat], marker=CAT_MARKERS[cat],
                       s=12, alpha=0.6, linewidths=0, label=cat)

        # Mark the detail segment time range
        seg_start_h = hours[detail_seg]
        seg_end_h = seg_start_h + meta.get("segment_duration_sec", 30) / 3600
        ax3.axvspan(seg_start_h, seg_end_h, alpha=0.15, color='gray', label='detail window')

        ax3.set_xlabel("Time (hours from recording start)", fontsize=10)
        ax3.set_ylabel("var_id", fontsize=10)
        ax3.set_title(f"EHR Events ({len(ehr_events)} total)", fontsize=10, loc='left')

        # Y-axis labels: show variable names for the var_ids that appear
        unique_vids = sorted(set(int(v) for v in var_ids))
        if len(unique_vids) <= 40:
            ax3.set_yticks(unique_vids)
            ax3.set_yticklabels([f"{vid} {VAR_NAMES.get(vid, '?')}" for vid in unique_vids],
                               fontsize=7)
        ax3.legend(loc='upper right', fontsize=8, ncol=len(plotted_cats) + 1)
    else:
        ax3.text(0.5, 0.5, "No EHR events", transform=ax3.transAxes, ha='center')

    # ---------- Panel 4: Waveform envelope (full recording overview) ----------
    ax4 = fig.add_subplot(gs[3])

    if "PLETH40" in channels:
        pleth = channels["PLETH40"]
        # Compute per-segment stats (subsample for speed)
        step = max(1, n_seg // 2000)
        seg_indices = np.arange(0, n_seg, step)
        seg_hours = hours[seg_indices]

        means = []
        stds = []
        nan_fracs = []
        for idx in seg_indices:
            seg = pleth[idx].astype(np.float32)
            nf = np.isnan(seg).mean()
            nan_fracs.append(nf)
            if nf < 1.0:
                valid = seg[~np.isnan(seg)]
                means.append(np.mean(valid))
                stds.append(np.std(valid))
            else:
                means.append(np.nan)
                stds.append(np.nan)

        means = np.array(means)
        stds = np.array(stds)
        nan_fracs = np.array(nan_fracs)

        valid = ~np.isnan(means)
        if np.any(valid):
            ax4.fill_between(seg_hours[valid], means[valid] - stds[valid],
                            means[valid] + stds[valid], alpha=0.3, color='#1565C0')
            ax4.plot(seg_hours[valid], means[valid], color='#1565C0', linewidth=0.5)

        # Mark gaps (large time_ms jumps)
        if n_seg > 1:
            diffs = np.diff(time_ms)
            gap_mask = diffs > stride_sec * 1000 * 1.5
            gap_indices = np.where(gap_mask)[0]
            for gi in gap_indices:
                gap_h = hours[gi]
                ax4.axvline(gap_h, color='red', alpha=0.4, linewidth=1, linestyle='--')

        # Mark detail window
        ax4.axvline(hours[detail_seg], color='gray', linewidth=2, alpha=0.5)

        ax4.set_ylabel("PLETH40\nenvelope", fontsize=10)
        ax4.set_xlabel("Time (hours from recording start)", fontsize=10)
        gap_count = meta.get("n_gaps", 0)
        ax4.set_title(f"Signal overview  |  {gap_count} gaps (red dashes)  |  "
                      f"gray line = detail segment", fontsize=10, loc='left')

    # ---------- Save ----------
    if out_path is None:
        out_path = OUT_DIR / f"vis_{patient_id}.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Print summary
    print(f"\n--- {patient_id} ---")
    print(f"Segments: {n_seg}, Duration: {total_hours:.1f}h, EHR events: {len(ehr_events)}")
    if len(ehr_events) > 0:
        for cat in ["lab", "vital", "action", "score"]:
            cat_mask = np.array([VAR_CATS.get(int(vid), "") == cat for vid in ehr_events['var_id']])
            n = int(np.sum(cat_mask))
            if n > 0:
                vids = set(int(v) for v in ehr_events['var_id'][cat_mask])
                names = [VAR_NAMES.get(v, f"?{v}") for v in sorted(vids)]
                print(f"  {cat:7s}: {n:5d} events  ({', '.join(names)})")
    for ch_name in sorted(channels.keys()):
        arr = channels[ch_name]
        nan_pct = np.isnan(arr.astype(np.float32)).mean() * 100
        print(f"  {ch_name}: shape={arr.shape}, NaN={nan_pct:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Visualize one patient's waveform + EHR")
    parser.add_argument("patient_dir", help="Path to patient directory")
    parser.add_argument("--seg", type=int, default=None, help="Detail segment index")
    parser.add_argument("--hour", type=float, default=None, help="Detail time (hours from start)")
    parser.add_argument("--out", type=str, default=None, help="Output PNG path")
    args = parser.parse_args()

    if not os.path.isdir(args.patient_dir):
        print(f"Not a directory: {args.patient_dir}")
        sys.exit(1)

    plot_patient(args.patient_dir, detail_seg=args.seg, detail_hour=args.hour, out_path=args.out)


if __name__ == "__main__":
    main()
