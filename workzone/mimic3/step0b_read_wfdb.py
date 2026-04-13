#!/usr/bin/env python3
"""
Step 0b: Read WFDB headers from known MIMIC-III records.
Tries multiple record types to find channel names and sample rates.

Run: python workzone/mimic3/step0b_read_wfdb.py
"""
import os
import json
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO_ROOT / "workzone" / "outputs" / "mimic3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WAV_ROOT = "/labs/hulab/MIMIC_waveform_matched_subset/physionet.org/files/mimic3wdb-matched/1.0"

import wfdb

# Known files from exploration report
KNOWN_RECORDS = [
    # Segment records (should have signal info)
    "p00/p000020/3544749_0001",
    "p00/p000020/3544749_0002",
    "p00/p000020/3544749_0003",
    "p00/p000030/3524877_0001",
    "p01/p010013/3609176_0001",
    "p01/p010013/3609176_0002",
    "p01/p010023/3247161_0001",
    # Layout headers
    "p00/p000020/3544749_layout",
    "p00/p000030/3524877_layout",
    # Master multi-segment headers
    "p00/p000020/p000020-2183-04-28-17-47",
    "p00/p000030/p000030-2172-10-16-12-22",
    # Numerics headers
    "p00/p000020/p000020-2183-04-28-17-47n",
]

results = []
channels_seen = {}

for rel_path in KNOWN_RECORDS:
    full_path = os.path.join(WAV_ROOT, rel_path)
    hea_file = full_path + ".hea"

    if not os.path.exists(hea_file):
        results.append({"record": rel_path, "status": "FILE_NOT_FOUND"})
        print(f"  NOT FOUND: {rel_path}.hea")
        continue

    # First just read the raw .hea text
    try:
        with open(hea_file) as f:
            raw_header = f.read()
    except Exception as e:
        raw_header = f"READ ERROR: {e}"

    # Then try wfdb.rdheader
    try:
        h = wfdb.rdheader(full_path)
        info = {
            "record": rel_path,
            "status": "OK",
            "record_name": h.record_name,
            "n_sig": h.n_sig,
            "sig_name": h.sig_name,
            "fs": h.fs,
            "sig_len": h.sig_len,
            "units": h.units,
            "fmt": getattr(h, "fmt", None),
            "comments": getattr(h, "comments", None),
            "seg_name": getattr(h, "seg_name", None),
            "seg_len": getattr(h, "seg_len", None),
            "raw_header_first_3_lines": raw_header.split("\n")[:3],
        }

        if h.sig_name:
            for ch in h.sig_name:
                channels_seen[ch] = channels_seen.get(ch, 0) + 1
            duration = h.sig_len / h.fs if h.fs and h.sig_len else None
            print(f"  OK {rel_path}: channels={h.sig_name} fs={h.fs}Hz n_sig={h.n_sig} len={h.sig_len} dur={duration:.0f}s" if duration else f"  OK {rel_path}: channels={h.sig_name} fs={h.fs}Hz n_sig={h.n_sig}")
        else:
            print(f"  OK {rel_path}: n_sig={h.n_sig} fs={h.fs} (no sig_name -- multi-segment master?)")

        results.append(info)

    except Exception as e:
        results.append({
            "record": rel_path,
            "status": "ERROR",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "raw_header_first_3_lines": raw_header.split("\n")[:3],
        })
        print(f"  ERROR {rel_path}: {e}")

# Also try wfdb.rdrecord on one segment to see actual signal data
print("\n=== Trying wfdb.rdrecord on one segment ===")
try:
    rec = wfdb.rdrecord(os.path.join(WAV_ROOT, "p00/p000020/3544749_0001"))
    rec_info = {
        "sig_name": rec.sig_name,
        "fs": rec.fs,
        "n_sig": rec.n_sig,
        "sig_len": rec.sig_len,
        "units": rec.units,
        "p_signal_shape": list(rec.p_signal.shape) if rec.p_signal is not None else None,
        "d_signal_shape": list(rec.d_signal.shape) if rec.d_signal is not None else None,
        "fmt": rec.fmt,
    }
    print(f"  rdrecord OK: {rec_info}")
except Exception as e:
    rec_info = {"error": str(e), "traceback": traceback.format_exc()}
    print(f"  rdrecord ERROR: {e}")

# Summary
report = {
    "headers": results,
    "channels_seen": dict(sorted(channels_seen.items(), key=lambda x: -x[1])),
    "rdrecord_test": rec_info,
}

out_path = OUT_DIR / "wfdb_headers.json"
with open(out_path, "w") as f:
    json.dump(report, f, indent=2, default=str)

print(f"\nChannels seen: {channels_seen}")
print(f"Saved to: {out_path}")
