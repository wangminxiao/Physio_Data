#!/usr/bin/env python3
"""
Step 0: Explore MIMIC-III data structure on the remote server.

Run from the Physio_Data repo root:
    python workzone/mimic3/step0_explore.py

Outputs go to workzone/outputs/mimic3/
"""
import os
import sys
import json
import glob
from pathlib import Path

# Resolve paths relative to repo root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO_ROOT / "workzone" / "outputs" / "mimic3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Data paths (edit if different on your server) ----
WAV_ROOT = "/labs/hulab/MIMIC_waveform_matched_subset/physionet.org/files/mimic3wdb-matched/1.0"
EHR_ROOT = "/labs/hulab/MIMICIII-v1.4"

# Paths to check for existing processed data
EXISTING_PATHS = [
    "/opt/localdata100tb/UNIPHY_Plus/dataset/EST/MIMIC3_SPO2_I_40hz_v3_lab_vital",
    "/opt/localdata100tb/UNIPHY_Plus/dataset/EST/MIMIC3_SPO2_I_40hz_v3",
]


def explore_waveform_dir():
    """List waveform directory structure and find sample records."""
    print("=== 1. Waveform Directory ===")
    result = {"root": WAV_ROOT, "exists": os.path.isdir(WAV_ROOT)}

    if not result["exists"]:
        print(f"  NOT FOUND: {WAV_ROOT}")
        return result

    # Top-level subdirectories
    subdirs = sorted([d for d in os.listdir(WAV_ROOT) if os.path.isdir(os.path.join(WAV_ROOT, d))])
    result["top_level_dirs"] = subdirs[:20]
    result["total_top_dirs"] = len(subdirs)
    print(f"  Found {len(subdirs)} top-level dirs: {subdirs[:5]}...")

    # Find some .hea files
    hea_files = []
    for sd in subdirs[:3]:
        for root, dirs, files in os.walk(os.path.join(WAV_ROOT, sd)):
            for f in files:
                if f.endswith(".hea"):
                    hea_files.append(os.path.join(root, f))
                if len(hea_files) >= 10:
                    break
            if len(hea_files) >= 10:
                break
    result["sample_hea_files"] = hea_files[:10]
    print(f"  Found {len(hea_files)} sample .hea files")

    return result


def inspect_wfdb_headers():
    """Read a few WFDB headers to understand channels and sample rates."""
    print("\n=== 2. WFDB Header Inspection ===")

    try:
        import wfdb
    except ImportError:
        print("  ERROR: wfdb not installed. Run: pip install wfdb")
        return {"error": "wfdb not installed"}

    # Find .hea files (skip layout/master headers which have '_layout' or 'RECORDS')
    hea_files = []
    for sd in sorted(os.listdir(WAV_ROOT))[:5]:
        sd_path = os.path.join(WAV_ROOT, sd)
        if not os.path.isdir(sd_path):
            continue
        for root, dirs, files in os.walk(sd_path):
            for f in files:
                if f.endswith(".hea") and "_layout" not in f and "RECORDS" not in f:
                    hea_files.append(os.path.join(root, f))
                if len(hea_files) >= 20:
                    break
            if len(hea_files) >= 20:
                break

    headers = []
    channels_seen = {}

    for hea in hea_files[:10]:
        try:
            h = wfdb.rdheader(hea.replace(".hea", ""))
            info = {
                "file": hea,
                "record_name": h.record_name,
                "sig_name": h.sig_name,
                "fs": h.fs,
                "sig_len": h.sig_len,
                "n_sig": h.n_sig,
                "units": h.units,
                "duration_sec": h.sig_len / h.fs if h.fs else None,
            }
            headers.append(info)

            # Track channel frequency
            for ch in h.sig_name:
                channels_seen[ch] = channels_seen.get(ch, 0) + 1

            print(f"  {h.record_name}: {h.sig_name} @ {h.fs}Hz, {info['duration_sec']:.0f}s")
        except Exception as e:
            print(f"  SKIP {hea}: {e}")

    result = {
        "headers_inspected": len(headers),
        "headers": headers,
        "channel_frequency": dict(sorted(channels_seen.items(), key=lambda x: -x[1])),
    }
    print(f"  Channels seen: {result['channel_frequency']}")
    return result


def inspect_ehr_tables():
    """Check EHR CSV structure."""
    print("\n=== 3. EHR Tables ===")

    try:
        import pandas as pd
    except ImportError:
        print("  ERROR: pandas not installed")
        return {"error": "pandas not installed"}

    tables = {
        "LABEVENTS.csv.gz": "Lab measurements",
        "CHARTEVENTS.csv.gz": "Charted vitals/observations",
        "PATIENTS.csv": "Patient demographics",
        "ADMISSIONS.csv": "Hospital admissions",
        "DIAGNOSES_ICD.csv": "ICD diagnoses",
    }

    result = {}
    for fname, desc in tables.items():
        path = os.path.join(EHR_ROOT, fname)
        if not os.path.exists(path):
            result[fname] = {"error": f"not found at {path}"}
            print(f"  MISSING: {fname}")
            continue

        try:
            size_mb = os.path.getsize(path) / 1e6
            df = pd.read_csv(path, nrows=3)
            info = {
                "description": desc,
                "size_mb": round(size_mb, 1),
                "columns": list(df.columns),
                "dtypes": {c: str(d) for c, d in df.dtypes.items()},
                "sample_row": {c: str(v) for c, v in df.iloc[0].items()},
            }
            result[fname] = info
            print(f"  OK: {fname} ({size_mb:.0f} MB) - cols: {list(df.columns)}")
        except Exception as e:
            result[fname] = {"error": str(e)}
            print(f"  ERROR: {fname}: {e}")

    return result


def check_existing_processed():
    """Check for existing processed data from previous pipelines."""
    print("\n=== 4. Existing Processed Data ===")
    result = {}

    for path in EXISTING_PATHS:
        if os.path.isdir(path):
            files = os.listdir(path)
            total_size = sum(
                os.path.getsize(os.path.join(path, f))
                for f in files
                if os.path.isfile(os.path.join(path, f))
            )
            result[path] = {
                "exists": True,
                "n_files": len(files),
                "total_size_gb": round(total_size / 1e9, 2),
                "sample_files": files[:10],
            }
            print(f"  FOUND: {path} ({len(files)} files, {total_size/1e9:.1f} GB)")
        else:
            result[path] = {"exists": False}
            print(f"  NOT FOUND: {path}")

    return result


def check_packages():
    """Check available Python packages."""
    print("\n=== 5. Python Packages ===")
    packages = {}
    for pkg in ["numpy", "pandas", "scipy", "wfdb", "polars", "pyarrow", "torch"]:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "unknown")
            packages[pkg] = ver
            print(f"  {pkg}: {ver}")
        except ImportError:
            packages[pkg] = None
            print(f"  {pkg}: NOT INSTALLED")
    return packages


def main():
    print(f"Physio_Data Step 0: MIMIC-III Exploration")
    print(f"Server: {os.uname().nodename}")
    print(f"Output: {OUT_DIR}\n")

    report = {
        "server": os.uname().nodename,
        "waveform": explore_waveform_dir(),
        "wfdb_headers": inspect_wfdb_headers(),
        "ehr_tables": inspect_ehr_tables(),
        "existing_processed": check_existing_processed(),
        "packages": check_packages(),
    }

    out_path = OUT_DIR / "exploration_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n=== Done ===")
    print(f"Report saved to: {out_path}")
    print(f"\nNext: git add workzone/outputs/ && git commit -m 'MIMIC-III exploration' && git push")


if __name__ == "__main__":
    main()
