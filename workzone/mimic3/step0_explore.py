#!/usr/bin/env python3
"""
Step 0: Explore MIMIC-III data structure on the remote server.

Run from the Physio_Data repo root:
    python workzone/mimic3/step0_explore.py

Outputs go to workzone/outputs/mimic3/
"""
import os
import json
import traceback
from pathlib import Path

# Resolve paths relative to repo root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO_ROOT / "workzone" / "outputs" / "mimic3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Data paths (edit if different on your server) ----
WAV_ROOT = "/labs/hulab/MIMIC_waveform_matched_subset/physionet.org/files/mimic3wdb-matched/1.0"
EHR_ROOT = "/labs/hulab/MIMICIII-v1.4"


# ==========================================================================
# 1. Waveform directory structure
# ==========================================================================
def explore_waveform_dir():
    print("=== 1. Waveform Directory ===")
    result = {"root": WAV_ROOT, "exists": os.path.isdir(WAV_ROOT)}
    if not result["exists"]:
        print(f"  NOT FOUND: {WAV_ROOT}")
        return result

    # Top-level dirs (p00, p01, ...)
    subdirs = sorted(d for d in os.listdir(WAV_ROOT) if os.path.isdir(os.path.join(WAV_ROOT, d)))
    result["top_level_dirs"] = subdirs
    result["total_top_dirs"] = len(subdirs)
    print(f"  {len(subdirs)} top-level dirs: {subdirs[:5]}...")

    # Walk into first few patient dirs and list ALL files
    sample_patients = []
    for sd in subdirs[:2]:
        sd_path = os.path.join(WAV_ROOT, sd)
        for patient_dir in sorted(os.listdir(sd_path))[:3]:
            patient_path = os.path.join(sd_path, patient_dir)
            if not os.path.isdir(patient_path):
                continue
            files = sorted(os.listdir(patient_path))
            sample_patients.append({
                "path": patient_path,
                "patient_id": patient_dir,
                "files": files,
                "n_files": len(files),
            })
            print(f"  {patient_dir}: {len(files)} files -> {files[:5]}...")

    result["sample_patients"] = sample_patients

    # Count total patients across all top-level dirs
    total_patients = 0
    for sd in subdirs:
        sd_path = os.path.join(WAV_ROOT, sd)
        total_patients += sum(1 for d in os.listdir(sd_path) if os.path.isdir(os.path.join(sd_path, d)))
    result["total_patients"] = total_patients
    print(f"  Total patient directories: {total_patients}")

    return result


# ==========================================================================
# 2. WFDB header inspection -- try multiple records, report all errors
# ==========================================================================
def inspect_wfdb_headers():
    print("\n=== 2. WFDB Header Inspection ===")
    try:
        import wfdb
    except ImportError:
        print("  ERROR: wfdb not installed")
        return {"error": "wfdb not installed"}

    # Collect candidate .hea files from several patients
    candidates = []
    for sd in sorted(os.listdir(WAV_ROOT))[:3]:
        sd_path = os.path.join(WAV_ROOT, sd)
        if not os.path.isdir(sd_path):
            continue
        for patient_dir in sorted(os.listdir(sd_path))[:5]:
            patient_path = os.path.join(sd_path, patient_dir)
            if not os.path.isdir(patient_path):
                continue
            for f in sorted(os.listdir(patient_path)):
                if f.endswith(".hea"):
                    candidates.append(os.path.join(patient_path, f))

    print(f"  Found {len(candidates)} .hea files to try")

    headers = []
    errors = []
    channels_seen = {}

    for hea_path in candidates[:30]:
        record_path = hea_path[:-4]  # strip .hea
        fname = os.path.basename(hea_path)

        # Skip layout headers
        if "_layout" in fname:
            continue

        try:
            h = wfdb.rdheader(record_path)

            # Skip if no signals (master multi-segment headers)
            if h.n_sig is None or h.n_sig == 0:
                errors.append({"file": fname, "error": "n_sig is None/0 (master header?)"})
                continue
            if h.sig_name is None:
                errors.append({"file": fname, "error": "sig_name is None"})
                continue

            info = {
                "file": hea_path,
                "record_name": h.record_name,
                "sig_name": h.sig_name,
                "fs": h.fs,
                "sig_len": h.sig_len,
                "n_sig": h.n_sig,
                "units": h.units,
                "duration_sec": round(h.sig_len / h.fs, 1) if h.fs and h.sig_len else None,
            }
            headers.append(info)
            for ch in h.sig_name:
                channels_seen[ch] = channels_seen.get(ch, 0) + 1
            print(f"  OK {fname}: {h.sig_name} @ {h.fs}Hz, {info['duration_sec']}s")

        except Exception as e:
            errors.append({"file": fname, "error": str(e), "traceback": traceback.format_exc()})
            print(f"  FAIL {fname}: {e}")

    result = {
        "candidates_found": len(candidates),
        "headers_inspected": len(headers),
        "headers": headers[:15],
        "errors": errors[:10],
        "channel_frequency": dict(sorted(channels_seen.items(), key=lambda x: -x[1])),
    }
    print(f"\n  Summary: {len(headers)} succeeded, {len(errors)} failed/skipped")
    print(f"  Channels seen: {result['channel_frequency']}")
    return result


# ==========================================================================
# 3. EHR tables -- list ALL files first, then try to read known tables
# ==========================================================================
def inspect_ehr_tables():
    print("\n=== 3. EHR Tables ===")

    result = {"root": EHR_ROOT, "exists": os.path.isdir(EHR_ROOT)}
    if not result["exists"]:
        print(f"  NOT FOUND: {EHR_ROOT}")
        return result

    # First: list everything in the directory
    all_files = sorted(os.listdir(EHR_ROOT))
    result["all_files"] = all_files
    print(f"  Directory listing ({len(all_files)} files):")
    for f in all_files:
        size = os.path.getsize(os.path.join(EHR_ROOT, f))
        print(f"    {f} ({size / 1e6:.1f} MB)")

    # Try to read tables -- search for various naming patterns
    try:
        import pandas as pd
    except ImportError:
        print("  ERROR: pandas not installed")
        result["tables"] = {"error": "pandas not installed"}
        return result

    # Look for files matching these keywords (case-insensitive)
    keywords = ["labevents", "chartevents", "patients", "admissions", "diagnoses", "d_labitems"]
    tables = {}

    for f in all_files:
        f_lower = f.lower()
        for kw in keywords:
            if kw in f_lower:
                path = os.path.join(EHR_ROOT, f)
                try:
                    size_mb = os.path.getsize(path) / 1e6
                    df = pd.read_csv(path, nrows=3)
                    tables[f] = {
                        "size_mb": round(size_mb, 1),
                        "columns": list(df.columns),
                        "dtypes": {c: str(d) for c, d in df.dtypes.items()},
                        "sample_row": {c: str(v) for c, v in df.iloc[0].items()},
                    }
                    print(f"  READ OK: {f} -> cols: {list(df.columns)}")
                except Exception as e:
                    tables[f] = {"size_mb": round(size_mb, 1), "error": str(e)}
                    print(f"  READ FAIL: {f} -> {e}")
                break

    result["tables"] = tables
    return result


# ==========================================================================
# 4. Check existing processed data + inspect one NPZ
# ==========================================================================
def check_existing_processed():
    print("\n=== 4. Existing Processed Data ===")

    check_paths = [
        "/opt/localdata100tb/UNIPHY_Plus/dataset/EST/MIMIC3_SPO2_I_40hz_v3_lab_vital",
        "/opt/localdata100tb/UNIPHY_Plus/dataset/EST/MIMIC3_SPO2_I_40hz_v3",
    ]

    result = {}
    for path in check_paths:
        if not os.path.isdir(path):
            result[path] = {"exists": False}
            print(f"  NOT FOUND: {path}")
            continue

        files = sorted(os.listdir(path))
        result[path] = {
            "exists": True,
            "n_files": len(files),
            "sample_files": files[:5],
        }
        print(f"  FOUND: {path} ({len(files)} files)")

        # Inspect one NPZ to see what keys/shapes are inside
        if files:
            import numpy as np
            npz_path = os.path.join(path, files[0])
            try:
                data = np.load(npz_path, allow_pickle=True)
                npz_info = {}
                for k in data.files:
                    arr = data[k]
                    npz_info[k] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
                result[path]["sample_npz"] = {"file": files[0], "keys": npz_info}
                print(f"  NPZ keys: { {k: v['shape'] for k, v in npz_info.items()} }")
            except Exception as e:
                result[path]["sample_npz"] = {"error": str(e)}
                print(f"  NPZ read error: {e}")

    return result


# ==========================================================================
# 5. Python packages
# ==========================================================================
def check_packages():
    print("\n=== 5. Python Packages ===")
    packages = {}
    for pkg in ["numpy", "pandas", "scipy", "wfdb", "polars", "pyarrow", "torch", "matplotlib"]:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "unknown")
            packages[pkg] = ver
            print(f"  {pkg}: {ver}")
        except ImportError:
            packages[pkg] = None
            print(f"  {pkg}: NOT INSTALLED")
    return packages


# ==========================================================================
# Main
# ==========================================================================
def main():
    print(f"Physio_Data Step 0: MIMIC-III Exploration (v2)")
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

    print(f"\n{'='*60}")
    print(f"Report saved to: {out_path}")
    print(f"\nNext:")
    print(f"  git add workzone/outputs/")
    print(f"  git commit -m 'MIMIC-III exploration v2'")
    print(f"  git push")


if __name__ == "__main__":
    main()
