"""Structural scan of UCSF data on bedanalysis (Step 0b, first pass).

Run on bedanalysis. Writes dataset_profile.json next to itself.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
from collections import Counter
from glob import glob
from pathlib import Path

import polars as pl

RNG = random.Random(42)

UCSF_ROOT = "/labs/hulab/UCSF"
RDB = f"{UCSF_ROOT}/rdb_new"
OUT = Path("/labs/hulab/mxwang/Physio_Data/workzone/ucsf/explore/dataset_profile.json")

BINFILEPY_SRC = "/labs/hulab/mxwang/data/ucsf_EHR/bedanalysis_waveformExtraction"
sys.path.insert(0, BINFILEPY_SRC)
from binfilepy import BinFile  # type: ignore  # noqa: E402
from vitalfilepy.vitalfile import VitalFile  # type: ignore  # noqa: E402


def list_wynton_folders() -> list[str]:
    folders = []
    for name in sorted(os.listdir(UCSF_ROOT)):
        p = os.path.join(UCSF_ROOT, name)
        if os.path.isdir(p) and re.match(r"\d{4}-\d{2}-deid$", name):
            folders.append(name)
    return folders


def scan_wynton_counts(folders: list[str]) -> dict:
    out = {}
    for w in folders:
        wp = os.path.join(UCSF_ROOT, w)
        pats = [d for d in os.listdir(wp) if d.startswith("DE") and os.path.isdir(os.path.join(wp, d))]
        out[w] = {"n_patients": len(pats)}
    return out


def sample_patient_dirs(folders: list[str], k: int = 20) -> list[str]:
    pool = []
    for w in folders:
        wp = os.path.join(UCSF_ROOT, w)
        try:
            for d in os.listdir(wp):
                if d.startswith("DE"):
                    pool.append(os.path.join(wp, d))
        except OSError:
            pass
    RNG.shuffle(pool)
    return pool[:k]


def vital_suffixes_in_patient(pat_dir: str) -> Counter:
    c: Counter = Counter()
    for bed in os.listdir(pat_dir):
        bed_p = os.path.join(pat_dir, bed)
        if not os.path.isdir(bed_p):
            continue
        for f in os.listdir(bed_p):
            if not f.endswith(".vital"):
                continue
            m = re.match(r"DE\d+_\d{14}_\d{5}_(.+)\.vital$", f)
            if m:
                c[m.group(1)] += 1
    return c


def probe_adibin_sample(pat_dirs: list[str], n: int = 8) -> list[dict]:
    probes = []
    for pd in pat_dirs:
        if len(probes) >= n:
            break
        for bed in os.listdir(pd):
            bed_p = os.path.join(pd, bed)
            if not os.path.isdir(bed_p):
                continue
            adibs = [f for f in os.listdir(bed_p) if f.endswith(".adibin")]
            if not adibs:
                continue
            p = os.path.join(bed_p, adibs[0])
            try:
                with BinFile(p, "r") as fh:
                    fh.readHeader()
                    h = fh.header
                    chans = [c.Title if c.Title else f"Unnamed{i}" for i, c in enumerate(fh.channels)]
                    probes.append({
                        "path": p,
                        "channels": chans,
                        "fs_hz": 1.0 / h.secsPerTick,
                        "samples_per_channel": int(h.SamplesPerChannel),
                        "duration_sec": float(h.SamplesPerChannel * h.secsPerTick),
                        "start": f"{h.Year:04d}-{h.Month:02d}-{h.Day:02d}T{h.Hour:02d}:{h.Minute:02d}:{float(h.Second):09.6f}",
                    })
            except Exception as e:
                probes.append({"path": p, "error": str(e)})
            break
    return probes


def probe_vital_sample(pat_dirs: list[str], n: int = 8) -> list[dict]:
    probes = []
    seen_suffixes = set()
    for pd in pat_dirs:
        if len(probes) >= n:
            break
        for bed in os.listdir(pd):
            bed_p = os.path.join(pd, bed)
            if not os.path.isdir(bed_p):
                continue
            vitals = [f for f in os.listdir(bed_p) if f.endswith(".vital")]
            # prefer unseen suffix variety
            vitals.sort(key=lambda f: f.split("_")[-1])
            for f in vitals:
                m = re.match(r"DE\d+_\d{14}_\d{5}_(.+)\.vital$", f)
                suffix = m.group(1) if m else f
                if suffix in seen_suffixes and len(seen_suffixes) < 10:
                    continue
                p = os.path.join(bed_p, f)
                try:
                    with VitalFile(p, "r") as fh:
                        fh.readHeader()
                        h = fh.header
                        n_samp = int(fh.numSamplesInFile)
                        raw = fh.readVitalDataBuf(min(200, n_samp))
                        # tuples are (value, offset_sec, sentinel, constant); first two fields used
                        vals = [float(t[0]) for t in raw]
                        offs = [float(t[1]) for t in raw]
                        data = list(zip(vals, offs))
                        diffs = sorted(offs[i + 1] - offs[i] for i in range(len(offs) - 1))
                        median_dt = diffs[len(diffs) // 2] if diffs else None
                        probes.append({
                            "path": p,
                            "suffix": suffix,
                            "n_samples_in_file": n_samp,
                            "first_5": [(v, o) for v, o in zip(vals[:5], offs[:5])],
                            "last_5": [(v, o) for v, o in zip(vals[-5:], offs[-5:])] if len(vals) > 5 else [],
                            "median_dt_sec": median_dt,
                            "val_min": min(vals) if vals else None,
                            "val_max": max(vals) if vals else None,
                            "start": f"{h.Year:04d}-{h.Month:02d}-{h.Day:02d}T{h.Hour:02d}:{h.Minute:02d}:{float(h.Second):09.6f}",
                        })
                    seen_suffixes.add(suffix)
                except Exception as e:
                    probes.append({"path": p, "error": str(e)})
                break
            if len(probes) >= n:
                break
    return probes


def probe_ehr_headers() -> dict:
    out: dict = {}
    for sub in ["Filtered_Lab_New", "Filtered_Medication_Orders_New", "Filtered_Diagnoses_New",
                "Filtered_Encounters_New", "Filtered_Billing_New", "Filtered_Procedure_Orders_New",
                "FLOWSHEETVALUEFACT"]:
        d = os.path.join(RDB, sub)
        if not os.path.isdir(d):
            continue
        files = sorted(glob(os.path.join(d, "*.txt")))
        if not files:
            out[sub] = {"n_shards": 0}
            continue
        first = files[0]
        try:
            with open(first, "r", encoding="latin-1") as fh:
                header = fh.readline().strip()
            out[sub] = {"n_shards": len(files), "first_shard": first, "header": header}
        except Exception as e:
            out[sub] = {"n_shards": len(files), "error": str(e)}
    return out


def probe_flowsheet_rowkeys(shards_to_scan: int = 3, top_k: int = 40) -> dict:
    d = os.path.join(RDB, "FLOWSHEETVALUEFACT")
    files = sorted(glob(os.path.join(d, "*.txt")))[:shards_to_scan]
    c: Counter = Counter()
    n_rows = 0
    for f in files:
        with open(f, "r", encoding="latin-1") as fh:
            fh.readline()
            for line in fh:
                n_rows += 1
                key = line.split(",", 1)[0]
                c[key] += 1
    return {"shards_scanned": len(files), "rows_seen": n_rows,
            "unique_rowkeys": len(c), "top_rowkeys": c.most_common(top_k)}


def probe_offset_xlsx() -> dict:
    xlsx = os.path.join(UCSF_ROOT, "encounter_date_offset_table_ver_Apr2024.xlsx")
    import pandas as pd
    pdf = pd.read_excel(xlsx, engine="openpyxl")
    cols = list(pdf.columns)
    out = {
        "path": xlsx,
        "n_rows": int(len(pdf)),
        "columns": cols,
    }
    if "Patient_ID_GE" in cols:
        out["n_unique_patient_id_ge"] = int(pdf["Patient_ID_GE"].nunique())
    if "Encounter_ID" in cols:
        out["n_unique_encounter_id"] = int(pdf["Encounter_ID"].nunique())
    if "Patient_ID" in cols:
        out["n_unique_patient_id"] = int(pdf["Patient_ID"].nunique())
    if "Wynton_folder" in cols:
        out["wynton_folders_in_table"] = sorted(pdf["Wynton_folder"].dropna().unique().tolist())[:5]
        out["n_wynton_folders_in_table"] = int(pdf["Wynton_folder"].nunique())
    return out


def main():
    folders = list_wynton_folders()
    print(f"[wynton] {len(folders)} folders")

    wynton_counts = scan_wynton_counts(folders)
    total_patients_in_fs = sum(v["n_patients"] for v in wynton_counts.values())
    print(f"[wynton] total DE* dirs (filesystem): {total_patients_in_fs}")

    pat_sample = sample_patient_dirs(folders, k=20)
    print(f"[sample] {len(pat_sample)} patient dirs sampled")

    all_suffixes: Counter = Counter()
    per_pat_suffix = {}
    for pd in pat_sample[:10]:
        c = vital_suffixes_in_patient(pd)
        per_pat_suffix[os.path.basename(pd)] = dict(c)
        all_suffixes.update(c)
    print(f"[suffixes] {len(all_suffixes)} distinct suffixes across sample")

    adibin_probes = probe_adibin_sample(pat_sample, n=8)
    print(f"[adibin] {len(adibin_probes)} probed")

    vital_probes = probe_vital_sample(pat_sample, n=10)
    print(f"[vital] {len(vital_probes)} probed")

    ehr_headers = probe_ehr_headers()
    print(f"[ehr] {len(ehr_headers)} tables")

    flowsheet_rowkeys = probe_flowsheet_rowkeys(shards_to_scan=3)
    print(f"[flowsheet] top rowkeys scanned: {flowsheet_rowkeys['unique_rowkeys']} unique")

    offset = probe_offset_xlsx()
    print(f"[offset] {offset['n_rows']} rows, {offset['n_unique_patient_id_ge']} unique Patient_ID_GE")

    out = {
        "ucsf_root": UCSF_ROOT,
        "wynton_folders": {"n_folders": len(folders), "folders": folders,
                            "total_de_dirs": total_patients_in_fs,
                            "per_folder": wynton_counts},
        "vital_suffixes_observed": all_suffixes.most_common(),
        "vital_suffix_per_patient_sample": per_pat_suffix,
        "adibin_probes": adibin_probes,
        "vital_probes": vital_probes,
        "ehr_tables": ehr_headers,
        "flowsheet_rowkeys_sample": flowsheet_rowkeys,
        "offset_table": offset,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"[done] wrote {OUT}")


if __name__ == "__main__":
    main()
