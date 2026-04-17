"""Diagnose the two demo-run failures:
1. Which candidates have NON-zero vital file timestamps?
2. Does our Patient_ID (214688334614038) have labs anywhere in Filtered_Lab_New?
"""
from __future__ import annotations

import os
import re
import sys
from glob import glob

sys.path.insert(0, "/labs/hulab/mxwang/data/ucsf_EHR/bedanalysis_waveformExtraction")
from vitalfilepy.vitalfile import VitalFile  # type: ignore

# --- 1) For each of our top 5 candidates, report how many vital filenames are zero-timestamped.
CANDIDATES = [
    ("2017-03-deid", "214688354794344"),   # current pick (broken)
    ("2017-03-deid", "747223141294381"),
    ("2017-10-deid", "969040437292716"),
    ("2016-10-deid", "299830937914959"),
    ("2017-08-deid", "56577667285165"),
]

print("=== vital-file timestamp check across top 5 candidates ===")
for wyn, pid in CANDIDATES:
    pat_dir = f"/labs/hulab/UCSF/{wyn}/DE{pid}"
    total = zero = nonzero = 0
    example = None
    for bed in sorted(os.listdir(pat_dir)):
        bed_p = os.path.join(pat_dir, bed)
        if not os.path.isdir(bed_p):
            continue
        for f in os.listdir(bed_p):
            if not f.endswith(".vital"):
                continue
            total += 1
            m = re.match(r"DE\d+_(\d{14})_\d{5}_.+\.vital$", f)
            if m:
                if m.group(1) == "00000000000000":
                    zero += 1
                else:
                    nonzero += 1
                    if example is None:
                        example = f
    print(f"  {pid}  total={total:3d}  nonzero_ts={nonzero:3d}  zero_ts={zero:3d}  "
          f"ex={example}")

# --- 2) For pat 214688354794344, try one NON-zero vital file if any exists (sanity check reader)
print("\n=== verify non-zero vital read works ===")
for wyn, pid in CANDIDATES:
    pat_dir = f"/labs/hulab/UCSF/{wyn}/DE{pid}"
    found = None
    for bed in sorted(os.listdir(pat_dir)):
        bed_p = os.path.join(pat_dir, bed)
        if not os.path.isdir(bed_p):
            continue
        for f in os.listdir(bed_p):
            if not f.endswith(".vital"):
                continue
            m = re.match(r"DE\d+_(\d{14})_\d{5}_.+\.vital$", f)
            if m and m.group(1) != "00000000000000":
                found = os.path.join(bed_p, f)
                break
        if found:
            break
    if not found:
        print(f"  {pid}: NO non-zero vital files")
        continue
    try:
        with VitalFile(found, "r") as fh:
            fh.readHeader()
            h = fh.header
            print(f"  {pid}: ok {found.split('/')[-1]} start={h.Year}-{h.Month}-{h.Day} "
                  f"{h.Hour}:{h.Minute}:{h.Second} nSamples={fh.numSamplesInFile}")
    except Exception as e:
        print(f"  {pid}: ERROR {e}")

# --- 3) Count lab rows for Patient_ID 214688334614038 across Filtered_Lab_New.
print("\n=== lab row count for Patient_ID 214688334614038 ===")
pat_id = "214688334614038"
lab_dir = "/labs/hulab/UCSF/rdb_new/Filtered_Lab_New"
shards = sorted(glob(os.path.join(lab_dir, "*.txt")))
print(f"  total shards: {len(shards)}  (will fast-grep all with substring scan)")
total_hits = 0
by_year = {}
for s in shards:
    year = os.path.basename(s).split("_")[0]
    with open(s, "r", encoding="latin-1") as fh:
        fh.readline()
        hits = 0
        for line in fh:
            if pat_id in line:
                hits += 1
    total_hits += hits
    by_year[year] = by_year.get(year, 0) + hits
print(f"  total lines containing substring: {total_hits}")
print(f"  per year: {dict(sorted(by_year.items()))}")

# Also grep Medication_Orders and FLOWSHEETVALUEFACT (fast)
for sub in ["Filtered_Medication_Orders_New", "FLOWSHEETVALUEFACT"]:
    d = f"/labs/hulab/UCSF/rdb_new/{sub}"
    total = 0
    for s in sorted(glob(os.path.join(d, "*.txt"))):
        with open(s, "r", encoding="latin-1") as fh:
            fh.readline()
            for line in fh:
                if pat_id in line:
                    total += 1
    print(f"  {sub}: {total} lines contain substring")
