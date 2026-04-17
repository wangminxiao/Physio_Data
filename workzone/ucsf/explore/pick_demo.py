"""Pick one good demo patient with vital_flag=1, .adibin files, and EHR data."""
from __future__ import annotations

import json
import os
import re
from glob import glob
from pathlib import Path

import pandas as pd

UCSF = "/labs/hulab/UCSF"
OUT = Path("/labs/hulab/mxwang/Physio_Data/workzone/ucsf/explore/demo_candidate.json")

xlsx = f"{UCSF}/encounter_date_offset_table_ver_Apr2024.xlsx"
df = pd.read_excel(xlsx, engine="openpyxl")
df = df[df["vital_flag"] == 1].copy()
# reasonable LOS: short enough for a tractable demo (2-8 hours bed time)
df = df[(df["Encounter_LOS"] >= 1) & (df["Encounter_LOS"] <= 20)]
# prefer recent years for data quality (2016-2018)
df["_year"] = df["Wynton_folder"].astype(str).str.slice(0, 4)
df = df[df["_year"].isin(["2016", "2017"])]

print(f"candidates after filters: {len(df)}")
candidates = []
for _, row in df.sample(n=min(30, len(df)), random_state=42).iterrows():
    wf = row["Wynton_folder"]
    pid_ge = str(row["Patient_ID_GE"])
    pat_dir = f"{UCSF}/{wf}/DE{pid_ge}"
    if not os.path.isdir(pat_dir):
        continue
    # check for MRN-Mapping
    mrn = os.path.join(pat_dir, "MRN-Mapping.csv")
    if not os.path.isfile(mrn):
        continue
    # count adibin and vital files
    adibin_n = 0
    vital_n = 0
    ar_present = False
    suffixes = set()
    bed_subdirs = []
    for bed in sorted(os.listdir(pat_dir)):
        bed_p = os.path.join(pat_dir, bed)
        if not os.path.isdir(bed_p):
            continue
        bed_subdirs.append(bed)
        for f in os.listdir(bed_p):
            if f.endswith(".adibin"):
                adibin_n += 1
            elif f.endswith(".vital"):
                vital_n += 1
                m = re.match(r"DE\d+_\d{14}_\d{5}_(.+)\.vital$", f)
                if m:
                    sfx = m.group(1)
                    suffixes.add(sfx)
                    if sfx.startswith("AR"):
                        ar_present = True
    if adibin_n == 0 or vital_n == 0:
        continue
    candidates.append({
        "Patient_ID_GE": pid_ge,
        "Patient_ID": str(row["Patient_ID"]),
        "Encounter_ID": str(row["Encounter_ID"]),
        "Wynton_folder": wf,
        "offset": int(row["offset"]),
        "offset_GE": int(row["offset_GE"]),
        "offset_delta_days": int(row["offset_GE"] - row["offset"]),
        "encounter_start": str(row["Encounter_Start_time"]),
        "LOS_days": int(row["Encounter_LOS"]),
        "vital_flag": int(row["vital_flag"]),
        "adibin_count": adibin_n,
        "vital_count": vital_n,
        "vital_suffixes": sorted(suffixes),
        "has_arterial": ar_present,
        "bed_subdirs": bed_subdirs,
        "pat_dir": pat_dir,
    })

# prefer candidates with arterial line + manageable file count
candidates.sort(key=lambda c: (
    -int(c["has_arterial"]),
    abs(c["adibin_count"] - 30),  # target ~30 adibin files (few hours)
))

print(f"usable: {len(candidates)}")
for c in candidates[:5]:
    print(f"  {c['Patient_ID_GE']} enc={c['Encounter_ID']} adibin={c['adibin_count']} "
          f"vital={c['vital_count']} AR={c['has_arterial']} delta={c['offset_delta_days']}d LOS={c['LOS_days']}d")

pick = candidates[0] if candidates else None

# inspect MRN-Mapping for the pick
if pick:
    mrn_path = os.path.join(pick["pat_dir"], "MRN-Mapping.csv")
    mrn_df = pd.read_csv(mrn_path, on_bad_lines="skip")
    mrn_df.columns = [c.strip() for c in mrn_df.columns]
    # Group by WaveCycleUID, get one representative row
    wave_cycles = []
    for wcuid, g in mrn_df.groupby("WaveCycleUID"):
        bed_in = g["BedTransfer_In"].min()
        bed_out = g["BedTransfer_Out"].max()
        wstart = g["WaveStartTime"].min()
        wstop = g["WaveStopTime"].max()
        wave_cycles.append({
            "WaveCycleUID": int(wcuid),
            "UnitBed": g["UnitBed"].iloc[0] if "UnitBed" in g.columns else None,
            "BedTransfer_In": str(bed_in),
            "BedTransfer_Out": str(bed_out),
            "WaveStartTime": str(wstart),
            "WaveStopTime": str(wstop),
            "n_rows_in_MRN_Mapping": len(g),
        })
    pick["wave_cycles"] = wave_cycles

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as fh:
    json.dump({"n_candidates": len(candidates), "top_5": candidates[:5], "pick": pick}, fh, indent=2, default=str)
print(f"[done] wrote {OUT}")
