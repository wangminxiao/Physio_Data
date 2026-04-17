"""Resolve the 4 UCSF Step 0b blockers before Step 0c."""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from glob import glob
from pathlib import Path

UCSF_ROOT = "/labs/hulab/UCSF"
RDB = f"{UCSF_ROOT}/rdb_new"
RDB_DB = f"{UCSF_ROOT}/rdb_database"
OUT = Path("/labs/hulab/mxwang/Physio_Data/workzone/ucsf/explore/dataset_profile_v2.json")


def probe_flowsheet_rowdim() -> dict:
    """Load FLOWSHEETROWDIM and find matches for our target vitals."""
    path = os.path.join(RDB_DB, "FLOWSHEETROWDIM_New", "FLOWSHEETROWDIM_New.csv")
    import csv
    targets = {
        "HR":          [r"\bHEART RATE\b", r"\bHR\b"],
        "SpO2":        [r"PULSE OXIMETRY", r"\bSPO2\b", r"\bSAO2\b"],
        "RR":          [r"RESPIRATION", r"RESPIRATORY RATE", r"\bRESP RATE\b"],
        "Temperature": [r"\bTEMPERATURE\b", r"\bTEMP\b"],
        "NBPs":        [r"BLOOD PRESSURE SYSTOLIC", r"\bSBP\b", r"\bNBP SYST"],
        "NBPd":        [r"BLOOD PRESSURE DIASTOLIC", r"\bDBP\b", r"\bNBP DIAS"],
        "NBPm":        [r"BLOOD PRESSURE MEAN", r"\bMAP\b", r"\bNBP MEAN"],
        "ABPs":        [r"ARTERIAL.*SYSTOLIC", r"\bABP SYST"],
        "ABPd":        [r"ARTERIAL.*DIASTOLIC", r"\bABP DIAS"],
        "ABPm":        [r"ARTERIAL.*MEAN", r"\bABP MEAN"],
        "CVP":         [r"\bCVP\b", r"CENTRAL VENOUS"],
        "Height":      [r"\bHEIGHT\b"],
        "Weight":      [r"\bWEIGHT\b"],
        "GCS":         [r"\bGCS\b", r"GLASGOW COMA"],
        "pain":        [r"PAIN SCORE"],
    }
    hits: dict = {k: [] for k in targets}
    n = 0
    # header has BOM; csv.DictReader handles it
    with open(path, "r", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            n += 1
            name = (row.get("Name") or "").upper()
            if not name:
                continue
            for tgt, patterns in targets.items():
                if any(re.search(p, name) for p in patterns):
                    if len(hits[tgt]) < 10:  # cap
                        hits[tgt].append({
                            "FlowsheetRowKey": row.get("FlowsheetRowKey"),
                            "Name": row.get("Name"),
                            "Abbreviation": row.get("Abbreviation"),
                            "ValueType": row.get("ValueType"),
                            "Unit": row.get("Unit"),
                        })
    return {"total_rowdim_rows": n, "candidate_matches": hits}


def probe_top_flowsheet_rowkeys_with_names(shards: int = 5, top_k: int = 40) -> list[dict]:
    """Top flowsheet rowkeys by row count, resolved to names via FLOWSHEETROWDIM."""
    import csv
    # build lookup
    rowdim_path = os.path.join(RDB_DB, "FLOWSHEETROWDIM_New", "FLOWSHEETROWDIM_New.csv")
    lookup: dict = {}
    with open(rowdim_path, "r", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            lookup[row["FlowsheetRowKey"]] = (row.get("Name"), row.get("Abbreviation"), row.get("Unit"), row.get("ValueType"))
    # count top rowkeys
    files = sorted(glob(os.path.join(RDB, "FLOWSHEETVALUEFACT", "*.txt")))[:shards]
    c: Counter = Counter()
    for f in files:
        with open(f, "r", encoding="latin-1") as fh:
            fh.readline()
            for line in fh:
                c[line.split(",", 1)[0]] += 1
    out = []
    for k, n in c.most_common(top_k):
        name, abbr, unit, vt = lookup.get(k, (None, None, None, None))
        out.append({"key": k, "count_in_sample": n, "name": name, "abbr": abbr, "unit": unit, "value_type": vt})
    return out


def probe_offset_table() -> dict:
    """Examine offset/offset_GE/vital_flag distributions."""
    import pandas as pd
    xlsx = os.path.join(UCSF_ROOT, "encounter_date_offset_table_ver_Apr2024.xlsx")
    df = pd.read_excel(xlsx, engine="openpyxl")
    out: dict = {"n_rows": int(len(df)), "columns": list(df.columns)}

    sample_rows = df.head(3).to_dict(orient="records")
    out["first_3_rows"] = [
        {k: (str(v) if not isinstance(v, (int, float, bool)) or (isinstance(v, float) and not (v == v)) else v)
         for k, v in r.items()} for r in sample_rows
    ]

    if "offset" in df.columns:
        offset = df["offset"].dropna()
        out["offset_stats"] = {
            "n_non_null": int(len(offset)),
            "min": str(offset.min()) if len(offset) else None,
            "max": str(offset.max()) if len(offset) else None,
            "dtype": str(offset.dtype),
            "sample_unique": [str(v) for v in offset.drop_duplicates().head(10).tolist()],
        }
    if "offset_GE" in df.columns:
        ge = df["offset_GE"].dropna()
        out["offset_GE_stats"] = {
            "n_non_null": int(len(ge)),
            "min": str(ge.min()) if len(ge) else None,
            "max": str(ge.max()) if len(ge) else None,
            "dtype": str(ge.dtype),
            "sample_unique": [str(v) for v in ge.drop_duplicates().head(10).tolist()],
        }
    if "offset" in df.columns and "offset_GE" in df.columns:
        both = df[["offset", "offset_GE"]].dropna()
        if len(both):
            diff = both["offset"].astype(str) == both["offset_GE"].astype(str)
            out["offset_equals_offset_GE"] = {
                "n_pairs": int(len(both)),
                "n_equal": int(diff.sum()),
            }
    if "vital_flag" in df.columns:
        vf = df["vital_flag"]
        out["vital_flag_counts"] = {str(k): int(v) for k, v in vf.value_counts(dropna=False).items()}
    if "Wynton_folder" in df.columns:
        out["rows_per_year"] = {}
        years = df["Wynton_folder"].dropna().astype(str).str.slice(0, 4).value_counts().to_dict()
        for k, v in sorted(years.items()):
            out["rows_per_year"][k] = int(v)
        # 2018-month breakdown
        mask = df["Wynton_folder"].astype(str).str.startswith("2018-")
        out["rows_per_2018_month"] = {
            str(k): int(v) for k, v in df.loc[mask, "Wynton_folder"].value_counts().sort_index().items()
        }
    if "Encounter_Start_time" in df.columns:
        out["encounter_start_min"] = str(df["Encounter_Start_time"].min())
        out["encounter_start_max"] = str(df["Encounter_Start_time"].max())
    return out


def probe_top_loinc(shards: int = 5, top_k: int = 50) -> dict:
    """Top LOINC codes and common names from Filtered_Lab_New."""
    files = sorted(glob(os.path.join(RDB, "Filtered_Lab_New", "*.txt")))[:shards]
    loinc_cnt: Counter = Counter()
    common_cnt: Counter = Counter()
    loinc_to_name: dict = {}
    n_rows = 0
    for f in files:
        with open(f, "r", encoding="latin-1") as fh:
            header = fh.readline().strip().split(",")
            try:
                i_loinc = header.index("LOINC_Code")
            except ValueError:
                i_loinc = -1
            try:
                i_common = header.index("Lab_Common_Name")
            except ValueError:
                i_common = -1
            try:
                i_name = header.index("LOINC_Name")
            except ValueError:
                i_name = -1
            for line in fh:
                n_rows += 1
                parts = line.rstrip("\n").split(",")
                if len(parts) < len(header):
                    continue
                if i_loinc >= 0 and parts[i_loinc]:
                    loinc_cnt[parts[i_loinc]] += 1
                    if parts[i_loinc] not in loinc_to_name and i_name >= 0 and parts[i_name]:
                        loinc_to_name[parts[i_loinc]] = parts[i_name]
                if i_common >= 0 and parts[i_common]:
                    common_cnt[parts[i_common]] += 1
    top_loinc = [{"loinc": k, "count": n, "loinc_name": loinc_to_name.get(k)} for k, n in loinc_cnt.most_common(top_k)]
    top_common = [{"common_name": k, "count": n} for k, n in common_cnt.most_common(top_k)]
    return {"shards_scanned": len(files), "rows_seen": n_rows, "top_loinc": top_loinc, "top_common_name": top_common}


def probe_top_medications(shards: int = 5, top_k: int = 40) -> dict:
    files = sorted(glob(os.path.join(RDB, "Filtered_Medication_Orders_New", "*.txt")))[:shards]
    gen_cnt: Counter = Counter()
    class_cnt: Counter = Counter()
    route_cnt: Counter = Counter()
    n_rows = 0
    for f in files:
        with open(f, "r", encoding="latin-1") as fh:
            header = fh.readline().strip().split(",")
            try:
                i_gen = header.index("Medication_Generic_Name")
            except ValueError:
                i_gen = -1
            try:
                i_cls = header.index("Medication_Therapeutic_Class")
            except ValueError:
                i_cls = -1
            try:
                i_route = header.index("Medication_Route")
            except ValueError:
                i_route = -1
            for line in fh:
                n_rows += 1
                parts = line.rstrip("\n").split(",")
                if len(parts) < len(header):
                    continue
                if i_gen >= 0 and parts[i_gen]:
                    gen_cnt[parts[i_gen]] += 1
                if i_cls >= 0 and parts[i_cls]:
                    class_cnt[parts[i_cls]] += 1
                if i_route >= 0 and parts[i_route]:
                    route_cnt[parts[i_route]] += 1
    return {
        "shards_scanned": len(files),
        "rows_seen": n_rows,
        "top_generic_names": gen_cnt.most_common(top_k),
        "top_therapeutic_class": class_cnt.most_common(20),
        "top_routes": route_cnt.most_common(20),
    }


def main():
    out: dict = {}
    print("[rowdim] probing FLOWSHEETROWDIM_New...", flush=True)
    out["flowsheet_rowdim"] = probe_flowsheet_rowdim()
    print(f"  total rowdim rows: {out['flowsheet_rowdim']['total_rowdim_rows']}", flush=True)

    print("[rowkeys] top rowkeys resolved to names...", flush=True)
    out["top_flowsheet_rowkeys_resolved"] = probe_top_flowsheet_rowkeys_with_names(shards=5, top_k=50)

    print("[offset] examining offset/offset_GE/vital_flag...", flush=True)
    out["offset_analysis"] = probe_offset_table()

    print("[labs] top LOINC codes...", flush=True)
    out["top_loinc"] = probe_top_loinc(shards=5, top_k=50)

    print("[meds] top generic names...", flush=True)
    out["top_medications"] = probe_top_medications(shards=5, top_k=40)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"[done] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
