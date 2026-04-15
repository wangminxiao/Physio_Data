#!/usr/bin/env python3
"""
Post-stage: Emit demographics.csv at {output_dir}/demographics.csv.

One row per admission-level patient_id = {SUBJECT_ID}_{HADM_ID}, matching
the key that UniphyDataset / patient_store uses.

Columns:
  patient_id       str   e.g. "12345_67890"  (index when read by UNIPHY)
  subject_id       int
  hadm_id          int
  gender           str   "M" | "F" | ""      (categorical)
  age_years        float years at admission, capped at 89 per MIMIC policy
  ethnicity        str   categorical (raw MIMIC labels, consumer can re-bin)
  insurance        str   categorical
  admission_type   str   categorical
  icd9_primary     str   SEQ_NUM==1 code, "" if missing
  died_in_hospital int   0 | 1 (from ADMISSIONS.HOSPITAL_EXPIRE_FLAG)

Run:  python workzone/mimic3/post_demographics.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
with open(REPO_ROOT / "workzone" / "configs" / "server_paths.yaml") as f:
    cfg = yaml.safe_load(f)

EHR_ROOT = cfg["mimic3"]["raw_ehr_dir"]
PROCESSED_ROOT = Path(cfg["mimic3"]["output_dir"])

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _load(name: str, **kwargs) -> pd.DataFrame:
    p = os.path.join(EHR_ROOT, f"{name}.csv.gz")
    if not os.path.exists(p):
        p = os.path.join(EHR_ROOT, f"{name}.csv")
    return pd.read_csv(p, **kwargs)


def main():
    t0 = time.time()
    log.info(f"Emitting demographics.csv -> {PROCESSED_ROOT}")

    # 1. Patient dirs -> (subject_id, hadm_id)
    pids = []
    for d in sorted(PROCESSED_ROOT.iterdir()):
        if not d.is_dir() or not (d / "meta.json").exists():
            continue
        try:
            sid, hid = d.name.split("_")
            pids.append({"patient_id": d.name, "subject_id": int(sid), "hadm_id": int(hid)})
        except (ValueError, TypeError):
            continue
    pid_df = pd.DataFrame(pids)
    log.info(f"  {len(pid_df)} patients")

    # 2. Join PATIENTS (gender, dob). MIMIC shifts DOBs of patients 89+ by
    # ~300 years, which overflows datetime64[ns]. Parse to python date and
    # compute age via (year,month,day) arithmetic to avoid the overflow.
    patients = _load("PATIENTS", usecols=["SUBJECT_ID", "GENDER", "DOB"])
    patients["DOB_date"] = pd.to_datetime(patients["DOB"], errors="coerce", format="mixed").dt.date
    # For DOBs pandas can't represent (year > 2262), fall back to str parsing.
    missing = patients["DOB_date"].isna()
    if missing.any():
        from datetime import datetime as _dt
        def _parse(s):
            try:
                return _dt.strptime(str(s).split(" ")[0], "%Y-%m-%d").date()
            except Exception:
                return None
        patients.loc[missing, "DOB_date"] = patients.loc[missing, "DOB"].map(_parse)
    out = pid_df.merge(patients, left_on="subject_id", right_on="SUBJECT_ID", how="left")

    # 3. Join ADMISSIONS (ethnicity, insurance, admission_type, admittime, expire flag)
    admissions = _load("ADMISSIONS", usecols=[
        "SUBJECT_ID", "HADM_ID", "ADMITTIME", "ETHNICITY", "INSURANCE",
        "ADMISSION_TYPE", "HOSPITAL_EXPIRE_FLAG",
    ])
    admissions["ADMITTIME"] = pd.to_datetime(admissions["ADMITTIME"])
    admissions["ADMITTIME_date"] = admissions["ADMITTIME"].dt.date
    out = out.merge(
        admissions,
        left_on=["subject_id", "hadm_id"],
        right_on=["SUBJECT_ID", "HADM_ID"],
        how="left",
        suffixes=("", "_adm"),
    )

    # 4. Primary ICD9 (SEQ_NUM == 1)
    diag = _load("DIAGNOSES_ICD", usecols=["SUBJECT_ID", "HADM_ID", "SEQ_NUM", "ICD9_CODE"])
    diag_primary = (
        diag[diag["SEQ_NUM"] == 1]
        .drop_duplicates(subset=["SUBJECT_ID", "HADM_ID"])[["SUBJECT_ID", "HADM_ID", "ICD9_CODE"]]
    )
    out = out.merge(
        diag_primary,
        left_on=["subject_id", "hadm_id"],
        right_on=["SUBJECT_ID", "HADM_ID"],
        how="left",
        suffixes=("", "_d"),
    )

    # 5. Compute age via date-of-year arithmetic (overflow-safe for shifted DOBs).
    # Cap at 89 per MIMIC de-identification policy (ages >89 are shifted +300y).
    def _age_years(adm, dob):
        if adm is None or dob is None or pd.isna(adm) or pd.isna(dob):
            return np.nan
        days = (adm - dob).days
        return days / 365.25
    age_yr = np.array(
        [_age_years(a, d) for a, d in zip(out["ADMITTIME_date"], out["DOB_date"])],
        dtype="float64",
    )
    age_yr = np.clip(age_yr, 0, 89.0)
    out["age_years"] = np.round(age_yr, 2)

    # 6. Select + rename
    final = pd.DataFrame({
        "patient_id":       out["patient_id"],
        "subject_id":       out["subject_id"].astype("Int64"),
        "hadm_id":          out["hadm_id"].astype("Int64"),
        "gender":           out["GENDER"].fillna("").astype(str),
        "age_years":        out["age_years"],
        "ethnicity":        out["ETHNICITY"].fillna("").astype(str),
        "insurance":        out["INSURANCE"].fillna("").astype(str),
        "admission_type":   out["ADMISSION_TYPE"].fillna("").astype(str),
        "icd9_primary":     out["ICD9_CODE"].fillna("").astype(str),
        "died_in_hospital": out["HOSPITAL_EXPIRE_FLAG"].fillna(0).astype(int),
    })

    out_path = PROCESSED_ROOT / "demographics.csv"
    final.to_csv(out_path, index=False)
    log.info(f"  Wrote {out_path} ({len(final)} rows)")
    log.info(f"  Null gender:    {(final['gender'] == '').sum()}")
    log.info(f"  Null ethnicity: {(final['ethnicity'] == '').sum()}")
    log.info(f"  Null icd9:      {(final['icd9_primary'] == '').sum()}")
    log.info(f"  Null age:       {final['age_years'].isna().sum()}")
    log.info(f"  Time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
