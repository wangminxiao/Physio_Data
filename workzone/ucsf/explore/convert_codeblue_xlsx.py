"""One-off: SAUCSFCodeBlue_FirstEvent_2013_2018_final.xlsx -> parquet (needs openpyxl: run with the S4M env).
Columns kept: EID (== offset-table Encounter_ID), TypeCode (CPA/ME/ARC), CodeTime (MATLAB datenum), Unit, Age, Gender.
MRN is dropped."""
import sys, pandas as pd
src, dst = sys.argv[1], sys.argv[2]
df = pd.read_excel(src, dtype=str)
df = df[[c for c in ["EID", "TypeCode", "CodeTime", "Unit", "Age", "Gender"] if c in df.columns]].copy()
df["EID"] = df["EID"].str.strip(); df["TypeCode"] = df["TypeCode"].str.strip(); df["CodeTime"] = pd.to_numeric(df["CodeTime"], errors="coerce")
df.to_parquet(dst, index=False)
print(f"wrote {dst}: {len(df)} rows; TypeCode counts: {df['TypeCode'].value_counts().to_dict()}; CodeTime nulls: {int(df['CodeTime'].isna().sum())}")
