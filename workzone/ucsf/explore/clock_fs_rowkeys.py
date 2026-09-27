import polars as pl, collections, glob, os, json, pandas as pd
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
F = "/mnt/localdata/storage/UCSF/rdb_new/FLOWSHEETVALUEFACT"; D = "/mnt/localdata/storage/UCSF/rdb_database/FLOWSHEETROWDIM_New/FLOWSHEETROWDIM_New.csv"
dim = pl.read_csv(D, infer_schema_length=0, encoding="utf8-lossy"); dim = dim.rename({c: c.strip("﻿") for c in dim.columns})
name = dict(zip(dim["FlowsheetRowKey"].to_list(), dim["Name"].to_list())); unit = dict(zip(dim["FlowsheetRowKey"].to_list(), dim["Unit"].to_list()))
cnt = collections.Counter()
for f in sorted(glob.glob(f"{F}/2015_03_1*.txt"))[:12]:
    df = pl.read_csv(f, infer_schema_length=0, encoding="utf8-lossy", truncate_ragged_lines=True)
    cnt.update(df["FlowsheetRowKey"].to_list())
print("top row keys in 12 shards of 2015_03:")
for k, n in cnt.most_common(40): print(f"  {k:>6} {n:>7}  {name.get(k,'?')[:60]!r}  unit={unit.get(k,'')}")
hr_keys = [k for k, nm in name.items() if nm and nm.strip().upper() in ("PULSE", "HEART RATE", "R HEART RATE", "HR")]
print("candidate HR keys by exact name:", [(k, name[k]) for k in hr_keys])
# ---- ADT side check from the offset table (no scanning)
LA = ZoneInfo("America/Los_Angeles")
off = pd.read_parquet("/mnt/localdata/storage/mxwang/Physio_Data/workzone/outputs/ucsf_all/encounter_offset_table.parquet")
for c in ["Encounter_Start_time", "alarm_time", "bed_transferin_time", "bin_file_time"]:
    print(c, "sample:", off[c].dropna().astype(str).head(3).tolist())
bt = pd.to_datetime(off["bed_transferin_time"], errors="coerce"); bf = pd.to_datetime(off["bin_file_time"], errors="coerce"); es = pd.to_datetime(off["Encounter_Start_time"], errors="coerce")
dmin = (bt - bf).dt.total_seconds() / 60
print("bed_transferin - bin_file (min): n", int(dmin.notna().sum()), "quantiles", dmin.quantile([.05,.25,.5,.75,.95]).round(0).tolist())
print("Encounter_Start(EHR cal) - bin_file(GE cal) days: quantiles", ((es - bf).dt.total_seconds()/86400).quantile([.05,.25,.5,.75,.95]).round(1).tolist())
ds = lambda t: int(t.replace(tzinfo=LA).dst().total_seconds() != 0)
grp = collections.defaultdict(list)
for t_bt, t_bf, o, og in zip(bt, bf, off["offset"], off["offset_GE"]):
    if pd.isna(t_bt) or pd.isna(t_bf): continue
    t_real = (t_bf.to_pydatetime().replace(tzinfo=LA).astimezone(timezone.utc) + timedelta(days=float(og))).astimezone(LA).replace(tzinfo=None)
    grp[(ds(t_real), ds(t_bf.to_pydatetime()))].append((t_bt - t_bf).total_seconds() / 60)
import numpy as np
for k in sorted(grp):
    v = np.array(grp[k]); print(f"(dst real, dst GE)={k}: n={v.size} bed_transferin-bin_file median={np.median(v):.0f} min; share within ±15 of 0/-60/+60: {np.mean(np.abs(v)<=15):.2f}/{np.mean(np.abs(v+60)<=15):.2f}/{np.mean(np.abs(v-60)<=15):.2f}")
