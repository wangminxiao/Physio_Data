import pandas as pd, collections, os, glob
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
LA = ZoneInfo("America/Los_Angeles")
off = pd.read_parquet("/mnt/localdata/storage/mxwang/Physio_Data/workzone/outputs/ucsf_all/encounter_offset_table.parquet")
print("columns:", list(off.columns)); print("rows:", len(off))
off["offset"] = pd.to_numeric(off["offset"], errors="coerce"); off["offset_GE"] = pd.to_numeric(off["offset_GE"], errors="coerce")
print("offset==offset_GE:", int((off["offset"] == off["offset_GE"]).sum()), " differ:", int((off["offset"] != off["offset_GE"]).sum()), " nan:", int(off["offset_GE"].isna().sum() + off["offset"].isna().sum()))
d = (off["offset_GE"] - off["offset"]).dropna(); print("offset_GE-offset days: describe", d.describe().round(1).to_dict()); print("value counts top:", d.value_counts().head(8).to_dict())
# Encounter start (EHR-shifted calendar) -> DST states of EHR-shifted date vs GE-shifted date (naive) vs real date
col = [c for c in off.columns if "Start" in c or "start" in c]; print("start cols:", col)
st = pd.to_datetime(off[col[0]], errors="coerce")
res = collections.Counter()
for t_ehr, o, og in zip(st, off["offset"], off["offset_GE"]):
    if pd.isna(t_ehr) or pd.isna(o) or pd.isna(og): res["nan"] += 1; continue
    t_ge = t_ehr - timedelta(days=float(og - o)); t_real = t_ehr + timedelta(days=float(o))
    ds = lambda t: int(t.to_pydatetime().replace(tzinfo=LA).dst().total_seconds() != 0)
    res[(ds(t_real), ds(t_ehr), ds(t_ge))] += 1
print("(dst real, dst ehr-shifted, dst ge-shifted) counts:", dict(res))
n_mis_ehr_ge = sum(v for k, v in res.items() if k != "nan" and k[1] != k[2]); n_mis_real_ge = sum(v for k, v in res.items() if k != "nan" and k[0] != k[2])
print(f"encounters where EHR-shifted and GE-shifted dates differ in DST: {n_mis_ehr_ge}; real vs GE-shifted differ: {n_mis_real_ge}")
for root in ["/mnt/localdata/storage/UCSF/rdb_new", "/mnt/localdata/storage/UCSF"]:
    try: print(root, "->", sorted(os.listdir(root))[:40])
    except Exception as e: print(root, "ERR", e)
for pat in ["/mnt/localdata/storage/UCSF/rdb_new/*low*", "/mnt/localdata/storage/UCSF/rdb_new/*LOW*", "/mnt/localdata/storage/UCSF/*low*"]:
    for p in glob.glob(pat):
        fs = os.listdir(p); print(p, "n files", len(fs), "first", fs[:3], "size of first MB", round(os.path.getsize(os.path.join(p, fs[0]))/1e6, 2) if fs else None)
