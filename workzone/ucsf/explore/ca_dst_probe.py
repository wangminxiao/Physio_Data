import json, pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
E = "/mnt/localdata/storage/mxwang/data/ucsf_EHR"
off = pd.read_parquet("/mnt/localdata/storage/mxwang/Physio_Data/workzone/outputs/ucsf_all/encounter_offset_table.parquet")
off["pid"] = off["Patient_ID_GE"].astype(str).str.strip().str.replace(r"^DE", "", regex=True); off["Encounter_ID"] = off["Encounter_ID"].astype(str).str.strip(); off["offset_GE"] = pd.to_numeric(off["offset_GE"], errors="coerce")
xl = pd.read_excel(f"{E}/SAUCSFCodeBlue_FirstEvent_2013_2018_final.xlsx", dtype=str); xl["EID"] = xl["EID"].str.strip()
m = xl.merge(off[["Encounter_ID", "pid", "offset_GE"]], left_on="EID", right_on="Encounter_ID")
ev = json.load(open("/projects/mwang80/staging/ca_events_corrected.json"))
def dn(x): d = float(x); return datetime.fromordinal(int(d)) + timedelta(days=d % 1) - timedelta(days=366)
LA = ZoneInfo("America/Los_Angeles")
print("pid\tcode_local\tdst\toffset_GE\tfrac_offset\tcols")
seen=set()
for r in m.itertuples(index=False):
    if r.pid in ev and r.TypeCode.strip()=="CPA" and r.pid not in seen:
        seen.add(r.pid); t = dn(r.CodeTime); dst = int(t.replace(tzinfo=LA).dst().total_seconds()!=0)
        print(f"{r.pid}\t{t:%Y-%m-%d %H:%M:%S}\t{dst}\t{r.offset_GE}\t{float(r.offset_GE)%1:.4f}\t{len(xl.columns)}")
print("xlsx columns:", list(xl.columns))
