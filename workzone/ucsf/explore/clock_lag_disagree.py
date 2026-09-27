import json, collections
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
LA = ZoneInfo("America/Los_Angeles"); S = "/projects/mwang80/staging"; ROOT = "/mnt/localdata100tb/physio_data/ucsf_all"
lags = json.load(open(f"{S}/ca_entity_lags.json")); ev2 = json.load(open(f"{S}/ca_events_corrected_v2.json")); dl = json.load(open(f"{S}/ca_dst_delta.json"))
man = {q["entity_id"]: q for q in json.load(open(f"{ROOT}/manifest.json"))}
import pandas as pd
off = pd.read_parquet("/mnt/localdata/storage/mxwang/Physio_Data/workzone/outputs/ucsf_all/encounter_offset_table.parquet")
off["pid"] = off["Patient_ID_GE"].astype(str).str.strip().str.replace(r"^DE", "", regex=True); og = dict(zip(off["pid"], pd.to_numeric(off["offset_GE"], errors="coerce")))
ms2dt = lambda ms: datetime(1970, 1, 1) + timedelta(milliseconds=int(ms))
def ds(t): return int(t.replace(tzinfo=LA).dst().total_seconds() != 0)
def to_real(t_ge, o): return (t_ge.replace(tzinfo=LA).astimezone(timezone.utc) + timedelta(days=float(o))).astimezone(LA).replace(tzinfo=None)
by_p = collections.defaultdict(list)
for ent, v in lags.items(): by_p[v["pid"]].append((ent, v))
def show(pid):
    print(f"--- patient {pid}: delta {dl[pid]['delta_min']} | event GE {ms2dt(ev2[pid]):%Y-%m-%d %H:%M} | offset_GE {og.get(pid)}")
    for ent, v in sorted(by_p[pid], key=lambda kv: man[kv[0]]["wave_start_ms"]):
        q = man[ent]; s, e = ms2dt(q["wave_start_ms"]), ms2dt(q["wave_end_ms"]); o = og.get(pid)
        rs, re_ = (to_real(s, o), to_real(e, o)) if o == o else (None, None)
        strad = (ds(rs) != ds(re_)) if rs else None
        covers = q["wave_start_ms"] <= ev2[pid] <= q["wave_end_ms"] + 30000
        print(f"   {ent} {q.get('unit')} {v['folder']} GE {s:%m-%d %H:%M}..{e:%m-%d %H:%M} real {rs:%Y-%m-%d} dst(real s/e)={ds(rs)}/{ds(re_)} straddles={strad} covers_event={covers} lag={v['lag']} votes={v['votes']} {v['note'][:40]}")
for pid in ["669941987999192", "62174208153448", "249219710629842", "726929272216222", "977095334947518", "3165754646471", "329361886408588"]: show(pid)
print("\n===== mixed patients")
for pid, items in by_p.items():
    ls = {v["lag"] for _, v in items if v["lag"] is not None}
    if len(ls) > 1: show(pid)
