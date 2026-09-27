# Final CA event times on the entity grid: measured lag (v3) where available; otherwise the refined model
# lag = (dst(GE at cycle START) - dst(real at event)) * 60 min  [the grid does not observe DST switches on the GE calendar].
import json, collections, pandas as pd
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
LA = ZoneInfo("America/Los_Angeles"); S = "/projects/mwang80/staging"; ROOT = "/mnt/localdata100tb/physio_data/ucsf_all"
ev2 = json.load(open(f"{S}/ca_events_corrected_v2.json")); ev3 = json.load(open(f"{S}/ca_events_corrected_v3.json")); dl = json.load(open(f"{S}/ca_dst_delta.json")); lag3 = json.load(open(f"{S}/ca_event_lags_v3.json"))
summ = json.load(open(f"{S}/ca_t0_plots_v2/summary.json")); cover = {e["entity"].split("_")[0]: e["entity"] for e in summ}
man = {q["entity_id"]: q for q in json.load(open(f"{ROOT}/manifest.json"))}
off = pd.read_parquet("/mnt/localdata/storage/mxwang/Physio_Data/workzone/outputs/ucsf_all/encounter_offset_table.parquet"); off["pid"] = off["Patient_ID_GE"].astype(str).str.strip().str.replace(r"^DE", "", regex=True); og = dict(zip(off["pid"], pd.to_numeric(off["offset_GE"], errors="coerce")))
ms2dt = lambda ms: datetime(1970, 1, 1) + timedelta(milliseconds=int(ms))
def ds(t): return int(t.replace(tzinfo=LA).dst().total_seconds() != 0)
def to_real(t_ge, o): return (t_ge.replace(tzinfo=LA).astimezone(timezone.utc) + timedelta(days=float(o))).astimezone(LA).replace(tzinfo=None)
final = {}; src = collections.Counter(); changed = []
for pid, t2 in ev2.items():
    d = dl[pid]["delta_min"]; naive = t2 - int(d * 60000); r = lag3.get(pid)
    if r and r.get("lag") is not None: final[pid] = naive + int(r["lag"] * 60000); src["measured"] += 1; continue
    ent = cover.get(pid)
    if ent is None: final[pid] = t2; src["model v2 (no cycle)"] += 1; continue
    s = ms2dt(man[ent]["wave_start_ms"]); tE = ms2dt(t2); o = og.get(pid)
    d_ref = (ds(s) - ds(to_real(tE, o))) * 60
    final[pid] = naive + int(d_ref * 60000); src["refined model" if d_ref != d else "model (same as v2)"] += 1
    if d_ref != d: changed.append((pid, ent, man[ent].get("wynton_folder"), "delta v2", d, "refined", d_ref))
print("sources:", dict(src)); print("events changed vs v2/v3 by the refined model:", changed)
n_diff3 = sum(1 for p in final if final[p] != ev3[p]); print("events differing from v3:", n_diff3)
json.dump(final, open(f"{S}/ca_events_final.json", "w")); print("RERUN" if n_diff3 else "NO_RERUN")
