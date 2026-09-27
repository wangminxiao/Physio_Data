# v2: de-identification day shift applied in ABSOLUTE time (UTC), then expressed as LA wall clock at the shifted date.
import json, numpy as np, pandas as pd, polars as pl, collections
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
LA = ZoneInfo("America/Los_Angeles")
E = "/mnt/localdata/storage/mxwang/data/ucsf_EHR"; ROOT = "/mnt/localdata100tb/physio_data/ucsf_all"
off = pd.read_parquet("/mnt/localdata/storage/mxwang/Physio_Data/workzone/outputs/ucsf_all/encounter_offset_table.parquet")
off["pid"] = off["Patient_ID_GE"].astype(str).str.strip().str.replace(r"^DE", "", regex=True); off["Encounter_ID"] = off["Encounter_ID"].astype(str).str.strip(); off["offset_GE"] = pd.to_numeric(off["offset_GE"], errors="coerce")
xl = pd.read_excel(f"{E}/SAUCSFCodeBlue_FirstEvent_2013_2018_final.xlsx", dtype=str); xl["EID"] = xl["EID"].str.strip()
m = xl.merge(off[["Encounter_ID", "pid", "offset_GE"]], left_on="EID", right_on="Encounter_ID")
def dn(x): d = float(x); return datetime.fromordinal(int(d)) + timedelta(days=d % 1) - timedelta(days=366)
vw = pl.read_csv(f"{E}/bedanalysis_waveformExtraction/Output_new/ValidWaveTime_allEnc_eventtime.csv", infer_schema_length=0); vw = vw.rename({c: c.strip() for c in vw.columns})
csv_ev = {}
for p, e in zip(vw["Patient_ID_GE"].to_list(), vw["EventTime"].fill_null("").to_list()):
    p = p.strip()[2:] if p.strip().startswith("DE") else p.strip(); e = e.strip()
    if e and e != "-1" and p not in csv_ev: csv_ev[p] = e
ev, ev_naive, chk = {}, {}, []
for r in m.itertuples(index=False):
    if r.pid in csv_ev and pd.notna(r.offset_GE) and r.TypeCode.strip() == "CPA" and r.pid not in ev:
        L = dn(r.CodeTime); N = float(r.offset_GE)
        naive = L - timedelta(days=N)
        L_utc = L.replace(tzinfo=LA).astimezone(timezone.utc)
        absol = (L_utc - timedelta(days=N)).astimezone(LA).replace(tzinfo=None)
        ev[r.pid] = int(absol.replace(tzinfo=timezone.utc).timestamp() * 1000); ev_naive[r.pid] = int(naive.replace(tzinfo=timezone.utc).timestamp() * 1000)
        et = datetime.strptime(csv_ev[r.pid][:19], "%Y-%m-%dT%H:%M:%S")
        chk.append((round((et - (L_utc.replace(tzinfo=None) - timedelta(days=N))).total_seconds() / 60), round((et - absol).total_seconds() / 60), round((et - naive).total_seconds() / 60)))
print("events:", len(ev), "| absolute-naive (min):", collections.Counter(int((ev[p]-ev_naive[p])/60000) for p in ev))
print("CSV EventTime - [UTC(L) - N days]  (model A, expect 0):", collections.Counter(c[0] for c in chk))
print("CSV EventTime - absolute-shift wall clock (expect 420/480):", collections.Counter(c[1] for c in chk))
print("CSV EventTime - naive wall clock:", collections.Counter(c[2] for c in chk))
json.dump(ev, open("/projects/mwang80/staging/ca_events_corrected_v2.json", "w")); print("written v2 json")
man = json.load(open(f"{ROOT}/manifest.json")); by_pat = collections.defaultdict(list)
for q in man: by_pat[str(q["patient_id_ge"])].append(q)
cat = collections.Counter(); rows = []
for pid, t in ev.items():
    cyc = sorted(by_pat.get(pid, []), key=lambda q: q["wave_start_ms"])
    if not cyc: cat["no waveform"] += 1; continue
    inside = [q for q in cyc if q["wave_start_ms"] <= t <= q["wave_end_ms"] + 30000]
    t0 = cyc[0]["wave_start_ms"]; t1 = max(q["wave_end_ms"] + 30000 for q in cyc)
    if inside: pos = "inside a monitored cycle"
    elif t < t0: pos = "before first waveform"
    elif t > t1: pos = "after last waveform end (%s)" % ("<=30 min" if t - t1 <= 1.8e6 else "30 min-24 h" if t - t1 <= 86400e3 else ">24 h")
    else: pos = "in a gap between cycles"
    cat[pos] += 1
    n_tot = int((t1 - t0) // 30000) + 1; valid = np.zeros(n_tot, bool)
    for q in cyc:
        pl_ = np.load(f"{ROOT}/{q['entity_id']}/PLETH40.npy", mmap_mode="r"); n = pl_.shape[0]; vm = np.zeros(n, bool)
        for a in range(0, n, 4000): vm[a:a+4000] = np.isfinite(np.asarray(pl_[a:a+4000], dtype=np.float32)).mean(axis=1) >= 0.5
        k0 = int((q["wave_start_ms"] - t0) // 30000); valid[k0:k0 + n] |= vm[:max(0, n_tot - k0)]
    k = int(np.floor((t - t0) / 30000)); vb = valid[:max(0, min(k, n_tot))]; last = int(np.flatnonzero(vb)[-1]) if vb.any() else None
    gap = ((t - (t0 + (last + 1) * 30000)) / 60000) if last is not None else None; cov6 = float(valid[max(0, k-720):max(0,k)].mean()) if k > 0 else 0.0
    rows.append((gap, cov6, float(vb.sum() * 30 / 3600)))
print("event position (v2 times):", dict(cat))
g = np.array([r[0] for r in rows if r[0] is not None]); print(f"gap last PPG -> event (min): n={g.size} p10/p50/p90 = {np.percentile(g,10):.0f}/{np.median(g):.0f}/{np.percentile(g,90):.0f}; <=5: {int((g<=5).sum())} <=30: {int((g<=30).sum())} <=120: {int((g<=120).sum())}")
us = [r for r in rows if r[0] is not None and r[0] <= 30 and r[1] >= 0.5]; print(f"USABLE (gap<=30 min & PPG cov6>=0.5): {len(us)}; with >=24 h PPG before: {sum(1 for r in us if r[2] >= 24)}; >=12 h: {sum(1 for r in us if r[2] >= 12)}; PPG hours before p50 = {np.median([r[2] for r in us]) if us else 0:.1f}")
