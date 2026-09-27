# Charted NBP (flowsheet, EHR calendar) vs cuff NBP events in the .vital-derived nbp_events.npy (GE clock):
# a charted "sys/dia" pair should match a cuff reading (|d|<=2 mmHg both) within +-10 min at the right lag.
import polars as pl, numpy as np, pandas as pd, json, glob, os, sys, time, collections, random, re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
LA = ZoneInfo("America/Los_Angeles"); T0 = time.time()
F = "/mnt/localdata/storage/UCSF/rdb_new/FLOWSHEETVALUEFACT"; D = "/mnt/localdata/storage/UCSF/rdb_database/FLOWSHEETROWDIM_New/FLOWSHEETROWDIM_New.csv"; ROOT = "/mnt/localdata100tb/physio_data/ucsf_all"
MONTHS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["2015_03", "2015_11"]
dim = pl.read_csv(D, infer_schema_length=0, encoding="utf8-lossy"); dim = dim.rename({c: c.strip("﻿") for c in dim.columns})
name = dict(zip(dim["FlowsheetRowKey"].to_list(), dim["Name"].to_list()))
BP_KEYS = {k for k, nm in name.items() if nm and re.search(r"\b(BP|BLOOD PRESSURE|NIBP)\b", nm.upper()) and not re.search(r"ANE|\bOB\b|\bOR\b|RETIRED|ORTHOSTATIC|GOAL|TARGET|CUFF SIZE|SITE|LOCATION|POSITION|METHOD|ARTERIAL|\bART\b|\bABP\b|\bPA\b|\bLA\b|PULMONARY", nm.upper())}
print("BP row keys considered:", len(BP_KEYS), "e.g.", [(k, name[k][:35]) for k in list(BP_KEYS)[:12]])
EHR_MINUS_GE_DAYS = 12
def ds(t): return int(t.replace(tzinfo=LA).dst().total_seconds() != 0)
def to_real(t_ge, og): return (t_ge.replace(tzinfo=LA).astimezone(timezone.utc) + timedelta(days=float(og))).astimezone(LA).replace(tzinfo=None)
ms2dt = lambda ms: datetime(1970, 1, 1) + timedelta(milliseconds=int(ms))
off = pd.read_parquet("/mnt/localdata/storage/mxwang/Physio_Data/workzone/outputs/ucsf_all/encounter_offset_table.parquet")
off["pid"] = off["Patient_ID_GE"].astype(str).str.strip().str.replace(r"^DE", "", regex=True); off["wf"] = off["Wynton_folder"].astype(str).str.strip()
cnt = off.groupby(["pid", "wf"]).size(); uniq = set(cnt[cnt == 1].index)
link = {(r.pid, r.wf): (str(r.Encounter_ID).strip(), float(r.offset), float(r.offset_GE)) for r in off.itertuples(index=False) if (r.pid, r.wf) in uniq}
man = json.load(open(f"{ROOT}/manifest.json"))
mranges = [(datetime(int(m[:4]), int(m[5:7]), 1), datetime(int(m[:4]) + (int(m[5:7]) == 12), int(m[5:7]) % 12 + 1, 1)) for m in MONTHS]
sel = []
for q in man:
    key = (str(q["patient_id_ge"]), str(q.get("wynton_folder")))
    if key not in link: continue
    s_ge, e_ge = ms2dt(q["wave_start_ms"]), ms2dt(q["wave_end_ms"]); s_ehr, e_ehr = s_ge + timedelta(days=EHR_MINUS_GE_DAYS), e_ge + timedelta(days=EHR_MINUS_GE_DAYS)
    if not any(s_ehr < b and e_ehr > a for a, b in mranges): continue
    enc, o, og = link[key]; s_real, e_real = to_real(s_ge, og), to_real(e_ge, og)
    if ds(s_ge) != ds(e_ge) or ds(s_ehr) != ds(e_ehr) or ds(s_real) != ds(e_real): continue
    sel.append(dict(entity=q["entity_id"], enc=enc, unit=q.get("unit"), p1=(ds(s_ge) - ds(s_real)) * 60, p2=(ds(s_ge) - ds(s_ehr)) * 60))
random.seed(0); random.shuffle(sel); cells = collections.defaultdict(list)
for r in sel: cells[(r["p1"], r["p2"])].append(r)
sel = [r for k, v in cells.items() for r in v[:200]]; enc_set = {r["enc"] for r in sel}
print("cells:", {k: len(v) for k, v in cells.items()}, "| selected", len(sel), "entities", flush=True)
rows = []; nfiles = 0; keycount = collections.Counter()
for m in MONTHS:
    for f in sorted(glob.glob(f"{F}/{m}_*.txt")):
        nfiles += 1
        try: df = pl.read_csv(f, infer_schema_length=0, encoding="utf8-lossy", truncate_ragged_lines=True, columns=["FlowsheetRowKey", "Value", "FlowDate", "FlowTime", "encounter_ID"])
        except Exception as e: continue
        df = df.filter(pl.col("FlowsheetRowKey").is_in(list(BP_KEYS)) & pl.col("encounter_ID").is_in(list(enc_set)))
        if df.height: rows.append(df); keycount.update(df["FlowsheetRowKey"].to_list())
        if time.time() - T0 > 20 * 60: print("time guard"); break
fs = pl.concat(rows); print(f"scanned {nfiles} shards; BP rows {fs.height}; per key: {[(k, name[k][:30], n) for k, n in keycount.most_common(8)]}  [{time.time()-T0:.0f}s]", flush=True)
fs = fs.with_columns((pl.col("FlowDate") + " " + pl.col("FlowTime")).str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False).alias("t")).drop_nulls("t")
fs = fs.with_columns(pl.col("Value").str.extract(r"^\s*(\d{2,3})\s*/\s*(\d{2,3})", 1).cast(pl.Float64, strict=False).alias("sys"), pl.col("Value").str.extract(r"^\s*(\d{2,3})\s*/\s*(\d{2,3})", 2).cast(pl.Float64, strict=False).alias("dia")).drop_nulls(["sys", "dia"])
print("parsed sys/dia rows:", fs.height)
out = []; LAGS = (-60, 0, 60)
for r in sel:
    g = fs.filter(pl.col("encounter_ID") == r["enc"])
    if g.height < 5: continue
    try: ev = np.load(f"{ROOT}/{r['entity']}/nbp_events.npy")
    except Exception: continue
    if ev.size == 0: continue
    s = ev[ev["var_id"] == 157]; d = ev[ev["var_id"] == 158]
    if s.size < 5: continue
    dmap = dict(zip(d["time_ms"].tolist(), d["value"].tolist())); st = s["time_ms"].astype(np.int64); sv = s["value"].astype(np.float32); dv = np.array([dmap.get(int(t), np.nan) for t in st], dtype=np.float32)
    tc = np.array([int((t - timedelta(days=EHR_MINUS_GE_DAYS) - datetime(1970, 1, 1)).total_seconds() * 1000) for t in g["t"].to_list()]); cs = g["sys"].to_numpy(); cd = g["dia"].to_numpy()
    m = {}
    for L in LAGS:
        hit = 0
        for t_ms, a, b in zip(tc, cs, cd):
            c = t_ms + L * 60000; i0, i1 = np.searchsorted(st, c - 600000), np.searchsorted(st, c + 600000)
            if i1 > i0 and np.any((np.abs(sv[i0:i1] - a) <= 2) & (np.abs(dv[i0:i1] - b) <= 2)): hit += 1
        m[L] = hit
    n = len(tc); win = max(LAGS, key=lambda L: m[L]); srt = sorted(m.values(), reverse=True); margin = srt[0] - srt[1]
    out.append(dict(entity=r["entity"], unit=r["unit"], n_charted=n, p1=r["p1"], p2=r["p2"], win=win, hit_m60=m[-60], hit_0=m[0], hit_p60=m[60], margin=margin, hit_frac=round(srt[0] / n, 2)))
pd.DataFrame(out).to_csv("/projects/mwang80/staging/fs_nbp_probe_" + "_".join(MONTHS) + ".csv", index=False)
print(f"entities with charted NBP + cuff events: {len(out)}  [{time.time()-T0:.0f}s]")
oo = [o for o in out if o["margin"] >= 3]; print(f"clear winners (margin >= 3 matched readings): {len(oo)}; median hit fraction at winner {np.median([o['hit_frac'] for o in oo]) if oo else 0:.2f}")
tab = collections.defaultdict(collections.Counter)
for o in oo: tab[(o["p1"], o["p2"])][o["win"]] += 1
for k in sorted(tab): print(f"  predicted (H1 EHR-naive/GE-abs, H2 both-abs) = {k}: winner counts {dict(tab[k])}")
amb = [o for o in out if o["margin"] < 3]; print("ambiguous:", len(amb), "e.g.", amb[:3])
