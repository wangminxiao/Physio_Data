# Measure, per wave cycle of every CA patient, the EHR->monitor clock lag (-60/0/+60 min) from charted HR/SpO2/NBP vs monitor
# (vitals_hf / nbp_events). Then E_monitor_v3 = naive Code Blue time + measured lag (fallback: v2 absolute model).
import polars as pl, numpy as np, pandas as pd, json, glob, os, sys, time, collections
from datetime import datetime, timedelta
LA_DAYS = 12; T0 = time.time()
F = "/mnt/localdata/storage/UCSF/rdb_new/FLOWSHEETVALUEFACT"; ROOT = "/mnt/localdata100tb/physio_data/ucsf_all"; S = "/projects/mwang80/staging"
HR_KEYS = {"38524", "61799", "70230", "73515", "74163", "89443"}; SPO2_KEYS = {"2"}; BP_KEYS = {"32710"}
ev2 = json.load(open(f"{S}/ca_events_corrected_v2.json")); dl = json.load(open(f"{S}/ca_dst_delta.json")); summ = json.load(open(f"{S}/ca_t0_plots_v2/summary.json"))
cover = {e["entity"].split("_")[0]: e["entity"] for e in summ}
off = pd.read_parquet("/mnt/localdata/storage/mxwang/Physio_Data/workzone/outputs/ucsf_all/encounter_offset_table.parquet")
off["pid"] = off["Patient_ID_GE"].astype(str).str.strip().str.replace(r"^DE", "", regex=True); off["wf"] = off["Wynton_folder"].astype(str).str.strip()
cnt = off.groupby(["pid", "wf"]).size(); uniq = set(cnt[cnt == 1].index)
link = {(r.pid, r.wf): str(r.Encounter_ID).strip() for r in off.itertuples(index=False) if (r.pid, r.wf) in uniq}
man = json.load(open(f"{ROOT}/manifest.json")); ents = []
for q in man:
    pid = str(q["patient_id_ge"])
    if pid in ev2:
        enc = link.get((pid, str(q.get("wynton_folder"))))
        ents.append(dict(entity=q["entity_id"], pid=pid, enc=enc, folder=q.get("wynton_folder"), unit=q.get("unit"), s=q["wave_start_ms"], e=q["wave_end_ms"]))
enc_set = {x["enc"] for x in ents if x["enc"]}; print(f"CA patients {len(ev2)}; their entities {len(ents)}; linked encounters {len(enc_set)}; entities without link {sum(1 for x in ents if not x['enc'])}  [{time.time()-T0:.0f}s]", flush=True)
rows = []; nfiles = 0; KEYS = list(HR_KEYS | SPO2_KEYS | BP_KEYS); ENC = list(enc_set)
for f in sorted(glob.glob(f"{F}/*.txt")):
    nfiles += 1
    try: df = pl.read_csv(f, infer_schema_length=0, encoding="utf8-lossy", truncate_ragged_lines=True, columns=["FlowsheetRowKey", "Value", "FlowDate", "FlowTime", "encounter_ID"])
    except Exception: continue
    df = df.filter(pl.col("FlowsheetRowKey").is_in(KEYS) & pl.col("encounter_ID").is_in(ENC))
    if df.height: rows.append(df)
    if nfiles % 5000 == 0: print(f"  {nfiles} shards [{time.time()-T0:.0f}s]", flush=True)
    if time.time() - T0 > 24 * 60: print("TIME GUARD at", nfiles); break
fs = pl.concat(rows); print(f"scanned {nfiles} shards; rows {fs.height}  [{time.time()-T0:.0f}s]", flush=True)
fs = fs.with_columns((pl.col("FlowDate") + " " + pl.col("FlowTime")).str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False).alias("t")).drop_nulls("t")
fs = fs.with_columns(pl.col("Value").cast(pl.Float64, strict=False).alias("num"), pl.col("Value").str.extract(r"^\s*(\d{2,3})\s*/\s*(\d{2,3})", 1).cast(pl.Float64, strict=False).alias("sys"), pl.col("Value").str.extract(r"^\s*(\d{2,3})\s*/\s*(\d{2,3})", 2).cast(pl.Float64, strict=False).alias("dia"))
fs = fs.with_columns(((pl.col("t") - pl.duration(days=LA_DAYS)).dt.epoch("ms")).alias("t_naive_ms"))
LAGS = (-60, 0, 60)
def series_vote(tc, vc, tt, mon, lo, hi):
    m = (vc >= lo) & (vc <= hi); tc, vc = tc[m], vc[m]
    if tc.size < 8: return None
    mae = {}
    for L in LAGS:
        errs = []
        for t_ms, v in zip(tc, vc):
            c = t_ms + L * 60000; a, b = np.searchsorted(tt, c - 150000), np.searchsorted(tt, c + 150000); w = mon[a:b]; w = w[np.isfinite(w)]
            if w.size >= 10: errs.append(abs(float(np.median(w)) - v))
        mae[L] = float(np.mean(errs)) if len(errs) >= 8 else np.nan
    ok = [L for L in LAGS if not np.isnan(mae[L])]
    if len(ok) < 2: return None
    win = min(ok, key=lambda L: mae[L]); srt = sorted(mae[L] for L in ok); return (win, srt[1] - srt[0], mae)
out = []
for x in ents:
    if not x["enc"]: continue
    g = fs.filter(pl.col("encounter_ID") == x["enc"])
    if g.height == 0: out.append(dict(**x, votes={}, lag=None, note="no flowsheet rows")); continue
    try: hf = np.load(f"{ROOT}/{x['entity']}/vitals_hf.npy", mmap_mode="r"); tm = np.load(f"{ROOT}/{x['entity']}/time_ms.npy")
    except Exception: out.append(dict(**x, votes={}, lag=None, note="no vitals_hf")); continue
    tt = (tm[:, None] + np.arange(15)[None, :] * 2000).reshape(-1); votes = {}
    for nm, keys, idx, lo, hi, mm in (("HR", HR_KEYS, 0, 20, 250, 1.0), ("SPO2", SPO2_KEYS, 1, 50, 100, 1.0)):
        gg = g.filter(pl.col("FlowsheetRowKey").is_in(list(keys)) & pl.col("num").is_not_null())
        if gg.height >= 8:
            r = series_vote(gg["t_naive_ms"].to_numpy().astype(np.int64), gg["num"].to_numpy(), tt, np.asarray(hf[:, :, idx], dtype=np.float32).reshape(-1), lo, hi)
            if r and r[1] >= mm: votes[nm] = (r[0], round(r[1], 2))
    gb = g.filter(pl.col("FlowsheetRowKey").is_in(list(BP_KEYS)) & pl.col("sys").is_not_null())
    try: ev = np.load(f"{ROOT}/{x['entity']}/nbp_events.npy")
    except Exception: ev = np.zeros(0)
    if gb.height >= 5 and ev.size:
        s_ = ev[ev["var_id"] == 157]; d_ = ev[ev["var_id"] == 158]; dmap = dict(zip(d_["time_ms"].tolist(), d_["value"].tolist()))
        st = s_["time_ms"].astype(np.int64); sv = s_["value"].astype(np.float32); dv = np.array([dmap.get(int(t), np.nan) for t in st], dtype=np.float32)
        if st.size >= 5:
            hits = {}
            for L in LAGS:
                h = 0
                for t_ms, a, b in zip(gb["t_naive_ms"].to_numpy().astype(np.int64), gb["sys"].to_numpy(), gb["dia"].to_numpy()):
                    c = t_ms + L * 60000; i0, i1 = np.searchsorted(st, c - 600000), np.searchsorted(st, c + 600000)
                    if i1 > i0 and np.any((np.abs(sv[i0:i1] - a) <= 2) & (np.abs(dv[i0:i1] - b) <= 2)): h += 1
                hits[L] = h
            win = max(LAGS, key=lambda L: hits[L]); srt = sorted(hits.values(), reverse=True)
            if srt[0] - srt[1] >= 3: votes["NBP"] = (win, srt[0] - srt[1])
    lags = [v[0] for v in votes.values()]
    lag = lags[0] if lags and all(l == lags[0] for l in lags) else None
    out.append(dict(**x, votes=votes, lag=lag, note="" if lag is not None else ("conflict " + str(votes) if votes else "no clear vote")))
print(f"entities measured: {sum(1 for o in out if o['lag'] is not None)} / {len(out)}  [{time.time()-T0:.0f}s]")
# per patient
by_p = collections.defaultdict(list)
for o in out:
    if o["lag"] is not None: by_p[o["pid"]].append(o["lag"])
pat_lag = {p: (v[0] if all(l == v[0] for l in v) else "mixed") for p, v in by_p.items()}
print("patients with a measured lag:", len(pat_lag), "| mixed across cycles:", sum(1 for v in pat_lag.values() if v == "mixed"))
tab = collections.Counter()
for p, v in pat_lag.items():
    d = dl[p]["delta_min"]; tab[(d, v)] += 1
print("(delta from absolute model, measured lag) counts:", dict(sorted(tab.items(), key=lambda kv: str(kv[0]))))
ev3 = {}; src = collections.Counter(); flips = []
for p, t2 in ev2.items():
    d = dl[p]["delta_min"]; naive = t2 - int(d * 60000); ce = cover.get(p)
    lag_c = next((o["lag"] for o in out if o["entity"] == ce and o["lag"] is not None), None)
    lag = lag_c if lag_c is not None else (pat_lag.get(p) if pat_lag.get(p) not in (None, "mixed") else None)
    if lag is None: ev3[p] = t2; src["v2 (model)"] += 1
    else:
        ev3[p] = naive + int(lag * 60000); src["measured:covering" if lag_c is not None else "measured:other cycle"] += 1
        if lag != d: flips.append((p, ce, d, lag, [o["folder"] for o in out if o["entity"] == ce]))
print("v3 source:", dict(src)); print("events where measured lag != absolute model (delta -> measured):"); [print("  ", f) for f in flips]
json.dump(ev3, open(f"{S}/ca_events_corrected_v3.json", "w")); json.dump({o["entity"]: dict(pid=o["pid"], lag=o["lag"], votes=o["votes"], note=o["note"], folder=o["folder"]) for o in out}, open(f"{S}/ca_entity_lags.json", "w"), indent=1)
print("written v3 json + entity lags  [%.0fs]" % (time.time() - T0))
