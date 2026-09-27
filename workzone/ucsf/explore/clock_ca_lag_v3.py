# v3: per-event EHR->monitor lag measured from charted HR/SpO2/NBP within +-36 h of the event, restricted to charted points
# whose DST states (real and GE calendars) equal those at the event; E_monitor_v3 = naive + measured lag (fallback v2).
import polars as pl, numpy as np, pandas as pd, json, glob, os, time, collections
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
LA = ZoneInfo("America/Los_Angeles"); LA_DAYS = 12; T0 = time.time()
F = "/mnt/localdata/storage/UCSF/rdb_new/FLOWSHEETVALUEFACT"; ROOT = "/mnt/localdata100tb/physio_data/ucsf_all"; S = "/projects/mwang80/staging"
HR_KEYS = {"38524", "61799", "70230", "73515", "74163", "89443"}; SPO2_KEYS = {"2"}; BP_KEYS = {"32710"}
ev2 = json.load(open(f"{S}/ca_events_corrected_v2.json")); dl = json.load(open(f"{S}/ca_dst_delta.json")); summ = json.load(open(f"{S}/ca_t0_plots_v2/summary.json"))
cover = {e["entity"].split("_")[0]: e["entity"] for e in summ}
off = pd.read_parquet("/mnt/localdata/storage/mxwang/Physio_Data/workzone/outputs/ucsf_all/encounter_offset_table.parquet")
off["pid"] = off["Patient_ID_GE"].astype(str).str.strip().str.replace(r"^DE", "", regex=True); off["wf"] = off["Wynton_folder"].astype(str).str.strip()
cnt = off.groupby(["pid", "wf"]).size(); uniq = set(cnt[cnt == 1].index)
link = {(r.pid, r.wf): str(r.Encounter_ID).strip() for r in off.itertuples(index=False) if (r.pid, r.wf) in uniq}; og = dict(zip(off["pid"], pd.to_numeric(off["offset_GE"], errors="coerce")))
man = {q["entity_id"]: q for q in json.load(open(f"{ROOT}/manifest.json"))}
enc_of = {}
for pid, ent in cover.items():
    q = man[ent]; e = link.get((pid, str(q.get("wynton_folder"))))
    if e: enc_of[pid] = e
print(f"events with covering cycle {len(cover)}; with encounter link {len(enc_of)}  [{time.time()-T0:.0f}s]", flush=True)
PQ = f"{S}/ca_flowsheet_rows.parquet"
if os.path.exists(PQ): fs = pl.read_parquet(PQ)
else:
    rows = []; KEYS = list(HR_KEYS | SPO2_KEYS | BP_KEYS); ENC = list(set(enc_of.values()))
    for i, f in enumerate(sorted(glob.glob(f"{F}/*.txt"))):
        try: df = pl.read_csv(f, infer_schema_length=0, encoding="utf8-lossy", truncate_ragged_lines=True, columns=["FlowsheetRowKey", "Value", "FlowDate", "FlowTime", "encounter_ID"])
        except Exception: continue
        df = df.filter(pl.col("FlowsheetRowKey").is_in(KEYS) & pl.col("encounter_ID").is_in(ENC))
        if df.height: rows.append(df)
        if time.time() - T0 > 22 * 60: print("TIME GUARD", i); break
    fs = pl.concat(rows); fs.write_parquet(PQ)
print(f"flowsheet rows {fs.height}  [{time.time()-T0:.0f}s]", flush=True)
fs = fs.with_columns((pl.col("FlowDate") + " " + pl.col("FlowTime")).str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False).alias("t")).drop_nulls("t")
fs = fs.with_columns(pl.col("Value").cast(pl.Float64, strict=False).alias("num"), pl.col("Value").str.extract(r"^\s*(\d{2,3})\s*/\s*(\d{2,3})", 1).cast(pl.Float64, strict=False).alias("sys"), pl.col("Value").str.extract(r"^\s*(\d{2,3})\s*/\s*(\d{2,3})", 2).cast(pl.Float64, strict=False).alias("dia"))
fs = fs.with_columns(((pl.col("t") - pl.duration(days=LA_DAYS)).dt.epoch("ms")).alias("t_naive_ms"))
ms2dt = lambda ms: datetime(1970, 1, 1) + timedelta(milliseconds=int(ms))
def ds(t): return int(t.replace(tzinfo=LA).dst().total_seconds() != 0)
def to_real(t_ge, o): return (t_ge.replace(tzinfo=LA).astimezone(timezone.utc) + timedelta(days=float(o))).astimezone(LA).replace(tzinfo=None)
LAGS = (-60, 0, 60)
def mae_vote(tc, vc, tt, mon, lo, hi):
    m = (vc >= lo) & (vc <= hi); tc, vc = tc[m], vc[m]
    if tc.size < 6: return None
    mae = {}
    for L in LAGS:
        errs = []
        for t_ms, v in zip(tc, vc):
            c = t_ms + L * 60000; a, b = np.searchsorted(tt, c - 150000), np.searchsorted(tt, c + 150000); w = mon[a:b]; w = w[np.isfinite(w)]
            if w.size >= 10: errs.append(abs(float(np.median(w)) - v))
        mae[L] = float(np.mean(errs)) if len(errs) >= 6 else np.nan
    ok = [L for L in LAGS if not np.isnan(mae[L])]
    if len(ok) < 2: return None
    win = min(ok, key=lambda L: mae[L]); srt = sorted(mae[L] for L in ok); return win, srt[1] - srt[0]
res = {}
for pid, ent in cover.items():
    enc = enc_of.get(pid)
    if not enc: res[pid] = dict(lag=None, votes={}, note="no encounter link"); continue
    E2 = ev2[pid]; d = dl[pid]["delta_min"]; E_naive = E2 - int(d * 60000); o = og.get(pid)
    tE = ms2dt(E2); dE = (ds(tE), ds(to_real(tE, o)))
    g = fs.filter((pl.col("encounter_ID") == enc) & (pl.col("t_naive_ms") >= E_naive - 36 * 3600000) & (pl.col("t_naive_ms") <= E_naive + 36 * 3600000))
    if g.height == 0: res[pid] = dict(lag=None, votes={}, note="no charted rows within 36 h"); continue
    keep = [(ds(ms2dt(t)), ds(to_real(ms2dt(t), o))) == dE for t in g["t_naive_ms"].to_list()]
    g = g.filter(pl.Series(keep))
    if g.height == 0: res[pid] = dict(lag=None, votes={}, note="no rows in same DST regime"); continue
    hf = np.load(f"{ROOT}/{ent}/vitals_hf.npy", mmap_mode="r"); tm = np.load(f"{ROOT}/{ent}/time_ms.npy"); tt = (tm[:, None] + np.arange(15)[None, :] * 2000).reshape(-1); votes = {}
    for nm, keys, idx, lo, hi in (("HR", HR_KEYS, 0, 20, 250), ("SPO2", SPO2_KEYS, 1, 50, 100)):
        gg = g.filter(pl.col("FlowsheetRowKey").is_in(list(keys)) & pl.col("num").is_not_null())
        if gg.height >= 6:
            r = mae_vote(gg["t_naive_ms"].to_numpy().astype(np.int64), gg["num"].to_numpy(), tt, np.asarray(hf[:, :, idx], dtype=np.float32).reshape(-1), lo, hi)
            if r and r[1] >= 1.0: votes[nm] = (r[0], round(r[1], 2))
    gb = g.filter(pl.col("FlowsheetRowKey").is_in(list(BP_KEYS)) & pl.col("sys").is_not_null())
    if gb.height >= 4 and os.path.exists(f"{ROOT}/{ent}/nbp_events.npy"):
        ev = np.load(f"{ROOT}/{ent}/nbp_events.npy")
        if ev.size:
            s_ = ev[ev["var_id"] == 157]; d_ = ev[ev["var_id"] == 158]; dmap = dict(zip(d_["time_ms"].tolist(), d_["value"].tolist()))
            st = s_["time_ms"].astype(np.int64); sv = s_["value"].astype(np.float32); dv = np.array([dmap.get(int(t), np.nan) for t in st], dtype=np.float32); hits = {}
            for L in LAGS:
                h = 0
                for t_ms, a, b in zip(gb["t_naive_ms"].to_numpy().astype(np.int64), gb["sys"].to_numpy(), gb["dia"].to_numpy()):
                    c = t_ms + L * 60000; i0, i1 = np.searchsorted(st, c - 600000), np.searchsorted(st, c + 600000)
                    if i1 > i0 and np.any((np.abs(sv[i0:i1] - a) <= 2) & (np.abs(dv[i0:i1] - b) <= 2)): h += 1
                hits[L] = h
            win = max(LAGS, key=lambda L: hits[L]); srt = sorted(hits.values(), reverse=True)
            if srt[0] - srt[1] >= 3: votes["NBP"] = (win, srt[0] - srt[1])
    lags = [v[0] for v in votes.values()]; lag = lags[0] if lags and all(l == lags[0] for l in lags) else None
    res[pid] = dict(lag=lag, votes=votes, note="" if lag is not None else ("conflict" if votes else "no clear vote"), n_rows=g.height)
tab = collections.Counter((dl[p]["delta_min"], r["lag"]) for p, r in res.items()); print("(delta model, measured lag) over 178 events:", dict(sorted(tab.items(), key=lambda kv: str(kv[0]))))
print("unmeasured reasons:", collections.Counter(r["note"] for r in res.values() if r["lag"] is None))
ev3 = {}; src = collections.Counter()
for pid, t2 in ev2.items():
    r = res.get(pid); d = dl[pid]["delta_min"]
    if r and r["lag"] is not None: ev3[pid] = t2 - int(d * 60000) + int(r["lag"] * 60000); src["measured"] += 1
    else: ev3[pid] = t2; src["model (v2)"] += 1
print("v3 sources:", dict(src)); print("events where measured != model:")
for pid, r in res.items():
    if r["lag"] is not None and r["lag"] != dl[pid]["delta_min"]: print("  ", pid, cover[pid], man[cover[pid]].get("wynton_folder"), "delta", dl[pid]["delta_min"], "measured", r["lag"], r["votes"], "rows", r.get("n_rows"))
json.dump(ev3, open(f"{S}/ca_events_corrected_v3.json", "w")); json.dump(res, open(f"{S}/ca_event_lags_v3.json", "w"), indent=1); print("written  [%.0fs]" % (time.time() - T0))
