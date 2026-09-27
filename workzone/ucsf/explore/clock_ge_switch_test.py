# Does the EHR->grid lag change across a GE-calendar DST switch that falls INSIDE a wave cycle?
import polars as pl, numpy as np, pandas as pd, json, os, collections
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
LA = ZoneInfo("America/Los_Angeles"); ROOT = "/mnt/localdata100tb/physio_data/ucsf_all"; S = "/projects/mwang80/staging"
HR_KEYS = {"38524", "61799", "70230", "73515", "74163", "89443"}; BP_KEYS = {"32710"}
fs = pl.read_parquet(f"{S}/ca_flowsheet_rows.parquet")
fs = fs.with_columns((pl.col("FlowDate") + " " + pl.col("FlowTime")).str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False).alias("t")).drop_nulls("t")
fs = fs.with_columns(pl.col("Value").cast(pl.Float64, strict=False).alias("num"), pl.col("Value").str.extract(r"^\s*(\d{2,3})\s*/\s*(\d{2,3})", 1).cast(pl.Float64, strict=False).alias("sys"), pl.col("Value").str.extract(r"^\s*(\d{2,3})\s*/\s*(\d{2,3})", 2).cast(pl.Float64, strict=False).alias("dia"), ((pl.col("t") - pl.duration(days=12)).dt.epoch("ms")).alias("tn"))
off = pd.read_parquet("/mnt/localdata/storage/mxwang/Physio_Data/workzone/outputs/ucsf_all/encounter_offset_table.parquet")
off["pid"] = off["Patient_ID_GE"].astype(str).str.strip().str.replace(r"^DE", "", regex=True); off["wf"] = off["Wynton_folder"].astype(str).str.strip()
cnt = off.groupby(["pid", "wf"]).size(); uniq = set(cnt[cnt == 1].index)
link = {(r.pid, r.wf): str(r.Encounter_ID).strip() for r in off.itertuples(index=False) if (r.pid, r.wf) in uniq}; og = dict(zip(off["pid"], pd.to_numeric(off["offset_GE"], errors="coerce")))
encs = set(fs["encounter_ID"].to_list()); man = json.load(open(f"{ROOT}/manifest.json"))
ms2dt = lambda ms: datetime(1970, 1, 1) + timedelta(milliseconds=int(ms))
def ds(t): return int(t.replace(tzinfo=LA).dst().total_seconds() != 0)
def to_real(t_ge, o): return (t_ge.replace(tzinfo=LA).astimezone(timezone.utc) + timedelta(days=float(o))).astimezone(LA).replace(tzinfo=None)
def switch_inside(a, b):   # first DST transition datetime strictly inside (a, b), naive LA wall clock
    t = a.replace(minute=0, second=0, microsecond=0)
    while t < b:
        if ds(t) != ds(t + timedelta(hours=1)): return t + timedelta(hours=1)
        t += timedelta(hours=1)
    return None
def lag_of(g, ent):
    hf = np.load(f"{ROOT}/{ent}/vitals_hf.npy", mmap_mode="r"); tm = np.load(f"{ROOT}/{ent}/time_ms.npy"); tt = (tm[:, None] + np.arange(15)[None, :] * 2000).reshape(-1); mon = np.asarray(hf[:, :, 0], dtype=np.float32).reshape(-1)
    gg = g.filter(pl.col("FlowsheetRowKey").is_in(list(HR_KEYS)) & (pl.col("num") >= 20) & (pl.col("num") <= 250)); res = {}
    for L in (-60, 0, 60):
        errs = []
        for t_ms, v in zip(gg["tn"].to_numpy().astype(np.int64), gg["num"].to_numpy()):
            c = t_ms + L * 60000; a, b = np.searchsorted(tt, c - 150000), np.searchsorted(tt, c + 150000); w = mon[a:b]; w = w[np.isfinite(w)]
            if w.size >= 10: errs.append(abs(float(np.median(w)) - v))
        res[L] = (float(np.mean(errs)), len(errs)) if len(errs) >= 5 else (np.nan, len(errs))
    ok = [L for L in res if not np.isnan(res[L][0])]
    if len(ok) < 2: return None
    win = min(ok, key=lambda L: res[L][0]); srt = sorted(res[L][0] for L in ok); return win, round(srt[1] - srt[0], 2), res[win][1]
n = 0
for q in man:
    pid = str(q["patient_id_ge"]); enc = link.get((pid, str(q.get("wynton_folder"))))
    if enc not in encs: continue
    s, e = ms2dt(q["wave_start_ms"]), ms2dt(q["wave_end_ms"]); o = og.get(pid)
    if o != o or (e - s) < timedelta(hours=12): continue
    sw_ge = switch_inside(s, e); rs, re_ = to_real(s, o), to_real(e, o); sw_real = switch_inside(rs, re_)
    if sw_ge is None and sw_real is None: continue
    kind = ("GE-switch" if sw_ge else "") + ("+" if sw_ge and sw_real else "") + ("real-switch" if sw_real else "")
    sw = sw_ge if sw_ge else None
    g = fs.filter(pl.col("encounter_ID") == enc); sms = int((s - datetime(1970, 1, 1)).total_seconds() * 1000); ems = int((e - datetime(1970, 1, 1)).total_seconds() * 1000)
    if sw_ge:
        swms = int((sw_ge - datetime(1970, 1, 1)).total_seconds() * 1000)
        before = lag_of(g.filter((pl.col("tn") >= sms) & (pl.col("tn") < swms - 3600000)), q["entity_id"]); after = lag_of(g.filter((pl.col("tn") > swms + 3600000) & (pl.col("tn") <= ems)), q["entity_id"])
        print(f"{kind:22s} {q['entity_id']} {q.get('unit')} GE {s:%Y-%m-%d %H:%M}..{e:%m-%d %H:%M} real {rs:%Y-%m-%d} GE-switch at {sw_ge:%m-%d %H:%M} | lag before {before} after {after} | model dst(real)={ds(rs)}/{ds(re_)} dst(GE)={ds(s)}->{ds(e)}")
    else:
        rsw_ge = ms2dt(int(((sw_real.replace(tzinfo=LA).astimezone(timezone.utc) - timedelta(days=float(o))).timestamp()) * 1000)) if False else None
        # real switch inside cycle: locate it on the GE grid via UTC (absolute model): t_ge = to_LA(UTC(sw_real) - o days)
        t_ge_sw = (sw_real.replace(tzinfo=LA).astimezone(timezone.utc) - timedelta(days=float(o))).astimezone(LA).replace(tzinfo=None); swms = int((t_ge_sw - datetime(1970, 1, 1)).total_seconds() * 1000)
        before = lag_of(g.filter((pl.col("tn") >= sms) & (pl.col("tn") < swms - 3600000)), q["entity_id"]); after = lag_of(g.filter((pl.col("tn") > swms + 3600000) & (pl.col("tn") <= ems)), q["entity_id"])
        print(f"{kind:22s} {q['entity_id']} {q.get('unit')} GE {s:%Y-%m-%d %H:%M}..{e:%m-%d %H:%M} real {rs:%Y-%m-%d} real-switch at {sw_real:%m-%d %H:%M} (GE {t_ge_sw:%m-%d %H:%M}) | lag before {before} after {after} | model dst(real)={ds(rs)}->{ds(re_)} dst(GE)={ds(s)}/{ds(e)}")
    n += 1
print("cycles examined:", n)
