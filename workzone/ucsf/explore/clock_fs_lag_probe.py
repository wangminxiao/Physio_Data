# Flowsheet (EHR-side, shifted by `offset`) charted HR vs monitor HR (vitals_hf, GE clock, shifted by offset_GE):
# which lag (-60 / 0 / +60 min) makes them agree, per entity, vs the lag predicted by two hypotheses.
import polars as pl, numpy as np, pandas as pd, json, glob, os, sys, time, collections, random
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
LA = ZoneInfo("America/Los_Angeles"); T0 = time.time()
F = "/mnt/localdata/storage/UCSF/rdb_new/FLOWSHEETVALUEFACT"; ROOT = "/mnt/localdata100tb/physio_data/ucsf_all"
MONTHS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["2015_03", "2015_11"]
HR_KEYS = {"38524", "61799", "70230", "73515", "74163", "89443"}; SPO2_KEYS = {"2"}
EHR_MINUS_GE_DAYS = 12
def ds(t): return int(t.replace(tzinfo=LA).dst().total_seconds() != 0)
def to_real(t_ge, og): return (t_ge.replace(tzinfo=LA).astimezone(timezone.utc) + timedelta(days=float(og))).astimezone(LA).replace(tzinfo=None)
ms2dt = lambda ms: datetime(1970, 1, 1) + timedelta(milliseconds=int(ms))
off = pd.read_parquet("/mnt/localdata/storage/mxwang/Physio_Data/workzone/outputs/ucsf_all/encounter_offset_table.parquet")
off["pid"] = off["Patient_ID_GE"].astype(str).str.strip().str.replace(r"^DE", "", regex=True); off["wf"] = off["Wynton_folder"].astype(str).str.strip()
cnt = off.groupby(["pid", "wf"]).size(); uniq = set(cnt[cnt == 1].index)
link = {(r.pid, r.wf): (str(r.Encounter_ID).strip(), str(r.Patient_ID).strip(), float(r.offset), float(r.offset_GE)) for r in off.itertuples(index=False) if (r.pid, r.wf) in uniq}
man = json.load(open(f"{ROOT}/manifest.json"))
# month windows in the EHR calendar
mranges = []
for m in MONTHS:
    y, mo = int(m[:4]), int(m[5:7]); a = datetime(y, mo, 1); b = datetime(y + (mo == 12), mo % 12 + 1, 1); mranges.append((a, b))
sel = []
for q in man:
    key = (str(q["patient_id_ge"]), str(q.get("wynton_folder")))
    if key not in link: continue
    s_ge, e_ge = ms2dt(q["wave_start_ms"]), ms2dt(q["wave_end_ms"]); s_ehr, e_ehr = s_ge + timedelta(days=EHR_MINUS_GE_DAYS), e_ge + timedelta(days=EHR_MINUS_GE_DAYS)
    if not any(s_ehr < b and e_ehr > a for a, b in mranges): continue
    enc, pid_ehr, o, og = link[key]
    s_real, e_real = to_real(s_ge, og), to_real(e_ge, og)
    if ds(s_ge) != ds(e_ge) or ds(s_ehr) != ds(e_ehr) or ds(s_real) != ds(e_real): continue   # cycle straddles a DST switch in some calendar -> skip
    p1 = (ds(s_ge) - ds(s_real)) * 60; p2 = (ds(s_ge) - ds(s_ehr)) * 60   # H1: EHR naive / GE absolute ; H2: both absolute
    sel.append(dict(entity=q["entity_id"], enc=enc, unit=q.get("unit"), s_ge=s_ge, e_ge=e_ge, og=og, o=o, p1=p1, p2=p2, dst=(ds(s_real), ds(s_ehr), ds(s_ge))))
random.seed(0); random.shuffle(sel)
cells = collections.defaultdict(list)
for r in sel: cells[(r["p1"], r["p2"])].append(r)
print("candidate entities per (P1,P2) cell:", {k: len(v) for k, v in cells.items()})
sel = [r for k, v in cells.items() for r in v[:150]]
enc_set = {r["enc"] for r in sel}; print(f"selected {len(sel)} entities / {len(enc_set)} encounters  [{time.time()-T0:.0f}s]", flush=True)
# ---- scan shards
rows = []; nfiles = 0; keycount = collections.Counter()
for m in MONTHS:
    for f in sorted(glob.glob(f"{F}/{m}_*.txt")):
        nfiles += 1
        try:
            df = pl.read_csv(f, infer_schema_length=0, encoding="utf8-lossy", truncate_ragged_lines=True, columns=["FlowsheetRowKey", "Value", "FlowDate", "FlowTime", "encounter_ID"])
        except Exception as e: print("read fail", os.path.basename(f), e); continue
        df = df.filter(pl.col("FlowsheetRowKey").is_in(list(HR_KEYS | SPO2_KEYS)) & pl.col("encounter_ID").is_in(list(enc_set)))
        if df.height: rows.append(df); keycount.update(df["FlowsheetRowKey"].to_list())
        if time.time() - T0 > 20 * 60: print("time guard hit at", nfiles, "files"); break
    if nfiles % 500 == 0: print(f"  scanned {nfiles} files [{time.time()-T0:.0f}s]", flush=True)
fs = pl.concat(rows) if rows else None
print(f"scanned {nfiles} shards, matched rows {fs.height if fs is not None else 0}, per key {dict(keycount)}  [{time.time()-T0:.0f}s]", flush=True)
if fs is None: sys.exit(0)
fs = fs.with_columns(pl.col("Value").cast(pl.Float64, strict=False)).drop_nulls("Value")
fs = fs.with_columns((pl.col("FlowDate") + " " + pl.col("FlowTime")).str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False).alias("t")).drop_nulls("t")
by_enc = {k: g for k, g in fs.group_by("encounter_ID")} if False else {}
for enc in enc_set: by_enc[enc] = fs.filter(pl.col("encounter_ID") == enc)
LAGS = list(range(-120, 121, 5)); out = []
for r in sel:
    g = by_enc.get(r["enc"]);
    if g is None or g.height == 0: continue
    try:
        hf = np.load(f"{ROOT}/{r['entity']}/vitals_hf.npy", mmap_mode="r"); tm = np.load(f"{ROOT}/{r['entity']}/time_ms.npy")
    except Exception as e: continue
    hr = np.asarray(hf[:, :, 0], dtype=np.float32).reshape(-1); tt = (tm[:, None] + np.arange(15)[None, :] * 2000).reshape(-1)   # 2-s grid
    for sig, keys, lo, hi in (("HR", HR_KEYS, 20, 250), ("SPO2", SPO2_KEYS, 50, 100)):
        gg = g.filter(pl.col("FlowsheetRowKey").is_in(list(keys)) & (pl.col("Value") >= lo) & (pl.col("Value") <= hi))
        if gg.height < 8: continue
        if sig == "SPO2": mon = np.asarray(hf[:, :, 1], dtype=np.float32).reshape(-1)
        else: mon = hr
        tc = np.array([int((t - timedelta(days=EHR_MINUS_GE_DAYS) - datetime(1970, 1, 1)).total_seconds() * 1000) for t in gg["t"].to_list()]); vc = gg["Value"].to_numpy()
        res = {}
        for L in LAGS:
            errs = []
            for t_ms, v in zip(tc, vc):
                c = t_ms + L * 60000; a, b = np.searchsorted(tt, c - 150000), np.searchsorted(tt, c + 150000)
                w = mon[a:b]; w = w[np.isfinite(w)]
                if w.size >= 10: errs.append(abs(float(np.median(w)) - v))
            res[L] = (float(np.mean(errs)) if len(errs) >= 8 else np.nan, len(errs))
        if all(np.isnan(res[L][0]) for L in (-60, 0, 60)): continue
        three = {L: res[L][0] for L in (-60, 0, 60)}; win = min((L for L in three if not np.isnan(three[L])), key=lambda L: three[L])
        srt = sorted(v for v in three.values() if not np.isnan(v)); margin = (srt[1] - srt[0]) if len(srt) > 1 else np.nan
        best = min((L for L in LAGS if not np.isnan(res[L][0])), key=lambda L: res[L][0])
        out.append(dict(entity=r["entity"], unit=r["unit"], sig=sig, n=int(res[0][1] or res[win][1]), p1=r["p1"], p2=r["p2"], dst=r["dst"], win=win, margin=round(margin, 2), best=best, mae_m60=round(three[-60], 2) if not np.isnan(three[-60]) else None, mae_0=round(three[0], 2) if not np.isnan(three[0]) else None, mae_p60=round(three[60], 2) if not np.isnan(three[60]) else None))
print(f"entities with a lag estimate: {len(out)}  [{time.time()-T0:.0f}s]")
pd.DataFrame(out).to_csv("/projects/mwang80/staging/fs_lag_probe_" + "_".join(MONTHS) + ".csv", index=False)
for sig in ("HR", "SPO2"):
    oo = [o for o in out if o["sig"] == sig and o["margin"] is not None and o["margin"] >= 1.0]
    print(f"\n== {sig}: entities with clear winner (margin>=1 bpm/%): {len(oo)}")
    tab = collections.defaultdict(collections.Counter)
    for o in oo: tab[(o["p1"], o["p2"])][o["win"]] += 1
    for k in sorted(tab): print(f"  predicted (H1 EHR-naive/GE-abs, H2 both-abs) = {k}: winner counts {dict(tab[k])}")
    fine = collections.Counter(o["best"] for o in oo); print("  fine-grid best lag top:", fine.most_common(8))
