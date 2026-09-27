"""Read-only probe of UCSF raw file formats (.adibin CFWB waveforms, .vital numerics).

Samples ~12 patients spread across cohort folders, one bed sub-directory each, and reports
ONLY aggregates: header layouts, per-channel native-rate evidence, .vital cadence/units/
sentinels, and adibin-vs-vital time alignment. No patient identifiers are printed.
"""
import os, sys, struct, csv, random, time, re
from collections import Counter, defaultdict
from datetime import datetime, timezone
import numpy as np

ROOT = "/mnt/localdata/storage/UCSF"
N_PAT = int(sys.argv[1]) if len(sys.argv) > 1 else 12
CHUNK_SEC = 300          # waveform chunk analysed per file
random.seed(1)
t0 = time.time()

CFWB_FMT = "<4si d iiiii dd iiii"; CFWB_SIZE = struct.calcsize(CFWB_FMT)      # 68
CH_FMT = "<32s32s4d"; CH_SIZE = struct.calcsize(CH_FMT)                        # 96
VH_FMT = "<16s8s8s4s iiiii d"; VH_SIZE = struct.calcsize(VH_FMT)              # 56
DTYPES = {1: np.float64, 2: np.float32, 3: np.int16}

def cstr(b): return b.split(b"\0", 1)[0].decode("latin-1")

def to_ms(y, mo, d, h, mi, s):
    try:
        si = int(s); return int(datetime(y, mo, d, h, mi, si, tzinfo=timezone.utc).timestamp() * 1000) + int(round((s - si) * 1000))
    except Exception:
        return None

def read_adibin_header(p):
    with open(p, "rb") as f:
        hb = f.read(CFWB_SIZE)
        magic, ver, spt, y, mo, d, h, mi, sec, trig, nch, nsamp, tch, fmt = struct.unpack(CFWB_FMT, hb)
        chans = []
        for _ in range(nch):
            t, u, sc, off, hi, lo = struct.unpack(CH_FMT, f.read(CH_SIZE))
            chans.append((cstr(t), cstr(u), sc, off, hi, lo))
    return dict(magic=magic, ver=ver, spt=spt, start_ms=to_ms(y, mo, d, h, mi, sec), trig=trig, nch=nch,
                nsamp=nsamp, tch=tch, fmt=fmt, chans=chans, hdr_bytes=CFWB_SIZE + CH_SIZE * nch,
                dur_s=nsamp * spt)

def chan_metrics(raw, spt):
    """raw: 1-D int16/float chunk of one channel. Returns dict of native-rate evidence."""
    m = {}
    if raw.dtype == np.int16:
        gap = (raw == -32767) | (raw == -32768)
    else:
        gap = ~np.isfinite(raw)
    m["gap_pct"] = 100.0 * gap.mean()
    x = raw[~gap].astype(np.float64)
    if x.size < 2400:
        return m
    d1 = np.diff(x); m["dup_pct"] = 100.0 * (d1 == 0).mean()
    d2 = np.diff(x, 2); m["lin_pct"] = 100.0 * (np.abs(d2) < 1e-9).mean()
    # run lengths of constant value
    chg = np.flatnonzero(d1 != 0); runs = np.diff(chg) if chg.size > 1 else np.array([1])
    m["run_mode"] = int(Counter(runs.tolist()).most_common(1)[0][0]) if runs.size else 1
    fs = 1.0 / spt; nblk = int(x.size // fs)
    if nblk >= 5:
        blk = x[: int(nblk * fs)].reshape(nblk, int(fs))
        m["distinct_per_s"] = float(np.median([len(np.unique(b)) for b in blk]))
        m["changes_per_s"] = float(np.median((np.diff(blk, axis=1) != 0).sum(axis=1)))
    # spectral: longest gap-free stretch, up to 60 s
    seg = x[: min(x.size, int(60 * fs))]
    seg = seg - seg.mean()
    if seg.std() > 0:
        P = np.abs(np.fft.rfft(seg * np.hanning(seg.size))) ** 2; fr = np.fft.rfftfreq(seg.size, spt); tot = P.sum() + 1e-12
        for lo, hi in ((0, 15), (15, 30), (30, 60), (60, 120)):
            m[f"pow{lo}-{hi}"] = 100.0 * P[(fr >= lo) & (fr < hi)].sum() / tot
    m["quant_step"] = float(np.min(np.abs(d1[d1 != 0]))) if (d1 != 0).any() else 0.0
    return m

def analyse_adibin_data(p, h, want=None):
    dt = DTYPES.get(h["fmt"]); out = {}
    if dt is None or h["nsamp"] <= 0: return out
    n_chunk = min(h["nsamp"], int(CHUNK_SEC / h["spt"]))
    start = max(0, (h["nsamp"] - n_chunk) // 2)
    mm = np.memmap(p, dtype=dt, mode="r", offset=h["hdr_bytes"], shape=(h["nsamp"], h["nch"]))
    block = np.array(mm[start:start + n_chunk, :]); del mm
    for i, ch in enumerate(h["chans"]):
        out[ch[0]] = chan_metrics(block[:, i], h["spt"])
    return out

def read_vital(p, max_dt=200_000):
    size = os.path.getsize(p)
    if size < VH_SIZE: return None
    with open(p, "rb") as f:
        lab, uom, unit, bed, y, mo, d, h, mi, sec = struct.unpack(VH_FMT, f.read(VH_SIZE))
    n = (size - VH_SIZE) // 32
    r = dict(label=cstr(lab), uom=cstr(uom), unit=cstr(unit), bed=cstr(bed), zero_time=(y == 0),
             start_ms=to_ms(y, mo, d, h, mi, sec), n=n)
    if n <= 0: return r
    mm = np.memmap(p, dtype=np.float64, mode="r", offset=VH_SIZE, shape=(n, 4))
    val = np.array(mm[:, 0]); off = np.array(mm[:, 1]); lo = np.array(mm[:, 2]); hi = np.array(mm[:, 3]); del mm
    r["first_off"] = float(off[0]); r["last_off"] = float(off[-1]); r["off_monotonic"] = bool(np.all(np.diff(off) >= 0))
    dts = np.diff(off)
    if dts.size > max_dt: dts = dts[np.linspace(0, dts.size - 1, max_dt).astype(int)]
    r["dts"] = dts
    sent = (val == -999999) | (val <= -99999)
    r["sent_pct"] = 100.0 * sent.mean(); r["val"] = val[~sent]
    r["lo_top"] = Counter(np.round(lo[:5000], 3).tolist()).most_common(2); r["hi_top"] = Counter(np.round(hi[:5000], 3).tolist()).most_common(2)
    r["off_int_pct"] = 100.0 * np.mean(np.abs(off[:20000] - np.round(off[:20000])) < 1e-6)
    return r

# ---------------- sample patients ----------------
folders = sorted(d for d in os.listdir(ROOT) if d.endswith("-deid"))
pick_idx = sorted(set(np.linspace(0, len(folders) - 1, N_PAT).astype(int).tolist()))
picked = []
for fi in pick_idx:
    fp = os.path.join(ROOT, folders[fi])
    des = [d for d in os.listdir(fp) if d.startswith("DE")]
    random.shuffle(des)
    for de in des[:25]:
        dp = os.path.join(fp, de)
        subs = [s for s in os.scandir(dp) if s.is_dir()]
        random.shuffle(subs)
        for s in subs:
            names = os.listdir(s.path)
            if any(n.endswith(".adibin") for n in names) and any(n.endswith("_HR.vital") for n in names):
                picked.append((folders[fi], dp, s.path, names)); break
        if picked and picked[-1][1] == dp: break
print(f"sampled {len(picked)} patients from folders {[folders[i][:7] for i in pick_idx]}  t={time.time()-t0:.0f}s", flush=True)

# ---------------- MRN-Mapping / Alarms schema ----------------
map_cols = Counter(); alarm_cols = Counter(); ts_pat = Counter(); rows_per = []; uids_per = []
for _, dp, _, _ in picked:
    m = os.path.join(dp, "MRN-Mapping.csv")
    if os.path.isfile(m):
        with open(m, newline="", encoding="latin-1") as fh:
            r = csv.reader(fh); hdr = next(r, []); map_cols[tuple(hdr)] += 1
            rows = list(r); rows_per.append(len(rows))
            if "WaveCycleUID" in hdr:
                i = hdr.index("WaveCycleUID"); uids_per.append(len({x[i] for x in rows if len(x) > i}))
            for c in ("WaveStartTime", "BedTransfer_In"):
                if c in hdr:
                    ci = hdr.index(c)
                    good = [x for x in rows if len(x) > ci and x[ci].strip()]
                    if good: ts_pat[c + ":" + re.sub(r"\d", "#", good[0][ci])] += 1
            short = sum(1 for x in rows if len(x) < len(hdr)); ts_pat["short_rows"] += short
    a = os.path.join(dp, "Alarms.csv")
    if os.path.isfile(a):
        with open(a, newline="", encoding="latin-1") as fh:
            alarm_cols[tuple(next(csv.reader(fh), []))] += 1
print("\n== MRN-Mapping.csv columns:", [list(k) for k in map_cols][:2], "| rows/patient:", rows_per, "| unique UID/patient:", uids_per)
print("   timestamp patterns:", dict(ts_pat))
print("== Alarms.csv columns:", [list(k) for k in alarm_cols][:1], "present in", sum(alarm_cols.values()), "patients")

# ---------------- .adibin headers ----------------
hdr_keys = Counter(); ch_tuples = Counter(); ch_meta = defaultdict(Counter); durs = []; gaps = []; files_per_uid = []
nfiles_per_pat = []; overlap_neg = 0; data_stats = defaultdict(lambda: defaultdict(list)); adibin_cov = {}
for fi, (fold, dp, sp, names) in enumerate(picked):
    ad = sorted(n for n in names if n.endswith(".adibin"))
    nfiles_per_pat.append(len(ad)); hs = []
    for n in ad:
        try: h = read_adibin_header(os.path.join(sp, n))
        except Exception as e: hdr_keys[("ERR", str(e)[:40])] += 1; continue
        hs.append((n, h))
        hdr_keys[(h["magic"], h["ver"], round(1 / h["spt"], 3), h["fmt"], h["nch"], h["tch"])] += 1
        ch_tuples[tuple(c[0] for c in h["chans"])] += 1
        for c in h["chans"]: ch_meta[c[0]][(c[1], c[2], c[3], c[4], c[5])] += 1
        durs.append(h["dur_s"])
    by_uid = Counter(n.rsplit("_", 1)[-1][:-7] for n, _ in hs); files_per_uid.extend(by_uid.values())
    hs_ok = sorted([x for x in hs if x[1]["start_ms"]], key=lambda x: x[1]["start_ms"])
    for (n1, a), (n2, b) in zip(hs_ok, hs_ok[1:]):
        g = (b["start_ms"] - (a["start_ms"] + a["dur_s"] * 1000)) / 1000.0; gaps.append(g); overlap_neg += g < -1
    if hs_ok:
        adibin_cov[fi] = (hs_ok[0][1]["start_ms"], max(x[1]["start_ms"] + x[1]["dur_s"] * 1000 for x in hs_ok))
    # data-level metrics on the 2 longest files
    for n, h in sorted(hs, key=lambda x: -x[1]["dur_s"])[:2]:
        try:
            for ch, m in analyse_adibin_data(os.path.join(sp, n), h).items():
                for k, v in m.items(): data_stats[ch][k].append(v)
        except Exception as e:
            hdr_keys[("DATA_ERR", str(e)[:40])] += 1
print(f"\n== .adibin headers (magic, version, fs_Hz, DataFormat[3=int16], NChannels, TimeChannel) -> n files")
for k, v in hdr_keys.most_common(8): print("  ", k, v)
print("== channel title tuples -> n files"); [print("  ", k, v) for k, v in ch_tuples.most_common(6)]
print("== per-channel header (Units, scale, offset, RangeHigh, RangeLow) -> n files")
for ch in sorted(ch_meta): print(f"   {ch:6s}", ch_meta[ch].most_common(2))
d = np.array(durs); g = np.array(gaps)
print(f"== file duration s: n={d.size} min={d.min():.0f} p50={np.median(d):.0f} p90={np.percentile(d,90):.0f} max={d.max():.0f}; files/patient={nfiles_per_pat}; files/UID p50={np.median(files_per_uid):.0f} max={max(files_per_uid)}")
if g.size: print(f"== gap between consecutive files s: p10={np.percentile(g,10):.1f} p50={np.median(g):.1f} p90={np.percentile(g,90):.1f} max={g.max():.0f}; overlaps(<-1s)={overlap_neg}")
print("\n== per-channel native-rate evidence (median over sampled files; chunk=%ds from file middle)" % CHUNK_SEC)
print("   ch     gap%%  dup%%  lin%%  runMode distinct/s changes/s quant  pow<15 15-30 30-60 60-120 (%% of power)")
for ch in sorted(data_stats):
    s = data_stats[ch]; f = lambda k, fmt="%5.1f": (fmt % np.median(s[k])) if s.get(k) else "   - "
    print(f"   {ch:6s} {f('gap_pct')} {f('dup_pct')} {f('lin_pct')} {f('run_mode','%4.0f'):>7s} {f('distinct_per_s','%6.0f'):>9s} {f('changes_per_s','%6.0f'):>9s} {f('quant_step','%5.2f'):>6s}  {f('pow0-15')} {f('pow15-30')} {f('pow30-60')} {f('pow60-120')}")
print("   (60 Hz native + sample-hold -> dup~75%, runMode 4, distinct/s<=60; 60 Hz + linear interp -> lin~75%; true 240 Hz -> dup small, distinct/s>>60)")

# ---------------- .vital ----------------
vs = defaultdict(lambda: defaultdict(list)); vmeta = defaultdict(Counter); vital_cov = {}
for fi, (fold, dp, sp, names) in enumerate(picked):
    for n in sorted(x for x in names if x.endswith(".vital")):
        suf = n.rsplit("_", 1)[-1][:-6]
        try: r = read_vital(os.path.join(sp, n))
        except Exception as e: vmeta[suf][("ERR", str(e)[:30])] += 1; continue
        if r is None: vmeta[suf][("EMPTY",)] += 1; continue
        vmeta[suf][(r["label"], r["uom"], r["unit"], "zeroT" if r["zero_time"] else "T", "off_int" if r.get("off_int_pct", 0) > 99 else "off_frac")] += 1
        s = vs[suf]; s["n"].append(r["n"])
        if "dts" in r and r["dts"].size:
            dts = r["dts"]; s["dt_p50"].append(np.median(dts)); s["dt_p5"].append(np.percentile(dts, 5)); s["dt_p95"].append(np.percentile(dts, 95))
            s["dt_max"].append(dts.max()); s["dt_eq2"].append(100.0 * np.mean(np.abs(dts - 2.0) < 1e-6)); s["dt_le0"].append(100.0 * np.mean(dts <= 0))
            s["mono"].append(r["off_monotonic"]); s["span_h"].append((r["last_off"] - r["first_off"]) / 3600); s["first_off"].append(r["first_off"])
            s["sent"].append(r["sent_pct"]); v = r["val"]
            if v.size: s["v_p1"].append(np.percentile(v, 1)); s["v_p50"].append(np.median(v)); s["v_p99"].append(np.percentile(v, 99)); s["v_min"].append(v.min()); s["v_max"].append(v.max())
            s["lo"].append(r["lo_top"]); s["hi"].append(r["hi_top"])
        if suf == "HR" and r.get("start_ms") and "first_off" in r:
            vital_cov[fi] = (r["start_ms"] + r["first_off"] * 1000, r["start_ms"] + r["last_off"] * 1000)
print("\n== .vital by suffix: header (Label, Uom, Unit, time-zero?, offset int/frac) -> n files")
for suf in sorted(vmeta): print(f"   {suf:8s}", vmeta[suf].most_common(2))
print("\n== .vital cadence/value stats (medians over files)")
print("   suffix   files  n/file  dt_p50 dt_p5 dt_p95 dt_max  dt==2%%  dt<=0%% mono span_h first_off sent%%   v_p1   v_p50   v_p99   v_min   v_max  low(top) high(top)")
for suf in sorted(vs):
    s = vs[suf]; med = lambda k, fmt="%6.1f": (fmt % np.median(s[k])) if s.get(k) else "   -  "
    lo = s["lo"][0][:1] if s.get("lo") else "-"; hi = s["hi"][0][:1] if s.get("hi") else "-"
    print(f"   {suf:8s} {len(s['n']):5d} {med('n','%7.0f')} {med('dt_p50')} {med('dt_p5')} {med('dt_p95')} {med('dt_max','%6.0f')} {med('dt_eq2')} {med('dt_le0')} {str(all(s.get('mono',[True]))):>5s} {med('span_h')} {med('first_off','%9.1f')} {med('sent')} {med('v_p1','%7.1f')} {med('v_p50','%7.1f')} {med('v_p99','%7.1f')} {med('v_min','%7.1f')} {med('v_max','%7.1f')}  {lo} {hi}")

# ---------------- alignment ----------------
print("\n== alignment per patient: adibin coverage vs HR.vital coverage (hours), start delta (s = vital_first - adibin_first)")
for fi in sorted(adibin_cov):
    a0, a1 = adibin_cov[fi]
    if fi in vital_cov:
        v0, v1 = vital_cov[fi]; ov = max(0, min(a1, v1) - max(a0, v0)) / 3600e3
        print(f"   P{fi:02d} adibin {((a1-a0)/3600e3):7.1f} h | vital {((v1-v0)/3600e3):7.1f} h | overlap {ov:7.1f} h | start delta {((v0-a0)/1e3):9.0f} s | end delta {((v1-a1)/1e3):9.0f} s")
    else:
        print(f"   P{fi:02d} adibin {((a1-a0)/3600e3):7.1f} h | no HR.vital timing")
print(f"\ndone t={time.time()-t0:.0f}s")
