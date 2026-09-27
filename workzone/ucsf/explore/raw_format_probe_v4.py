"""UCSF raw-format probe v4 (read-only, aggregates only).
a) SPO2 upsampling formula: within-block ratios of the 4-sample pattern + raw diff excerpts (numbers only).
b) Filename 14-digit timestamp vs header time (adibin, vital); per-UID consistency.
c) Zero-header-time .vital anchor candidates: filename ts / first adibin start / MRN-Mapping WaveStartTime.
d) Mixed-offset files: continuity between relative and absolute parts; implied calendar shift (aggregate days only).
"""
import os, struct, csv, random, time
from collections import Counter, defaultdict
from datetime import datetime, timezone
import numpy as np
ROOT = "/mnt/localdata/storage/UCSF"; t0 = time.time()
CFWB_FMT = "<4si d iiiii dd iiii"; CFWB_SIZE = struct.calcsize(CFWB_FMT); CH_FMT = "<32s32s4d"; CH_SIZE = struct.calcsize(CH_FMT)
VH_FMT = "<16s8s8s4s iiiii d"; VH_SIZE = struct.calcsize(VH_FMT); EPOCH_0001_S = 62135596800
def cstr(b): return b.split(b"\0", 1)[0].decode("latin-1")
def to_ms(y, mo, d, h, mi, s):
    try: si = int(s); return int(datetime(y, mo, d, h, mi, si, tzinfo=timezone.utc).timestamp() * 1000) + int(round((s - si) * 1000))
    except Exception: return None
def fname_ts_ms(name):
    for t in name.split("_"):
        if len(t) == 14 and t.isdigit():
            try: return int(datetime.strptime(t, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc).timestamp() * 1000)
            except ValueError: return None
    return None
def parse_ts(s):
    s = s.strip()
    for f in ("%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M"):
        try: return int(datetime.strptime(s, f).replace(tzinfo=timezone.utc).timestamp() * 1000)
        except ValueError: pass
    return None
def read_adibin_header(p):
    with open(p, "rb") as f:
        magic, ver, spt, y, mo, d, h, mi, sec, trig, nch, nsamp, tch, fmt = struct.unpack(CFWB_FMT, f.read(CFWB_SIZE))
        chans = [struct.unpack(CH_FMT, f.read(CH_SIZE)) for _ in range(nch)]
    return dict(spt=spt, start_ms=to_ms(y, mo, d, h, mi, sec), nch=nch, nsamp=nsamp, fmt=fmt, hdr_bytes=CFWB_SIZE + CH_SIZE * nch,
                dur_s=nsamp * spt, titles=[cstr(t) for t, *_ in chans])
folders = sorted(d for d in os.listdir(ROOT) if d.endswith("-deid")); random.seed(23)
picked = []
for fi in np.linspace(0, len(folders) - 1, 40).astype(int):
    fp = os.path.join(ROOT, folders[fi]); des = [d for d in os.listdir(fp) if d.startswith("DE")]; random.shuffle(des)
    for de in des[:30]:
        dp = os.path.join(fp, de); done = False
        for s in os.scandir(dp):
            if not s.is_dir(): continue
            names = os.listdir(s.path)
            if any(n.endswith(".adibin") for n in names) and any(n.endswith(".vital") for n in names):
                picked.append((dp, s.path, names)); done = True; break
        if done: break
print(f"patients={len(picked)}")
ratios = defaultdict(list); excerpts = []; ad_fn_delta = []; vi_fn_delta = []; vi_fn_per_uid = Counter()
anch = defaultdict(list); mixed = defaultdict(list); n_zero = n_hdr = 0
for dp, sp, names in picked:
    # MRN-Mapping windows
    wins = {}
    m = os.path.join(dp, "MRN-Mapping.csv")
    if os.path.isfile(m):
        with open(m, newline="", encoding="latin-1") as fh:
            r = csv.reader(fh); h = [c.strip() for c in next(r, [])]
            if "WaveCycleUID" in h and "WaveStartTime" in h:
                iu, i0 = h.index("WaveCycleUID"), h.index("WaveStartTime")
                for row in r:
                    if len(row) > max(iu, i0):
                        a = parse_ts(row[i0])
                        if a: u = row[iu].strip(); wins[u] = min(wins.get(u, a), a)
    ad = []
    for n in names:
        if n.endswith(".adibin"):
            try: H = read_adibin_header(os.path.join(sp, n)); ad.append((n, H))
            except Exception: pass
    ad_ok = [x for x in ad if x[1]["start_ms"] and x[1]["nsamp"] > 0 and x[1]["fmt"] == 3]
    if not ad_ok: continue
    a0 = min(H["start_ms"] for _, H in ad_ok); a1 = max(H["start_ms"] + H["dur_s"] * 1000 for _, H in ad_ok)
    uid_a0 = defaultdict(list)
    for n, H in ad_ok:
        ft = fname_ts_ms(n)
        if ft: ad_fn_delta.append((ft - H["start_ms"]) / 1000)
        uid_a0[n.rsplit("_", 1)[-1][:-7]].append(H["start_ms"])
    # a) SPO2 block ratios on the longest file
    n, H = max(ad_ok, key=lambda x: x[1]["dur_s"])
    if "SPO2" in H["titles"]:
        i = H["titles"].index("SPO2"); nck = min(H["nsamp"], 240 * 120); st = max(0, (H["nsamp"] - nck) // 2)
        mm = np.memmap(os.path.join(sp, n), dtype=np.int16, mode="r", offset=H["hdr_bytes"], shape=(H["nsamp"], H["nch"]))
        x = np.array(mm[st:st + nck, i]).astype(np.float64); del mm
        for ph in range(4):
            xs = x[ph:]; nb = (xs.size - 1) // 4
            b = xs[: nb * 4 + 1].reshape(-1)  # blocks start at ph
            x0, x1, x2, x3, x4 = (b[k:nb * 4:4] for k in range(5)) if False else (b[0:nb*4:4], b[1:nb*4:4], b[2:nb*4:4], b[3:nb*4:4], b[4:nb*4+1:4])
            D = x4 - x0; ok = np.abs(D) >= 16
            if ok.sum() > 50:
                ratios[(ph, "r1=(x1-x0)/D")].append(float(np.median(((x1 - x0) / D)[ok]))); ratios[(ph, "r2=(x2-x0)/D")].append(float(np.median(((x2 - x0) / D)[ok]))); ratios[(ph, "r3=(x3-x0)/D")].append(float(np.median(((x3 - x0) / D)[ok])))
        if len(excerpts) < 3:
            d = np.diff(x[:33]).astype(int); excerpts.append(d.tolist())
    # vitals
    for n in names:
        if not n.endswith(".vital"): continue
        p = os.path.join(sp, n); suf = n.rsplit("_", 1)[-1][:-6]; uid = n.rsplit("_", 2)[-2]
        with open(p, "rb") as f: lab, uom, unit, bed, y, mo, d, h, mi, sec = struct.unpack(VH_FMT, f.read(VH_SIZE))
        nn = (os.path.getsize(p) - VH_SIZE) // 32
        if nn <= 1: continue
        mm = np.memmap(p, dtype=np.float64, mode="r", offset=VH_SIZE, shape=(nn, 4)); off = np.array(mm[:, 1]); del mm
        ft = fname_ts_ms(n); zero = (y == 0); hdr_ms = None if zero else to_ms(y, mo, d, h, mi, sec)
        is_abs = off > 6.0e10; rel = off[~is_abs]; ab = off[is_abs]
        vi_fn_per_uid[(dp, uid, ft)] += 1
        if hdr_ms and ft: vi_fn_delta.append((ft - hdr_ms) / 1000); n_hdr += 1
        if zero: n_zero += 1
        if suf not in ("HR", "SPO2-%", "RESP", "NBP-S", "AR1-S", "TMP-1", "CUFF"): continue
        if rel.size > 1:
            cands = {"fname_ts": ft, "adibin_a0": a0, "uid_adibin_start": min(uid_a0[uid]) if uid in uid_a0 else None, "mrn_WaveStart": wins.get(uid)}
            if hdr_ms: cands["hdr_time"] = hdr_ms
            for cname, c in cands.items():
                if c is None: continue
                key = ("zeroT" if zero else "hdrT", suf, cname)
                anch[key].append(((c + rel[0] * 1000 - a0) / 1000, (c + rel[-1] * 1000 - a1) / 1000))
        if rel.size > 1 and ab.size > 1:
            anchor = ft if ft else a0
            t_rel_last = anchor + rel[-1] * 1000; t_abs_first = (ab[0] - EPOCH_0001_S) * 1000
            mixed[suf].append(((t_abs_first - t_rel_last) / 1000, (t_abs_first - t_rel_last) / 86400e3, (ab[-1] - ab[0]) / 3600, rel.size / nn * 100,
                               float(np.median(np.diff(ab))), float((np.diff(ab) < 0).mean() * 100)))
print("\na) SPO2 4-sample block structure: median ratio of (x_k - x_0)/(x_4 - x_0) for blocks starting at each phase (|x4-x0|>=16 LSB)")
for ph in range(4):
    ks = [k for k in ratios if k[0] == ph]
    if ks: print(f"   phase {ph}: " + "  ".join(f"{k[1]}={np.median(ratios[k]):.3f}" for k in sorted(ks)))
print("   (pure linear x4 -> 0.25/0.50/0.75 at the anchor phase; hold -> 0/0/0; 'hold then halving' -> 0/0.5/0.75)")
print("   raw first-difference excerpts (32 consecutive SPO2 samples, 3 files):")
for e in excerpts: print("   ", e)
print(f"\nb) filename 14-digit timestamp vs header time (s): adibin n={len(ad_fn_delta)} median={np.median(ad_fn_delta):.0f} p10={np.percentile(ad_fn_delta,10):.0f} p90={np.percentile(ad_fn_delta,90):.0f} | vital(hdrT) n={len(vi_fn_delta)} median={np.median(vi_fn_delta):.0f} p10={np.percentile(vi_fn_delta,10):.0f} p90={np.percentile(vi_fn_delta,90):.0f}")
per_uid_ts = Counter(); 
for (dpp, uid, ft), c in vi_fn_per_uid.items(): per_uid_ts[(dpp, uid)] += 1
print(f"   distinct filename timestamps per (patient, UID) among vital files: {Counter(per_uid_ts.values()).most_common(4)}   [1 = all vital files of a UID share one ts]")
print(f"   vital files: zero-header-time={n_zero}, with header time={n_hdr}")
print("\nc) anchor test: (anchor + first_rel_off) - adibin_first  and  (anchor + last_rel_off) - adibin_last, seconds, median [p10, p90]")
for key in sorted(anch):
    arr = np.array(anch[key])
    if key[1] in ("HR", "NBP-S", "AR1-S"):
        print(f"   {key[0]} {key[1]:6s} {key[2]:17s} n={arr.shape[0]:3d}  start {np.median(arr[:,0]):9.0f} [{np.percentile(arr[:,0],10):9.0f},{np.percentile(arr[:,0],90):9.0f}]   end {np.median(arr[:,1]):9.0f} [{np.percentile(arr[:,1],10):9.0f},{np.percentile(arr[:,1],90):9.0f}]")
print("\nd) mixed files (relative part then absolute part): abs_first - (anchor=fname_ts + rel_last): seconds | days; abs span h; rel part %; abs median dt; abs neg-dt %")
for suf in sorted(mixed):
    arr = np.array(mixed[suf])
    print(f"   {suf:7s} n={arr.shape[0]:2d}  gap s median={np.median(arr[:,0]):12.0f} | days median={np.median(arr[:,1]):7.1f} [p10 {np.percentile(arr[:,1],10):7.1f}, p90 {np.percentile(arr[:,1],90):7.1f}] | abs span h={np.median(arr[:,2]):6.1f} | rel%={np.median(arr[:,3]):5.1f} | abs dt={np.median(arr[:,4]):4.1f} | neg%={np.median(arr[:,5]):5.2f}")
print("   (days ~ 0 => absolute part continues the same shifted calendar; days >> 0 => absolute part is on a different (likely real) calendar)")
print(f"\ndone t={time.time()-t0:.0f}s")
