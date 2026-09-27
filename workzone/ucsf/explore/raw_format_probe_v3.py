"""UCSF raw-format probe v3 (read-only, aggregates only).
1) Upsampling diagnosis per waveform channel: RMS reconstruction error (LSB) when the 240 Hz stream is rebuilt from
   every P-th sample by linear interpolation or by sample-and-hold (best phase), P in 2,3,4,6,8.
2) .vital offset semantics per file: relative seconds vs absolute seconds-since-0001-01-01 vs mixed; cross-tab with
   zero header time; converted absolute times checked against adibin coverage.
"""
import os, struct, random, time
from collections import Counter, defaultdict
from datetime import datetime, timezone
import numpy as np
ROOT = "/mnt/localdata/storage/UCSF"; t0 = time.time()
CFWB_FMT = "<4si d iiiii dd iiii"; CFWB_SIZE = struct.calcsize(CFWB_FMT); CH_FMT = "<32s32s4d"; CH_SIZE = struct.calcsize(CH_FMT)
VH_FMT = "<16s8s8s4s iiiii d"; VH_SIZE = struct.calcsize(VH_FMT)
EPOCH_0001_S = 62135596800  # seconds from 0001-01-01 to 1970-01-01 (proleptic Gregorian)
def cstr(b): return b.split(b"\0", 1)[0].decode("latin-1")
def to_ms(y, mo, d, h, mi, s):
    try: si = int(s); return int(datetime(y, mo, d, h, mi, si, tzinfo=timezone.utc).timestamp() * 1000) + int(round((s - si) * 1000))
    except Exception: return None
def read_adibin_header(p):
    with open(p, "rb") as f:
        magic, ver, spt, y, mo, d, h, mi, sec, trig, nch, nsamp, tch, fmt = struct.unpack(CFWB_FMT, f.read(CFWB_SIZE))
        chans = [struct.unpack(CH_FMT, f.read(CH_SIZE)) for _ in range(nch)]
    return dict(spt=spt, start_ms=to_ms(y, mo, d, h, mi, sec), nch=nch, nsamp=nsamp, fmt=fmt, hdr_bytes=CFWB_SIZE + CH_SIZE * nch,
                dur_s=nsamp * spt, titles=[cstr(t) for t, *_ in chans])
def recon_err(x, P, mode):
    """min over phase of RMS(x - reconstruction from x[phase::P]) in LSB."""
    best = None
    for ph in range(P):
        idx = np.arange(ph, x.size, P)
        if idx.size < 10: continue
        if mode == "lin": r = np.interp(np.arange(idx[0], idx[-1] + 1), idx, x[idx]); seg = x[idx[0]:idx[-1] + 1]
        else: r = np.repeat(x[idx], P)[: x.size - idx[0]]; seg = x[idx[0]:idx[0] + r.size]
        e = float(np.sqrt(np.mean((seg - r) ** 2))); best = e if best is None or e < best else best
    return best
folders = sorted(d for d in os.listdir(ROOT) if d.endswith("-deid")); random.seed(11)
picked = []
for fi in np.linspace(0, len(folders) - 1, 30).astype(int):
    fp = os.path.join(ROOT, folders[fi]); des = [d for d in os.listdir(fp) if d.startswith("DE")]; random.shuffle(des)
    for de in des[:30]:
        dp = os.path.join(fp, de); done = False
        for s in os.scandir(dp):
            if not s.is_dir(): continue
            names = os.listdir(s.path)
            if any(n.endswith(".adibin") for n in names) and any(n.endswith(".vital") for n in names):
                picked.append((s.path, names)); done = True; break
        if done: break
print(f"patients={len(picked)}")
W = defaultdict(lambda: defaultdict(list)); phase4 = defaultdict(list)
V = defaultdict(Counter); Vdelta = defaultdict(list); Vswitch = []; Vex = Counter(); nvit = 0
for sp, names in picked:
    ad = []
    for n in names:
        if n.endswith(".adibin"):
            try: H = read_adibin_header(os.path.join(sp, n)); ad.append((n, H))
            except Exception: pass
    ad_ok = [x for x in ad if x[1]["start_ms"] and x[1]["nsamp"] > 0 and x[1]["fmt"] == 3]
    if not ad_ok: continue
    a0 = min(H["start_ms"] for _, H in ad_ok); a1 = max(H["start_ms"] + H["dur_s"] * 1000 for _, H in ad_ok)
    for n, H in sorted(ad_ok, key=lambda x: -x[1]["dur_s"])[:2]:
        nck = min(H["nsamp"], 240 * 240); st = max(0, (H["nsamp"] - nck) // 2)
        mm = np.memmap(os.path.join(sp, n), dtype=np.int16, mode="r", offset=H["hdr_bytes"], shape=(H["nsamp"], H["nch"]))
        blk = np.array(mm[st:st + nck, :]); del mm
        for i, t in enumerate(H["titles"]):
            raw = blk[:, i]; ok = (raw != -32767) & (raw != -32768)
            if ok.mean() < 0.99: continue
            x = raw.astype(np.float64); x[~ok] = np.nan
            if np.nanstd(x) < 1: continue
            x = np.where(np.isnan(x), np.nanmean(x), x)
            d1 = np.diff(x); rms_d1 = float(np.sqrt(np.mean(d1 ** 2))) or 1.0
            W[t]["rms_d1"].append(rms_d1)
            for P in (2, 3, 4, 6, 8):
                W[t][f"lin{P}"].append(recon_err(x, P, "lin")); W[t][f"hold{P}"].append(recon_err(x, P, "hold"))
            dup = (d1 == 0); d2 = np.abs(np.diff(x, 2)) < 1e-9
            phase4[t].append((np.array([dup[k::4].mean() for k in range(4)]) * 100, np.array([d2[k::4].mean() for k in range(4)]) * 100))
    hr_start = None
    for n in names:
        if not n.endswith(".vital"): continue
        p = os.path.join(sp, n); suf = n.rsplit("_", 1)[-1][:-6]
        with open(p, "rb") as f: lab, uom, unit, bed, y, mo, d, h, mi, sec = struct.unpack(VH_FMT, f.read(VH_SIZE))
        nn = (os.path.getsize(p) - VH_SIZE) // 32
        if nn <= 0: V[suf][("empty",)] += 1; continue
        nvit += 1
        mm = np.memmap(p, dtype=np.float64, mode="r", offset=VH_SIZE, shape=(nn, 4)); off = np.array(mm[:, 1]); val = np.array(mm[:, 0]); del mm
        zero = (y == 0); hdr_ms = None if zero else to_ms(y, mo, d, h, mi, sec)
        is_abs = off > 6.0e10; fa = float(is_abs.mean())
        kind = "abs" if fa > 0.999 else "rel" if fa < 0.001 else "mixed"
        V[suf][("zeroT" if zero else "hdrT", kind)] += 1
        if kind == "mixed":
            j = int(np.flatnonzero(np.diff(is_abs.astype(int)) != 0)[0]); Vswitch.append((suf, round(100 * j / nn, 1), int(is_abs[0]), int(off[0] < 1)))
        # where does the file start in absolute time, under each interpretation, relative to adibin coverage (s)?
        if kind in ("abs", "mixed") and is_abs.any():
            t_abs0 = (off[is_abs][0] - EPOCH_0001_S) * 1000; t_abs1 = (off[is_abs][-1] - EPOCH_0001_S) * 1000
            Vdelta[("abs->start-a0", suf)].append((t_abs0 - a0) / 1000); Vdelta[("abs->end-a1", suf)].append((t_abs1 - a1) / 1000)
        if kind in ("rel", "mixed") and (~is_abs).any():
            r0, r1 = off[~is_abs][0], off[~is_abs][-1]
            Vdelta[("rel_first_off", suf)].append(r0); Vdelta[("rel_span_h", suf)].append((r1 - r0) / 3600)
            if hdr_ms: Vdelta[("hdr+rel->start-a0", suf)].append((hdr_ms + r0 * 1000 - a0) / 1000); Vdelta[("hdr+rel->end-a1", suf)].append((hdr_ms + r1 * 1000 - a1) / 1000)
            elif kind == "rel": Vdelta[("zeroT_rel: a0+off end-a1", suf)].append((a0 + r1 * 1000 - a1) / 1000)
        if suf == "HR" and hdr_ms: hr_start = hdr_ms
        # value sanity on absolute-offset files
        Vex[(kind, "val>-99999 %", round(100 * float((val > -99999).mean())))] += 1
print("\n1) waveform upsampling diagnosis: RMS reconstruction error in LSB (median over files); rms(diff)=typical sample-to-sample step")
print("   ch      n  rms_d1 | lin P=2   3     4     6     8  | hold P=2   3     4     6     8")
for t in sorted(W, key=lambda k: -len(W[k]["rms_d1"])):
    w = W[t]; m = lambda k: float(np.median(w[k]))
    print(f"   {t:6s} {len(w['rms_d1']):3d} {m('rms_d1'):6.1f} | " + " ".join(f"{m(f'lin{P}'):5.2f}" for P in (2, 3, 4, 6, 8)) + " | " + " ".join(f"{m(f'hold{P}'):5.2f}" for P in (2, 3, 4, 6, 8)))
print("   verdict rule: lin{P} <= ~0.5 LSB (rounding) while lin{2P} >> => linear interpolation from 240/P Hz; hold{P} ~0 => sample-and-hold from 240/P Hz; all errors ~rms_d1 => native 240 Hz")
print("\n   per-phase (P=4) dup% and zero-2nd-diff% vectors, median over files:")
for t in ("SPO2", "RR", "AR1", "AR2", "CVP2", "II"):
    if t in phase4:
        dv = np.median(np.array([a for a, _ in phase4[t]]), axis=0); lv = np.median(np.array([b for _, b in phase4[t]]), axis=0)
        print(f"   {t:5s} dup%[{', '.join(f'{v:5.1f}' for v in dv)}]  lin%[{', '.join(f'{v:5.1f}' for v in lv)}]")
print(f"\n2) .vital offset semantics ({nvit} files): (header time, offset kind) -> n   [abs = seconds since 0001-01-01]")
tot = Counter()
for suf in V:
    for k, v in V[suf].items(): tot[k] += v
print("   ALL:", dict(tot))
for suf in ("HR", "SPO2-%", "RESP", "NBP-S", "AR1-S", "AR2-S", "TMP-1", "CUFF", "ST-II"):
    if suf in V: print(f"   {suf:7s}", dict(V[suf]))
print("   mixed files (suffix, switch position % into file, starts_abs?, first_off<1?):", Vswitch[:10])
print("   value validity by kind:", dict(Vex))
print("\n   timing checks (s unless noted; median [p10, p90]) vs adibin coverage of the same bed dir:")
for key in sorted(Vdelta):
    arr = np.array(Vdelta[key])
    if key[1] in ("HR", "NBP-S", "AR1-S", "AR2-S", "TMP-1", "SPO2-%"):
        print(f"   {key[0]:26s} {key[1]:7s} n={arr.size:3d}  {np.median(arr):12.1f} [{np.percentile(arr,10):12.1f}, {np.percentile(arr,90):12.1f}]")
print(f"\ndone t={time.time()-t0:.0f}s")
