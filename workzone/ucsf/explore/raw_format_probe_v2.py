"""UCSF raw-format probe v2 (read-only, aggregates only).
A) wide header scan (5 DE dirs per cohort folder): adibin header/channel/unit stats, vital zero-time counts,
   filename token structure, UIDs per bed dir, MRN-Mapping window vs file coverage.
B) data-level tests on 24 patients: interpolation/hold phase test per waveform channel, raw value ranges,
   .vital value-change intervals, negative offsets, zero-time anchor test.
"""
import os, sys, struct, csv, random, time, re
from collections import Counter, defaultdict
from datetime import datetime, timezone
import numpy as np
ROOT = "/mnt/localdata/storage/UCSF"
t0 = time.time()
CFWB_FMT = "<4si d iiiii dd iiii"; CFWB_SIZE = struct.calcsize(CFWB_FMT); CH_FMT = "<32s32s4d"; CH_SIZE = struct.calcsize(CH_FMT)
VH_FMT = "<16s8s8s4s iiiii d"; VH_SIZE = struct.calcsize(VH_FMT)
def cstr(b): return b.split(b"\0", 1)[0].decode("latin-1")
def to_ms(y, mo, d, h, mi, s):
    try: si = int(s); return int(datetime(y, mo, d, h, mi, si, tzinfo=timezone.utc).timestamp() * 1000) + int(round((s - si) * 1000))
    except Exception: return None
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
    chans = [(cstr(t), cstr(u), sc, off, hi, lo) for t, u, sc, off, hi, lo in chans]
    return dict(spt=spt, start_ms=to_ms(y, mo, d, h, mi, sec), nch=nch, nsamp=nsamp, fmt=fmt, chans=chans,
                hdr_bytes=CFWB_SIZE + CH_SIZE * nch, dur_s=nsamp * spt, tch=tch)
def read_vital_header(p):
    with open(p, "rb") as f: lab, uom, unit, bed, y, mo, d, h, mi, sec = struct.unpack(VH_FMT, f.read(VH_SIZE))
    n = (os.path.getsize(p) - VH_SIZE) // 32
    return dict(label=cstr(lab), uom=cstr(uom), zero=(y == 0), start_ms=to_ms(y, mo, d, h, mi, sec), n=n)
def tok_pattern(name):
    stem = name.rsplit(".", 1)[0]; toks = stem.split("_")
    return "_".join(("D%d" % len(t)) if t.isdigit() else ("DE+%d" % (len(t) - 2) if t.startswith("DE") and t[2:].isdigit() else "S") for t in toks)

folders = sorted(d for d in os.listdir(ROOT) if d.endswith("-deid"))
random.seed(7)
# ---------------- A) wide header scan ----------------
hdr = Counter(); chan_presence = Counter(); chan_meta = defaultdict(Counter); durs = []; zero_nsamp = 0; n_ad = 0
tok_ad = Counter(); tok_vi = Counter(); uids_per_bed = []; vital_per_bed_suffix = []; vzero = Counter(); vtot = Counter()
win_start_delta = []; win_stop_delta = []; win_missing = 0; map_hdr = Counter(); n_bed = 0; ad_per_uid = []
for fold in folders:
    fp = os.path.join(ROOT, fold); des = [d for d in os.listdir(fp) if d.startswith("DE")]; random.shuffle(des)
    for de in des[:5]:
        dp = os.path.join(fp, de)
        # MRN-Mapping windows keyed by UID
        wins = {}
        m = os.path.join(dp, "MRN-Mapping.csv")
        if os.path.isfile(m):
            with open(m, newline="", encoding="latin-1") as fh:
                r = csv.reader(fh); h = [c.strip() for c in next(r, [])]; map_hdr[tuple(h)] += 1
                if "WaveCycleUID" in h:
                    iu, i0, i1 = h.index("WaveCycleUID"), h.index("WaveStartTime") if "WaveStartTime" in h else None, h.index("WaveStopTime") if "WaveStopTime" in h else None
                    for row in r:
                        if len(row) > iu and i0 is not None and i1 is not None and len(row) > max(i0, i1):
                            a, b = parse_ts(row[i0]), parse_ts(row[i1])
                            if a and b:
                                u = row[iu].strip(); wins.setdefault(u, [a, b]); wins[u][0] = min(wins[u][0], a); wins[u][1] = max(wins[u][1], b)
        for s in os.scandir(dp):
            if not s.is_dir(): continue
            names = os.listdir(s.path); n_bed += 1
            cov = defaultdict(lambda: [None, None]); per_uid = Counter()
            for n in names:
                if n.endswith(".adibin"):
                    n_ad += 1; tok_ad[tok_pattern(n)] += 1
                    try: H = read_adibin_header(os.path.join(s.path, n))
                    except Exception as e: hdr[("ERR", str(e)[:30])] += 1; continue
                    hdr[(round(1 / H["spt"], 2) if H["spt"] else 0, H["fmt"], H["tch"])] += 1
                    durs.append(H["dur_s"]); zero_nsamp += H["nsamp"] == 0
                    for c in H["chans"]: chan_presence[c[0]] += 1; chan_meta[c[0]][(c[1], c[2], c[3], c[4], c[5])] += 1
                    uid = n.rsplit("_", 1)[-1][:-7]; per_uid[uid] += 1
                    if H["start_ms"]:
                        e = H["start_ms"] + H["dur_s"] * 1000; c = cov[uid]
                        c[0] = H["start_ms"] if c[0] is None else min(c[0], H["start_ms"]); c[1] = e if c[1] is None else max(c[1], e)
                elif n.endswith(".vital"):
                    tok_vi[tok_pattern(n)] += 1; suf = n.rsplit("_", 1)[-1][:-6]; vtot[suf] += 1
                    try: V = read_vital_header(os.path.join(s.path, n)); vzero[suf] += V["zero"]
                    except Exception: pass
            uids_per_bed.append(len(per_uid)); ad_per_uid.extend(per_uid.values())
            vs = Counter(n.rsplit("_", 1)[-1][:-6] for n in names if n.endswith(".vital"))
            vital_per_bed_suffix.append(max(vs.values()) if vs else 0)
            for uid, (a, b) in cov.items():
                if uid in wins and a is not None:
                    win_start_delta.append((a - wins[uid][0]) / 1000); win_stop_delta.append((b - wins[uid][1]) / 1000)
                elif a is not None: win_missing += 1
print(f"A) header scan: folders={len(folders)} bed_dirs={n_bed} adibin_files={n_ad} t={time.time()-t0:.0f}s")
print("   (fs_Hz, DataFormat, TimeChannel) -> n:", hdr.most_common(5))
d = np.array(durs); print(f"   duration s: p10={np.percentile(d,10):.0f} p50={np.median(d):.0f} p90={np.percentile(d,90):.0f} p99={np.percentile(d,99):.0f} max={d.max():.0f}; zero-sample files={zero_nsamp}")
print(f"   channel presence (% of files): " + ", ".join(f"{k}:{100*v/n_ad:.0f}" for k, v in chan_presence.most_common(16)))
print("   channel header variants (Units, scale, offset, RangeHigh, RangeLow) -> n:")
for ch, _ in chan_presence.most_common(12): print(f"     {ch:5s}", chan_meta[ch].most_common(3))
print("   adibin filename token pattern:", tok_ad.most_common(3)); print("   vital  filename token pattern:", tok_vi.most_common(3))
print(f"   UIDs per bed dir: {Counter(uids_per_bed).most_common(6)}; adibin files per UID p50={np.median(ad_per_uid):.0f} p90={np.percentile(ad_per_uid,90):.0f} max={max(ad_per_uid)}")
print(f"   max vital files per suffix per bed dir: {Counter(vital_per_bed_suffix).most_common(5)}")
print("   vital zero-header-time % by suffix: " + ", ".join(f"{k}:{100*vzero[k]/vtot[k]:.0f}%({vtot[k]})" for k in sorted(vtot, key=lambda k: -vtot[k])[:20]))
print("   MRN-Mapping header variants:", [(list(k)[:4], v) for k, v in map_hdr.most_common(3)])
ws, we = np.array(win_start_delta), np.array(win_stop_delta)
if ws.size: print(f"   adibin coverage vs MRN-Mapping window (s): start delta p10/p50/p90 = {np.percentile(ws,10):.0f}/{np.median(ws):.0f}/{np.percentile(ws,90):.0f}, stop delta p10/p50/p90 = {np.percentile(we,10):.0f}/{np.median(we):.0f}/{np.percentile(we,90):.0f}; UIDs with adibin but no mapping window: {win_missing}/{ws.size+win_missing}")

# ---------------- B) data-level tests ----------------
def phase_test(x):
    """Return (best_period, lin% at non-anchor phases, lin% at anchor phase, hold%) for periods 2,3,4,8."""
    d2 = np.abs(np.diff(x, 2)) < 1e-9; d1 = np.diff(x) == 0; best = None
    for P in (2, 3, 4, 8):
        ph = np.array([d2[k::P].mean() for k in range(P)]) * 100; hp = np.array([d1[k::P].mean() for k in range(P)]) * 100
        contrast = ph.max() - ph.min(); hc = hp.max() - hp.min()
        if best is None or contrast > best[1]: best = (P, contrast, ph, hp, hc)
    P, contrast, ph, hp, hc = best
    return P, contrast, float(np.sort(ph)[-1]), float(np.sort(ph)[0]), float(hc), float(d1.mean() * 100)
picked = []
for fi in np.linspace(0, len(folders) - 1, 24).astype(int):
    fp = os.path.join(ROOT, folders[fi]); des = [d for d in os.listdir(fp) if d.startswith("DE")]; random.shuffle(des)
    for de in des[:25]:
        dp = os.path.join(fp, de); done = False
        for s in os.scandir(dp):
            if not s.is_dir(): continue
            names = os.listdir(s.path)
            if any(n.endswith(".adibin") for n in names) and any(n.endswith(".vital") for n in names):
                picked.append((dp, s.path, names)); done = True; break
        if done: break
print(f"\nB) data tests on {len(picked)} patients  t={time.time()-t0:.0f}s")
wave = defaultdict(lambda: defaultdict(list)); vstat = defaultdict(lambda: defaultdict(list)); anchor = Counter(); anchor_ex = []
for dp, sp, names in picked:
    ad = []
    for n in names:
        if n.endswith(".adibin"):
            try: ad.append((n, read_adibin_header(os.path.join(sp, n))))
            except Exception: pass
    ad_ok = [x for x in ad if x[1]["start_ms"] and x[1]["nsamp"] > 0]
    if not ad_ok: continue
    a0 = min(x[1]["start_ms"] for x in ad_ok); a1 = max(x[1]["start_ms"] + x[1]["dur_s"] * 1000 for x in ad_ok)
    for n, H in sorted(ad_ok, key=lambda x: -x[1]["dur_s"])[:2]:
        if H["fmt"] != 3: continue
        nck = min(H["nsamp"], 240 * 300); st = max(0, (H["nsamp"] - nck) // 2)
        mm = np.memmap(os.path.join(sp, n), dtype=np.int16, mode="r", offset=H["hdr_bytes"], shape=(H["nsamp"], H["nch"]))
        blk = np.array(mm[st:st + nck, :]); del mm
        for i, c in enumerate(H["chans"]):
            raw = blk[:, i]; ok = (raw != -32767) & (raw != -32768); x = raw[ok].astype(np.float64)
            if x.size < 4800 or x.std() == 0: continue
            P, contrast, lin_hi, lin_lo, hold_c, dup = phase_test(x)
            w = wave[c[0]]; w["P"].append(P); w["contrast"].append(contrast); w["lin_hi"].append(lin_hi); w["lin_lo"].append(lin_lo); w["hold_c"].append(hold_c); w["dup"].append(dup)
            w["raw_p1"].append(np.percentile(x, 1)); w["raw_p50"].append(np.median(x)); w["raw_p99"].append(np.percentile(x, 99)); w["scale"].append(c[2])
            w["phys_p1"].append(c[2] * (np.percentile(x, 1) + c[3])); w["phys_p99"].append(c[2] * (np.percentile(x, 99) + c[3]))
    # vitals
    hr_start = None
    vfiles = []
    for n in names:
        if not n.endswith(".vital"): continue
        p = os.path.join(sp, n); suf = n.rsplit("_", 1)[-1][:-6]
        try: V = read_vital_header(p)
        except Exception: continue
        if V["n"] <= 0: continue
        mm = np.memmap(p, dtype=np.float64, mode="r", offset=VH_SIZE, shape=(V["n"], 4)); val = np.array(mm[:, 0]); off = np.array(mm[:, 1]); del mm
        vfiles.append((suf, V, val, off))
        if suf == "HR" and not V["zero"]: hr_start = V["start_ms"]
    for suf, V, val, off in vfiles:
        s = vstat[suf]; dts = np.diff(off); neg = dts < 0
        s["neg_files"].append(int(neg.any())); s["neg_n"].append(int(neg.sum())); s["neg_min"].append(float(dts.min()) if dts.size else 0)
        if neg.any():
            j = int(np.flatnonzero(neg)[0]); s["neg_back_to"].append(float(off[j + 1]))
        good = val > -99999
        v = val[good]; o = off[good]
        if v.size > 2:
            chg = np.flatnonzero(np.diff(v) != 0)
            s["dup"].append(100 * (np.diff(v) == 0).mean())
            if chg.size > 1: s["chg_int_s"].append(float(np.median(np.diff(o[chg + 1]))))
        s["first_off"].append(float(off[0])); s["zero"].append(int(V["zero"]))
        # anchor test for zero-time files
        if V["zero"]:
            for name, anc in (("HR_hdr", hr_start), ("adibin_first", a0)):
                if anc is None: continue
                lo, hi = anc + off[0] * 1000, anc + off[-1] * 1000
                inside = (lo >= a0 - 120e3) and (hi <= a1 + 120e3)
                anchor[(suf, name, "inside" if inside else "outside")] += 1
                if len(anchor_ex) < 6: anchor_ex.append((suf, name, round((lo - a0) / 3600e3, 2), round((hi - a1) / 3600e3, 2)))
        elif V["start_ms"]:
            s["hdr_vs_adibin_s"].append((V["start_ms"] + off[0] * 1000 - a0) / 1000)
print("\n   waveform channel: interpolation/hold phase test (median over files)")
print("   ch     n  period  contrast  lin%(interp phases)  lin%(anchor phase)  hold-contrast  dup%   raw p1/p50/p99      scale   phys p1..p99")
for ch in sorted(wave, key=lambda c: -len(wave[c]["P"])):
    w = wave[ch]; med = lambda k: float(np.median(w[k]))
    Pm = Counter(w["P"]).most_common(1)[0]
    print(f"   {ch:5s} {len(w['P']):3d}   {Pm[0]}x({Pm[1]})   {med('contrast'):6.1f}      {med('lin_hi'):6.1f}             {med('lin_lo'):6.1f}            {med('hold_c'):6.1f}     {med('dup'):5.1f}   {med('raw_p1'):7.0f}/{med('raw_p50'):6.0f}/{med('raw_p99'):6.0f}   {med('scale'):5.2f}  {med('phys_p1'):8.1f}..{med('phys_p99'):8.1f}")
print("   read: period Px with high contrast and lin%(interp phases)~100 => linear interpolation from 240/P Hz; hold-contrast high => sample-and-hold from 240/P Hz; contrast~0 => native 240 Hz")
print("\n   .vital: value-change interval, negative offsets, zero-time, header-vs-adibin start (medians)")
print("   suffix  files zero%  dup%   chg_int_s  neg_files  neg_n  neg_min_s  neg_back_to_off  hdr_start-adibin_start_s")
for suf in sorted(vstat, key=lambda k: -len(vstat[k]["zero"])):
    s = vstat[suf]; med = lambda k, f="%8.1f": (f % np.median(s[k])) if s.get(k) else "    -   "
    print(f"   {suf:7s} {len(s['zero']):4d} {100*np.mean(s['zero']):5.0f} {med('dup','%6.1f')} {med('chg_int_s','%10.1f')} {sum(s['neg_files']):8d} {sum(s['neg_n']):7d} {med('neg_min','%9.1f')} {med('neg_back_to','%12.1f')} {med('hdr_vs_adibin_s','%12.1f')}")
print("\n   zero-time anchor test (does anchor+offset window fall inside adibin coverage +-2min?):", dict(anchor))
print("   examples (suffix, anchor, start_h vs adibin_start, end_h vs adibin_end):", anchor_ex)
print(f"\ndone t={time.time()-t0:.0f}s")
