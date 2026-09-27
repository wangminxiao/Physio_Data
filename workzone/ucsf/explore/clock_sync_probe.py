"""Are .adibin waveforms and .vital numerics on the same clock? (read-only, aggregates only)
ECG-derived HR (R peaks on lead II) vs HR.vital, and AR1 waveform per-2s sys/dia/mean vs AR1-S/D/M.vital.
Cross-correlation lag search: positive lag = numeric value lags the waveform-derived value."""
import os, struct, random, time
from datetime import datetime, timezone
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
ROOT = "/mnt/localdata/storage/UCSF"; t0 = time.time()
CFWB_FMT = "<4si d iiiii dd iiii"; CFWB_SIZE = struct.calcsize(CFWB_FMT); CH_FMT = "<32s32s4d"; CH_SIZE = struct.calcsize(CH_FMT)
VH_FMT = "<16s8s8s4s iiiii d"; VH_SIZE = struct.calcsize(VH_FMT)
CHUNK_S = 900; GRID = 2.0
def cstr(b): return b.split(b"\0", 1)[0].decode("latin-1")
def to_ms(y, mo, d, h, mi, s):
    try: si = int(s); return int(datetime(y, mo, d, h, mi, si, tzinfo=timezone.utc).timestamp() * 1000) + int(round((s - si) * 1000))
    except Exception: return None
def read_adibin_header(p):
    with open(p, "rb") as f:
        magic, ver, spt, y, mo, d, h, mi, sec, trig, nch, nsamp, tch, fmt = struct.unpack(CFWB_FMT, f.read(CFWB_SIZE))
        chans = [struct.unpack(CH_FMT, f.read(CH_SIZE)) for _ in range(nch)]
    return dict(spt=spt, start_ms=to_ms(y, mo, d, h, mi, sec), sec_frac=sec - int(sec), nch=nch, nsamp=nsamp, fmt=fmt,
                hdr_bytes=CFWB_SIZE + CH_SIZE * nch, dur_s=nsamp * spt, titles=[cstr(t) for t, *_ in chans], scales=[c[2] for c in chans])
def read_vital(p):
    with open(p, "rb") as f: lab, uom, unit, bed, y, mo, d, h, mi, sec = struct.unpack(VH_FMT, f.read(VH_SIZE))
    n = (os.path.getsize(p) - VH_SIZE) // 32
    if y == 0 or n < 10: return None
    mm = np.memmap(p, dtype=np.float64, mode="r", offset=VH_SIZE, shape=(n, 4)); val = np.array(mm[:, 0]); off = np.array(mm[:, 1]); del mm
    ok = (off < 6e10) & (val > -99999); return to_ms(y, mo, d, h, mi, sec), val[ok], off[ok]
def ecg_hr_series(x, fs, t_grid):
    b, a = butter(3, [5 / (fs / 2), 25 / (fs / 2)], btype="band"); y = filtfilt(b, a, x)
    e = np.convolve(y ** 2, np.ones(int(0.12 * fs)) / int(0.12 * fs), mode="same")
    pk, pr = find_peaks(e, distance=int(0.3 * fs))
    if pk.size < 50: return None
    h = e[pk]; keep = h >= 0.3 * np.percentile(h, 90); tb = pk[keep] / fs
    out = np.full(t_grid.size, np.nan)
    for i, t in enumerate(t_grid):
        m = (tb >= t - 5) & (tb <= t + 5)
        if m.sum() >= 4: out[i] = 60.0 * (m.sum() - 1) / (tb[m][-1] - tb[m][0])
    return out
def lag_scan(ref, num_t, num_v, t_grid, max_lag=40):
    """ref on t_grid; numeric samples (num_t, num_v) absolute seconds. Returns (best_lag, corr_best, corr0, mae_best, mae0, n)."""
    res = []
    for lag in np.arange(-max_lag, max_lag + GRID, GRID):
        v = np.interp(t_grid + lag, num_t, num_v, left=np.nan, right=np.nan)
        ok = np.isfinite(v) & np.isfinite(ref)
        if ok.sum() < 100: res.append((lag, np.nan, np.nan, 0)); continue
        c = np.corrcoef(ref[ok], v[ok])[0, 1]; res.append((lag, c, float(np.mean(np.abs(ref[ok] - v[ok]))), int(ok.sum())))
    res = [r for r in res if np.isfinite(r[1])]
    if not res: return None
    best = max(res, key=lambda r: r[1]); zero = min(res, key=lambda r: abs(r[0]))
    return best[0], best[1], zero[1], best[2], zero[2], best[3]
folders = sorted(d for d in os.listdir(ROOT) if d.endswith("-deid")); random.seed(5)
rows_hr, rows_bp, frac_secs = [], [], []
for fi in np.linspace(0, len(folders) - 1, 40).astype(int):
    if len(rows_hr) >= 14: break
    fp = os.path.join(ROOT, folders[fi]); des = [d for d in os.listdir(fp) if d.startswith("DE")]; random.shuffle(des)
    for de in des[:40]:
        dp = os.path.join(fp, de); got = False
        for s in os.scandir(dp):
            if not s.is_dir(): continue
            names = os.listdir(s.path)
            hr_files = [n for n in names if n.endswith("_HR.vital")]
            if not hr_files: continue
            ad = []
            for n in names:
                if n.endswith(".adibin"):
                    try: H = read_adibin_header(os.path.join(s.path, n)); ad.append((n, H))
                    except Exception: pass
            ad = [x for x in ad if x[1]["start_ms"] and x[1]["fmt"] == 3 and x[1]["dur_s"] >= CHUNK_S + 120 and "II" in x[1]["titles"]]
            if not ad: continue
            n, H = max(ad, key=lambda x: x[1]["dur_s"]); uid = n.rsplit("_", 1)[-1][:-7]
            hrf = [x for x in hr_files if f"_{uid}_" in x]
            if not hrf: continue
            V = read_vital(os.path.join(s.path, hrf[0]))
            if V is None: continue
            vstart_ms, vval, voff = V
            fs = 1 / H["spt"]; st = max(0, (H["nsamp"] - int(CHUNK_S * fs)) // 2); nck = int(CHUNK_S * fs)
            mm = np.memmap(os.path.join(s.path, n), dtype=np.int16, mode="r", offset=H["hdr_bytes"], shape=(H["nsamp"], H["nch"]))
            blk = np.array(mm[st:st + nck, :]); del mm
            t_abs0 = H["start_ms"] / 1000 + st / fs  # absolute seconds of chunk start
            t_grid = t_abs0 + np.arange(10, CHUNK_S - 10, GRID)
            ii = blk[:, H["titles"].index("II")].astype(np.float64); ii[(ii == -32767) | (ii == -32768)] = np.nan
            if np.isnan(ii).mean() > 0.05: continue
            ii = np.where(np.isnan(ii), np.nanmean(ii), ii)
            ref = ecg_hr_series(ii, fs, t_grid - t_abs0 + 0)  # relative seconds within chunk
            if ref is None: continue
            num_t = vstart_ms / 1000 + voff
            r = lag_scan(ref, num_t, vval, t_grid)
            if r is None: continue
            rows_hr.append(r); frac_secs.append(H["sec_frac"]); got = True
            # ABP
            if "AR1" in H["titles"]:
                ar = blk[:, H["titles"].index("AR1")].astype(np.float64); ar[(ar == -32767) | (ar == -32768)] = np.nan; ar *= H["scales"][H["titles"].index("AR1")]
                nbin = int((CHUNK_S - 20) / GRID); rel = t_grid - t_abs0
                sys_ = np.full(nbin, np.nan); dia_ = np.full(nbin, np.nan); mean_ = np.full(nbin, np.nan)
                for i in range(nbin):
                    seg = ar[int((rel[i] - 1) * fs): int((rel[i] + 1) * fs)]
                    if np.isfinite(seg).mean() > 0.9: sys_[i] = np.nanpercentile(seg, 98); dia_[i] = np.nanpercentile(seg, 2); mean_[i] = np.nanmean(seg)
                for suf, ref_bp in (("AR1-S", sys_), ("AR1-D", dia_), ("AR1-M", mean_)):
                    vf = [x for x in names if x.endswith(f"_{uid}_{suf}.vital")]
                    if not vf: continue
                    VB = read_vital(os.path.join(s.path, vf[0]))
                    if VB is None: continue
                    rb = lag_scan(ref_bp, VB[0] / 1000 + VB[2], VB[1], t_grid)
                    if rb: rows_bp.append((suf,) + rb)
            break
        if got: break
print(f"patients with class-A HR.vital + long II chunk: {len(rows_hr)}   (chunk {CHUNK_S}s from file middle; grid {GRID}s; lag>0 = numeric lags waveform)")
print("   ECG-derived HR vs HR.vital:   best_lag_s  corr@best  corr@0  MAE@best  MAE@0   n")
for r in rows_hr: print(f"                                  {r[0]:8.0f}   {r[1]:7.3f}  {r[2]:6.3f}   {r[3]:6.2f}  {r[4]:6.2f} {r[5]:5d}")
a = np.array(rows_hr); print(f"   median: lag={np.median(a[:,0]):.0f}s corr@best={np.median(a[:,1]):.3f} corr@0={np.median(a[:,2]):.3f} MAE@best={np.median(a[:,3]):.2f} bpm MAE@0={np.median(a[:,4]):.2f} bpm | lag within +-4s: {int(np.sum(np.abs(a[:,0])<=4))}/{a.shape[0]}")
print(f"   adibin header Second fractional part: {np.round(frac_secs,3).tolist()}")
if rows_bp:
    print("\n   AR1 waveform per-2s sys/dia/mean vs AR1-S/D/M.vital:")
    for r in rows_bp: print(f"      {r[0]:6s} lag={r[1]:4.0f}s corr@best={r[2]:.3f} corr@0={r[3]:.3f} MAE@best={r[4]:5.1f} MAE@0={r[5]:5.1f} mmHg n={r[6]}")
print(f"done t={time.time()-t0:.0f}s")
