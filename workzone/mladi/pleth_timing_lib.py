"""ECG -> Pleth timing on the canonical store: beat detectors, beat pairing and the IntelliVue re-sync sawtooth.

Ported from Physio_HNET (kept identical in method so the store and the model side agree):
  ecg_beats / ppg_beats  <- scripts/dev/mladi_pat_verify.py (raw-H5 independent detector), vectorised and
                            run here on II120 / PLETH40
  estimate_offset, pair_nearest <- model/hnet_wav/beat_align.py
  fit_sawtooth, eval_sawtooth   <- model/hnet_wav/sawtooth.py
R = signed extreme of the 0.5-40 Hz ECG within +-40 ms of the 5-20 Hz energy peak (parabola); Pleth foot =
intersecting tangent at the steepest upstroke (10 Hz low-pass). The ECG -> Pleth delay (~1.2 s) exceeds one
inter-beat interval, so beats are paired through the sharpest peak of all R-to-foot differences (with a prior),
then nearest within 120 ms. Between resets r_k the device part of the delay is
saw(t) = a (t - r_k) - a P / 2 (zero-mean over a cycle).
"""
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np


# ------------------------------------------------------------------ detectors

def _bp(x, fs, lo, hi, order=2):
    from scipy.signal import butter, filtfilt
    if lo:
        b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], "band")
    else:
        b, a = butter(order, hi / (fs / 2), "low")
    return filtfilt(b, a, x)


def _fill(v):
    """NaN / implausible samples interpolated (flat stretches carry no beats); None if < 50 % finite."""
    v = np.asarray(v, np.float64)
    ok = np.isfinite(v) & (np.abs(v) < 1e3)
    if ok.mean() < 0.5:
        return None
    if not ok.all():
        i = np.arange(v.size)
        v = np.interp(i, i[ok], v[ok])
    return v


def _parab_vec(y, k):
    k = np.asarray(k, np.int64)
    out = k.astype(np.float64)
    m = (k > 0) & (k < y.size - 1)
    y0, y1, y2 = y[k[m] - 1], y[k[m]], y[k[m] + 1]
    den = y0 - 2 * y1 + y2
    d = np.where(den != 0, 0.5 * (y0 - y2) / np.where(den != 0, den, 1.0), 0.0)
    out[m] = k[m] + np.clip(d, -1, 1)
    return out


def ecg_beats(v, fs) -> Optional[np.ndarray]:
    """R times (s from the first sample)."""
    from scipy.signal import find_peaks
    x = _fill(v)
    if x is None or x.size < fs * 20:
        return None
    e = _bp(x, fs, 5, min(20, 0.45 * fs)) ** 2
    k = max(1, int(0.06 * fs)); e = np.convolve(e, np.ones(k) / k, "same")
    pk, _ = find_peaks(e, distance=int(0.27 * fs), height=0.25 * np.percentile(e, 99))
    if pk.size == 0:
        return np.zeros(0)
    xf = _bp(x, fs, 0.5, min(40, 0.45 * fs))
    w = max(1, int(0.04 * fs)); W = 5 * w
    pk = pk[(pk - W >= 0) & (pk + W < xf.size)]
    if pk.size == 0:
        return np.zeros(0)
    base = np.median(xf[pk[:, None] + np.arange(-W, W + 1)[None]], axis=1)
    seg = xf[pk[:, None] + np.arange(-w, w + 1)[None]] - base[:, None]
    j = np.argmax(np.abs(seg), axis=1)
    s = np.sign(seg[np.arange(pk.size), j]); s[s == 0] = 1.0
    kk = pk - w + j
    # parabola on the polarity-corrected signal
    out = kk.astype(np.float64)
    m = (kk > 0) & (kk < xf.size - 1)
    y0, y1, y2 = s[m] * xf[kk[m] - 1], s[m] * xf[kk[m]], s[m] * xf[kk[m] + 1]
    den = y0 - 2 * y1 + y2
    out[m] = kk[m] + np.clip(np.where(den != 0, 0.5 * (y0 - y2) / np.where(den != 0, den, 1.0), 0.0), -1, 1)
    return out / fs


def ppg_beats(v, fs) -> Optional[np.ndarray]:
    """Foot times (s from the first sample) by the intersecting tangent at the steepest upstroke."""
    from scipy.signal import find_peaks
    x = _fill(v)
    if x is None or x.size < fs * 20:
        return None
    x = _bp(x, fs, None, min(10, 0.45 * fs))
    d = np.gradient(x) * fs
    if abs(np.percentile(d, 0.5)) > np.percentile(d, 99.5) * 1.3:        # inverted trace
        x, d = -x, -d
    pk, _ = find_peaks(d, distance=int(0.27 * fs), height=0.3 * np.percentile(d, 99))
    w = int(0.35 * fs)
    pk = pk[pk - w >= 0]
    if pk.size == 0:
        return np.zeros(0)
    km = _parab_vec(d, pk)
    xm = np.interp(km, np.arange(x.size), x); dm = np.interp(km, np.arange(d.size), d)
    xmin = x[pk[:, None] + np.arange(-w, 1)[None]].min(axis=1)
    ok = dm > 0
    tfo = km[ok] - (xm[ok] - xmin[ok]) / dm[ok] * fs
    tfo = np.maximum(tfo, pk[ok] - w)
    return tfo / fs


# ------------------------------------------------------------------ pairing (beat_align.py)

LO_S, HI_S, BIN_S = -1.5, 3.5, 0.010


def estimate_offset(t_ecg, t_ppg, prior: Optional[float] = None, *, lo: float = LO_S, hi: float = HI_S,
                    bin_s: float = BIN_S, alias_frac: float = 0.8, min_sep: float = 0.25) -> Optional[Tuple[float, float, int]]:
    r = np.asarray(t_ecg, float); f = np.asarray(t_ppg, float)
    if r.size < 10 or f.size < 10:
        return None
    d = (f[None, :] - r[:, None]).ravel()
    d = d[(d >= lo) & (d <= hi)]
    if d.size < 10:
        return None
    nb = int(round((hi - lo) / bin_s))
    h, edges = np.histogram(d, bins=nb, range=(lo, hi))
    hs = np.convolve(h, np.ones(3) / 3, mode="same")
    centers = edges[:-1] + bin_s / 2
    peaks = np.where((hs > 0) & (hs >= np.r_[hs[1:], 0]) & (hs >= np.r_[0, hs[:-1]]))[0]
    if peaks.size == 0:
        return None
    top = peaks[np.argmax(hs[peaks])]
    pick = top
    if prior is not None:
        cand = peaks[hs[peaks] >= alias_frac * hs[top]]
        pick = cand[np.argmin(np.abs(centers[cand] - prior))]
    far = np.abs(centers - centers[pick]) > min_sep
    h2 = float(hs[far].max()) if far.any() else 0.0
    near = d[np.abs(d - centers[pick]) <= 0.05]
    return float(np.median(near)), (h2 / hs[pick] if hs[pick] > 0 else 1.0), int(near.size)


def pair_nearest(t_ecg, t_ppg, offset: float, tol: float = 0.15):
    r = np.asarray(t_ecg, float); f0 = np.asarray(t_ppg, float)
    idx = np.full(r.size, -1, np.int64); res = np.zeros(r.size)
    if r.size == 0 or f0.size == 0:
        return idx, res
    order = np.argsort(f0, kind="stable"); f = f0[order]
    tgt = r + offset
    if f.size > 1:
        j = np.clip(np.searchsorted(f, tgt), 1, f.size - 1)
        j = np.where(np.abs(f[j - 1] - tgt) < np.abs(f[j] - tgt), j - 1, j)
    else:
        j = np.zeros(r.size, np.int64)
    ok = np.abs(f[j] - tgt) <= tol
    idx[ok] = order[j[ok]]; res[ok] = f[j[ok]] - tgt[ok]
    return idx, res


# ------------------------------------------------------------------ sawtooth (sawtooth.py)

def _bin_medians(t, d, bin_s):
    b = np.floor((t - t[0]) / bin_s).astype(np.int64)
    nb = int(b.max()) + 1
    m = np.full(nb, np.nan)
    o = np.argsort(b, kind="stable"); bs = b[o]; ds = d[o]
    cut = np.r_[0, np.nonzero(np.diff(bs))[0] + 1, bs.size]
    for i in range(cut.size - 1):
        m[bs[cut[i]]] = np.median(ds[cut[i]:cut[i + 1]])
    return t[0] + bin_s * (np.arange(nb) + 0.5), m


def fit_sawtooth(t, d, *, bin_s=2.0, thr=0.012, min_gap=20.0, max_gap=300.0, min_resets=3) -> dict:
    t = np.asarray(t, float); d = np.asarray(d, float)
    m_ = np.isfinite(t) & np.isfinite(d)
    t, d = t[m_], d[m_]
    out = {"resets": np.zeros(0), "ramp": 0.0, "period": float("nan"), "ok": False, "n": int(t.size), "jump": float("nan")}
    if t.size < 60:
        return out
    o = np.argsort(t, kind="stable"); t, d = t[o], d[o]
    tm, m = _bin_medians(t, d, bin_s)
    R, i, nb = [], 3, m.size
    while i < nb - 3:
        pre, post = m[i - 3:i], m[i:i + 3]
        if np.isfinite(pre).sum() >= 2 and np.isfinite(post).sum() >= 2 \
                and np.nanmedian(post) - np.nanmedian(pre) < -thr \
                and (not R or tm[i] - bin_s / 2 - R[-1] >= min_gap):
            j0, j1 = max(1, i - 1), min(nb - 1, i + 2)
            dd = [m[j] - m[j - 1] if np.isfinite(m[j]) and np.isfinite(m[j - 1]) else 0.0 for j in range(j0, j1)]
            k = j0 + int(np.argmin(dd))
            R.append(tm[k] - bin_s / 2)
            i = k + int(min_gap / bin_s)
            continue
        i += 1
    R = np.array(R)
    gaps = np.diff(R); gaps = gaps[(gaps > min_gap) & (gaps < max_gap)]
    period = float(np.median(gaps)) if gaps.size else float("nan")
    slopes = []
    for a, b in zip(R[:-1], R[1:]):
        if b - a < max_gap:
            s = (t > a + 2) & (t < b - 1)
            if s.sum() >= 20:
                slopes.append(np.polyfit(t[s], d[s], 1)[0])
    ramp = float(np.median(slopes)) if slopes else 0.0
    ok = bool(R.size >= min_resets and np.isfinite(period) and ramp > 0)
    jump = float("nan")
    if R.size:
        with np.errstate(all="ignore"):
            js = [np.nanmedian(m[(tm > r) & (tm < r + 6)]) - np.nanmedian(m[(tm > r - 6) & (tm < r)]) for r in R]
        jump = float(np.nanmedian(js)) if np.isfinite(js).any() else float("nan")
    out.update(resets=R, ramp=ramp if ok else 0.0, period=period, ok=ok, jump=jump)
    return out


def eval_sawtooth(t, resets, ramp, period):
    """saw(t) in seconds; 0 where there is nothing to apply. Readers: per run, with that run's resets."""
    t = np.asarray(t, float); resets = np.asarray(resets, float)
    if resets.size == 0 or not ramp or not np.isfinite(period) or period <= 0:
        return np.zeros_like(t)
    k = np.searchsorted(resets, t, side="right") - 1
    last = np.where(k >= 0, resets[np.clip(k, 0, None)], np.nan)
    before = k < 0
    if before.any():
        n = np.ceil((resets[0] - t[before]) / period)
        last[before] = resets[0] - n * period
    since = t - last
    since = np.where(since > 1.5 * period, np.mod(since, period), since)
    return ramp * since - ramp * period / 2
