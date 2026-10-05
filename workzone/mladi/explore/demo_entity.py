#!/usr/bin/env python3
"""MLADI Step 0c: one encounter, raw DWC HDF5 -> canonical format, plotted for visual alignment.

  time base   time_ms = time_origin (root .meta, America/New_York, EDT/EST/LMT as stamped) + t * 1000;
              every audata time column (waveforms, numerics, /ehr) is seconds from that origin.
  grid        the earlier pretrain_wav_v2 rows (__meta.json seg_list): segment i = mmap row i, so
              label caches and the e1 split keep indexing the same seconds. Consecutive rows are
              processed as one stretch on the block's own sample grid (start + n / src_fs), the way
              data_preparing_v2 did, then resample_poly -- and NO band-pass (canonical = raw).
              Check: band-passing the canonical row as data_preparing_v2 did must reproduce the
              mmap row (correlation per row reported).
  vitals_hf   data/numerics at 1.024 s -> [N_seg, 30, n_var] 1-s slots (last wins); ABP from ART.*
              first, then ABP.*. NBP -> nbp_events (value changes).
  ehr         /ehr factors decoded per file (audata levels); a subset of charted vitals, labs and
              vasopressors mapped to registry ids for the demo; split into baseline / recent /
              events / future around the waveform span.
  alignment   charted vs monitor at lags -60..+60 min (1-min medians): Pulse vs HR, Systolic BP vs
              NBP, arterial systolic vs ART systolic. The best lag should be ~0.

Writes <intermediate>/explore/demo_<base>/ (canonical arrays, summary.json, demo.png). Nothing
leaves PSC from here.
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np

RAW = "/ocean/projects/med250003p/shared/mladi_extract_2023_waves"
WAV = "/ocean/projects/med250003p/shared/pretrain_wav_v2"
INTER = "/ocean/projects/med250003p/mwang11/Physio_Data/workzone/outputs/mladi"
NY = ZoneInfo("America/New_York")
CH = {"PLETH40": ("Pleth", 125, 40, 1200, (0.5, 12.0)), "II120": ("II", 500, 120, 3600, (0.5, 50.0))}
NUM = [(150, "HR_hf", ["HR.HR"]), (151, "SpO2_hf", ["SpO₂.SpO₂"]), (152, "RR_hf", ["RR.RR"]),
       (153, "ABPs_hf", ["ART.Systolic", "ABP.ABPs"]), (154, "ABPd_hf", ["ART.Diastolic", "ABP.ABPd"]),
       (155, "ABPm_hf", ["ART.Mean", "ABP.ABPm"]), (156, "PULSE_hf", ["SpO₂.Pulse"]),
       (113, "PR_art", ["ART.Pulse", "ABP.Pulse"]), (160, "CVP_hf", ["CVP.CVPm"]), (164, "PVCrate_hf", ["PVC.PVC"])]
NBP = [(157, "NBP.NBPs"), (158, "NBP.NBPd"), (159, "NBP.NBPm")]
LOW_RATE = {"Pulse": 100, "O2 Saturation": 101, "Respiratory Rate": 102, "Temperature Metric": 103,
            "Systolic BP": 104, "Diastolic BP": 105, "Mean blood pressure": 106, "Glasgow Coma Score": 108,
            "Arterial Systolic Pr": 110, "Arterial Diastolic P": 111, "Mean arterial pressu": 112}
LABS = {"K": 0, "Ca": 1, "Na": 2, "Glucose": 3, "Lactate": 4, "Lactate, Whole Blood": 4, "Cr": 5,
        "Bili, Total": 6, "Platelets": 7, "WBC": 8, "Hgb": 9, "INR": 10, "BUN": 11}
MEDS = {"norepinephrine": 207, "epinephrine": 208, "phenylephrine": 209, "vasopressin": 211}
EV = np.dtype([("time_ms", "i8"), ("seg_idx", "i4"), ("var_id", "u2"), ("value", "f4")])
DAY = 24 * 3600 * 1000


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def origin_ms(f):
    s = json.loads(f.attrs[".meta"])["time_origin"]
    stamp, tz = s.rsplit(" ", 1)
    dt = datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=NY)
    return int(round(dt.timestamp() * 1000)), s, tz, dt.tzname()


def find(ds, t, lo=0):
    hi = ds.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if ds[mid]["time"] < t:
            lo = mid + 1
        else:
            hi = mid
    return lo


def factor(ds):
    a = ds[:]
    cm = json.loads(ds.attrs[".meta"]).get("columns", {}) if ".meta" in ds.attrs else {}
    out = {}
    for c in a.dtype.names:
        lev = cm.get(c, {}).get("levels")
        if lev:
            codes = a[c].astype(int)
            out[c] = np.array([lev[k] if 0 <= k < len(lev) else None for k in codes], dtype=object)
        else:
            out[c] = a[c]
    return out


E1 = "/ocean/projects/med250003p/shared/data_cache/e1_mladi/mladi_encounter_index.json"


def pick(n_scan=4000):
    e1 = json.load(open(E1))["encounters"]
    import h5py
    files = sorted(glob.glob(os.path.join(RAW, "*.h5")))
    files = files[len(files) // 2:] + files[: len(files) // 2]
    for p in files[:n_scan]:
        b = os.path.basename(p)[:-3]
        if e1.get(b, {}).get("split") != "train" or not os.path.exists(os.path.join(WAV, b + "__meta.json")):
            continue
        with h5py.File(p, "r") as f:
            need = ["data/waveforms/Pleth", "data/waveforms/II", "data/numerics/ART.Systolic", "data/numerics/NBP.NBPs",
                    "ehr/lab_results", "ehr/low_rate", "ehr/medications"]
            if not all(k in f for k in need):
                continue
            m = json.loads(f["data/waveforms/Pleth"].attrs[".meta"])["dwc_meta"]
            h = (m["maxTime"] - m["minTime"]) / 3600
            if 24 <= h <= 96:
                return b
    raise SystemExit("no encounter matched")


def main():
    import h5py
    from scipy.signal import butter, filtfilt, resample_poly
    global RAW, WAV, INTER, E1
    ap = argparse.ArgumentParser(); ap.add_argument("--base", default=None)
    for k in ("raw", "wav", "inter", "e1"):
        ap.add_argument("--" + k, default=None)
    a = ap.parse_args()
    RAW, WAV, INTER, E1 = a.raw or RAW, a.wav or WAV, a.inter or INTER, a.e1 or E1
    base = a.base or pick()
    out = os.path.join(INTER, "explore", f"demo_{base}"); os.makedirs(out, exist_ok=True)
    log("entity", base)
    meta = json.load(open(os.path.join(WAV, base + "__meta.json")))
    seg = np.array([[s[0], s[1], s[2]] for s in meta["seg_list"]], float)
    seg = seg[np.argsort(seg[:, 2], kind="stable")]
    start_s = seg[:, 2]; n_seg = start_s.size
    S = {"base": base, "n_seg": int(n_seg)}
    with h5py.File(os.path.join(RAW, base + ".h5"), "r") as f:
        o_ms, o_str, tz, tz_resolved = origin_ms(f)
        S.update(time_origin=o_str[:4] + "-..", tz_stamped=tz, tz_resolved=tz_resolved)
        time_ms = o_ms + np.round(start_s * 1000).astype(np.int64)
        assert np.all(np.diff(time_ms) > 0)
        # ---- waveforms on the mmap grid, no band-pass
        runs, r0 = [], 0
        for i in range(1, n_seg + 1):
            if i == n_seg or abs(start_s[i] - start_s[i - 1] - 30) > 1e-3 or seg[i, 0] != seg[i - 1, 0]:
                runs.append((r0, i)); r0 = i
        S["runs"] = len(runs)
        arrays = {}
        for name, (key, sfs, tfs, L, band) in CH.items():
            ds = f["data/waveforms/" + key]
            X = np.full((n_seg, L), np.nan, np.float32)
            for (i0, i1) in runs:
                t0, t1 = start_s[i0], start_s[i1 - 1] + 30
                j0 = find(ds, t0 - 1); j1 = find(ds, t1 + 1, j0)
                if j1 - j0 < 10:
                    continue
                blk = ds[j0:j1]; tt = blk["time"].astype(float); vv = blk["value"].astype(float)
                bad = ~np.isfinite(vv) | (np.abs(vv) > 1e3)
                g = t0 + np.arange(int(round((t1 - t0) * sfs))) / sfs
                ok = ~bad
                if ok.sum() < 10:
                    continue
                y = np.interp(g, tt[ok], vv[ok])
                if bad.any():                    # samples that sat on an invalid code stay NaN
                    nb = np.interp(g, tt, bad.astype(float)) > 0.5
                    y[nb] = np.nan
                nan = np.isnan(y)
                yz = resample_poly(np.where(nan, 0.0, y), tfs, sfs)
                if nan.any():
                    nm = resample_poly(nan.astype(float), tfs, sfs) > 0.1
                    yz[nm] = np.nan
                n = min((i1 - i0) * L, yz.size)
                X[i0:i0 + n // L] = yz[: (n // L) * L].reshape(-1, L)
            arrays[name] = X.astype(np.float16)
            # grid check against the earlier band-passed mmap
            mm = glob.glob(os.path.join(WAV, f"{base}_{key}_{tfs}Hz_*_mmap.npy"))
            if mm:
                M = np.load(mm[0], mmap_mode="r")
                b_, a_ = butter(4, [band[0] / (tfs / 2), min(band[1] / (tfs / 2), 0.99)], "band")
                rs = []
                for i in np.linspace(0, n_seg - 1, 40).astype(int):
                    x = X[i].astype(float)
                    if np.isfinite(x).all() and np.std(M[i]) > 0:
                        rs.append(np.corrcoef(filtfilt(b_, a_, x), np.asarray(M[i], float))[0, 1])
                S[f"{name}_vs_mmap_r_p10_p50"] = [float(np.percentile(rs, 10)), float(np.median(rs))] if rs else None
                log(f"{name}: band-passed canonical vs mmap rows r p10/p50 {S[f'{name}_vs_mmap_r_p10_p50']}")
            S[f"{name}_nan_frac"] = float(np.isnan(X).mean())
        # ---- vitals_hf, 1-s slots
        slot, n_slot = 1.0, 30
        V = np.full((n_seg, n_slot, len(NUM)), np.nan, np.float32)
        src = {}
        for j, (vid, vname, keys) in enumerate(NUM):
            for k in keys:
                if "data/numerics/" + k in f:
                    a_ = f["data/numerics/" + k][:]
                    t = a_["time"].astype(float); v = a_["value"].astype(float)
                    i = np.searchsorted(start_s, t, side="right") - 1
                    sl = np.floor((t - start_s[np.clip(i, 0, None)]) / slot).astype(int)
                    m = (i >= 0) & (sl >= 0) & (sl < n_slot) & np.isfinite(v)
                    fill = np.isnan(V[i[m], sl[m], j])
                    V[i[m][fill], sl[m][fill], j] = v[m][fill]
                    src[vname] = src.get(vname, []) + [k]
        S["vitals_hf_coverage"] = {NUM[j][1]: float(np.isfinite(V[:, :, j]).mean()) for j in range(len(NUM))}
        S["vitals_hf_sources"] = src
        # ---- nbp events
        nbp = []
        for vid, k in NBP:
            if "data/numerics/" + k in f:
                a_ = f["data/numerics/" + k][:]
                t = a_["time"].astype(float); v = a_["value"].astype(float)
                keep = np.r_[True, np.diff(v) != 0] & np.isfinite(v)
                for tt, vv in zip(t[keep], v[keep]):
                    nbp.append((o_ms + int(round(tt * 1000)), vid, vv))
        # ---- ehr
        events = []
        if "ehr/low_rate" in f:
            d = factor(f["ehr/low_rate"])
            for nm, t, v in zip(d["eventName"], d["date"], d["resultVal"]):
                if nm in LOW_RATE and np.isfinite(v):
                    events.append((o_ms + int(round(t * 1000)), LOW_RATE[nm], float(v)))
        if "ehr/lab_results" in f:
            d = factor(f["ehr/lab_results"])
            for nm, t, v in zip(d["eventDisp"], d["time"], d["resultVal"]):
                if nm in LABS and np.isfinite(v):
                    events.append((o_ms + int(round(t * 1000)), LABS[nm], float(v)))
        if "ehr/medications" in f:
            d = factor(f["ehr/medications"])
            for nm, t, v in zip(d["catalogDisp"], d["time"], d["dose"]):
                if nm in MEDS and np.isfinite(t):
                    events.append((o_ms + int(round(t * 1000)), MEDS[nm], float(v) if np.isfinite(v) else np.nan))
        demo_age = None
        if "ehr/demographic" in f:
            d = factor(f["ehr/demographic"])
            S["demographic_cols"] = sorted(d.keys())
            demo_age = d.get("age")
    ev = np.array(sorted(events), dtype=[("time_ms", "i8"), ("var_id", "u2"), ("value", "f4")])
    w0, w1 = time_ms[0], time_ms[-1] + 30000
    parts = {"ehr_baseline": ev[(ev["time_ms"] < w0 - DAY) & (ev["time_ms"] >= w0 - 30 * DAY)],
             "ehr_recent": ev[(ev["time_ms"] >= w0 - DAY) & (ev["time_ms"] < w0)],
             "ehr_events": ev[(ev["time_ms"] >= w0) & (ev["time_ms"] <= w1)],
             "ehr_future": ev[(ev["time_ms"] > w1) & (ev["time_ms"] <= w1 + 7 * DAY)]}
    sent = {"ehr_baseline": np.iinfo(np.int32).min, "ehr_recent": np.iinfo(np.int32).min + 1,
            "ehr_future": np.iinfo(np.int32).min + 2}
    for k, p in parts.items():
        arr = np.zeros(p.size, EV)
        arr["time_ms"], arr["var_id"], arr["value"] = p["time_ms"], p["var_id"], p["value"]
        arr["seg_idx"] = (np.clip(np.searchsorted(time_ms, p["time_ms"], side="right") - 1, 0, n_seg - 1)
                          if k == "ehr_events" else sent[k])
        np.save(os.path.join(out, k + ".npy"), arr)
    S["ehr_counts"] = {k: int(v.size) for k, v in parts.items()}
    np.save(os.path.join(out, "time_ms.npy"), time_ms)
    for k, v in arrays.items():
        np.save(os.path.join(out, k + ".npy"), np.ascontiguousarray(v))
    np.save(os.path.join(out, "vitals_hf.npy"), V)
    nb = np.array(nbp, dtype=[("time_ms", "i8"), ("var_id", "u2"), ("value", "f4")])
    S["n_nbp_events"] = int(nb.size)

    # ---- alignment: charted vs monitor, 1-min medians, lags -60..60 min
    def minute_series(t_ms, v):
        k = ((t_ms - w0) // 60000).astype(int); n = int((w1 - w0) // 60000) + 1
        s = np.full(n, np.nan); m = (k >= 0) & (k < n) & np.isfinite(v)
        k, v = k[m], v[m]
        if k.size:
            o = np.argsort(k, kind="stable"); k, v = k[o], v[o]
            u, i0 = np.unique(k, return_index=True)
            for kk, a_, b_ in zip(u, i0, np.r_[i0[1:], k.size]):
                s[kk] = np.median(v[a_:b_])
        return s
    flat_t = (time_ms[:, None] + (np.arange(n_slot) * 1000)[None, :]).ravel()
    mon = {vname: minute_series(flat_t, V[:, :, j].ravel()) for j, (_, vname, _) in enumerate(NUM)}
    mon["NBPs"] = None
    if nb.size:                       # cuff readings are sparse: each one stands for +-5 min around it
        m = nb["var_id"] == 157
        base_s = minute_series(nb["time_ms"][m], nb["value"][m].astype(float))
        dil = base_s.copy()
        for sh in range(1, 6):
            for src_ in (np.r_[base_s[sh:], [np.nan] * sh], np.r_[[np.nan] * sh, base_s[:-sh]]):
                dil = np.where(np.isnan(dil), src_, dil)
        mon["NBPs"] = dil
    E = parts["ehr_events"]
    pairs = [("charted Pulse vs HR_hf", 100, "HR_hf"), ("charted Systolic BP vs NBPs", 104, "NBPs"),
             ("charted arterial systolic vs ABPs_hf", 110, "ABPs_hf")]
    S["alignment"] = {}
    for lab, vid, mk in pairs:
        ch = E[E["var_id"] == vid]
        ms = mon.get(mk)
        if ch.size < 5 or ms is None:
            continue
        k = ((ch["time_ms"] - w0) // 60000).astype(int)
        res = []
        for lag in range(-60, 61):
            kk = k + lag; ok = (kk >= 0) & (kk < ms.size)
            x = ch["value"][ok].astype(float); y = ms[kk[ok]]; g = np.isfinite(y)
            if g.sum() >= 5:
                res.append((lag, float(np.median(np.abs(x[g] - y[g]))), int(g.sum())))
        if res:
            best = min(res, key=lambda r: r[1])
            at0 = [r for r in res if r[0] == 0]
            S["alignment"][lab] = {"best_lag_min": best[0], "mad_at_best": best[1],
                                   "mad_at_0": at0[0][1] if at0 else None, "n": best[2]}
            log(f"alignment {lab}: best lag {best[0]} min (median |diff| {best[1]:.1f}), at 0: {at0[0][1] if at0 else None}")

    # ---- figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(16, 15)); gs = fig.add_gridspec(5, 3, height_ratios=[1, 1, 1.2, 1.2, 1.0], hspace=0.55, top=0.95)
    hours = (np.arange(int((w1 - w0) // 60000) + 1)) / 60
    picks = [n_seg // 4, n_seg // 2, 3 * n_seg // 4]
    for c, i in enumerate(picks):
        for r, name in enumerate(("PLETH40", "II120")):
            ax = fig.add_subplot(gs[r, c]); x = arrays[name][i].astype(float); fs = CH[name][2]
            n = 10 * fs; ax.plot(np.arange(n) / fs, x[:n], lw=0.8, c="#2a78d6" if r == 0 else "k")
            ax.set_title(f"{name} segment {i} ({(time_ms[i] - w0) / 3.6e6:.1f} h), raw", fontsize=9)
            ax.set_xlabel("s", fontsize=8)
    ax = fig.add_subplot(gs[2, :])
    ax.plot(hours[: mon["HR_hf"].size], mon["HR_hf"], lw=0.7, c="k", label="HR_hf (monitor, 1-min median)")
    ax.plot(hours[: mon["SpO2_hf"].size], mon["SpO2_hf"], lw=0.7, c="#2a78d6", label="SpO2_hf")
    for vid, c_, lab in ((100, "#d62728", "charted Pulse"), (101, "#17becf", "charted SpO2")):
        ch = E[E["var_id"] == vid]
        ax.scatter((ch["time_ms"] - w0) / 3.6e6, ch["value"], s=10, c=c_, label=lab, zorder=3)
    ax.legend(fontsize=7, ncol=4, loc="upper right"); ax.set_ylabel("bpm / %"); ax.set_title("monitor numerics vs charted vitals (hours from waveform start)", fontsize=10)
    ax = fig.add_subplot(gs[3, :], sharex=ax)
    for vname, c_ in (("ABPs_hf", "#d62728"), ("ABPm_hf", "#9467bd"), ("ABPd_hf", "#1f77b4")):
        ax.plot(hours[: mon[vname].size], mon[vname], lw=0.6, c=c_, label=vname + " (ART/ABP numerics)")
    if nb.size:
        m = nb["var_id"] == 157
        ax.scatter((nb["time_ms"][m] - w0) / 3.6e6, nb["value"][m], marker="s", s=14, c="k", label="NBPs (monitor cuff)", zorder=3)
    for vid, c_, lab in ((104, "#ff7f0e", "charted Systolic BP"), (110, "#e377c2", "charted arterial systolic")):
        ch = E[E["var_id"] == vid]
        ax.scatter((ch["time_ms"] - w0) / 3.6e6, ch["value"], s=9, c=c_, label=lab, zorder=4)
    ax.legend(fontsize=7, ncol=3, loc="upper right"); ax.set_ylabel("mmHg"); ax.set_title("arterial line numerics, cuff NBP and charted pressures", fontsize=10)
    ax = fig.add_subplot(gs[4, :], sharex=ax)
    cov = np.zeros(hours.size)
    k = ((time_ms - w0) // 60000).astype(int); cov[k[k < cov.size]] = 1
    ax.fill_between(hours, -1, -0.4, where=cov > 0, color="0.8", step="mid", label="waveform segments")
    names = {**{v: k for k, v in LABS.items()}, **{v: k for k, v in MEDS.items()}}
    ids = sorted({int(x) for x in np.unique(E["var_id"]) if x < 100 or x >= 200})
    for r, vid in enumerate(ids):
        ch = E[E["var_id"] == vid]
        ax.scatter((ch["time_ms"] - w0) / 3.6e6, np.full(ch.size, r), s=8, marker="|", c="#d62728" if vid >= 200 else "k")
    ax.set_yticks(range(len(ids))); ax.set_yticklabels([names.get(v, str(v)) for v in ids], fontsize=7)
    ax.set_xlabel("hours from waveform start"); ax.set_title("lab results (black) and vasopressor administrations (red)", fontsize=10)
    al = "; ".join(f"{k}: best lag {v['best_lag_min']} min" for k, v in S["alignment"].items())
    fig.suptitle(f"MLADI demo encounter (canonical, raw H5 -> 30-s grid of pretrain_wav_v2)\n{al}", fontsize=11, x=0.02, ha="left", y=0.995)
    fig.savefig(os.path.join(out, "demo.png"), dpi=100, bbox_inches="tight"); plt.close(fig)
    json.dump(S, open(os.path.join(out, "summary.json"), "w"), indent=1, default=str)
    log("summary " + json.dumps({k: v for k, v in S.items() if k not in ("base",)}, default=str)[:3000])
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
