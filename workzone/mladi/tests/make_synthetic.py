#!/usr/bin/env python3
"""Build a synthetic MLADI-like tree for testing the stages (no patient data): one audata-style HDF5
(26 h, a 10-min gap, Pleth invalid-code stretch, ART + NBP numerics, factor-coded /ehr tables with units,
routes, doses, infusions, demographic), its pretrain_wav_v2-style mmaps + __meta.json (band-passed, two
blocks) and an e1 index.

    python workzone/mladi/tests/make_synthetic.py <out_dir>
    -> <out>/h5/<base>.h5, <out>/wav/<base>_*_mmap.npy + __meta.json, <out>/e1.json
"""
import json, os, sys
import h5py
import numpy as np
from scipy.signal import butter, filtfilt, resample_poly

D = sys.argv[1]
for s in ("h5", "wav"):
    os.makedirs(os.path.join(D, s), exist_ok=True)
rng = np.random.default_rng(1)
base = "20210910_555_777"
T0, H, gap = 1000.0, 26 * 3600, (10 * 3600, 10 * 3600 + 600)


def keep(t):
    return (t < T0 + gap[0]) | (t >= T0 + gap[1])


def slow(t):
    return np.sin(2 * np.pi * (np.asarray(t) - T0) / (6 * 3600))


def table(f, name, rows, dtype, factors):
    ds = f.create_dataset(f"ehr/{name}", data=np.array(rows, dtype=dtype))
    ds.attrs[".meta"] = json.dumps({"columns": {c: {"type": "factor", "levels": lv} for c, lv in factors.items()}})


with h5py.File(f"{D}/h5/{base}.h5", "w") as f:
    f.attrs[".meta"] = json.dumps({"audata_version": "1.1", "time_origin": "2018-10-22 04:00:00.000000 EDT"})
    for nm, fs in (("II", 500), ("Pleth", 125)):
        t = T0 + np.arange(0, H, 1 / fs); t = t[keep(t)]
        ph = np.cumsum(np.r_[0, np.diff(t)]) * (1.2 + 0.1 * slow(t))
        v = (np.sin(2 * np.pi * ph) ** 8 if nm == "II" else 0.5 + 0.2 * np.sin(2 * np.pi * ph)) + 0.3 * np.sin(2 * np.pi * 0.05 * t)
        if nm == "Pleth":
            v[(t > T0 + 5000) & (t < T0 + 5003)] = -268365824.0
        a = np.zeros(t.size, dtype=[("time", "f8"), ("value", "f4")]); a["time"] = t; a["value"] = v
        ds = f.create_dataset(f"data/waveforms/{nm}", data=a, chunks=(200000,))
        ds.attrs[".meta"] = json.dumps({"dwc_meta": {"label": nm, "samplePeriod": 1000 / fs, "minTime": float(t[0]), "maxTime": float(t[-1]),
                                                     "lowEdgeFrequency": 0.05, "highEdgeFrequency": 150.0}})
    tn = T0 + np.arange(0, H, 1.024); tn = tn[keep(tn)]
    for nm, val in (("HR.HR", 72 + 8 * slow(tn)), ("SpO₂.SpO₂", 96 + 0 * tn), ("RR.RR", 16 + 0 * tn),
                    ("ART.Systolic", 120 + 15 * slow(tn)), ("ART.Diastolic", 65 + 5 * slow(tn)), ("ART.Mean", 85 + 8 * slow(tn)),
                    ("ART.Pulse", 72 + 8 * slow(tn)), ("SpO₂.Pulse", 72 + 8 * slow(tn)), ("Perf.Perf", 2.5 + 0 * tn), ("PVC.PVC", 0 * tn)):
        a = np.zeros(tn.size, dtype=[("time", "f8"), ("value", "f4")]); a["time"] = tn; a["value"] = val
        f.create_dataset(f"data/numerics/{nm}", data=a).attrs[".meta"] = json.dumps({"dwc_meta": {"label": nm}})
    tb = T0 + np.arange(600, H, 3600.0)
    nbp = {"NBP.NBPs": 118 + 15 * slow(tb), "NBP.NBPd": 64 + 5 * slow(tb), "NBP.NBPm": 84 + 8 * slow(tb)}
    for nm, val in nbp.items():
        a = np.zeros(tb.size, dtype=[("time", "f8"), ("value", "f4")]); a["time"] = tb; a["value"] = np.round(val)
        f.create_dataset(f"data/numerics/{nm}", data=a)
    # charted vitals: SBP/DBP copy the cuff readings (validated), pulse hourly, temperature in F and C
    lr_lv = ["Pulse", "Systolic BP", "Diastolic BP", "Temperature Metric", "Temperature", "Oxygen % (FiO2)",
             "Positive end expiratory pressure (PEEP)", "RRT Vent Status"]
    un_lv = ["bpm", "mmHg", "DegC", "DegF", "%", "cmH2O"]
    rows = []
    for k, t in enumerate(tb):
        rows += [(t, 1, float(np.round(nbp["NBP.NBPs"][k])), 1), (t, 2, float(np.round(nbp["NBP.NBPd"][k])), 1)]
    for t in T0 + np.arange(1800, H, 3600.0):
        rows += [(t, 0, float(72 + 8 * slow(t)), 0), (t, 3, 37.0, 2), (t, 4, 98.6, 3), (t, 5, 40.0, 4), (t, 6, 5.0, 5), (t, 7, np.nan, -1)]
    table(f, "low_rate", rows, [("date", "f8"), ("eventName", "i1"), ("resultVal", "f8"), ("resultUnit", "i1")],
          {"eventName": lr_lv, "resultUnit": un_lv})
    lab_lv = ["K", "Lactate, Whole Blood", "Cr", "Ca", "Ionized Calcium Whole Blood"]
    lab_un = ["mMol/L", "mg/dL"]
    labs = [(T0 + t, 0, 0, 4.1, 0) for t in range(-3600, H, 6 * 3600)] + [(T0 + t, 0, 1, 1.8, 0) for t in range(7200, H, 8 * 3600)] + \
           [(T0 + 9000, 0, 2, 1.1, 1), (T0 + 9000, 0, 3, 8.9, 1), (T0 + 9000, 0, 4, 1.15, 0), (T0 + 9000, 0, 0, 41.0, 1)]
    table(f, "lab_results", labs, [("time", "f8"), ("orderedAs", "i1"), ("eventDisp", "i1"), ("resultVal", "f8"), ("resultUnit", "i1")],
          {"orderedAs": ["BMP"], "eventDisp": lab_lv, "resultUnit": lab_un})
    med_lv = ["norepinephrine", "insulin regular", "potassium chloride", "Lactated Ringers", "lidocaine topical"]
    du_lv = ["mcg", "Unit(s)", "mEq", "mL", "Application"]
    rt_lv = ["IV", "subQ", "Topically"]
    meds = [(T0 + t, 0, 0, 8.0, 0, np.nan, -1) for t in range(20000, 40000, 900)] + \
           [(T0 + 30000, 1, 1, 4.0, 1, np.nan, -1), (T0 + 31000, 2, 0, 20.0, 2, np.nan, -1),
            (T0 + 32000, 3, 0, 1000.0, 3, 1000.0, 0), (T0 + 33000, 4, 2, 1.0, 4, np.nan, -1)]
    table(f, "medications", meds, [("time", "f8"), ("catalogDisp", "i1"), ("route", "i1"), ("dose", "f8"), ("doseUnit", "i1"),
                                   ("volumeDose", "f8"), ("volumeDoseUnit", "i1")],
          {"catalogDisp": med_lv, "route": rt_lv, "doseUnit": du_lv, "volumeDoseUnit": ["mL"]})
    io = [(T0 + t, 0, 120.0, 0, 0) for t in range(3600, H, 3600)] + [(T0 + 50000, 1, 300.0, 0, 1)]
    table(f, "infusions_and_outputs", io, [("time", "f8"), ("name", "i1"), ("volume", "f8"), ("unit", "i1"), ("detail", "i1")],
          {"name": ["Urine Output", "Blood Products/Colloids"], "unit": ["mL"], "detail": ["Foley Catheter", "Red Blood Cells 1 unit"]})
    table(f, "demographic", [(0, 61.0, T0 - 2 * 86400, T0 + H + 86400)], [("sex", "i1"), ("age", "f8"), ("regDate", "f8"), ("dischDate", "f8")],
          {"sex": ["Male"]})
# pretrain_wav_v2-style mmaps (band-passed, as data_preparing_v2) + seg_list
seg, rowsP, rowsE = [], [], []
with h5py.File(f"{D}/h5/{base}.h5", "r") as f:
    for b, (bs, be) in enumerate(((T0, T0 + gap[0]), (T0 + gap[1], T0 + H))):
        for key, fs, tfs, band, L, store in (("Pleth", 125, 40, (0.5, 12), 1200, rowsP), ("II", 500, 120, (0.5, 50), 3600, rowsE)):
            a = f[f"data/waveforms/{key}"][:]; m = (a["time"] >= bs) & (a["time"] <= be)
            g = bs + np.arange(int((be - bs) * fs)) / fs
            y = np.interp(g, a["time"][m], a["value"][m].astype(float))
            bb, aa = butter(4, [band[0] / (fs / 2), band[1] / (fs / 2)], "band"); y = filtfilt(bb, aa, y)
            y = resample_poly(y, tfs, fs); n = y.size // L; store.append(y[:n * L].reshape(n, L))
        n = min(rowsP[-1].shape[0], rowsE[-1].shape[0]); rowsP[-1] = rowsP[-1][:n]; rowsE[-1] = rowsE[-1][:n]
        seg += [[b, len(seg) + k, bs + 30 * k] for k in range(n)]
P = np.clip(np.concatenate(rowsP), -1e3, 1e3).astype(np.float16); E = np.concatenate(rowsE).astype(np.float16)
np.save(f"{D}/wav/{base}_Pleth_40Hz_{len(P)}_1200_mmap.npy", P); np.save(f"{D}/wav/{base}_II_120Hz_{len(E)}_3600_mmap.npy", E)
json.dump({"seg_list": seg}, open(f"{D}/wav/{base}__meta.json", "w"))
json.dump({"encounters": {base: {"split": "train"}}}, open(f"{D}/e1.json", "w"))
print("built", base, len(seg), "rows")
