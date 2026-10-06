#!/usr/bin/env python3
"""Synthetic ECG + Pleth entity with a KNOWN ECG -> Pleth delay for testing Stage C3 (no patient data).

Two runs (2 h and 40 min, 20-min gap); RR 0.8 s with HRV; R -> foot = 1.20 s + sawtooth (ramp 29 ms/min, reset
every 66 s, zero-mean) + a slow physiological wander (+-15 ms, 30-min period) + 3 ms jitter. ECG: Gaussian QRS
(one run inverted) + T wave; Pleth: raised-cosine upstroke + exponential decay; both with noise; written at
120 / 40 Hz as II120 / PLETH40 rows with time_ms and a minimal meta.json.

    python workzone/mladi/tests/make_synthetic_timing.py <out_root>   -> <out_root>/synth_timing_0001/
"""
import json, os, sys
import numpy as np

root = sys.argv[1]; od = os.path.join(root, "synth_timing_0001"); os.makedirs(od, exist_ok=True)
rng = np.random.default_rng(7)
RAMP, PERIOD, D0 = 29.0 / 60000.0, 66.0, 1.20          # s/s, s, s
runs = [(0.0, 7200.0), (8400.0, 10800.0)]
T0 = 1.7e12                                              # grid ms of the first segment
E_rows, P_rows, tms, truth = [], [], [], []
for ri, (a, b) in enumerate(runs):
    nseg = int((b - a) // 30)
    dur = nseg * 30.0
    te = np.arange(int(dur * 120)) / 120.0; tp = np.arange(int(dur * 40)) / 40.0
    ecg = 0.02 * rng.standard_normal(te.size); ppg = 0.01 * rng.standard_normal(tp.size)
    phase0 = rng.uniform(0, PERIOD)
    r = 0.3
    while r < dur - 3:
        sign = -1.0 if ri == 1 else 1.0
        ecg += sign * 1.0 * np.exp(-0.5 * ((te - r) / 0.012) ** 2) + 0.25 * np.exp(-0.5 * ((te - r - 0.3) / 0.05) ** 2)
        tg = a + r
        saw = RAMP * ((tg + phase0) % PERIOD) - RAMP * PERIOD / 2
        phys = 0.015 * np.sin(2 * np.pi * tg / 1800.0)
        f = r + D0 + saw + phys + 0.003 * rng.standard_normal()
        u = (tp - f)
        up = (u >= 0) & (u < 0.15)
        ppg[up] += 0.5 * (1 - np.cos(np.pi * u[up] / 0.15))
        dn = u >= 0.15
        ppg[dn] += np.exp(-(u[dn] - 0.15) / 0.35) * (u[dn] < 1.5)
        truth.append((tg, f - r, saw))
        r += 0.8 + 0.05 * np.sin(2 * np.pi * tg / 20.0) + 0.02 * rng.standard_normal()
    E_rows.append(ecg[: nseg * 3600].reshape(nseg, 3600)); P_rows.append(ppg[: nseg * 1200].reshape(nseg, 1200))
    tms.append(T0 + (a + 30.0 * np.arange(nseg)) * 1000)
np.save(os.path.join(od, "II120.npy"), np.concatenate(E_rows).astype(np.float16))
np.save(os.path.join(od, "PLETH40.npy"), np.concatenate(P_rows).astype(np.float16))
np.save(os.path.join(od, "time_ms.npy"), np.concatenate(tms).astype(np.int64))
json.dump({"entity_id": "synth_timing_0001", "n_seg": int(sum(x.shape[0] for x in E_rows))}, open(os.path.join(od, "meta.json"), "w"))
tr = np.array(truth); np.save(os.path.join(od, "_truth.npy"), tr)
print("segments", sum(x.shape[0] for x in E_rows), "beats", tr.shape[0], "true ramp ms/min 29.0 period 66 level 1200 ms")
