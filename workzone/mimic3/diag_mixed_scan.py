#!/usr/bin/env python
"""Per-window max classification: how many patients mix [0,1] and [0,4] PLETH scales?"""
import os, sys, warnings
import numpy as np
import multiprocessing as mp

ROOT = '/opt/localdata100tb/physio_data/mimic3'
K = 256
warnings.filterwarnings('ignore', category=RuntimeWarning)

def scan_one(ent):
    p = os.path.join(ROOT, ent, 'PLETH40.npy')
    try:
        arr = np.load(p, mmap_mode='r')
        n = arr.shape[0]
        idx = np.unique(np.linspace(0, n - 1, min(K, n)).astype(int))
        sub = np.asarray(arr[idx], dtype=np.float32)
        wmax = np.nanmax(sub, axis=1)
        wmax = wmax[np.isfinite(wmax)]
        if len(wmax) == 0:
            return ent, -1, -1
        f_unit = float(np.mean(wmax <= 1.1))
        f_quad = float(np.mean(wmax > 1.1))
        return ent, f_unit, f_quad
    except Exception:
        return ent, -1, -1

ents = sorted(e for e in os.listdir(ROOT)
              if os.path.exists(os.path.join(ROOT, e, 'PLETH40.npy')))
pure_unit = pure_quad = mixed = bad = 0
mixed_list = []
with mp.Pool(16) as pool:
    for ent, fu, fq in pool.imap_unordered(scan_one, ents, chunksize=16):
        if fu < 0:
            bad += 1
        elif fq >= 0.05 and fu >= 0.05:
            mixed += 1
            mixed_list.append((ent, round(fu, 2), round(fq, 2)))
        elif fq > fu:
            pure_quad += 1
        else:
            pure_unit += 1
print(f"pure [0,1]: {pure_unit}  pure [0,4]: {pure_quad}  MIXED: {mixed}  bad: {bad}")
for m in mixed_list[:20]:
    print("mixed example:", m)
with open('/labs/hulab/mxwang/Physio_Data/workzone/mimic3/diag_out/mixed_patients.txt', 'w') as f:
    for ent, fu, fq in mixed_list:
        f.write(f"{ent}\t{fu}\t{fq}\n")
