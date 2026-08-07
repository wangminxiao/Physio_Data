#!/usr/bin/env python
"""Diagnostic: per-record PLETH40 / II120 amplitude scan over an extracted store.

For each entity dir: subsample up to K segments evenly spaced, compute per-window
nanmean/nanstd, aggregate per record (median across windows). Also record the max
sampled value (upper-bound check: [0,1] vs [0,4] populations).

Usage: python diag_amp_scan.py <store_root> <out_csv> [channels...]
"""
import os
import sys
import json
import csv
import warnings
import numpy as np
import multiprocessing as mp

ROOT = sys.argv[1]
OUT = sys.argv[2]
CHANNELS = sys.argv[3:] if len(sys.argv) > 3 else ['PLETH40', 'II120']
K = 128

warnings.filterwarnings('ignore', message='Mean of empty slice')
warnings.filterwarnings('ignore', message='Degrees of freedom')
warnings.filterwarnings('ignore', category=RuntimeWarning)


def scan_one(ent):
    d = os.path.join(ROOT, ent)
    row = {'entity': ent}
    try:
        with open(os.path.join(d, 'meta.json')) as f:
            meta = json.load(f)
        row['source_path'] = meta.get('source_path', '')
    except Exception:
        row['source_path'] = ''
    for ch in CHANNELS:
        p = os.path.join(d, ch + '.npy')
        pre = ch.lower()
        for suf in ('n_seg', 'wmean_med', 'wstd_med', 'vmax', 'vmin', 'nan_frac'):
            row[f'{pre}_{suf}'] = ''
        if not os.path.exists(p):
            continue
        try:
            arr = np.load(p, mmap_mode='r')
            n = arr.shape[0]
            idx = np.unique(np.linspace(0, n - 1, min(K, n)).astype(int))
            sub = np.asarray(arr[idx], dtype=np.float32)
            wm = np.nanmean(sub, axis=1)
            ws = np.nanstd(sub, axis=1)
            ok = np.isfinite(wm) & np.isfinite(ws) & (ws > 0)
            row[f'{pre}_n_seg'] = n
            row[f'{pre}_nan_frac'] = round(float(np.mean(np.isnan(sub))), 4)
            if ok.sum() > 0:
                row[f'{pre}_wmean_med'] = round(float(np.median(wm[ok])), 5)
                row[f'{pre}_wstd_med'] = round(float(np.median(ws[ok])), 5)
                row[f'{pre}_vmax'] = round(float(np.nanmax(sub)), 4)
                row[f'{pre}_vmin'] = round(float(np.nanmin(sub)), 4)
        except Exception as e:
            row[f'{pre}_n_seg'] = f'ERR:{e}'
    return row


def main():
    ents = sorted(
        e for e in os.listdir(ROOT)
        if os.path.isdir(os.path.join(ROOT, e))
        and os.path.exists(os.path.join(ROOT, e, CHANNELS[0] + '.npy'))
    )
    print(f'{len(ents)} entities with {CHANNELS[0]} under {ROOT}', flush=True)
    fields = ['entity', 'source_path']
    for ch in CHANNELS:
        pre = ch.lower()
        fields += [f'{pre}_{s}' for s in ('n_seg', 'wmean_med', 'wstd_med', 'vmax', 'vmin', 'nan_frac')]
    with mp.Pool(16) as pool, open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, row in enumerate(pool.imap_unordered(scan_one, ents, chunksize=16)):
            w.writerow(row)
            if (i + 1) % 500 == 0:
                print(f'{i + 1}/{len(ents)}', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
