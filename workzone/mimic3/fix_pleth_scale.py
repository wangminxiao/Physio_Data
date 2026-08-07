#!/usr/bin/env python
"""Fix MIMIC-III PLETH40 4x scale: rescale 12-bit-source records to NU [0,1].

Entities in diag_out/group_high.txt come from 12-bit source segments whose .hea
kept gain=1023 (not rescaled to 4095), so stored physical PLETH spans [0,4.003].
Multiply PLETH40.npy by 1023/4095 to unify the store at NU [0,1].

Safety:
- Idempotent: entities whose meta.json carries `pleth_scale_fix` are skipped;
  additionally an entity is only scaled when its observed nanmax > 2.
- Atomic: writes tmp file in the entity dir, then os.replace.
- Invertible: exact factor recorded in meta.json (undo = divide by it).
- II120 and all other files untouched.

Usage: python fix_pleth_scale.py [--limit N] [--workers 16] [--dry-run]
"""
import argparse
import json
import os
import warnings
import multiprocessing as mp

import numpy as np

ROOT = '/opt/localdata100tb/physio_data/mimic3'
GROUP_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'diag_out', 'group_high.txt')
FACTOR = 1023.0 / 4095.0  # = 1 / 4.00293
warnings.filterwarnings('ignore', category=RuntimeWarning)

DRY_RUN = False


def fix_one(ent):
    d = os.path.join(ROOT, ent)
    meta_path = os.path.join(d, 'meta.json')
    npy_path = os.path.join(d, 'PLETH40.npy')
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get('pleth_scale_fix'):
            return ent, 'ALREADY_FIXED', None
        arr = np.load(npy_path)
        vmax = float(np.nanmax(arr))
        if not vmax > 2.0:
            return ent, 'ANOMALY_VMAX_LE_2', vmax
        if DRY_RUN:
            return ent, 'WOULD_FIX', vmax
        fixed = np.ascontiguousarray(
            (arr.astype(np.float32) * FACTOR).astype(np.float16))
        assert fixed.shape == arr.shape and fixed.dtype == np.float16
        assert fixed.flags['C_CONTIGUOUS']
        tmp = os.path.join(d, '.PLETH40.tmp.npy')  # .npy suffix: np.save must not append
        np.save(tmp, fixed)
        os.replace(tmp, npy_path)
        meta['pleth_scale_fix'] = {
            'factor': FACTOR,
            'reason': 'source 12-bit segments with gain=1023: physical [0,4.003] -> NU [0,1]',
            'date': '2026-08-07',
        }
        tmp_meta = meta_path + '.tmp'
        with open(tmp_meta, 'w') as f:
            json.dump(meta, f, indent=2)
        os.replace(tmp_meta, meta_path)
        return ent, 'FIXED', vmax
    except Exception as e:
        return ent, f'ERR:{e}', None


def main():
    global DRY_RUN
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    DRY_RUN = args.dry_run

    with open(GROUP_FILE) as f:
        ents = [l.strip() for l in f if l.strip()]
    if args.limit:
        ents = ents[:args.limit]
    print(f'{len(ents)} entities to fix (dry_run={DRY_RUN})', flush=True)

    counts = {}
    with mp.Pool(args.workers, initializer=_init, initargs=(DRY_RUN,)) as pool:
        for i, (ent, status, vmax) in enumerate(
                pool.imap_unordered(fix_one, ents, chunksize=8)):
            counts[status.split(':')[0]] = counts.get(status.split(':')[0], 0) + 1
            if status.startswith(('ERR', 'ANOMALY')):
                print(f'  {ent}: {status} vmax={vmax}', flush=True)
            if (i + 1) % 250 == 0:
                print(f'{i + 1}/{len(ents)} {counts}', flush=True)
    print(f'DONE {counts}', flush=True)


def _init(dry):
    global DRY_RUN
    DRY_RUN = dry


if __name__ == '__main__':
    main()
