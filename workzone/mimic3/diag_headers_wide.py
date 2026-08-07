#!/usr/bin/env python
"""Wide sweep: first PLETH segment header res/gain for N patients per group."""
import os, glob, json, random
import wfdb
from collections import Counter

STORE = '/opt/localdata100tb/physio_data/mimic3'
SP = '/labs/hulab/mxwang/Physio_Data/workzone/mimic3/diag_out'

groups = {}
for name in ('low', 'high'):
    with open(f'{SP}/group_{name}.txt') as f:
        ents = [l.strip() for l in f if l.strip()]
    random.seed(42)
    groups[name] = random.sample(ents, 40)

for name, ents in groups.items():
    cnt = Counter()
    for ent in ents:
        try:
            with open(os.path.join(STORE, ent, 'meta.json')) as f:
                src = json.load(f)['source_path']
        except Exception:
            cnt['no_meta'] += 1
            continue
        found = False
        segs = sorted(g for g in glob.glob(os.path.join(src, '*_[0-9]*.hea')) if 'layout' not in g)
        for sp in segs:
            try:
                h = wfdb.rdheader(sp[:-4])
            except Exception:
                continue
            if h.sig_name and 'PLETH' in h.sig_name:
                i = h.sig_name.index('PLETH')
                cnt[f'res={h.adc_res[i]} gain={h.adc_gain[i]}'] += 1
                found = True
                break
        if not found:
            cnt['no_pleth_seg'] += 1
    print(name, dict(cnt))
