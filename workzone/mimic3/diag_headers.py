#!/usr/bin/env python
"""Dump PLETH/II signal-spec fields from source WFDB segment headers for sample patients."""
import os, sys, glob
import wfdb

samples = {
    'low':  ['p01/p012631','p01/p012632','p01/p012788','p01/p012795','p01/p012806'],
    'high': ['p01/p012712','p01/p012733','p01/p012797','p01/p012869','p01/p013123'],
}
ROOT = '/labs/hulab/MIMIC_waveform_matched_subset/physionet.org/files/mimic3wdb-matched/1.0'

for grp, pats in samples.items():
    for pat in pats:
        pdir = os.path.join(ROOT, pat)
        # numbered segment headers, skip layout and numerics
        segs = sorted(g for g in glob.glob(os.path.join(pdir, '*_[0-9]*.hea'))
                      if 'layout' not in g)
        shown = 0
        for sp in segs:
            try:
                h = wfdb.rdheader(sp[:-4])
            except Exception:
                continue
            if not h.sig_name or 'PLETH' not in h.sig_name:
                continue
            i = h.sig_name.index('PLETH')
            fields = (f"PLETH gain={h.adc_gain[i]} base={h.baseline[i]} zero={h.adc_zero[i]} "
                      f"res={h.adc_res[i]} fmt={h.fmt[i]} units={h.units[i]}")
            if 'II' in h.sig_name:
                j = h.sig_name.index('II')
                fields += (f" | II gain={h.adc_gain[j]} base={h.baseline[j]} units={h.units[j]}")
            print(f"{grp:4s} {pat} {os.path.basename(sp):24s} {fields}")
            shown += 1
            if shown >= 3:
                break
        if shown == 0:
            print(f"{grp:4s} {pat} NO_PLETH_SEGMENT_FOUND")
