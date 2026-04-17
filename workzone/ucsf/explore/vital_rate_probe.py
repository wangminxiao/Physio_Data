"""Probe sample rate per .vital suffix across many patients.

Old code assumed BP 0.5 Hz. Earlier single probe showed AR at 0.5 Hz. Verify for
all suffixes and see whether the rate varies by suffix or by patient.
"""
from __future__ import annotations

import os
import re
import sys
from collections import defaultdict
from statistics import median

sys.path.insert(0, "/labs/hulab/mxwang/data/ucsf_EHR/bedanalysis_waveformExtraction")
from vitalfilepy.vitalfile import VitalFile  # type: ignore

UCSF = "/labs/hulab/UCSF"

# candidates from demo probe
CANDIDATES = [
    ("2017-03-deid", "214688354794344"),
    ("2017-03-deid", "747223141294381"),
    ("2017-10-deid", "969040437292716"),
    ("2016-10-deid", "299830937914959"),
    ("2017-08-deid", "56577667285165"),
    ("2013-03-deid", "174730580456963"),   # older cohort with non-zero headers
]

# {suffix: [median_dt, ...]}  from different files
per_suffix_dts: dict[str, list[float]] = defaultdict(list)
per_suffix_examples: dict[str, dict] = {}
total_files = 0

for wyn, pid in CANDIDATES:
    pat_dir = f"{UCSF}/{wyn}/DE{pid}"
    if not os.path.isdir(pat_dir):
        continue
    for bed in sorted(os.listdir(pat_dir)):
        bed_p = os.path.join(pat_dir, bed)
        if not os.path.isdir(bed_p):
            continue
        for f in sorted(os.listdir(bed_p)):
            if not f.endswith(".vital"):
                continue
            m = re.match(r"DE\d+_\d{14}_\d{5}_(.+)\.vital$", f)
            if not m:
                continue
            sfx = m.group(1)
            path = os.path.join(bed_p, f)
            try:
                with VitalFile(path, "r") as fh:
                    fh.readHeader()
                    n = int(fh.numSamplesInFile)
                    if n < 10:
                        continue
                    # read up to 2000 samples to estimate cadence
                    raw = fh.readVitalDataBuf(min(2000, n))
                    offs = [float(t[1]) for t in raw]
                    if len(offs) < 3:
                        continue
                    diffs = [offs[i + 1] - offs[i] for i in range(len(offs) - 1)]
                    diffs = [d for d in diffs if d > 0]
                    if not diffs:
                        continue
                    med = median(diffs)
                    per_suffix_dts[sfx].append(med)
                    if sfx not in per_suffix_examples:
                        per_suffix_examples[sfx] = {
                            "path": path,
                            "n_samples": n,
                            "median_dt_sec": med,
                            "val_range": (min(float(t[0]) for t in raw if t[0] > -999999),
                                           max(float(t[0]) for t in raw if t[0] > -999999)),
                        }
                    total_files += 1
            except Exception as e:
                pass

# summarise
print(f"{'suffix':<10}  {'n_files':>7}  {'median_dt(sec)':>15}  {'min_dt':>8}  {'max_dt':>8}  {'example path':<90}")
for sfx in sorted(per_suffix_dts):
    dts = per_suffix_dts[sfx]
    ex = per_suffix_examples[sfx]
    print(f"{sfx:<10}  {len(dts):>7d}  {median(dts):>15.3f}  {min(dts):>8.3f}  {max(dts):>8.3f}  "
          f"{ex['path'][-85:]:<90}")

print(f"\ntotal files probed: {total_files}")
