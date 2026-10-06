#!/usr/bin/env python3
"""Value census for selected /ehr item names: rows, numeric-parsable share, top raw values, units, per name.
Reads up to --files H5 files that have /ehr/<table>.

    python workzone/mladi/explore/name_values.py --table low_rate --name-col eventName --names "PEEP|Vent|FiO2"
"""
import argparse, collections, json, os, re, sys
import h5py
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import cfg, factor, num  # noqa: E402

C = cfg()
ap = argparse.ArgumentParser()
ap.add_argument("--table", default="low_rate"); ap.add_argument("--name-col", default="eventName")
ap.add_argument("--val-cols", default="resultVal,resultUnit"); ap.add_argument("--names", required=True)
ap.add_argument("--files", type=int, default=400); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
pat = re.compile(a.names, re.I); vc = re.split(r"[,:]", a.val_cols)
cnt = collections.Counter(); par = collections.Counter(); top = collections.defaultdict(collections.Counter)
unit = collections.defaultdict(collections.Counter); nf = 0
import random
FN = sorted(os.listdir(C["raw_h5_dir"])); random.Random(a.seed).shuffle(FN)
extra = collections.defaultdict(collections.Counter)
for fn in FN:
    if not fn.endswith(".h5") or nf >= a.files:
        continue
    try:
        with h5py.File(os.path.join(C["raw_h5_dir"], fn), "r") as f:
            if f"ehr/{a.table}" not in f or f[f"ehr/{a.table}"].shape[0] == 0:
                continue
            d = factor(f[f"ehr/{a.table}"]); nf += 1
            for i, nm in enumerate(d[a.name_col]):
                if nm is None or not pat.search(str(nm)):
                    continue
                cnt[nm] += 1; v = d[vc[0]][i]
                par[nm] += int(np.isfinite(num(v))); top[nm][str(v)[:30]] += 1
                if len(vc) > 1 and vc[1] in d:
                    unit[nm][str(d[vc[1]][i])] += 1
                for c in vc[2:]:
                    if c in d:
                        extra[nm][f"{c}={str(d[c][i])[:25]}"] += 1
    except Exception as ex:
        print("skip", fn[:12], ex)
print(f"files read {nf}")
for nm, n in cnt.most_common():
    print(json.dumps({"name": nm, "rows": n, "numeric": round(par[nm] / n, 3), "top": top[nm].most_common(8),
                      "units": unit[nm].most_common(3), "other": extra[nm].most_common(6)}), flush=True)
