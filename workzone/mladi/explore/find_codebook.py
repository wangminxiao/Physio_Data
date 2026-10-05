#!/usr/bin/env python3
"""MLADI: where do the integer codes in /ehr come from? Prints every attribute on the root, /ehr and
each /ehr table (and on a few data/numerics sets) of two files, and searches the shared project tree
for dictionary-like files. Read-only. Attribute VALUES are printed truncated: a codebook is a list of
clinical terms, not an identifier."""
import glob, json, os, sys
import h5py

RAW = "/ocean/projects/med250003p/shared/mladi_extract_2023_waves"
files = sorted(glob.glob(os.path.join(RAW, "*.h5")))
picks = [files[len(files) // 3], files[2 * len(files) // 3]]


def show(obj, name):
    for k, v in obj.attrs.items():
        v = v.decode(errors="replace") if isinstance(v, bytes) else v
        s = repr(v)
        print(f"  attr {name}@{k}: len {len(s)} :: {s[:1500]}")


for p in picks:
    print("=" * 30, os.path.basename(p)[:8], flush=True)
    with h5py.File(p, "r") as f:
        show(f, "/")
        for g in ("ehr", "data", "data/numerics", "dwc"):
            if g in f:
                show(f[g], "/" + g)
        if "ehr" in f:
            for k in f["ehr"].keys():
                show(f["ehr"][k], f"/ehr/{k}")
                d = f["ehr"][k]
                if hasattr(d, "dtype") and d.dtype.names:
                    print(f"  /ehr/{k} dtype: {d.dtype}")
        for k in list(f.get("data/numerics", {}).keys())[:2]:
            show(f["data/numerics"][k], f"/data/numerics/{k}")
print("=" * 30, "non-h5 files next to the raw files:", flush=True)
for q in sorted(os.listdir(RAW)):
    if not q.endswith(".h5"):
        print("  ", q)
print("=" * 30, "dictionary-like files under the shared tree (depth 4):", flush=True)
root = "/ocean/projects/med250003p/shared"
for dp, dn, fn in os.walk(root):
    if dp.count(os.sep) - root.count(os.sep) >= 4:
        dn[:] = []
    dn[:] = [d for d in dn if d not in ("pretrain_wav_v2", "data_cache", "mladi_extract_2023_waves")]
    for x in fn:
        lx = x.lower()
        if any(w in lx for w in ("vocab", "dict", "codebook", "mapping", "lookup", "categor", "schema", "readme", "legend", "enum")):
            print("  ", os.path.join(dp, x))
