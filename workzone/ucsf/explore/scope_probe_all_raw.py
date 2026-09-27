"""Read-only scope probe for UCSF raw: wave-cycle count + per-patient file-type coverage (aggregates only)."""
import os, csv, time, random
from collections import Counter
ROOT = "/mnt/localdata/storage/UCSF"
t0 = time.time()
folders = sorted(d for d in os.listdir(ROOT) if d.endswith("-deid"))
n_de = n_map = rows = 0
uids, pids, all_de = set(), set(), []
for f in folders:
    fp = os.path.join(ROOT, f)
    for e in os.scandir(fp):
        if not (e.is_dir() and e.name.startswith("DE")):
            continue
        n_de += 1; pids.add(e.name); all_de.append(e.path)
        m = os.path.join(e.path, "MRN-Mapping.csv")
        if not os.path.isfile(m):
            continue
        n_map += 1
        with open(m, newline="", encoding="latin-1") as fh:
            r = csv.reader(fh); hdr = next(r, None)
            if not hdr or "WaveCycleUID" not in hdr:
                continue
            i = hdr.index("WaveCycleUID")
            for row in r:
                if len(row) > i and row[i]:
                    rows += 1; uids.add((e.name, row[i]))
print(f"folders={len(folders)} DE_dirs={n_de} unique_pid={len(pids)} with_MRN-Mapping={n_map} "
      f"mapping_rows={rows} unique_(pid,WaveCycleUID)={len(uids)}  t={time.time()-t0:.0f}s", flush=True)

random.seed(0)
sample = random.sample(all_de, min(400, len(all_de)))
c = Counter(); n_sub = 0
for d in sample:
    has = set()
    for sub in os.scandir(d):
        if not sub.is_dir():
            continue
        n_sub += 1
        for fn in os.listdir(sub.path):
            if fn.endswith(".adibin"):
                has.add("adibin")
            elif fn.endswith(".vital"):
                has.add("v:" + fn.rsplit("_", 1)[-1][:-6])
    for h in has:
        c[h] += 1
    core = all(k in has for k in ("adibin", "v:HR", "v:SPO2-%", "v:RESP"))
    bpA = any(k in has for k in ("v:AR1-S", "v:AR2-S", "v:AR3-S")); bpN = "v:NBP-S" in has
    c["_core(adibin+HR+SPO2+RESP)"] += core
    c["_core+ABP(AR*-S)"] += core and bpA
    c["_core+NBP-S"] += core and bpN
    c["_core+(ABP or NBP)"] += core and (bpA or bpN)
print(f"sample_DE={len(sample)} bed_subdirs={n_sub}")
for k, v in sorted(c.items(), key=lambda kv: -kv[1])[:45]:
    print(f"  {k:32s} {v:5d}  {100*v/len(sample):5.1f}%")
print(f"t={time.time()-t0:.0f}s")
