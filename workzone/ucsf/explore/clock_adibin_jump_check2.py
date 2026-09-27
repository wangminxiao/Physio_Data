import polars as pl, numpy as np, json, os, struct, collections
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
LA = ZoneInfo("America/Los_Angeles"); OUT = "/mnt/localdata/storage/mxwang/Physio_Data/workzone/outputs/ucsf_all"; ROOT = "/mnt/localdata100tb/physio_data/ucsf_all"; RAW = "/mnt/localdata/storage/UCSF"
HDR = struct.Struct("<4sidiiiiiddiiii")
def hdr(path):
    with open(path, "rb") as f: b = f.read(HDR.size)
    magic, ver, spt, y, mo, d, h, mi, sec, trig, nch, nsamp, tch, fmt = HDR.unpack(b)
    t = datetime(y, mo, d, h, mi) + timedelta(seconds=float(sec)); return int((t - datetime(1970, 1, 1)).total_seconds() * 1000), int(nsamp * spt * 1000)
df = pl.read_parquet(f"{OUT}/valid_wave_window.parquet"); man = json.load(open(f"{ROOT}/manifest.json")); ms2dt = lambda ms: datetime(1970, 1, 1) + timedelta(milliseconds=int(ms))
def ds(t): return int(t.replace(tzinfo=LA).dst().total_seconds() != 0)
strad = {q["entity_id"] for q in man if ds(ms2dt(q["wave_start_ms"])) != ds(ms2dt(q["wave_end_ms"]))}
sub = df.filter(pl.col("entity_id").is_in(list(strad)))
res = collections.Counter(); ex = collections.defaultdict(list); nread = 0
for row in sub.iter_rows(named=True):
    recs = []
    for rel in row["adibin_files"]:
        try: s, d = hdr(os.path.join(RAW, rel)); recs.append((s, d)); nread += 1
        except Exception as e: pass
    recs.sort()
    if len(recs) < 2: res["<2 files"] += 1; continue
    t = ms2dt(recs[0][0]).replace(minute=0, second=0, microsecond=0); end = ms2dt(recs[-1][0] + recs[-1][1]); sw = None
    while t < end:
        if ds(t) != ds(t + timedelta(hours=1)): sw = t + timedelta(hours=1); break
        t += timedelta(hours=1)
    if sw is None: res["no switch inside files"] += 1; continue
    swms = int((sw - datetime(1970, 1, 1)).total_seconds() * 1000); spring = ds(sw) == 1; gaps = []
    for (s0, d0), (s1, d1) in zip(recs[:-1], recs[1:]):
        e0 = s0 + d0
        if abs(e0 - swms) <= 3 * 3600000 or abs(s1 - swms) <= 3 * 3600000: gaps.append(round((s1 - e0) / 60000, 1))
    if not gaps: res["single file spans switch"] += 1; ex["single file spans switch"].append(row["entity_id"]); continue
    if any(45 <= g <= 75 for g in gaps): kind = "gap +60 at switch (spring)" if spring else "gap +60 at switch (FALL)"
    elif any(-75 <= g <= -45 for g in gaps): kind = "overlap -60 at switch (fall)" if not spring else "overlap -60 at switch (SPRING)"
    elif any(abs(g) >= 45 for g in gaps): kind = "other big gap near switch"
    else: kind = "continuous across switch (%s)" % ("spring" if spring else "fall")
    res[kind] += 1
    if len(ex[kind]) < 3: ex[kind].append((row["entity_id"], gaps[:8]))
print("headers read:", nread); print("straddling entities by adibin boundary behaviour at the GE-calendar switch:", dict(res))
for k, v in ex.items(): print("  e.g.", k, v)
