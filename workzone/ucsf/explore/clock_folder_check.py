import numpy as np, pandas as pd, json, collections
ROOT = "/mnt/localdata100tb/physio_data/ucsf_all"
man = {q["entity_id"]: q for q in json.load(open(f"{ROOT}/manifest.json"))}
fold = lambda e: man.get(e, {}).get("wynton_folder")
print("entities per cohort folder (2015-10 .. 2016-06):", {k: v for k, v in sorted(collections.Counter(q["wynton_folder"] for q in man.values()).items()) if "2015-1" in k or "2016-0" in k})
print("\n== NBP probe: p1 x win for folders 2015-11 .. 2016-04")
df = pd.read_csv("/projects/mwang80/staging/fs_nbp_probe_2014_11_2015_03_2015_11_2016_03.csv"); df = df[df.margin >= 3].copy(); df["folder"] = df.entity.map(fold)
for f in ["2015-11-deid", "2015-12-deid", "2016-01-deid", "2016-02-deid", "2016-03-deid", "2016-04-deid"]:
    s = df[df.folder == f]
    if len(s): print(f, dict(collections.Counter(zip(s.p1, s.win))))
print("\n== HR probe: p1 x win for the same folders")
h = pd.read_csv("/projects/mwang80/staging/fs_lag_probe_2015_03_2015_11.csv"); h = h[(h.sig == "HR") & (h.margin >= 1)].copy(); h["folder"] = h.entity.map(fold)
for f in ["2015-11-deid", "2015-12-deid", "2016-01-deid", "2016-02-deid", "2016-03-deid", "2016-04-deid"]:
    s = h[h.folder == f]
    if len(s): print(f, dict(collections.Counter(zip(s.p1, s.win))))
print("\n== CA events (v2) in folders 2015-11 .. 2016-04")
s = json.load(open("/projects/mwang80/staging/ca_t0_plots_v2/summary.json")); d = json.load(open("/projects/mwang80/staging/ca_dst_delta.json"))
for e in s:
    f = fold(e["entity"])
    if f in ("2015-11-deid", "2015-12-deid", "2016-01-deid", "2016-02-deid", "2016-03-deid", "2016-04-deid"):
        print(f, e["entity"], e["unit"], "delta", d[e["entity"].split("_")[0]]["delta_min"], "tB", None if e["tB"] is None else round(e["tB"], 1), "tD", None if e["tD"] is None else round(e["tD"], 1), e["quality"])
print("\n== all CA events with |tB| in [100,140] (possible 2-h cases):")
for e in s:
    if e["tB"] is not None and 100 <= abs(e["tB"]) <= 140: print(fold(e["entity"]), e["entity"], e["unit"], "delta", d[e["entity"].split("_")[0]]["delta_min"], "tB", round(e["tB"], 1), e["quality"])
