# Does the adibin clock model depend on the recording era (vital timing class A vs B/C, cohort folder)?
import numpy as np, pandas as pd, json, collections
ROOT = "/mnt/localdata100tb/physio_data/ucsf_all"
man = {q["entity_id"]: q for q in json.load(open(f"{ROOT}/manifest.json"))}
def cls_of(ent):
    try: f = json.load(open(f"{ROOT}/{ent}/meta.json")).get("vitals_hf", {}).get("files", {})
    except Exception: return "?"
    c = {v.get("class") for k, v in f.items() if k.startswith("HR")}; return "".join(sorted(x for x in c if x)) or "?"
print("===== A. charted NBP lag vs prediction, by HR-file timing class and cohort folder")
df = pd.read_csv("/projects/mwang80/staging/fs_nbp_probe_2014_11_2015_03_2015_11_2016_03.csv"); df = df[df.margin >= 3].copy()
df["cls"] = df.entity.map(cls_of); df["folder"] = df.entity.map(lambda e: man.get(e, {}).get("wynton_folder")); df["agree"] = df.win == df.p1
print(pd.crosstab([df.cls, df.p1], df.win).to_string())
print("\nagreement by class:", df.groupby("cls").agree.agg(["sum", "count"]).to_dict("index"))
print("\nfolder (year) x class among p1!=0 rows, agree/disagree:")
sub = df[df.p1 != 0]; print(pd.crosstab([sub.folder.str[:4], sub.cls], sub.agree).to_string())
print("\ndisagreeing rows' folders:", collections.Counter(df[~df.agree].folder).most_common(10))
print("\n===== B. HR-lag probe (2015_03, 2015_11) by class")
h = pd.read_csv("/projects/mwang80/staging/fs_lag_probe_2015_03_2015_11.csv"); h = h[(h.sig == "HR") & (h.margin >= 1)].copy(); h["cls"] = h.entity.map(cls_of); h["agree"] = h.win == h.p1
print(pd.crosstab([h.cls, h.p1], h.win).to_string())
print("\n===== C. CA events (v2 times): B-marker offset by (delta, class, folder year)")
s = json.load(open("/projects/mwang80/staging/ca_t0_plots_v2/summary.json")); d = json.load(open("/projects/mwang80/staging/ca_dst_delta.json"))
rows = []
for e in s:
    pid = e["entity"].split("_")[0]; rows.append(dict(entity=e["entity"], unit=e["unit"], cls=cls_of(e["entity"]), folder=man.get(e["entity"], {}).get("wynton_folder"), delta=d[pid]["delta_min"], tB=e["tB"], tD=e["tD"], q=e["quality"]))
c = pd.DataFrame(rows); c["band"] = pd.cut(c.tB, [-999, -135, -105, -75, -45, -15, 15, 45, 75, 105, 135, 999], labels=["<-135", "-120", "-90", "-60", "-30", "0", "+30", "+60", "+90", "+120", ">135"])
print(pd.crosstab([c.cls, c.delta], c.band.astype(str)).to_string())
print("\nCA events by class:", collections.Counter(c.cls), "| folder year x class:", pd.crosstab(c.folder.str[:4], c.cls).to_dict())
print("\nclass B/C events with delta!=0 and a B marker:"); print(c[(c.cls != "A") & (c.delta != 0) & c.tB.notna()][["entity", "unit", "cls", "folder", "delta", "tB", "q"]].to_string(index=False))
