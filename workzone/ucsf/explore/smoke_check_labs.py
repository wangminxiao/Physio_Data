"""Smoke check for Stage D/E on ucsf_all: labs/partitions per entity, subset relation to the CA-cohort store.
   python workzone/ucsf/explore/smoke_check_labs.py [--pick N --year YYYY] | --entities a,b,c"""
import argparse, json, os, sys
import numpy as np, polars as pl
from datetime import datetime
STORE = "/mnt/localdata100tb/physio_data/ucsf_all"; CA = "/mnt/localdata100tb/physio_data/ucsf"
ap = argparse.ArgumentParser(); ap.add_argument("--entities", default=""); ap.add_argument("--pick", type=int, default=0); ap.add_argument("--year", type=int, default=2015)
a = ap.parse_args()
if a.pick:
    raw = pl.read_parquet("workzone/outputs/ucsf_all/stage_d_labs_raw.parquet"); pids = set(raw["patient_id"].to_list())
    l = pl.read_parquet("workzone/outputs/ucsf_all/ehr_link.parquet").filter(pl.col("encounter_id").is_not_null())
    rows = [r for r in l.select(["entity_id", "patient_id", "encounter_start_ehr_ms"]).iter_rows()
            if int(str(r[1]).strip()) in pids and datetime.utcfromtimestamp(r[2] / 1000).year == a.year and os.path.exists(f"{STORE}/{r[0]}/meta.json")]
    ca = [e for e, *_ in rows if os.path.exists(f"{CA}/{e}/labs_events.npy")]; other = [e for e, *_ in rows if e not in ca]
    print(",".join(ca[: a.pick] + other[: max(1, a.pick // 2)])); sys.exit(0)
for e in a.entities.split(","):
    d = f"{STORE}/{e}"; m = json.load(open(f"{d}/meta.json")); tm = np.load(f"{d}/time_ms.npy")
    labs = np.load(f"{d}/labs_events.npy"); parts = {k: np.load(f"{d}/ehr_{k}.npy") for k in ("baseline", "recent", "events", "future")}
    line = (f"{e} labs {labs.size} | " + " ".join(f"{k}={v.size}" for k, v in parts.items()) +
            f" | wave {len(tm) * 30 / 3600:.1f} h, admission {(m['admission_end_ms'] - m['admission_start_ms']) / 3.6e6:.0f} h, wave starts {(int(tm[0]) - m['admission_start_ms']) / 3.6e6:.1f} h after admission")
    p = f"{CA}/{e}/labs_events.npy"
    if os.path.exists(p):
        ca = np.load(p)
        A = set(zip(labs["time_ms"].tolist(), labs["var_id"].tolist(), np.round(labs["value"].astype(float), 3).tolist()))
        B = set(zip(ca["time_ms"].tolist(), ca["var_id"].tolist(), np.round(ca["value"].astype(float), 3).tolist()))
        line += f" | CA-store labs {ca.size}, subset={A <= B}, shared={len(A & B)}"
    ev = parts["events"]
    if ev.size:
        idx = np.searchsorted(tm, ev["time_ms"], side="right") - 1; line += f" | seg_idx ok={bool(np.array_equal(idx, ev['seg_idx']))}"
        line += f" | in-wave vars {sorted(set(ev['var_id'].tolist()))[:8]}"
    print(line)
