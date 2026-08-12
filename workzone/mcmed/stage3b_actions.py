#!/usr/bin/env python3
"""
MC-MED (ED) stage3b: extract 9-target driver ACTIONS to a per-patient ehr_actions.npy sidecar
(same dtype/var_ids as MIMIC/MOVER). NON-DESTRUCTIVE: ehr_events.npy untouched.

Source: orders.csv, Order_type startswith 'Medication', with First_admin_time (ISO-UTC, e.g.
'2253-07-16T00:48:57Z'; MC-MED shifts dates per-patient, internally consistent with time_ms).
entity_id == CSN. Drug via Procedure_name -> var_id (shared matcher).

ORDER-LEVEL only (no dose/rate, no stop) -> value = NaN (presence); vaso aggregate 200 = NaN
presence at each vaso administration. This is the coarsest of the 3 datasets (ED is pre-ICU).

Run:  python workzone/mcmed/stage3b_actions.py --patients-file <pids> [--out-root <scratch>]
"""
import os, json, argparse, datetime
import numpy as np

ORDERS = "/opt/localdata100tb/UNIPHY_Plus/raw_datasets/physionet.org/files/mc-med/1.0.1/data/orders.csv"
PROCESSED = "/opt/localdata100tb/physio_data/mcmed"
DTYPE = np.dtype([("time_ms", "int64"), ("seg_idx", "int32"), ("var_id", "uint16"), ("value", "float32")])
VASO_VARS = set(range(207, 214))


def drug_to_var(name):
    """Source drug name -> action var_id, or None. Excludes non-systemic routes
    (ophthalmic/nasal/nebulized/topical/irrigation/flush), decongestant tablets, and
    drug-in-vehicle piggyback preps (the drug, not a fluid/dextrose action)."""
    n = name.lower()
    if any(x in n for x in ["opht", "nasal", "naris", "nebu", "inhal", " inh ", "topical",
                            " tp ", "irrig", "swab", "flush", " drop", "ointment"]):
        return None
    if "pseudoephedrine" in n or "pseudoephed" in n:        return None   # oral decongestant
    if "lidocaine" in n or "bupivacaine" in n:              return None   # local anesthetics (+/- epi)
    piggyback = any(x in n for x in ["ivpb", "in sodium chloride", "in 0.9", "in 5 % dextrose",
                                     "in 5% dextrose", "in dextrose", "premix", "compounded"])
    if "norepinephrine" in n or "levophed" in n:            return 207
    if "epinephrine" in n or "adrenaline" in n:             return 208
    if "phenylephrine" in n or "synephrine" in n:           return 209
    if "dopamine" in n:                                     return 210
    if "vasopressin" in n:                                  return 211
    if "dobutamine" in n:                                   return 212
    if "ephedrine" in n:                                    return 213
    if "packed red" in n or "prbc" in n:                    return 214
    if "insulin" in n:                                      return 215
    if (any(x in n for x in ["dextrose 50", "dextrose 20", "dextrose 10", "d50", "d10w", "d20"])
            and not piggyback):                             return 216
    if "potassium chloride" in n or "kcl" in n:             return 217
    if "calcium" in n and ("chlor" in n or "gluc" in n):    return 218
    if "bicarb" in n:                                       return 219
    if "hypertonic" in n or "sodium chloride 3" in n or "nacl 3" in n: return 220
    if (not piggyback and any(f in n for f in ["plasmalyte", "plasma-lyte", "lactated ringer", "lr iv",
            "sodium chloride 0.9", "normal saline", "ns iv", "albumin", "normosol"])):        return 201
    return None


def load_orders():
    """CSN -> list of (var_id, admin_ms). Medications with First_admin_time."""
    import csv
    out = {}
    with open(ORDERS) as f:
        r = csv.reader(f); h = next(r)
        CI, OI, FI, PI = (h.index(c) for c in ["CSN", "Order_type", "First_admin_time", "Procedure_name"])
        for row in r:
            if len(row) <= max(CI, OI, FI, PI):
                continue
            if not row[OI].strip().lower().startswith("medication") or not row[FI].strip():
                continue
            vid = drug_to_var(row[PI])
            if vid is None:
                continue
            try:
                t = datetime.datetime.strptime(row[FI].strip(), "%Y-%m-%dT%H:%M:%SZ").replace(
                    tzinfo=datetime.timezone.utc).timestamp() * 1000
            except Exception:
                continue
            out.setdefault(row[CI].strip(), []).append((vid, t))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", nargs="*", default=None)
    ap.add_argument("--patients-file", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-root", default=None)
    a = ap.parse_args()
    out_root = a.out_root or PROCESSED

    print("loading MC-MED medication orders..."); orders = load_orders()
    print("  CSNs w/ driver-action orders:", len(orders))

    if a.patients_file:
        dirs = [l.strip() for l in open(a.patients_file) if l.strip()]
    elif a.patients:
        dirs = a.patients
    else:
        dirs = sorted(d for d in os.listdir(PROCESSED)
                      if os.path.isdir(os.path.join(PROCESSED, d)) and not d.startswith(".") and not d.startswith("tasks"))
        if a.limit:
            dirs = dirs[:a.limit]
    print("patient dirs:", len(dirs))

    n_ok = tot = n_del = 0
    for dn in dirs:
        cdir = os.path.join(PROCESSED, dn)
        od = os.path.join(out_root, dn)
        sidecar = os.path.join(od, "ehr_actions.npy")
        tp = os.path.join(cdir, "time_ms.npy")
        meds = orders.get(dn)
        ev = []
        if meds and os.path.exists(tp):
            time_ms = np.load(tp).astype("int64"); n_seg = len(time_ms)
            for vid, t in meds:
                s = int(np.searchsorted(time_ms, int(t), side="right") - 1)
                if 0 <= s < n_seg:
                    ev.append((int(t), s, int(vid), float("nan")))       # presence
                    if vid in VASO_VARS:
                        ev.append((int(t), s, 200, float("nan")))        # vaso presence aggregate
        if not ev:
            # idempotent: no valid actions for this patient -> drop any stale sidecar from an older run
            if os.path.exists(sidecar):
                os.remove(sidecar); n_del += 1
            continue
        arr = np.array(ev, dtype=DTYPE); arr.sort(order="time_ms")
        os.makedirs(od, exist_ok=True)
        np.save(sidecar, arr)
        n_ok += 1; tot += len(ev)
    print(f"done: {n_ok} patients, {tot:,} action events, {n_del} stale sidecars removed -> {out_root}")


if __name__ == "__main__":
    main()
