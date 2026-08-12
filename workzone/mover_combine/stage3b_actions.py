#!/usr/bin/env python3
"""
MOVER (OR) stage3b: extract 9-target driver ACTIONS to a per-patient ehr_actions.npy sidecar
(same dtype/var_ids as the MIMIC sidecar). NON-DESTRUCTIVE: ehr_events.npy untouched.

Two subsets (entity meta.source_dataset):
  mover_sis  : entity_id = PID  -> EMR/patient_medication.csv   (Drug_name, Start_time, End_time, Dose)
  mover_epic : entity_id = LOG_ID -> EPIC_EMR/EMR/patient_medications.csv (DISPLAY_NAME, MED_ACTION_TIME,
               MAR_ACTION_NM=='Given', ADMIN_SIG=dose)
Times are US/Pacific wall-clock -> UTC epoch ms (entity time_ms is UTC epoch). Drug -> var_id by name.

Value semantics: per-drug var (207-220, 201/202) = Dose (native); SIS infusions (End_time present) also
get value=0 at End; boluses are point events. Vaso aggregate var 200 = presence (value=NaN) at each vaso
administration (OR is bolus-based -> no continuous NE-eq; 200 just marks "vasopressor active", uniform
with the CV-presence convention). CV/EPIC doses that don't parse -> value=NaN.

Run:  python workzone/mover_combine/stage3b_actions.py --patients-file <pids> [--out-root <scratch>]
"""
import os, json, argparse, datetime
from zoneinfo import ZoneInfo
import numpy as np

RAW = "/opt/localdata100tb/UNIPHY_Plus/raw_datasets/MOVER"
SIS_MED = RAW + "/EMR/patient_medication.csv"
EPIC_MED = RAW + "/EPIC_EMR/EMR/patient_medications.csv"
PROCESSED = "/opt/localdata100tb/physio_data/mover_combine"
TZ = ZoneInfo("America/Los_Angeles")
DTYPE = np.dtype([("time_ms", "int64"), ("seg_idx", "int32"), ("var_id", "uint16"), ("value", "float32")])


def drug_to_var(name):
    """Map a drug name (SIS Drug_name or EPIC DISPLAY_NAME) to a var_id, or None."""
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
    if "hypertonic" in n or ("sodium chloride 3" in n) or ("nacl 3" in n): return 220
    if (not piggyback and any(f in n for f in ["plasmalyte", "plasma-lyte", "lactated ringer", "lr iv",
            "sodium chloride 0.9", "normal saline", "ns iv", "albumin", "normosol"])):        return 201
    return None


VASO_VARS = set(range(207, 214))


def _fnum(s):
    try:
        return float(s)
    except Exception:
        return float("nan")


def load_sis():
    """PID -> list of (var_id, start_ms, dose, end_ms_or_None)."""
    import csv
    out = {}
    with open(SIS_MED) as f:
        r = csv.reader(f); h = next(r)
        PI, ST, ET, DO, DN = (h.index(c) for c in ["PID", "Start_time", "End_time", "Dose", "Drug_name"])
        for row in r:
            if len(row) <= DN:
                continue
            vid = drug_to_var(row[DN])
            if vid is None:
                continue
            try:
                st = datetime.datetime.strptime(row[ST].strip(), "%m/%d/%y %H:%M").replace(tzinfo=TZ).timestamp() * 1000
            except Exception:
                continue
            en = None
            if row[ET].strip() not in ("", "\\N"):
                try:
                    en = datetime.datetime.strptime(row[ET].strip(), "%m/%d/%y %H:%M").replace(tzinfo=TZ).timestamp() * 1000
                except Exception:
                    en = None
            out.setdefault(row[PI], []).append((vid, st, _fnum(row[DO]), en))
    return out


def load_epic():
    """LOG_ID -> list of (var_id, t_ms, dose, None). Only MAR_ACTION_NM=='Given'."""
    import csv
    out = {}
    with open(EPIC_MED) as f:
        r = csv.reader(f); h = next(r)
        LI, DI, TI, MI, SI = (h.index(c) for c in ["LOG_ID", "DISPLAY_NAME", "MED_ACTION_TIME", "MAR_ACTION_NM", "ADMIN_SIG"])
        for row in r:
            if len(row) <= max(LI, DI, TI, MI, SI):
                continue
            if row[MI].strip() != "Given" or not row[TI].strip():
                continue
            vid = drug_to_var(row[DI])
            if vid is None:
                continue
            try:
                t = datetime.datetime.strptime(row[TI].strip(), "%Y-%m-%d %H:%M:%S").replace(tzinfo=TZ).timestamp() * 1000
            except Exception:
                continue
            out.setdefault(row[LI], []).append((vid, t, _fnum(row[SI]), None))
    return out


def build_events(meds, time_ms):
    """meds: list of (var_id, start_ms, dose, end_ms). -> list of (t_ms, seg, var, value)."""
    n_seg = len(time_ms); ev = []

    def add(t_ms, vid, val):
        s = int(np.searchsorted(time_ms, int(t_ms), side="right") - 1)
        if 0 <= s < n_seg:
            ev.append((int(t_ms), s, int(vid), float(val)))

    for vid, st, dose, en in meds:
        add(st, vid, dose)                       # per-drug dose at start
        if en is not None:
            add(en, vid, 0.0)                    # infusion stop
        if vid in VASO_VARS:
            add(st, 200, float("nan"))           # vaso aggregate = presence marker
            if en is not None:
                add(en, 200, float("nan"))
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", nargs="*", default=None)
    ap.add_argument("--patients-file", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-root", default=None)
    a = ap.parse_args()
    out_root = a.out_root or PROCESSED

    print("loading SIS meds..."); sis = load_sis(); print("  SIS pids w/ actions:", len(sis))
    print("loading EPIC meds..."); epic = load_epic(); print("  EPIC log_ids w/ actions:", len(epic))

    if a.patients_file:
        dirs = [l.strip() for l in open(a.patients_file) if l.strip()]
    elif a.patients:
        dirs = a.patients
    else:
        dirs = sorted(d for d in os.listdir(PROCESSED)
                      if os.path.isdir(os.path.join(PROCESSED, d)) and not d.startswith("."))
        if a.limit:
            dirs = dirs[:a.limit]
    print("patient dirs:", len(dirs))

    n_ok = tot = n_del = 0
    for dn in dirs:
        cdir = os.path.join(PROCESSED, dn)
        od = os.path.join(out_root, dn)
        sidecar = os.path.join(od, "ehr_actions.npy")
        mp = os.path.join(cdir, "meta.json"); tp = os.path.join(cdir, "time_ms.npy")
        ev = []
        if os.path.exists(mp) and os.path.exists(tp):
            src = json.load(open(mp)).get("source_dataset", "")
            meds = sis.get(dn) if "sis" in src else epic.get(dn)
            if meds:
                ev = build_events(meds, np.load(tp).astype("int64"))
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
