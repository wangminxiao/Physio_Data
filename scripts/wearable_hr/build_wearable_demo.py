#!/usr/bin/env python3
"""Per-subject demographics_unified.csv for the wearable datasets (mimic3 schema).

Needed so a mimic3-trained HR model (demo baked into the core SSM layers) loads
cleanly for zero-shot eval: the eval model must build the SAME demo encoder. Columns
match mimic3 demo_cols exactly: cats=[sex], conts=[age,height_cm,weight_kg,temp,hr,
rr,spo2,sbp,dbp]. Real age/sex/height/weight where available; missing conts -> NaN
(loader maps NaN->0.0). sex in {M,F,U}; arch is pinned by cat_encoders_path (the
mimic3 run's demo_encoder.json), so unseen sex strings just map to the pad/unknown id.

Index = subject id, IDENTICAL to build_wearable_hr.py (dalia_S1 / gyro_1 / wesad_S2).
Run on bedanalysis (raw pkl/readme live there), then transfer the CSVs to dream.
"""
import csv, glob, io, os, pickle, re, zipfile

DATA_ROOT = "/labs/hulab/mxwang/data/Wav_Agent"
OUT_ROOT = "/labs/hulab/mxwang/data/Wav_Agent/processed"
COLS = ["sex", "age", "height_cm", "weight_kg", "temp", "hr", "rr", "spo2", "sbp", "dbp"]


def _sex(s):
    s = (s or "").strip().lower()
    if s.startswith("m"):
        return "M"
    if s.startswith("f") or s.startswith("w"):
        return "F"
    return "U"


def rows_dalia():
    out = {}
    with zipfile.ZipFile(f"{DATA_ROOT}/PPG-DaLiA/data.zip") as z:
        for n in sorted(x for x in z.namelist() if x.endswith(".pkl")):
            sid = "dalia_S" + n.split("/S")[1].split("/")[0]
            q = pickle.load(io.BytesIO(z.read(n)), encoding="latin1").get("questionnaire", {})
            out[sid] = dict(sex=_sex(q.get("Gender")), age=q.get("AGE"),
                            height_cm=q.get("HEIGHT"), weight_kg=q.get("WEIGHT"))
    return out


def rows_wesad():
    out = {}
    for f in sorted(glob.glob(f"{DATA_ROOT}/WESAD/WESAD/S*/S*_readme.txt")):
        sid = "wesad_" + os.path.basename(os.path.dirname(f))
        t = open(f).read()
        def g(pat):
            m = re.search(pat, t, re.I)
            return m.group(1).strip() if m else None
        out[sid] = dict(sex=_sex(g(r"Gender:\s*(\w+)")),
                        age=g(r"Age:\s*([\d.]+)"),
                        height_cm=g(r"Height \(cm\):\s*([\d.]+)"),
                        weight_kg=g(r"Weight \(kg\):\s*([\d.]+)"))
    return out


def rows_gyro():
    # README: aggregate only (mean age 26.9). Neutral per-subject.
    out = {}
    for d in sorted(glob.glob(f"{OUT_ROOT}/gyro/gyro_*")):
        sid = os.path.basename(d)
        out[sid] = dict(sex="U", age=27, height_cm=None, weight_kg=None)
    return out


def write_csv(ds, rows):
    p = f"{OUT_ROOT}/{ds}/demographics_unified.csv"
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["subject_id"] + COLS)
        for sid in sorted(rows):
            r = rows[sid]
            w.writerow([sid] + [r.get(c, "") if r.get(c) is not None else "" for c in COLS])
    n_sex = sum(1 for r in rows.values() if r.get("sex") in ("M", "F"))
    print(f"{ds:6s} -> {p}  ({len(rows)} subjects, {n_sex} with real sex)")


if __name__ == "__main__":
    write_csv("dalia", rows_dalia())
    write_csv("wesad", rows_wesad())
    write_csv("gyro", rows_gyro())
