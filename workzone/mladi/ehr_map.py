"""MLADI /ehr -> registry var_id mapping (datasets/mladi/API.md, reviewed 2026-10-05).

Labs follow what the registry already does for the other cohorts (central lab plus whole-blood /
point-of-care for K, Na, glucose, lactate; total calcium only; central-lab creatinine; arterial blood gas
only). Units are checked per row on the decoded `resultUnit`; a row in a unit the variable does not
accept is dropped (counted); a missing unit is accepted and left to the registry range.
"""
from __future__ import annotations
import re
import numpy as np

# ---- labs: lab_results.eventDisp -> var_id
LABS = {
    0: ["K", "Potassium(K) Whole Blood", "Potassium iSTAT", "Potassium Level"],
    1: ["Ca", "Calcium Level"],
    2: ["Na", "Sodium(Na) Whole Blood", "Sodium Istat", "Sodium (Na) Level"],
    3: ["Glucose", "Glucose (bedside)", "Glucose POC", "Glucose iSTAT", "Glucose Level Whole Blood",
        "Glucose Whole Blood", "Glucose Level"],
    4: ["Lactate, Whole Blood", "Lactate", "Lactate Whole Blood", "Lactate Whole Blood (Syringe)"],
    5: ["Cr"],
    6: ["Bili, Total"],
    7: ["Platelets"],
    8: ["WBC"],
    9: ["Hgb", "Calc. Hemoglobin iSTAT", "Hemoglobin-Arterial"],
    10: ["INR"],
    11: ["BUN"],
    12: ["Albumin"],
    13: ["pHa"],
    14: ["PaO2", "Arterial pO2 (POC)"],
    15: ["PaCO2", "Arterial pCO2 (POC)"],
    16: ["CO2", "HCO3a", "HCO3"],
    17: ["AST/SGOT"],
    18: ["ALT/SGPT"],
}
LAB_NAME = {n: v for v, names in LABS.items() for n in names}
_MMOL = {"mmol/l", "meq/l"}
LAB_UNITS = {0: _MMOL, 1: {"mg/dl"}, 2: _MMOL, 3: {"mg/dl"}, 4: _MMOL, 5: {"mg/dl"}, 6: {"mg/dl"},
             7: {"x10e+09/l", "k/ul", "x10e+3/ul", "10*3/ul"}, 8: {"x10e+09/l", "k/ul", "x10e+3/ul", "10*3/ul"},
             9: {"g/dl", "gm/dl"}, 10: None, 11: {"mg/dl"}, 12: {"g/dl", "gm/dl"}, 13: None,
             14: {"mmhg", "mm hg"}, 15: {"mmhg", "mm hg"}, 16: _MMOL, 17: {"iu/l", "u/l"}, 18: {"iu/l", "u/l"}}

# ---- charted vitals: low_rate.eventName -> var_id
VITALS = {
    100: ["Pulse"], 101: ["O2 Saturation"], 102: ["Respiratory Rate"],
    103: ["Temperature Metric", "Temperature", "Temperature Conversi", "Temperature Conversion"],
    104: ["Systolic BP"], 105: ["Diastolic BP"], 106: ["Mean blood pressure"],
    107: ["Central Venous Press", "Central Venous Pressure"],
    108: ["Glasgow Coma Score", "Glascow Coma Score"],
    110: ["Arterial Systolic Pr", "Arterial Systolic Pressure"],
    111: ["Arterial Diastolic P", "Arterial Diastolic Pressure"],
    112: ["Mean arterial pressu", "Mean arterial pressure"],
    117: ["Oxygen per liter"],
}
VITAL_NAME = {n: v for v, names in VITALS.items() for n in names}

# ---- actions from low_rate
FIO2_NAMES = {"Oxygen % (FiO2)", "FIO2"}
PEEP_NAMES = {"Positive end expiratory pressure (PEEP)"}
VENT_NAMES = {"RRT Vent Status", "RRT Ventilator Type", "RRT Tidal Volume Set", "RRT Machine Rate", "RRT Total Rate"}

NONSYSTEMIC_ROUTES = re.compile(r"eye|ophth|nostril|nasal|aerosol|inhal|topical|swish|mucous|transdermal|"
                                r"irrig|otic|ear|vagin|clotted catheter|nerve block", re.I)


def unit_norm(u) -> str | None:
    if u is None:
        return None
    s = str(u).strip().lower().replace(" ", "")
    s = s.replace("mmol", "mmol").replace("gm/", "g/") if s else s
    return s or None


def unit_ok(vid: int, unit) -> bool:
    acc = LAB_UNITS.get(vid)
    u = unit_norm(unit)
    if acc is None or u is None or u in ("<na>", "nan", "none"):
        return True
    acc = {a.replace(" ", "") for a in acc} | {a.replace("g/dl", "gm/dl") for a in acc}
    return u in acc or u.replace("gm/", "g/") in acc


def temperature_c(name: str, value: float, unit) -> float:
    """Charted temperature -> deg C: Fahrenheit by unit or by value (> 50 is not a Celsius reading)."""
    u = (str(unit) if unit is not None else "").lower()
    if "f" in u.replace("ref", "") or value > 50:
        return (value - 32.0) * 5.0 / 9.0
    return value


def drug_to_var(name: str) -> int | None:
    """Drug name -> action var_id or None. The shared matcher of workzone/mcmed/stage3b_actions.py
    and mover_combine (vasopressors 207-213, PRBC 214, insulin 215, dextrose 216, KCl 217, calcium 218,
    bicarbonate 219, hypertonic saline 220, crystalloid / colloid bolus 202)."""
    n = (name or "").lower()
    if any(x in n for x in ["opht", "nasal", "naris", "nebu", "inhal", " inh ", "topical", " tp ", "irrig",
                            "swab", "flush", " drop", "ointment", "patch"]):
        return None
    if "pseudoephedrine" in n or "pseudoephed" in n:
        return None
    if "lidocaine" in n or "bupivacaine" in n:
        return None
    piggyback = any(x in n for x in ["ivpb", "in sodium chloride", "in 0.9", "in 5 % dextrose", "in 5% dextrose",
                                     "in dextrose", "premix", "compounded"])
    if "norepinephrine" in n or "levophed" in n: return 207
    if "epinephrine" in n or "adrenaline" in n: return 208
    if "phenylephrine" in n or "synephrine" in n: return 209
    if "dopamine" in n: return 210
    if "vasopressin" in n: return 211
    if "dobutamine" in n: return 212
    if "ephedrine" in n: return 213
    if "packed red" in n or "prbc" in n or "red blood cell" in n: return 214
    if "insulin" in n: return 215
    if any(x in n for x in ["dextrose 50", "dextrose 20", "dextrose 10", "d50", "d10w", "d20"]) and not piggyback: return 216
    if "potassium chloride" in n or "kcl" in n: return 217
    if "calcium" in n and ("chlor" in n or "gluc" in n): return 218
    if "bicarb" in n: return 219
    if "hypertonic" in n or "sodium chloride 3" in n or "nacl 3" in n: return 220
    if (not piggyback and any(f in n for f in ["plasmalyte", "plasma-lyte", "lactated ringer", "lr iv",
                                               "sodium chloride 0.9", "normal saline", "ns iv", "albumin",
                                               "normosol", "electrolyte solution"])): return 202
    return None


# native dose -> registry unit, where a conversion is exact; else the value is NaN (presence)
def action_value(vid: int, dose, unit, volume=None, volume_unit=None) -> float:
    from common import num
    d = num(dose); u = (str(unit) if unit is not None else "").strip().lower()
    if vid in range(207, 214):                 # registry rates; MLADI charts amounts -> presence
        return float("nan")
    if vid == 215:
        return d if u.startswith("unit") else float("nan")
    if vid in (217, 219):
        return d if u == "meq" else float("nan")
    if vid == 218:
        return d if u in ("gm", "g") else (d / 1000.0 if u == "mg" else float("nan"))
    if vid == 216:
        return d if u in ("gm", "g") else float("nan")
    if vid in (202, 220):
        v = num(volume); vu = (str(volume_unit) if volume_unit is not None else "").strip().lower()
        if np.isfinite(v) and vu == "ml":
            return v
        return d if u == "ml" else float("nan")
    return float("nan")
