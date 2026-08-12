#!/usr/bin/env python3
"""
Stage 3b (v2): 9-target action extraction — per-drug + aggregate, MV rate + CV presence.

Supersedes stage3b_extract_actions.py. Emits, per patient, into ehr_events.npy:
  200  vasopressor_rate     aggregate NE-equivalent (concurrent-sum step fn, with 0 at stop)
  207-213 per-drug vaso     raw rate (MV) at start, 0 at end; CV presence (value=NaN)
  201/202 fluid rate/bolus  crystalloids (225166 KCl bug REMOVED)
  214  prbc_transfusion     |  215 insulin | 216 dextrose_hi | 217 potassium_replacement
  218  calcium_replacement  |  219 sodium_bicarbonate | 220 hypertonic_saline
  203-206 FiO2/PEEP/mechvent/urine  (unchanged, MV/CHARTEVENTS/OUTPUTEVENTS)

Value semantics (uniform, per user decision):
  - MV numeric rate/amount stored as-is; vaso infusions also get a value=0 STOP at ENDTIME.
  - CV (CareVue): presence only -> value = NaN (rate units unreliable; ~8% of the BP cohort,
    which is ~92% metavision). NaN = "action occurred, magnitude unknown".

NON-DESTRUCTIVE: writes a SEPARATE per-patient `ehr_actions.npy` sidecar (same dtype as
ehr_events); ehr_events.npy / meta.json / waveforms are NEVER touched. Idempotent (overwrites
the sidecar). The adapter loads ehr_actions.npy when actions are needed. The OLD in-line
var 200-206 in ehr_events (from the legacy stage3b) are left as-is but deprecated/unread.
SCRATCH-SAFE: --out-root writes the sidecar to a copy dir; --patients/-file / --limit restrict.

Run (scratch test):
  python workzone/mimic3/stage3b_actions_v2.py \
     --patients 81378_147279 --out-root /labs/hulab/mxwang/tmp_local/stage3b_scratch
Full canonical run (writes ehr_actions.npy in-place; ehr_events untouched):
  python workzone/mimic3/stage3b_actions_v2.py
"""
import os, json, time, logging, argparse, shutil
from pathlib import Path
import numpy as np
import polars as pl
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO_ROOT / "workzone" / "outputs" / "mimic3"
import yaml
with open(REPO_ROOT / "workzone" / "configs" / "server_paths.yaml") as f:
    cfg = yaml.safe_load(f)
EHR_ROOT = cfg["mimic3"]["raw_ehr_dir"]
PROCESSED_ROOT = cfg["mimic3"]["output_dir"]

EHR_EVENT_DTYPE = np.dtype([("time_ms", "int64"), ("seg_idx", "int32"),
                            ("var_id", "uint16"), ("value", "float32")])

# ---- vasopressors: per-drug var_id, NE-factor, MV itemid, CV itemids ---------
# ne_factor converts native rate -> NE-eq mcg/kg/min. Phenylephrine mcg/min /80kg /10.
VASO = {
    207: dict(drug="Norepinephrine", ne=1.0,     mv=[221906], cv=[30047, 30120]),
    208: dict(drug="Epinephrine",    ne=1.0,     mv=[221289], cv=[30044, 30119, 30309]),
    209: dict(drug="Phenylephrine",  ne=0.1/80,  mv=[221749], cv=[30127, 30128]),
    210: dict(drug="Dopamine",       ne=0.01,    mv=[221662], cv=[30043, 30307]),
    211: dict(drug="Vasopressin",    ne=0.0,     mv=[222315], cv=[30051]),   # binary -> 0.1 NE-eq
    212: dict(drug="Dobutamine",     ne=0.0,     mv=[221653], cv=[30042, 30306]),  # inotrope, 0 NE-eq
}
VASO_MV_ITEM = {it: vid for vid, d in VASO.items() for it in d["mv"]}
VASO_CV_ITEM = {it: vid for vid, d in VASO.items() for it in d["cv"]}
NE_OF_VID = {vid: d["ne"] for vid, d in VASO.items()}

# ---- fluids (crystalloid) — KCl 225166 REMOVED --------------------------------
FLUID_MV = [225158, 225159, 225828]        # NaCl 0.9%, NaCl 0.45%, LR
FLUID_CV = [30018, 30021, 30160, 30352, 30013]  # CV crystalloids (presence)

# ---- lab-target actions: var_id -> (MV itemids, CV itemids), value=AMOUNT ------
LABACT = {
    214: dict(name="prbc_transfusion",     mv=[220996, 225168],                 cv=[30179, 30004, 42324, 46407]),
    215: dict(name="insulin",              mv=[223258, 223262, 223260, 223259, 223257, 223261], cv=[30045, 30100, 30310, 42763]),
    216: dict(name="dextrose_hi",          mv=[220950, 220952],                 cv=[30016, 30187, 44635]),
    217: dict(name="potassium_replacement",mv=[225166, 225925, 227522, 227536], cv=[30026, 30297, 40531]),
    218: dict(name="calcium_replacement",  mv=[221456, 227525, 228317],         cv=[30023, 30300, 30022]),
    219: dict(name="sodium_bicarbonate",   mv=[220995, 221211, 225165, 227533], cv=[30030, 30338]),
    220: dict(name="hypertonic_saline",    mv=[225161],                         cv=[]),
}

FIO2_ITEMID, PEEP_ITEMID = 223835, 220339
URINE_ITEMIDS = [226559, 226560, 226561, 226563, 226564, 226565,
                 226567, 226557, 226558, 227488, 227489]

DT_FMT = "%Y-%m-%d %H:%M:%S"
SKIP_CV = False  # set by --skip-cv (MV-only fast pass)


def _p(name):
    p = os.path.join(EHR_ROOT, name + ".csv.gz")
    return p if os.path.exists(p) else os.path.join(EHR_ROOT, name + ".csv")


def _ep_ms(dt_series):
    """polars datetime -> epoch ms (matches the pipeline's charttime_dt.timestamp()*1000)."""
    return (dt_series.dt.timestamp("ms")).cast(pl.Int64)


# ============================================================ vasopressors
def extract_vaso():
    """MV per-drug raw rate (start) + 0 (end) -> 207-212; aggregate NE-eq -> 200.
    CV per-drug presence (NaN) -> 207-212 and 200."""
    log.info("=== vasopressors (MV rate + stops + aggregate; CV presence) ===")
    mv = pl.scan_csv(_p("INPUTEVENTS_MV"), infer_schema_length=2000).filter(
        pl.col("ITEMID").is_in(list(VASO_MV_ITEM)) &
        pl.col("RATE").is_not_null() & pl.col("RATE").is_not_nan() & (pl.col("RATE") > 0) &
        (pl.col("STATUSDESCRIPTION") != "Rewritten")
    ).select(["SUBJECT_ID", "HADM_ID", "ITEMID", "STARTTIME", "ENDTIME", "RATE"]).collect()
    mv = mv.with_columns([
        pl.col("ITEMID").replace_strict(VASO_MV_ITEM, default=None).alias("var_id"),
        pl.col("ITEMID").replace_strict({it: NE_OF_VID[VASO_MV_ITEM[it]] for it in VASO_MV_ITEM}, default=0.0).alias("ne"),
        pl.col("STARTTIME").str.strptime(pl.Datetime, DT_FMT).alias("t_start"),
        pl.col("ENDTIME").str.strptime(pl.Datetime, DT_FMT).alias("t_end"),
    ])
    # per-drug: raw rate at start, 0 at end
    per_start = mv.select(["SUBJECT_ID", "HADM_ID", "var_id",
                           pl.col("t_start").alias("t"), pl.col("RATE").alias("value")])
    per_stop = mv.select(["SUBJECT_ID", "HADM_ID", "var_id",
                          pl.col("t_end").alias("t"), pl.lit(0.0).alias("value")])
    perdrug = pl.concat([per_start, per_stop])

    # aggregate 200: delta-cumsum of NE-eq per (subject,hadm).
    #   dobutamine (212, inotrope) EXCLUDED entirely (no NE-eq contribution, no spurious 200=0 rows);
    #   vasopressin (211) -> binary 0.1 ; others -> rate * ne_factor
    ne_start = mv.filter(pl.col("var_id") != 212).with_columns(
        pl.when(pl.col("var_id") == 211).then(pl.lit(0.1))
         .otherwise(pl.col("RATE") * pl.col("ne")).alias("neq")
    )
    d_up = ne_start.select(["SUBJECT_ID", "HADM_ID", pl.col("t_start").alias("t"), pl.col("neq").alias("delta")])
    d_dn = ne_start.select(["SUBJECT_ID", "HADM_ID", pl.col("t_end").alias("t"), (-pl.col("neq")).alias("delta")])
    # net delta per (patient,timestamp) BEFORE cumsum -> order-independent, one 200 event/timestamp
    # (avoids the transient double-count when a stop(-) and start(+) share a timestamp)
    deltas = (pl.concat([d_up, d_dn])
              .group_by(["SUBJECT_ID", "HADM_ID", "t"]).agg(pl.col("delta").sum())
              .sort(["SUBJECT_ID", "HADM_ID", "t"]))
    agg = deltas.with_columns(
        pl.col("delta").cum_sum().over(["SUBJECT_ID", "HADM_ID"]).alias("value")
    ).with_columns(
        pl.col("value").clip(0.0, 10.0).alias("value")   # clamp fp noise + outlier rates to physio_max
    ).select(["SUBJECT_ID", "HADM_ID", "t", pl.lit(200).cast(pl.Int32).alias("var_id"), "value"])

    # CV presence (value=NaN) per-drug + 200
    out = [perdrug.with_columns(pl.col("var_id").cast(pl.Int32)),
           agg.select(["SUBJECT_ID", "HADM_ID", "t", "var_id", "value"])]
    if SKIP_CV:
        res = pl.concat([o.select(["SUBJECT_ID", "HADM_ID", "t", "var_id", "value"]) for o in out])
        log.info(f"  vaso events (MV per-drug+agg, CV skipped): {len(res):,}")
        return res
    try:
        cv = pl.scan_csv(_p("INPUTEVENTS_CV"), infer_schema_length=2000).filter(
            pl.col("ITEMID").is_in(list(VASO_CV_ITEM)) & pl.col("CHARTTIME").is_not_null()
        ).select(["SUBJECT_ID", "HADM_ID", "ITEMID", "CHARTTIME"]).collect()
        cv = cv.with_columns([
            pl.col("ITEMID").replace_strict(VASO_CV_ITEM, default=None).alias("var_id"),
            pl.col("CHARTTIME").str.strptime(pl.Datetime, DT_FMT).alias("t"),
        ])
        cv_pd = cv.select(["SUBJECT_ID", "HADM_ID", pl.col("var_id").cast(pl.Int32), "t",
                           pl.lit(float("nan")).alias("value")])
        cv_agg = cv.select(["SUBJECT_ID", "HADM_ID", "t", pl.lit(200).cast(pl.Int32).alias("var_id"),
                            pl.lit(float("nan")).alias("value")])
        out += [cv_pd.select(["SUBJECT_ID", "HADM_ID", "var_id", "t", "value"]),
                cv_agg.select(["SUBJECT_ID", "HADM_ID", "t", "var_id", "value"])]
        log.info(f"  CV presence rows: {len(cv_pd):,}")
    except Exception as e:
        log.warning(f"  CV skipped: {e}")
    res = pl.concat([o.select(["SUBJECT_ID", "HADM_ID", "t", "var_id", "value"]) for o in out])
    log.info(f"  vaso events (per-drug+agg+cv): {len(res):,}")
    return res


# ============================================================ fluids / labact
def _mv_amount(itemids, var_id):
    df = pl.scan_csv(_p("INPUTEVENTS_MV"), infer_schema_length=2000).filter(
        pl.col("ITEMID").is_in(itemids) & pl.col("AMOUNT").is_not_null() &
        (pl.col("AMOUNT") > 0) & (pl.col("STATUSDESCRIPTION") != "Rewritten")
    ).select(["SUBJECT_ID", "HADM_ID", "STARTTIME", "AMOUNT"]).collect()
    return df.select(["SUBJECT_ID", "HADM_ID",
                      pl.col("STARTTIME").str.strptime(pl.Datetime, DT_FMT).alias("t"),
                      pl.lit(var_id).cast(pl.Int32).alias("var_id"),
                      pl.col("AMOUNT").cast(pl.Float64).alias("value")])


def _cv_presence(itemids, var_id):
    if not itemids or SKIP_CV:
        return None
    df = pl.scan_csv(_p("INPUTEVENTS_CV"), infer_schema_length=2000).filter(
        pl.col("ITEMID").is_in(itemids) & pl.col("CHARTTIME").is_not_null()
    ).select(["SUBJECT_ID", "HADM_ID", "CHARTTIME"]).collect()
    return df.select(["SUBJECT_ID", "HADM_ID",
                      pl.col("CHARTTIME").str.strptime(pl.Datetime, DT_FMT).alias("t"),
                      pl.lit(var_id).cast(pl.Int32).alias("var_id"),
                      pl.lit(float("nan")).alias("value")])


def extract_fluids():
    log.info("=== fluids (crystalloid; KCl 225166 removed) ===")
    df = pl.scan_csv(_p("INPUTEVENTS_MV"), infer_schema_length=2000).filter(
        pl.col("ITEMID").is_in(FLUID_MV) & pl.col("AMOUNT").is_not_null() &
        (pl.col("AMOUNT") > 0) & (pl.col("STATUSDESCRIPTION") != "Rewritten")
    ).select(["SUBJECT_ID", "HADM_ID", "STARTTIME", "AMOUNT", "RATE", "ORDERCATEGORYDESCRIPTION"]).collect()
    bolus = df.filter(pl.col("ORDERCATEGORYDESCRIPTION").str.contains("(?i)bolus")).select(
        ["SUBJECT_ID", "HADM_ID", pl.col("STARTTIME").str.strptime(pl.Datetime, DT_FMT).alias("t"),
         pl.lit(202).cast(pl.Int32).alias("var_id"), pl.col("AMOUNT").cast(pl.Float64).alias("value")]).filter(pl.col("value") <= 5000)
    inf = df.filter(~pl.col("ORDERCATEGORYDESCRIPTION").str.contains("(?i)bolus") &
                    pl.col("RATE").is_not_null() & (pl.col("RATE") > 0)).select(
        ["SUBJECT_ID", "HADM_ID", pl.col("STARTTIME").str.strptime(pl.Datetime, DT_FMT).alias("t"),
         pl.lit(201).cast(pl.Int32).alias("var_id"), pl.col("RATE").cast(pl.Float64).alias("value")]).filter(pl.col("value") <= 2000)
    parts = [bolus, inf]
    cv = _cv_presence(FLUID_CV, 201)
    if cv is not None:
        parts.append(cv)
    return pl.concat(parts)


def extract_labact():
    log.info("=== lab-target actions 214-220 (MV amount + CV presence) ===")
    parts = []
    for vid, d in LABACT.items():
        parts.append(_mv_amount(d["mv"], vid))
        cv = _cv_presence(d["cv"], vid)
        if cv is not None:
            parts.append(cv)
    return pl.concat(parts)


def extract_chart_out():
    """FiO2(203)/PEEP(204) from CHARTEVENTS, mechvent(205) inferred, urine(206) OUTPUTEVENTS."""
    log.info("=== FiO2/PEEP/mechvent/urine (unchanged) ===")
    ch = pl.scan_csv(_p("CHARTEVENTS"), infer_schema_length=1000).filter(
        pl.col("ITEMID").is_in([FIO2_ITEMID, PEEP_ITEMID]) &
        pl.col("VALUENUM").is_not_null() & pl.col("VALUENUM").is_not_nan()
    ).select(["SUBJECT_ID", "HADM_ID", "ITEMID", "CHARTTIME", "VALUENUM"]).collect()
    ch = ch.with_columns(pl.when(pl.col("ITEMID") == FIO2_ITEMID).then(pl.col("VALUENUM")/100.0)
                         .otherwise(pl.col("VALUENUM")).alias("VALUENUM"))
    ch = ch.with_columns(pl.when(pl.col("ITEMID") == FIO2_ITEMID).then(pl.lit(203)).otherwise(pl.lit(204)).alias("var_id"))
    ch = ch.filter(((pl.col("var_id") == 203) & (pl.col("VALUENUM").is_between(0.21, 1.0))) |
                   ((pl.col("var_id") == 204) & (pl.col("VALUENUM").is_between(0, 30))))
    ch = ch.with_columns(pl.col("CHARTTIME").str.strptime(pl.Datetime, DT_FMT).alias("t"))
    base = ch.select(["SUBJECT_ID", "HADM_ID", "t", pl.col("var_id").cast(pl.Int32),
                      pl.col("VALUENUM").cast(pl.Float64).alias("value")])
    vent = ch.filter(((pl.col("var_id") == 203) & (pl.col("VALUENUM") > 0.21)) |
                     ((pl.col("var_id") == 204) & (pl.col("VALUENUM") > 0))).select(
        ["SUBJECT_ID", "HADM_ID", "t"]).unique().with_columns([
        pl.lit(205).cast(pl.Int32).alias("var_id"), pl.lit(1.0).alias("value")])
    ur = pl.scan_csv(_p("OUTPUTEVENTS"), infer_schema_length=10000,
                     schema_overrides={"VALUE": pl.Float64}).filter(
        pl.col("ITEMID").is_in(URINE_ITEMIDS) & pl.col("VALUE").is_not_null() & pl.col("VALUE").is_not_nan()
    ).select(["SUBJECT_ID", "HADM_ID", "CHARTTIME", "VALUE"]).collect()
    ur = ur.filter(pl.col("VALUE").is_between(0, 2500)).select(
        ["SUBJECT_ID", "HADM_ID", pl.col("CHARTTIME").str.strptime(pl.Datetime, DT_FMT).alias("t"),
         pl.lit(206).cast(pl.Int32).alias("var_id"), pl.col("VALUE").alias("value")])
    return pl.concat([base, vent.select(base.columns), ur.select(base.columns)])


# ============================================================ merge
def write_actions_for_patient(canon_dir, out_dir, actions_df):
    """Write actions to a SEPARATE ehr_actions.npy sidecar (same dtype as ehr_events).
    NON-DESTRUCTIVE: ehr_events.npy and meta.json are never touched. Idempotent
    (overwrites ehr_actions.npy). Aligns to the canonical time_ms grid via searchsorted."""
    time_ms_path = os.path.join(canon_dir, "time_ms.npy")
    if not os.path.exists(time_ms_path):
        return -1
    time_ms = np.load(time_ms_path); n_seg = len(time_ms)
    new = []
    for r in actions_df.iter_rows(named=True):
        et = int(r["t_ms"])
        s = int(np.searchsorted(time_ms, et, side="right") - 1)
        if 0 <= s < n_seg:
            new.append((et, s, int(r["var_id"]), float(r["value"])))  # value may be NaN (CV presence)
    sidecar = os.path.join(out_dir, "ehr_actions.npy")
    if not new:
        # idempotent: no valid actions -> drop any stale sidecar from an older run
        if os.path.exists(sidecar):
            os.remove(sidecar)
        return 0
    arr = np.array(new, dtype=EHR_EVENT_DTYPE)
    arr.sort(order="time_ms")
    os.makedirs(out_dir, exist_ok=True)
    np.save(sidecar, arr)                                    # sidecar; ehr_events untouched
    return len(new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--patients", nargs="*", default=None, help="specific '{subj}_{hadm}' dirs")
    ap.add_argument("--patients-file", default=None, help="file with one '{subj}_{hadm}' dir per line")
    ap.add_argument("--out-root", default=None, help="write merged files here (scratch). default=canonical PROCESSED_ROOT (in-place)")
    ap.add_argument("--skip-chart", action="store_true", help="skip FiO2/PEEP/urine (203-206, unchanged; avoids the slow CHARTEVENTS scan)")
    ap.add_argument("--skip-cv", action="store_true", help="skip INPUTEVENTS_CV presence (MV-only, faster first pass)")
    a = ap.parse_args()
    global SKIP_CV
    SKIP_CV = a.skip_cv
    out_root = a.out_root or PROCESSED_ROOT
    inplace = (out_root == PROCESSED_ROOT)
    log.info(f"Stage3b v2 | out_root={out_root} | inplace={inplace}")

    _parts = [
        extract_vaso().select(["SUBJECT_ID", "HADM_ID", "t", "var_id", "value"]),
        extract_fluids().select(["SUBJECT_ID", "HADM_ID", "t", "var_id", "value"]),
        extract_labact().select(["SUBJECT_ID", "HADM_ID", "t", "var_id", "value"]),
    ]
    if not a.skip_chart:
        _parts.append(extract_chart_out().select(["SUBJECT_ID", "HADM_ID", "t", "var_id", "value"]))
    else:
        log.info("  (skipping FiO2/PEEP/mechvent/urine 203-206 per --skip-chart)")
    allacts = pl.concat(_parts).with_columns(_ep_ms(pl.col("t")).alias("t_ms"))
    log.info(f"total action events: {len(allacts):,}")
    for vid in sorted(set(allacts["var_id"].to_list())):
        log.info(f"  var {vid}: {allacts.filter(pl.col('var_id')==vid).height:,}")

    allacts = allacts.with_columns([
        pl.col("SUBJECT_ID").cast(pl.Int64, strict=False),
        pl.col("HADM_ID").cast(pl.Int64, strict=False),
    ]).filter(pl.col("SUBJECT_ID").is_not_null())
    grouped = {k[0]: v for k, v in allacts.partition_by("SUBJECT_ID", as_dict=True).items()}

    if a.patients_file:
        dirs = [l.strip() for l in open(a.patients_file) if l.strip()]
    elif a.patients:
        dirs = a.patients
    else:
        dirs = sorted(d for d in os.listdir(PROCESSED_ROOT)
                      if os.path.isdir(os.path.join(PROCESSED_ROOT, d)) and "_" in d and not d.startswith("."))
        if a.limit:
            dirs = dirs[:a.limit]
    log.info(f"patient dirs: {len(dirs)}")

    from tqdm import tqdm
    n_ok = tot = 0
    for dn in tqdm(dirs, unit="pat"):
        parts = dn.split("_")
        if len(parts) != 2:
            continue
        try:
            sid, hid = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        g = grouped.get(sid)
        if g is None or g.height == 0:
            continue
        gg = g.filter(pl.col("HADM_ID") == hid)
        if gg.height == 0:
            gg = g
        gpl = gg.select(["t_ms", "var_id", "value"])
        canon = os.path.join(PROCESSED_ROOT, dn)
        out = canon if inplace else os.path.join(out_root, dn)
        n = write_actions_for_patient(canon, out, gpl)
        if n > 0:
            n_ok += 1; tot += n
    log.info(f"done: {n_ok} patients, {tot:,} new events -> {out_root}")


if __name__ == "__main__":
    main()
