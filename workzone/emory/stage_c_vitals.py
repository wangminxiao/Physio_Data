#!/usr/bin/env python3
"""
Stage C — Emory dense vitals extraction from `_0n.mat` (WFDB numerics @ 0.5 Hz).

For each entity with Stage B meta.json on disk:
  1. Iterate wfdb_records_all (chronological).
  2. For each record, read {rec}_0n.{hea,mat} (record-level numerics file).
  3. For each channel whose name maps to a var_id (see CHANNEL_TO_VAR_ID),
     extract sparse events:
       - base_ms = (hdr.base_datetime + 30 years) treated as UTC
       - fs from hdr.fs (typically 0.5 Hz); time_ms = base_ms + i * (1000/fs)
       - value = p_signal[i, ch]  (physical=True applies ADC gain)
       - drop NaN / non-finite / zero / negative  (per API.md: 0/neg = missing)
  4. Concatenate across records, sort by (time_ms, var_id), write
     {out_dir}/{entity}/vitals_events.npy as structured
     (time_ms: int64, var_id: uint16, value: float32).
  5. Update meta.json with per-var / per-channel counts, unknown channels seen,
     and stage_c_version=1.
  6. Hold signals (NBP-S/D/M) are SKIPPED — use EHR SBP/DBP/MAP_CUFF (Stage D).

No episode clipping here (Stage B meta.json has wave_start/end only, not
admit/discharge). Stage E does the partition + seg_idx assignment.

Resume: skip entity if vitals_events.npy exists and meta.json has
stage_c_version >= 1 (unless --no-resume).

Run modes:
  python stage_c_vitals.py --limit 5 --workers 4     # smoke
  python stage_c_vitals.py --entity-id 1827183_359559206
  python stage_c_vitals.py                            # full run
"""
import os
import sys
import json
import time
import argparse
import logging
import traceback
import multiprocessing as mp
from datetime import timezone
from dateutil.relativedelta import relativedelta

import numpy as np
import polars as pl
import wfdb

UTC = timezone.utc
WFDB_ROOT = "/labs/collab/Waveform_Data/Waveform_Data"
OUT_ROOT = "/opt/localdata100tb/physio_data/emory"
COHORT_PARQUET = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/emory/valid_wave_window.parquet"
LOG_DIR = "/labs/hulab/mxwang/Physio_Data/workzone/emory/logs"
SUMMARY_JSON = "/labs/hulab/mxwang/Physio_Data/workzone/outputs/emory/stage_c_summary.json"

DEFAULT_WORKERS = 24  # half of 48 cores, shared cluster cap

# Canonical _0n sig_name → var_id (see datasets/emory/API.md §Vitals).
CHANNEL_TO_VAR_ID = {
    "HR":     100,
    "SPO2-%": 101,
    "RESP":   102,
    "TMP-1":  103, "TMP-2": 103, "TMP-3": 103, "TMP-4": 103,
    "BT":     103,  # blood temperature; same semantic (var_registry 103)
    "CO2-EX": 116,
    "CO2-IN": 118,
    "CO2-RR": 119,
    "ST-II":  120,
    "ST-I":   121,
    "ST-V":   122,
    # Invasive ABP lines collapse to same semantic (var 110/111/112)
    "AR1-S":  110, "AR2-S":  110, "ART-S":  110, "AR3-S": 110, "AR4-S": 110,
    "AR1-D":  111, "AR2-D":  111, "ART-D":  111, "AR3-D": 111, "AR4-D": 111,
    "AR1-M":  112, "AR2-M":  112, "ART-M":  112, "AR3-M": 112, "AR4-M": 112,
}

# Channels we deliberately ignore for first pass. Explicit list so any NEW
# channel shows up in meta.unknown_channels_seen instead of silently dropping.
SKIP_CHANNELS = {
    # hold signals — use EHR SBP/DBP/MAP_CUFF instead (see API.md cuff-cycle)
    "NBP-S", "NBP-D", "NBP-M", "CUFF",
    # low clinical value first pass
    "APNEA", "PVC",
    # extra ECG leads not in var_registry yet
    "ST-III", "ST-V1", "ST-V2", "ST-V3", "ST-V4", "ST-V5", "ST-V6",
    "ST-AVR", "ST-AVL", "ST-AVF",
    # redundant HR derivations
    "SPO2-R", "AR1-R", "AR2-R", "AR3-R", "AR4-R", "ART-R",
    # invasive pressures not in first-pass registry
    "CVP1", "CVP2", "CVP3", "CVP4",
    "ICP", "ICP1", "ICP2", "ICP3", "ICP4",
    "CPP", "CPP1", "CPP2", "CPP3", "CPP4",
    "SP1", "SP2", "SP3", "SP4",
    # Pulmonary artery (Swan-Ganz) and femoral lines — not in first-pass registry
    "PA1-S", "PA1-D", "PA1-M", "PA1-R",
    "PA2-S", "PA2-D", "PA2-M", "PA2-R",
    "PA3-S", "PA3-D", "PA3-M", "PA3-R",
    "PA4-S", "PA4-D", "PA4-M", "PA4-R",
    "FE1-S", "FE1-D", "FE1-M", "FE1-R",
    "FE2-S", "FE2-D", "FE2-M", "FE2-R",
    "FE3-S", "FE3-D", "FE3-M", "FE3-R",
    "FE4-S", "FE4-D", "FE4-M", "FE4-R",
    # Gas analytics, cardiac output, airway pressure — deferred
    "O2-EXP", "O2-INSP", "CO", "PAW",
    # Other temp variants not mapped (irrigation / inspired)
    "IT",
    # Depth-of-anesthesia / EEG-derived monitor channels — not first-pass
    "BIS", "EMG", "SQI", "SR",
    # Left atrial pressures — low prevalence, defer
    "LA1", "LA2", "LA3", "LA4",
}


def _salvage_sig_name(name) -> str:
    """Some Emory `.hea` files write temp units as `10/Deg C` — the space inside
    the units field pushes trailing tokens onto the sig_name (e.g.
    `'C 16 0 -32768 0 0 TMP-1'`, `'C 16 0 364 0 0 BT'`, `'C 16 0 263 0 0 IT'`).
    wfdb still reads the data column correctly (gain=10 is parsed) but hands
    back a garbled name. Recover the label as the last whitespace-separated
    token if the raw sig_name starts with `C ` (the leftover units character).

    Also handles the occasional `None` sig_name returned by wfdb on broken
    headers: return `""` so it falls through to the unknown bucket."""
    if not isinstance(name, str):
        return ""
    if name.startswith("C ") and " " in name:
        last = name.rsplit(" ", 1)[-1]
        if last in CHANNEL_TO_VAR_ID or last in SKIP_CHANNELS:
            return last
    return name

EVENT_DTYPE = np.dtype([
    ("time_ms", np.int64),
    ("var_id",  np.uint16),
    ("value",   np.float32),
])


def _0n_base_ms(hdr) -> int | None:
    if hdr.base_datetime is None:
        return None
    dt = hdr.base_datetime + relativedelta(years=30)
    return int(dt.replace(tzinfo=UTC).timestamp() * 1000)


def process_record(rec_id: str):
    """Extract mapped _0n events for one wfdb_record.

    Returns dict with keys:
      {"time", "var", "val"}          arrays (or all None on skip)
      {"per_var": {vid: n}, "per_channel": {name: n}}
      {"unknown": [names...], "reason": str|None, "sig_names": [...]}
    """
    cohort_prefix = rec_id.split("-")[0]
    rec_dir = f"{WFDB_ROOT}/{cohort_prefix}/{rec_id}"
    n_path = f"{rec_dir}/{rec_id}_0n"
    out = {"time": None, "var": None, "val": None,
           "per_var": {}, "per_channel": {}, "unknown": [],
           "reason": None, "sig_names": []}
    if not os.path.exists(n_path + ".hea"):
        out["reason"] = "no_0n_hea"
        return out
    try:
        hdr = wfdb.rdheader(n_path)
    except Exception as e:
        out["reason"] = f"hdr_err:{type(e).__name__}"
        return out
    out["sig_names"] = list(hdr.sig_name or [])
    base_ms = _0n_base_ms(hdr)
    if base_ms is None:
        out["reason"] = "no_base_datetime"
        return out
    fs = float(hdr.fs) if hdr.fs else 0.5
    if fs <= 0:
        out["reason"] = "bad_fs"
        return out

    # Pick columns we want
    wanted: list[tuple[int, str, int]] = []  # (orig_col_idx, name, var_id)
    unknown: list[str] = []
    for i, raw_name in enumerate(out["sig_names"]):
        name = _salvage_sig_name(raw_name)
        if not name or name in SKIP_CHANNELS:
            continue
        vid = CHANNEL_TO_VAR_ID.get(name)
        if vid is not None:
            wanted.append((i, name, vid))
        else:
            unknown.append(str(raw_name))
    out["unknown"] = unknown
    if not wanted:
        out["reason"] = "no_mapped_channels"
        return out

    try:
        rec = wfdb.rdrecord(n_path, physical=True,
                            channels=[i for i, _, _ in wanted])
    except Exception as e:
        out["reason"] = f"rdrec_err:{type(e).__name__}"
        return out
    p = rec.p_signal
    if p is None or p.size == 0:
        out["reason"] = "empty_signal"
        return out

    N = p.shape[0]
    dt_ms = 1000.0 / fs
    times_all = base_ms + (np.arange(N, dtype=np.int64) *
                           np.int64(round(dt_ms)))
    # Fractional correction when dt_ms is non-integer (e.g. fs=0.5 → 2000 exact;
    # fs=1 Hz → 1000 exact; leave float path for oddball rates)
    if abs(dt_ms - round(dt_ms)) > 1e-6:
        times_all = (base_ms + (np.arange(N) * dt_ms)).astype(np.int64)

    chunks_t: list[np.ndarray] = []
    chunks_var: list[np.ndarray] = []
    chunks_val: list[np.ndarray] = []
    per_var: dict[int, int] = {}
    per_channel: dict[str, int] = {}

    for col, (_, name, vid) in enumerate(wanted):
        v = p[:, col]
        # API.md: zero / negative = missing in _0n; also drop non-finite
        mask = np.isfinite(v) & (v > 0.0)
        if not mask.any():
            continue
        t = times_all[mask]
        vv = v[mask].astype(np.float32)
        chunks_t.append(t)
        chunks_val.append(vv)
        chunks_var.append(np.full(t.shape[0], vid, dtype=np.uint16))
        per_var[vid] = per_var.get(vid, 0) + int(t.shape[0])
        per_channel[name] = int(t.shape[0])

    if not chunks_t:
        out["reason"] = "all_filtered"
        return out

    out["time"] = np.concatenate(chunks_t)
    out["var"] = np.concatenate(chunks_var)
    out["val"] = np.concatenate(chunks_val)
    out["per_var"] = per_var
    out["per_channel"] = per_channel
    return out


def process_entity(row: dict, out_root: str = OUT_ROOT,
                   resume: bool = True) -> dict:
    entity_id = row["entity_id"]
    rec_ids = list(row["wfdb_records_all"]) if row.get("wfdb_records_all") is not None else []
    out_dir = f"{out_root}/{entity_id}"
    meta_path = f"{out_dir}/meta.json"
    events_path = f"{out_dir}/vitals_events.npy"

    if not os.path.exists(meta_path):
        return {"entity_id": entity_id, "status": "no_stage_b_meta"}

    # Resume safety net (worker-side). main() already pre-filters when resume=True.
    if resume and os.path.exists(events_path):
        try:
            with open(meta_path) as f:
                m = json.load(f)
            if m.get("stage_c_version", 0) >= 1:
                return {"entity_id": entity_id, "status": "resumed",
                        "n_events": int(m.get("vitals_0n", {}).get("n_events", 0))}
        except Exception:
            pass

    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except Exception as e:
        return {"entity_id": entity_id, "status": "meta_read_err",
                "error": f"{type(e).__name__}: {e}"}

    all_t, all_var, all_val = [], [], []
    per_var_total: dict[int, int] = {}
    per_channel_total: dict[str, int] = {}
    unknown_channels: set[str] = set()
    records_meta: list[dict] = []

    for rec_id in rec_ids:
        r = process_record(rec_id)
        rec_entry = {"record": rec_id, "sig_names": r["sig_names"]}
        if r["reason"] is not None:
            rec_entry["reason"] = r["reason"]
        if r["unknown"]:
            unknown_channels.update(r["unknown"])
            rec_entry["unknown"] = r["unknown"]
        records_meta.append(rec_entry)
        if r["time"] is None:
            continue
        all_t.append(r["time"])
        all_var.append(r["var"])
        all_val.append(r["val"])
        for k, v in r["per_var"].items():
            per_var_total[k] = per_var_total.get(k, 0) + v
        for k, v in r["per_channel"].items():
            per_channel_total[k] = per_channel_total.get(k, 0) + v

    if not all_t:
        events = np.empty(0, dtype=EVENT_DTYPE)
        np.save(events_path, events)
        meta["vitals_0n"] = {
            "n_events": 0,
            "n_records_scanned": len(rec_ids),
            "per_var_count": {},
            "per_channel_count": {},
            "unknown_channels_seen": sorted(unknown_channels),
            "records_meta": records_meta,
        }
        meta["stage_c_version"] = 1
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        return {"entity_id": entity_id, "status": "ok_empty",
                "n_events": 0, "n_records": len(rec_ids)}

    time_arr = np.concatenate(all_t)
    var_arr = np.concatenate(all_var)
    val_arr = np.concatenate(all_val)
    # Stable sort by time_ms, then by var_id for deterministic order at ties
    order = np.lexsort((var_arr, time_arr))
    events = np.empty(time_arr.shape[0], dtype=EVENT_DTYPE)
    events["time_ms"] = time_arr[order]
    events["var_id"]  = var_arr[order]
    events["value"]   = val_arr[order]
    np.save(events_path, events)

    meta["vitals_0n"] = {
        "n_events": int(events.shape[0]),
        "n_records_scanned": len(rec_ids),
        "per_var_count": {str(k): int(v) for k, v in sorted(per_var_total.items())},
        "per_channel_count": dict(sorted(per_channel_total.items())),
        "unknown_channels_seen": sorted(unknown_channels),
        "records_meta": records_meta,
    }
    meta["stage_c_version"] = 1
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    return {"entity_id": entity_id, "status": "ok",
            "n_events": int(events.shape[0]),
            "n_records": len(rec_ids),
            "n_vars": len(per_var_total)}


def _worker(args):
    row, out_root, resume = args
    try:
        return process_entity(row, out_root=out_root, resume=resume)
    except Exception as e:
        return {"entity_id": row.get("entity_id", "?"), "status": "error",
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(limit=3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--entity-id", type=str, default=None)
    ap.add_argument("--entities", type=str, default=None,
                    help="comma-separated entity_ids; overrides limit")
    ap.add_argument("--out-root", type=str, default=OUT_ROOT)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(f"{LOG_DIR}/stage_c_vitals.log")])
    log = logging.getLogger(__name__)
    log.info(f"Loading cohort parquet: {COHORT_PARQUET}")

    df = pl.read_parquet(COHORT_PARQUET).unique("entity_id", keep="first")
    if args.entity_id:
        df = df.filter(pl.col("entity_id") == args.entity_id)
    if args.entities:
        ids = [s.strip() for s in args.entities.split(",") if s.strip()]
        df = df.filter(pl.col("entity_id").is_in(ids))
    if args.limit and not (args.entity_id or args.entities):
        df = df.head(args.limit)

    rows = df.to_dicts()
    log.info(f"Entities to process (pre-resume): {len(rows)}  "
             f"workers: {args.workers}  resume={not args.no_resume}")

    if not args.no_resume:
        filtered = []
        skipped = 0
        for r in rows:
            ep = f"{args.out_root}/{r['entity_id']}"
            mpj = f"{ep}/meta.json"
            vpj = f"{ep}/vitals_events.npy"
            ok = False
            if os.path.exists(mpj) and os.path.exists(vpj):
                try:
                    with open(mpj) as f:
                        m = json.load(f)
                    if m.get("stage_c_version", 0) >= 1:
                        ok = True
                except Exception:
                    pass
            if ok:
                skipped += 1
            else:
                filtered.append(r)
        log.info(f"resume: skipping {skipped} entities already at stage_c_version>=1")
        rows = filtered

    if not rows:
        log.info("nothing to do")
        return

    worker_args = [(r, args.out_root, not args.no_resume) for r in rows]
    t0 = time.time()
    results: list[dict] = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, worker_args, chunksize=1)):
            results.append(r)
            if (i + 1) % 50 == 0 or i + 1 == len(rows):
                statuses: dict[str, int] = {}
                for x in results:
                    statuses[x["status"]] = statuses.get(x["status"], 0) + 1
                log.info(f"  {i+1}/{len(rows)}  elapsed {time.time()-t0:.0f}s  {statuses}")

    elapsed = time.time() - t0
    by_status: dict[str, list] = {}
    for r in results:
        by_status.setdefault(r["status"], []).append(r)
    summary = {
        "n_entities_processed": len(results),
        "elapsed_sec": round(elapsed, 1),
        "by_status": {s: len(v) for s, v in by_status.items()},
        "total_events": int(sum(r.get("n_events", 0) for r in results)),
        "workers": args.workers,
    }
    with open(SUMMARY_JSON, "w") as f:
        json.dump({"summary": summary,
                   "errors": [r for r in results if r["status"] in
                              {"error", "meta_read_err", "no_stage_b_meta"}][:50]},
                  f, indent=2, default=str)
    log.info(f"\n=== Stage C summary ===\n{json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
