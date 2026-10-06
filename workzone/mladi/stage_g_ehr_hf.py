#!/usr/bin/env python3
"""MLADI Stage G: ehr_hf.npy, the MIMIC-III-compatible monitor-numerics events, from vitals_hf.npy.

Same convention as workzone/ucsf/stage_g_ehr_hf.py (--cadence pair --include-nbp, as ucsf_all), with the slot
layout read from meta.vitals_hf (MLADI: 1-s slots, 30 per segment):
  every 2 segments = 60 s = one tick; for pair (2k, 2k+1) the reading nearest the pair-end time
  t_q = time_ms[2k+1] + 30 s within +-30 s is written at seg_idx = 2k+1, time_ms = t_q. Candidates are the
  slots of segment 2k+1 and of segment 2k+2 (slot centres); ties go to the earlier slot.
Variables: 150 HR_hf, 151 SpO2_hf, 152 RR_hf, 153-155 ABP s/d/m_hf, 156 PULSE_hf (+ 113 PR_art with
--include-pr-art), then nbp_events.npy appended (true cuff times). meta.json += ehr_hf {...}.
Kept separate from the UCSF script because the PSC env has no PyYAML / polars.

    python workzone/mladi/stage_g_ehr_hf.py --limit 5 --workers 4
"""
from __future__ import annotations
import argparse, json, os, sys, time, zlib
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import cfg, EHR_EVENT_DTYPE  # noqa: E402

T0 = time.time()
EHR_HF_VERSION = 1
SEG_MS = 30_000
MIMIC_IDS = (150, 151, 152, 153, 154, 155, 156)
_ARGS = None


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:6.0f}s]", *a, flush=True)


def project_pair(col: np.ndarray, time_ms: np.ndarray, slot_ms: int, tol_ms: float = 30_000.0):
    """col [n_seg, slots] -> (seg_idx, time_ms, value) under the pair-end convention (vectorised form of
    the UCSF loop: candidates = segment s then segment s+1, nearest finite slot centre within tol)."""
    n_seg, S = col.shape
    s = np.arange(1, n_seg, 2)
    if s.size == 0:
        return np.empty(0, int), np.empty(0, np.int64), np.empty(0, np.float32)
    t_q = time_ms[s] + SEG_MS
    off = np.arange(S) * slot_ms + slot_ms / 2
    nxt = np.minimum(s + 1, n_seg - 1)
    has_nxt = (s + 1) < n_seg
    cv = np.concatenate([col[s], np.where(has_nxt[:, None], col[nxt], np.nan)], axis=1)
    cc = np.concatenate([time_ms[s][:, None] + off[None], time_ms[nxt][:, None] + off[None]], axis=1)
    d = np.abs(cc - t_q[:, None]).astype(np.float64)
    d[~np.isfinite(cv)] = np.inf
    j = np.argmin(d, axis=1)
    dj = d[np.arange(s.size), j]
    ok = dj <= tol_ms
    return s[ok], t_q[ok].astype(np.int64), cv[np.arange(s.size), j][ok].astype(np.float32)


def one(od):
    eid = os.path.basename(od); mp = os.path.join(od, "meta.json")
    try:
        meta = json.load(open(mp))
        if meta.get("ehr_hf", {}).get("version") == EHR_HF_VERSION and os.path.exists(os.path.join(od, "ehr_hf.npy")):
            return {"entity_id": eid, "skipped": True}
        vh = meta.get("vitals_hf")
        if not vh or not os.path.exists(os.path.join(od, "vitals_hf.npy")):
            return {"entity_id": eid, "error": "no vitals_hf"}
        hf = np.load(os.path.join(od, "vitals_hf.npy"))
        tms = np.load(os.path.join(od, "time_ms.npy"))
        slot_ms = int(round(float(vh["slot_sec"]) * 1000))
        var_ids = list(vh["var_ids"])
        want = [v for v in var_ids if v in MIMIC_IDS or (_ARGS["pr_art"] and v == 113)]
        parts = []
        for vid in want:
            si, tq, v = project_pair(hf[:, :, var_ids.index(vid)], tms, slot_ms)
            a = np.zeros(si.size, EHR_EVENT_DTYPE); a["time_ms"] = tq; a["seg_idx"] = si; a["var_id"] = vid; a["value"] = v
            parts.append(a)
        if _ARGS["nbp"] and os.path.exists(os.path.join(od, "nbp_events.npy")):
            parts.append(np.load(os.path.join(od, "nbp_events.npy")))
        arr = np.concatenate(parts) if parts else np.empty(0, EHR_EVENT_DTYPE)
        arr.sort(order=["time_ms", "var_id"])
        np.save(os.path.join(od, "ehr_hf.npy"), arr)
        counts = {int(u): int((arr["var_id"] == u).sum()) for u in np.unique(arr["var_id"])}
        meta["ehr_hf"] = {"version": EHR_HF_VERSION, "file": "ehr_hf.npy", "dtype": "EHR_EVENT_DTYPE", "source": "vitals_hf.npy",
                          "cadence": "pair", "slot_sec": vh["slot_sec"],
                          "convention": "mimic3 stage3b pair-end: seg_idx=2k+1, t=time_ms[2k+1]+30s, nearest reading within 30 s",
                          "var_ids": sorted(counts), "n_events": int(arr.size), "per_var_count": counts,
                          "includes_nbp_events": bool(_ARGS["nbp"]), "includes_pr_art": bool(_ARGS["pr_art"])}
        json.dump(meta, open(mp + ".tmp", "w"), indent=1, default=str); os.replace(mp + ".tmp", mp)
        return {"entity_id": eid, "n": int(arr.size)}
    except Exception as ex:
        return {"entity_id": eid, "error": f"{type(ex).__name__}: {ex}"}


def _init(args):
    global _ARGS
    _ARGS = args


def main():
    C = cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=C["output_dir"]); ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--shard", default="0/1")
    ap.add_argument("--include-pr-art", action="store_true"); ap.add_argument("--no-nbp", action="store_true")
    a = ap.parse_args()
    k, n_sh = (int(x) for x in a.shard.split("/"))
    dirs = sorted(os.path.join(a.out, d) for d in os.listdir(a.out)
                  if zlib.crc32(d.encode()) % n_sh == k and os.path.exists(os.path.join(a.out, d, "vitals_hf.npy")))[: a.limit or None]
    log(f"{len(dirs)} entities (shard {a.shard})")
    err = done = tot = 0
    with Pool(a.workers, initializer=_init, initargs=({"pr_art": a.include_pr_art, "nbp": not a.no_nbp},)) as pool:
        for i, r in enumerate(pool.imap_unordered(one, dirs, chunksize=4), 1):
            if "error" in r:
                err += 1; log(f"ERROR {r['entity_id'][:8]}..: {r['error']}")
            elif not r.get("skipped"):
                done += 1; tot += r["n"]
            if i % 1000 == 0 or i == len(dirs):
                log(f"{i}/{len(dirs)} | written {done} | events {tot:,} | errors {err}")
    log("done")


if __name__ == "__main__":
    main()
