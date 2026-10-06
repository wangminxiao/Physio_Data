#!/usr/bin/env python3
"""MLADI Stage C2: monitor NBP streams that hold twin copies.

In ~3-4 % of 2020+ entities every cuff reading (s, d, pulse) reappears 240 or 300 min apart in
data/numerics NBP.* (explore/dup_check.py, nbp_twins_raw.py; waveforms and other numerics are not duplicated).
Which copy is the measurement cannot be decided reliably (cuff pulse vs HR leans to the earlier copy 425:250
with 1,110 undecided; the EHR clock from charted pulse vs monitor HR is 0 in 25/30, which makes the copy at
charting time the real one). So for a flagged entity ALL its monitor NBP events (157-159) are removed from
nbp_events.npy; the removed rows go to nbp_events_twins.npy, and meta.nbp_twins records the lag, the share of
readings in a pair and the action. meta.ehr_hf is cleared so Stage G rebuilds that entity's ehr_hf.npy.

Flag: >= 3 pairs and >= 20 % of readings in a pair, a pair being equal (s, d) at lag 240 or 300 min (+-90 s).

    python workzone/mladi/stage_c2_nbp_twins.py [--dry-run]
"""
from __future__ import annotations
import argparse, collections, json, os, sys, time
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import cfg, EHR_EVENT_DTYPE  # noqa: E402

T0 = time.time()
VERSION = "mladi-c2-1"
LAGS_MIN = (240, 300)
_DRY = False


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:6.0f}s]", *a, flush=True)


def twins(nb):
    """-> (best lag, n_pairs, share of readings in a pair) for (157, 158) readings."""
    s = nb[nb["var_id"] == 157]
    if s.size < 3:
        return None, 0, 0.0
    dmap = {int(r["time_ms"]): float(r["value"]) for r in nb[nb["var_id"] == 158]}
    t = s["time_ms"].astype(np.int64); v = s["value"].astype(float)
    dv = np.array([dmap.get(int(x), np.nan) for x in t])
    best = (None, 0, 0.0)
    for L in LAGS_MIN:
        inpair = np.zeros(t.size, bool); n = 0
        j = np.searchsorted(t, t + L * 60000 - 90000)
        for i in range(t.size):
            k = j[i]
            while k < t.size and t[k] <= t[i] + L * 60000 + 90000:
                if v[k] == v[i] and (dv[k] == dv[i] or not (np.isfinite(dv[k]) and np.isfinite(dv[i]))):
                    inpair[i] = inpair[k] = True; n += 1; break
                k += 1
        sh = float(inpair.mean())
        if sh > best[2]:
            best = (L, n, sh)
    return best


def one(od):
    eid = os.path.basename(od); mp = os.path.join(od, "meta.json")
    try:
        meta = json.load(open(mp))
        if meta.get("nbp_twins", {}).get("version") == VERSION:
            return {"entity_id": eid, "skipped": True, "flag": meta["nbp_twins"]["flag"]}
        p = os.path.join(od, "nbp_events.npy")
        if os.path.exists(os.path.join(od, "nbp_events_twins.npy")):
            nb = np.load(os.path.join(od, "nbp_events_twins.npy"))      # a previous run removed them: re-judge the originals
        else:
            nb = np.load(p)
        lag, n_pairs, share = twins(nb)
        flag = bool(n_pairs >= 3 and share >= 0.2)
        info = {"version": VERSION, "flag": flag, "lag_min": lag, "n_pairs": int(n_pairs), "share_in_pair": round(share, 3),
                "n_readings": int((nb["var_id"] == 157).sum()),
                "action": "all monitor NBP events moved to nbp_events_twins.npy" if flag else "none"}
        if _DRY:
            return {"entity_id": eid, "flag": flag, "info": info}
        if flag:
            np.save(os.path.join(od, "nbp_events_twins.npy"), nb)
            np.save(p, np.empty(0, EHR_EVENT_DTYPE))
            meta.pop("ehr_hf", None)
        meta["nbp_twins"] = info
        json.dump(meta, open(mp + ".tmp", "w"), indent=1, default=str); os.replace(mp + ".tmp", mp)
        return {"entity_id": eid, "flag": flag, "info": info}
    except Exception as ex:
        return {"entity_id": eid, "error": f"{type(ex).__name__}: {ex}"}


def _init(dry):
    global _DRY
    _DRY = dry


def main():
    C = cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=C["output_dir"]); ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    dirs = sorted(os.path.join(a.out, d) for d in os.listdir(a.out) if os.path.exists(os.path.join(a.out, d, "nbp_events.npy")))[: a.limit or None]
    log(f"{len(dirs)} entities{' (dry run)' if a.dry_run else ''}")
    n_flag = err = 0; lags = collections.Counter(); flagged = []
    with Pool(a.workers, initializer=_init, initargs=(a.dry_run,)) as pool:
        for i, r in enumerate(pool.imap_unordered(one, dirs, chunksize=8), 1):
            if "error" in r:
                err += 1; log(f"ERROR {r['entity_id'][:8]}..: {r['error']}")
            elif r.get("flag"):
                n_flag += 1; flagged.append(r["entity_id"])
                if "info" in r:
                    lags[r["info"]["lag_min"]] += 1
            if i % 2000 == 0 or i == len(dirs):
                log(f"{i}/{len(dirs)} | flagged {n_flag} | lags {dict(lags)} | errors {err}")
    C_int = cfg()["intermediate_dir"]; os.makedirs(C_int, exist_ok=True)
    json.dump(sorted(flagged), open(os.path.join(C_int, "stage_c2_nbp_twins_flagged.json"), "w"), indent=0)
    log("done")


if __name__ == "__main__":
    main()
