#!/usr/bin/env python3
"""Recompute time_ms.npy of every Stage B entity with clock.Grid.dwc_rows (runs: wall-rule start, raw
elapsed inside) and flag DST-crossing runs in meta.json:

  meta.dst_crossing_runs   [{"kind": spring|fall, "rows": [a, b]}] -- runs whose first and last row lie
                           on different sides of a DST change (raw time continuous through it)
  meta.time_rule_version   "runs-1"
  meta.ehr_dst_note        set when such a run exists: EHR events after the change may be 1 h off (charted
                           vitals inherit the monitor's labels; labs / meds follow the hospital wall clock)

Only time_ms changes (waveforms do not depend on it); entities whose runs do not cross DST get identical
values. Resumable (time_rule_version == runs-1 -> skip).
"""
import argparse, json, os, sys
from datetime import timedelta
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clock  # noqa: E402
from common import cfg  # noqa: E402

VERSION = "runs-1"
_WAV, _RAW = None, None


def _init(w, r):
    global _WAV, _RAW
    _WAV, _RAW = w, r


def one(od):
    import h5py
    e = os.path.basename(od); mp = os.path.join(od, "meta.json")
    try:
        m = json.load(open(mp))
        if not m.get("stage_b", {}).get("done") or m.get("time_rule_version") == VERSION:
            return (e, "skip", 0)
        seg = json.load(open(os.path.join(_WAV, e + "__meta.json")))["seg_list"]
        o = np.argsort([s[2] for s in seg], kind="stable")
        st = np.array([seg[i][2] for i in o], float); blk = np.array([seg[i][0] for i in o])
        with h5py.File(os.path.join(_RAW, e + ".h5"), "r") as f:
            W, label = clock.parse_origin(json.loads(f.attrs[".meta"])["time_origin"])
        new = clock.Grid(W, st[0]).dwc_rows(st, blk, W).astype(np.int64)
        assert np.all(np.diff(new) > 0)
        old = np.load(os.path.join(od, "time_ms.npy"))
        changed = old.shape != new.shape or not np.array_equal(old, new)
        if changed:
            np.save(os.path.join(od, "time_ms.npy"), new)
        cross = []
        for a, b in clock.runs(st, blk):
            s0 = (W + timedelta(seconds=float(st[a]))).replace(tzinfo=clock.NY).tzname()
            s1 = (W + timedelta(seconds=float(st[b - 1]))).replace(tzinfo=clock.NY).tzname()
            if s0 != s1:
                cross.append({"kind": "spring" if s0 == "EST" else "fall", "rows": [int(a), int(b)]})
        m["dst_crossing_runs"] = cross
        if cross:
            m["ehr_dst_note"] = "a run crosses a DST change with continuous raw time; EHR events after it may be 1 h off"
        m["time_rule"] = "workzone/mladi/clock.py Grid.dwc_rows (run start by the monitor wall rule, raw elapsed within a run)"
        m["time_rule_version"] = VERSION
        json.dump(m, open(mp + ".tmp", "w"), indent=1, default=str); os.replace(mp + ".tmp", mp)
        return (e, "changed" if changed else "same", len(cross))
    except Exception as ex:
        return (e, f"error {type(ex).__name__}: {ex}", 0)


def main():
    C = cfg()
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=C["output_dir"]); ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--raw-h5-dir", default=C["raw_h5_dir"]); ap.add_argument("--pretrain-wav-dir", default=C["pretrain_wav_dir"])
    a = ap.parse_args()
    dirs = sorted(os.path.join(a.out, d) for d in os.listdir(a.out) if os.path.exists(os.path.join(a.out, d, "meta.json")))
    import collections
    cnt = collections.Counter(); ncross = 0
    with Pool(a.workers, initializer=_init, initargs=(a.pretrain_wav_dir, a.raw_h5_dir)) as pool:
        for e, st, nc in pool.imap_unordered(one, dirs, chunksize=8):
            cnt[st if not st.startswith("error") else "error"] += 1; ncross += nc > 0
            if st.startswith("error"):
                print(e[:8], st, flush=True)
    print(f"regrid: {dict(cnt)} | entities with a DST-crossing run {ncross}", flush=True)


if __name__ == "__main__":
    main()
