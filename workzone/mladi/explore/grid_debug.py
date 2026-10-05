"""Where does the monitor-rule grid go backwards? For included entities whose base starts with PREFIX
(default 20190628), compute time_ms as Stage B does and print, around every non-increase, the raw
seconds, the wall clock, its DST state and the grid ms."""
import json, os, sys
from datetime import timedelta
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import clock
from common import cfg
C = cfg(); pre = sys.argv[1] if len(sys.argv) > 1 else "20190628"
import h5py
inv = [json.loads(l) for l in open(os.path.join(C["intermediate_dir"], "stage_a", "inventory.jsonl"))]
for r in inv:
    if not (r.get("included") and r["entity_id"].startswith(pre)):
        continue
    e = r["entity_id"]
    seg = json.load(open(os.path.join(C["pretrain_wav_dir"], e + "__meta.json")))["seg_list"]
    st = np.sort(np.array([s[2] for s in seg], float))
    with h5py.File(os.path.join(C["raw_h5_dir"], e + ".h5"), "r") as f:
        W, label = clock.parse_origin(json.loads(f.attrs[".meta"])["time_origin"])
    g = clock.Grid(W, st[0]); t = g.dwc(st, W)
    bad = np.flatnonzero(np.diff(t) <= 0)
    print(f"{e[:8]}.. origin {W.year} {label} | rows {st.size} | non-increasing at {bad.size} places", flush=True)
    for b in bad[:3]:
        for i in range(max(0, b - 3), min(st.size, b + 4)):
            w = W + timedelta(seconds=float(st[i]))
            st_ = w.replace(tzinfo=clock.NY).tzname()
            print(f"   row {i}: raw {st[i]:.1f} s | wall {w:%Y-%m-%d %H:%M:%S} {st_} | grid {t[i]} | step {(t[i] - t[i - 1]) / 1000 if i else 0:.0f} s | raw step {st[i] - st[i - 1] if i else 0:.0f} s", flush=True)
