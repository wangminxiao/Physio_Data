#!/usr/bin/env python3
"""One-off migration: Emory chart_vitals + ehr_* had CVP mislabeled as
var_id 115 (collided with var_registry's SPO2_pulse_rate). Flip to 107.

Operates in-place. Idempotent: runs once, subsequent runs are no-ops because
the 115→107 lookup yields nothing after the first pass.
"""
import os
import sys
import json
import numpy as np

OUT_ROOT = "/opt/localdata100tb/physio_data/emory"
OLD = 115
NEW = 107

FILES_TO_PATCH = [
    "chart_vitals_events.npy",
    "ehr_baseline.npy",
    "ehr_recent.npy",
    "ehr_events.npy",
    "ehr_future.npy",
]


def patch_file(path: str) -> int:
    if not os.path.exists(path):
        return 0
    arr = np.load(path)
    if arr.size == 0 or "var_id" not in arr.dtype.names:
        return 0
    mask = arr["var_id"] == OLD
    n = int(mask.sum())
    if n == 0:
        return 0
    arr["var_id"][mask] = NEW
    np.save(path, arr)
    return n


def patch_meta(meta_path: str, n_chart: int, n_ehr_events: int) -> None:
    """Update per_var_count dicts after renumbering."""
    if not os.path.exists(meta_path):
        return
    with open(meta_path) as f:
        meta = json.load(f)
    # chart_vitals.per_var_count
    cv = meta.get("chart_vitals")
    if cv and "per_var_count" in cv:
        d = cv["per_var_count"]
        if str(OLD) in d:
            v = d.pop(str(OLD))
            d[str(NEW)] = d.get(str(NEW), 0) + v
            cv["per_var_count"] = dict(sorted(d.items(), key=lambda x: int(x[0])))
    # also bump version flag so we know migration happened
    meta["cvp_migrated_115_to_107"] = True
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)


def main():
    n_entities = 0
    n_patched_entities = 0
    total_renames = {k: 0 for k in FILES_TO_PATCH}
    t0_entities = sorted(os.listdir(OUT_ROOT))
    print(f"scanning {len(t0_entities)} entries under {OUT_ROOT}")
    for i, d in enumerate(t0_entities, 1):
        edir = os.path.join(OUT_ROOT, d)
        if not os.path.isdir(edir):
            continue
        meta = os.path.join(edir, "meta.json")
        if not os.path.exists(meta):
            continue
        n_entities += 1
        n_here = {}
        for fname in FILES_TO_PATCH:
            p = os.path.join(edir, fname)
            n = patch_file(p)
            if n:
                n_here[fname] = n
                total_renames[fname] += n
        if n_here:
            n_patched_entities += 1
            patch_meta(meta, n_chart=n_here.get("chart_vitals_events.npy", 0),
                       n_ehr_events=n_here.get("ehr_events.npy", 0))
        if i % 1000 == 0 or i == len(t0_entities):
            print(f"  [{i}/{len(t0_entities)}] entities={n_entities} "
                  f"patched={n_patched_entities}", flush=True)
    print("--- Summary ---")
    print(f"entities scanned: {n_entities}")
    print(f"entities with CVP events renamed: {n_patched_entities}")
    for k, v in total_renames.items():
        print(f"  {k}: {v} var_id flips")


if __name__ == "__main__":
    main()
