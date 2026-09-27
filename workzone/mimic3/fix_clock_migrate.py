"""
MIMIC-III clock fix, step M1 — move `time_ms.npy` (and, provisionally, `ehr_hf.npy`) from the legacy local-epoch
base to the wall-clock base (datasets/CLOCK_FIX_PLAN_MIMIC_MOVER.md §1).

Legacy: `block_start_ms = wav_start.timestamp()` on a naive surrogate datetime -> epoch under America/New_York,
i.e. wall ms + 4 h (EDT surrogate date) or + 5 h (EST). Chart/lab events were written as wall ms, so every EHR
event sat 4-5 h before the waveform. The offset is one constant per entity (wav_start is evaluated once), so the
migration is an exact shift:

    delta_ms = wall_ms(naive wav_start) − legacy recording_start_ms        (= −4 h or −5 h)
    time_ms  += delta_ms ;  ehr_hf.time_ms += delta_ms (provisional; stage3b_extract_numerics rewrites it exactly)

Per entity meta.json gains: time_base="wall_clock", clock_shift_ms, clock_fix_version=1, recording_start_ms on
the new base, recording_start_legacy_ms, and a migrated source_path (old server prefix translated).
Idempotent: entities already carrying time_base == "wall_clock" are skipped.

Run (lab node):  python workzone/mimic3/fix_clock_migrate.py [--root STORE] [--limit N] [--entity-ids ...] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "workzone" / "common"))
from clock_utils import legacy_local_epoch_to_wall_ms, translate_path, NY  # noqa: E402


def migrate_entity(edir: Path, dry_run: bool) -> dict:
    mpath = edir / "meta.json"
    if not mpath.exists():
        return {"entity": edir.name, "status": "no_meta"}
    meta = json.loads(mpath.read_text())
    if meta.get("time_base") == "wall_clock":
        return {"entity": edir.name, "status": "already", "clock_shift_ms": meta.get("clock_shift_ms")}
    tm = np.load(edir / "time_ms.npy")
    legacy_start = int(meta.get("recording_start_ms", tm[0]))
    if legacy_start != int(tm[0]):
        return {"entity": edir.name, "status": "meta_mismatch", "detail": f"recording_start_ms {legacy_start} != time_ms[0] {int(tm[0])}"}
    wall_start, delta = legacy_local_epoch_to_wall_ms(legacy_start, NY)
    if delta not in (-4 * 3600000, -5 * 3600000):
        return {"entity": edir.name, "status": "unexpected_delta", "detail": f"delta_ms={delta}"}
    out = {"entity": edir.name, "status": "ok", "clock_shift_ms": delta, "n_seg": int(len(tm))}
    if dry_run:
        return out
    np.save(edir / "time_ms.npy", (tm + delta).astype(np.int64))
    hf_path = edir / "ehr_hf.npy"
    if hf_path.exists():
        hf = np.load(hf_path)
        if hf.size:
            hf["time_ms"] = hf["time_ms"] + delta
            np.save(hf_path, hf)
        out["ehr_hf_shifted"] = int(hf.size)
    meta["recording_start_legacy_ms"] = legacy_start
    meta["recording_start_ms"] = int(wall_start)
    meta["time_base"] = "wall_clock"
    meta["clock_shift_ms"] = int(delta)
    meta["clock_fix_version"] = 1
    meta["clock_fix_note"] = "2026-09 fix: legacy time_ms was naive surrogate time via datetime.timestamp() (America/New_York); now wall-clock ms like the EHR side"
    if meta.get("source_path"):
        meta["source_path_legacy"] = meta["source_path"]
        meta["source_path"] = translate_path(meta["source_path"])
    if "ehr_hf" not in meta and hf_path.exists():
        meta["hf_time_base"] = "wall_clock_provisional_shift"   # stage3b_extract_numerics replaces it by an exact recompute
    mpath.write_text(json.dumps(meta, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None, help="store root (default: server_paths.yaml mimic3.output_dir)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--entity-ids", nargs="*", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.root is None:
        cfg = yaml.safe_load((REPO_ROOT / "workzone" / "configs" / "server_paths.yaml").read_text())
        args.root = cfg["mimic3"]["output_dir"]
    root = Path(args.root)
    ids = args.entity_ids or sorted(d.name for d in root.iterdir() if d.is_dir() and (d / "meta.json").exists())
    if args.limit:
        ids = ids[:args.limit]
    print(f"root={root} entities={len(ids)} dry_run={args.dry_run}", flush=True)
    t0 = time.time(); by = {}; deltas = {}; bad = []
    for i, e in enumerate(ids, 1):
        r = migrate_entity(root / e, args.dry_run)
        by[r["status"]] = by.get(r["status"], 0) + 1
        if r["status"] == "ok":
            deltas[r["clock_shift_ms"]] = deltas.get(r["clock_shift_ms"], 0) + 1
        elif r["status"] not in ("already",):
            bad.append(r)
        if i % 500 == 0 or i == len(ids):
            print(f"  [{i}/{len(ids)}] {by} {time.time()-t0:.0f}s", flush=True)
    summary = {"stage": "fix_clock_migrate", "root": str(root), "dry_run": args.dry_run, "by_status": by,
               "delta_hours": {str(k / 3.6e6): v for k, v in deltas.items()}, "problems": bad[:50], "ran_at": time.strftime("%Y-%m-%d %H:%M")}
    out = REPO_ROOT / "workzone" / "outputs" / "mimic3"
    out.mkdir(parents=True, exist_ok=True)
    (out / ("fix_clock_migrate_summary%s.json" % ("_dryrun" if args.dry_run else ""))).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
