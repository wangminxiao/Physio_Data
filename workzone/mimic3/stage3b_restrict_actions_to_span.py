#!/usr/bin/env python3
"""Stage 3b post-step: restrict action events (var_id 200-299) to the waveform span.

Why this exists
---------------
stage3b_extract_actions.py aligns each event with

    seg_idx = np.searchsorted(time_ms, t, side="right") - 1

and keeps it when `0 <= seg_idx < n_seg`. That bound does NOT reject events after the
recording ends: any `t > time_ms[-1]` yields `seg_idx = n_seg - 1`, so post-waveform charting
is silently pinned to the last segment. On a fresh run the action rows therefore outnumber
the canonical store's by ~6x, and the surplus sits at one bogus segment.

The canonical store does not have that problem because stage3c_ehr_trajectory.py later
partitions the record into baseline / recent / events / future and keeps only in-waveform
actions in `ehr_events.npy`. Re-running stage3c is the heavier fix -- it also re-derives labs
and vitals from the stage2 parquets and needs ADMISSIONS.csv -- and is unnecessary when only
the action partition changed.

This step reproduces exactly the action half of what stage3c does. Verified on 50 entities:
restricting a fresh run to `[time_ms[0], time_ms[-1] + segment_duration]` reproduces the
canonical 200-206 rows byte-for-byte (50/50), with non-action rows untouched (50/50).

Idempotent: rows already inside the span are left alone, so re-running is a no-op.
Does NOT write ehr_recent/ehr_future -- out-of-span actions are dropped, not repartitioned.
Run stage3c if those partitions must also carry the new per-drug channels.

  python workzone/mimic3/stage3b_restrict_actions_to_span.py --root DIR [--limit N] [--dry-run]
"""
from __future__ import annotations
import argparse, json, os

import numpy as np

ACTION_LO, ACTION_HI = 200, 300      # [lo, hi)


def restrict_one(edir, write=True):
    """-> (status, n_dropped, n_kept_actions)"""
    tp = os.path.join(edir, "time_ms.npy")
    ep = os.path.join(edir, "ehr_events.npy")
    mp = os.path.join(edir, "meta.json")
    if not (os.path.isfile(tp) and os.path.isfile(ep)):
        return "skip_missing", 0, 0
    t = np.load(tp)
    if t.size == 0:
        return "skip_empty", 0, 0
    seg_ms = 30_000
    if os.path.isfile(mp):
        try:
            seg_ms = int(json.load(open(mp)).get("segment_duration_sec", 30)) * 1000
        except (ValueError, OSError):
            pass
    lo, hi = int(t[0]), int(t[-1]) + seg_ms

    e = np.load(ep)
    if e.size == 0:
        return "ok", 0, 0
    is_act = (e["var_id"] >= ACTION_LO) & (e["var_id"] < ACTION_HI)
    out_of_span = is_act & ((e["time_ms"] < lo) | (e["time_ms"] > hi))

    # Also drop exact duplicate ACTION rows. They arise when two source itemids are charted at
    # the same instant with the same value (observed: NS and LR both at fluid_rate 5.0), and
    # stage3c's partition merge removes them -- so keeping them would leave this output one row
    # off the canonical store for no informational gain. Restricted to actions: rows below 200
    # already reproduce canonical exactly and must not be touched.
    act_idx = np.flatnonzero(is_act & ~out_of_span)
    dup = np.zeros(e.shape, dtype=bool)
    if act_idx.size:
        _, first = np.unique(e[act_idx], return_index=True)
        keep_i = act_idx[np.sort(first)]
        d = np.ones(e.shape, dtype=bool)
        d[keep_i] = False
        dup = d & is_act & ~out_of_span

    drop = out_of_span | dup
    n_drop = int(drop.sum())
    n_keep = int((is_act & ~drop).sum())
    if n_drop and write:
        kept = e[~drop]
        kept.sort(order="time_ms")
        np.save(ep, kept)
        if os.path.isfile(mp):
            try:
                meta = json.load(open(mp))
                meta["n_ehr_events"] = int(len(kept))
                meta["n_action_events"] = n_keep
                json.dump(meta, open(mp, "w"), indent=2)
            except (ValueError, OSError):
                pass
    return "ok", n_drop, n_keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ents = sorted(n for n in os.listdir(args.root)
                  if "_" in n and os.path.isdir(os.path.join(args.root, n)))
    if args.limit:
        ents = ents[:args.limit]
    print(f"[restrict] {len(ents)} entities  "
          f"{'DRY-RUN (no write)' if args.dry_run else 'WRITE ehr_events.npy'}", flush=True)

    tot_drop = tot_keep = touched = 0
    hist = {}
    for i, name in enumerate(ents, 1):
        st, nd, nk = restrict_one(os.path.join(args.root, name), write=not args.dry_run)
        hist[st] = hist.get(st, 0) + 1
        tot_drop += nd
        tot_keep += nk
        touched += nd > 0
        if i % 1000 == 0:
            print(f"  {i}/{len(ents)} dropped={tot_drop} kept={tot_keep}", flush=True)
    print(f"[done] entities={len(ents)} touched={touched} "
          f"out_of_span_dropped={tot_drop} in_span_actions_kept={tot_keep} status={hist}",
          flush=True)


if __name__ == "__main__":
    main()
