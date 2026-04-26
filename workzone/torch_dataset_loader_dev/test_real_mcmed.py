"""Validate torch_dataset_loader against a real MC_MED processed dir.

Usage (run on the server where the canonical processed dir lives):

    python workzone/torch_dataset_loader_dev/test_real_mcmed.py \
        --processed_dir /opt/localdata100tb/physio_data/mcmed/processed \
        --target_var_id 100 --window_seconds 30 --alignment end \
        --batch_size 8 --max_entities 50

The script:
  1. lists entities under processed_dir,
  2. builds a PointEstimationDataset for the requested var,
  3. prints index stats,
  4. round-trips a few batches through DataLoader + physio_collate,
  5. sanity-checks shapes, dtypes, finite values, and channel sample counts.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from torch_dataset_loader import (
    ChannelSpec, PointEstimationDataset, physio_collate,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", required=True, type=Path)
    ap.add_argument("--target_var_id", type=int, default=100,
                    help="100=HR, 101=SpO2, 0=Potassium ...")
    ap.add_argument("--channels", nargs="+", default=["PLETH40:40", "II120:120"],
                    help="canonical_name:fs pairs")
    ap.add_argument("--window_seconds", type=float, default=30.0)
    ap.add_argument("--alignment", choices=["end", "middle", "start"], default="end")
    ap.add_argument("--split_file", type=Path, default=None)
    ap.add_argument("--split_name", type=str, default=None)
    ap.add_argument("--max_entities", type=int, default=None,
                    help="cap entities for fast iteration")
    ap.add_argument("--min_label_interval_ms", type=int, default=None)
    ap.add_argument("--physio_clip", type=float, nargs=2, default=None)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--n_batches", type=int, default=3)
    args = ap.parse_args()

    channels = []
    for c in args.channels:
        name, fs = c.split(":")
        channels.append(ChannelSpec(name=name, fs=int(fs)))

    entities = None
    if args.max_entities is not None and args.split_file is None:
        all_ents = sorted(p.name for p in args.processed_dir.iterdir()
                          if p.is_dir() and (p / "time_ms.npy").exists())
        entities = all_ents[: args.max_entities]
        print(f"[info] capped to first {len(entities)} entities")

    print(f"[1] building PointEstimationDataset(var={args.target_var_id}, "
          f"window={args.window_seconds}s, align={args.alignment})")
    t0 = time.time()
    ds = PointEstimationDataset(
        processed_dir=args.processed_dir,
        target_var_ids=args.target_var_id,
        channels=channels,
        window_seconds=args.window_seconds,
        alignment=args.alignment,
        entities=entities,
        split_file=args.split_file,
        split_name=args.split_name,
        physio_clip=tuple(args.physio_clip) if args.physio_clip else None,
        min_label_interval_ms=args.min_label_interval_ms,
        dataset_name="mcmed",
    )
    print(f"    built in {time.time()-t0:.1f}s; stats={ds.stats()}")
    if len(ds) == 0:
        raise SystemExit("no samples — check var_id, splits, channels.")

    print("[2] sampling __getitem__[0]")
    s = ds[0]
    for k, v in s["waveforms"].items():
        print(f"    {k}: shape={tuple(v.shape)} dtype={v.dtype} "
              f"finite={torch.isfinite(v).float().mean().item():.3f}")
    print(f"    label={s['label'].item():.3f} var_id={s['var_id']} "
          f"entity={s['entity_id']}")

    print(f"[3] DataLoader (bs={args.batch_size}, workers={args.num_workers})")
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=physio_collate,
    )
    expected = {c.name: int(round(args.window_seconds * c.fs)) for c in channels}
    seen = 0
    t0 = time.time()
    for i, batch in enumerate(loader):
        for c in channels:
            shape = tuple(batch["waveforms"][c.name].shape)
            assert shape == (len(batch["label"]), expected[c.name]), \
                f"{c.name}: got {shape}, expected (B, {expected[c.name]})"
        assert batch["label"].dtype == torch.float32
        assert batch["alignment"] == args.alignment
        seen += 1
        if i + 1 >= args.n_batches:
            break
    dt = time.time() - t0
    print(f"    {seen} batches OK ({dt:.2f}s, {seen*args.batch_size/dt:.1f} samples/s)")
    print("REAL-DATA TEST PASSED")


if __name__ == "__main__":
    main()
