"""End-to-end smoke test against synthetic canonical-layout fixtures.

Run from repo root: `python workzone/torch_dataset_loader_dev/test_smoke.py`
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from physio_data.ehr_trajectory import (
    FNAME_BASELINE, FNAME_EVENTS, FNAME_FUTURE, FNAME_RECENT,
)
from physio_data.schema import EHR_EVENT_DTYPE, SEGMENT_DUR_SEC

from torch_dataset_loader import (
    ChannelSpec, MultiPhysioDataset, PointEstimationDataset, physio_collate,
)
from torch_dataset_loader.multi import DatasetConfig
from torch_dataset_loader.windowing import locate_window


SEG_MS = SEGMENT_DUR_SEC * 1000


def _build_entity(root, entity_id, *, n_seg, base_t_ms,
                  channels, gap_after=None, events=None):
    d = root / entity_id
    d.mkdir(parents=True, exist_ok=True)

    time_ms = np.arange(n_seg, dtype=np.int64) * SEG_MS + base_t_ms
    if gap_after is not None:
        time_ms[gap_after + 1:] += 60_000
    np.save(d / "time_ms.npy", time_ms)

    for name, fs in channels.items():
        samples = fs * SEGMENT_DUR_SEC
        arr = np.random.default_rng(42).standard_normal(
            (n_seg, samples)
        ).astype(np.float16)
        np.save(d / f"{name}.npy", arr)

    ev = np.empty(len(events or []), dtype=EHR_EVENT_DTYPE)
    for i, (t, vid, val) in enumerate(events or []):
        seg = int(np.searchsorted(time_ms, t, side="right")) - 1
        ev[i] = (t, max(seg, 0), vid, val)
    ev.sort(order="time_ms")
    np.save(d / FNAME_EVENTS, ev)
    empty = np.empty(0, dtype=EHR_EVENT_DTYPE)
    np.save(d / FNAME_BASELINE, empty)
    np.save(d / FNAME_RECENT, empty)
    np.save(d / FNAME_FUTURE, empty)

    (d / "meta.json").write_text(json.dumps({
        "n_seg": int(n_seg),
        "channels": list(channels),
        "samples_per_seg": {k: v * SEGMENT_DUR_SEC for k, v in channels.items()},
        "n_events": int(len(ev)),
    }))


def _scenario(tmp):
    a = tmp / "cohortA" / "processed"
    b = tmp / "cohortB" / "processed"
    base_a = 1_700_000_000_000
    _build_entity(a, "patientA1",
        n_seg=10, base_t_ms=base_a,
        channels={"PLETH40": 40, "II120": 120},
        events=[
            (base_a + 60_000,  0, 4.2),
            (base_a + 150_000, 0, 4.5),
            (base_a + 270_000, 0, 4.0),
            (base_a + 90_000,  100, 88),
            (base_a + 200_000, 100, 92),
        ])
    _build_entity(a, "patientA2",
        n_seg=8, base_t_ms=base_a + 10_000_000,
        channels={"PLETH40": 40, "II120": 120},
        gap_after=3,
        events=[
            (base_a + 10_000_000 + 60_000,  0, 3.8),
            (base_a + 10_000_000 + 200_000, 0, 4.1),
        ])
    base_b = 1_800_000_000_000
    _build_entity(b, "patientB1",
        n_seg=12, base_t_ms=base_b,
        channels={"PLETH40": 40, "II120": 120},
        events=[(base_b + 90_000, 0, 5.1), (base_b + 240_000, 0, 4.7)])
    return tmp


def test_locate_window_alignments():
    time_ms = np.arange(10, dtype=np.int64) * SEG_MS + 1_000_000_000_000
    t_label = int(time_ms[5]) + 5_000

    span_end = locate_window(time_ms, t_label, 60_000, "end")
    assert span_end is not None and span_end.t_end_ms == t_label

    span_mid = locate_window(time_ms, t_label, 60_000, "middle")
    assert span_mid is not None and span_mid.t_start_ms == t_label - 30_000

    span_start = locate_window(time_ms, t_label, 60_000, "start")
    assert span_start is not None and span_start.t_start_ms == t_label

    assert locate_window(time_ms, int(time_ms[0]) - 1_000, 60_000, "end") is None
    assert locate_window(time_ms, int(time_ms[-1]) + SEG_MS - 100, 60_000, "start") is None
    print("  locate_window alignment + OOB OK")


def test_locate_window_gap_rejection():
    time_ms = np.arange(6, dtype=np.int64) * SEG_MS + 1_000_000_000_000
    time_ms[3:] += 60_000
    t_label = int(time_ms[3]) + 1_000
    assert locate_window(time_ms, t_label, 120_000, "end") is None
    t_label = int(time_ms[5]) + 5_000
    assert locate_window(time_ms, t_label, 30_000, "end") is not None
    print("  gap rejection OK")


def test_dataset_smoke(tmp):
    a = tmp / "cohortA" / "processed"
    ds = PointEstimationDataset(
        processed_dir=a, target_var_ids=0,
        channels=[ChannelSpec("PLETH40", 40), ChannelSpec("II120", 120)],
        window_seconds=60.0, alignment="end",
        physio_clip=(1.5, 9.0), dataset_name="cohortA",
    )
    print(f"  cohort A potassium samples: {ds.stats()}")
    assert ds.stats()["n_samples"] >= 1
    s = ds[0]
    assert s["waveforms"]["PLETH40"].shape[0] == 40 * 60
    assert s["waveforms"]["II120"].shape[0] == 120 * 60
    print("  single-sample shapes OK")


def test_multi_dataloader(tmp):
    a = tmp / "cohortA" / "processed"
    b = tmp / "cohortB" / "processed"
    ds = MultiPhysioDataset(
        cohorts=[
            DatasetConfig(name="cohortA", processed_dir=a),
            DatasetConfig(name="cohortB", processed_dir=b),
        ],
        target_var_ids=[0],
        channels=[ChannelSpec("PLETH40", 40), ChannelSpec("II120", 120)],
        window_seconds=60.0, alignment="middle", physio_clip=(1.5, 9.0),
    )
    print(f"  multi stats: {ds.stats()}")
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=physio_collate)
    batches = list(loader)
    assert batches
    bt = batches[0]
    assert bt["waveforms"]["PLETH40"].shape[1] == 40 * 60
    assert bt["label"].dtype == torch.float32
    print(f"  batched PLETH40: {tuple(bt['waveforms']['PLETH40'].shape)}, label={bt['label']}")


def main():
    print("[1] window unit tests")
    test_locate_window_alignments()
    test_locate_window_gap_rejection()

    tmp = Path(tempfile.mkdtemp(prefix="physio_loader_smoke_"))
    try:
        _scenario(tmp)
        print(f"[2] dataset smoke (fixtures at {tmp})")
        test_dataset_smoke(tmp)
        print("[3] multi + DataLoader")
        test_multi_dataloader(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
