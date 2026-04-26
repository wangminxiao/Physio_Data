"""PyTorch dataset wrappers over the canonical physio_data layout.

The canonical layout (see physio_data.schema and datasets/TEMPLATE_API.md):

    {processed_dir}/{entity_id}/
        {CHANNEL}.npy        [N_seg, samples_per_seg]  float16
        time_ms.npy          [N_seg]                    int64 UTC ms (seg starts)
        ehr_events.npy       structured (time_ms, seg_idx, var_id, value)
        meta.json
    {processed_dir}/manifest.json
    {processed_dir}/{pretrain,downstream}_splits.json

Public API:

    ChannelSpec(name, fs)
    PointEstimationDataset(...)         # one dataset
    MultiPhysioDataset(per_dataset_cfg) # several datasets, unified output
    physio_collate(batch)               # default collate for both
"""
from .windowing import Alignment, locate_window
from .dataset import ChannelSpec, PointEstimationDataset, Sample
from .multi import MultiPhysioDataset, physio_collate

__all__ = [
    "Alignment",
    "ChannelSpec",
    "MultiPhysioDataset",
    "PointEstimationDataset",
    "Sample",
    "locate_window",
    "physio_collate",
]
