"""Canonical format definitions for physio_data."""
import numpy as np

# Waveform arrays: [N_seg, samples_per_seg] float16, C-contiguous
WAVEFORM_DTYPE = np.float16

# Timestamps: [N_seg] int64, monotonically increasing, absolute milliseconds
TIME_DTYPE = np.int64

# EHR events: structured array, sorted by time_ms
EHR_EVENT_DTYPE = np.dtype([
    ('time_ms', 'int64'),     # actual measurement timestamp (absolute ms)
    ('seg_idx', 'int32'),     # aligned signal segment index
    ('var_id',  'uint16'),    # variable ID (lookup in var_registry.json)
    ('value',   'float32'),   # raw measured value
])

# Default segment duration
SEGMENT_DUR_SEC = 30
