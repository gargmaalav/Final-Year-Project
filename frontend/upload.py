"""
parse_uploaded_csv / load_uploaded_segment
============================================

Turns a user-uploaded CSV into a core.Segment the rest of the pipeline
(models/classify.py, models/forecast.py) already knows how to consume,
by reusing zenodo_biceps/loader.py's fully generic loader.to_segment()
(resample + bandpass, no dependency on the Zenodo file format).

Accepted formats:
  - two columns (time_s, signal): native sample rate is inferred from the
    time column.
  - one column (signal only): the caller must supply sample_rate_hz.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

import loader  # noqa: E402  to_segment (sys.path set up by the importing app)

TARGET_FS = 250  # matches models/classify.py's trained config


class UploadError(Exception):
    pass


def parse_uploaded_csv(file, sample_rate_hz: float | None) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (t_native, x_native, fs_native) from an uploaded CSV file-like."""
    try:
        df = pd.read_csv(io.BytesIO(file.getvalue()))
    except Exception as e:
        raise UploadError(f"couldn't read that as a CSV ({e})") from e

    if df.shape[1] == 0 or df.shape[0] < 2:
        raise UploadError("the file has no usable rows")

    if df.shape[1] >= 2:
        t = df.iloc[:, 0].to_numpy(float)
        x = df.iloc[:, 1].to_numpy(float)
        dt = np.median(np.diff(t))
        if not np.isfinite(dt) or dt <= 0:
            raise UploadError("the first column doesn't look like an "
                              "increasing time-in-seconds column")
        fs_native = 1.0 / dt
    else:
        if not sample_rate_hz:
            raise UploadError(
                "this file has a single column, so I need the recording's "
                "sample rate (Hz) -- set it in the sidebar and try again")
        x = df.iloc[:, 0].to_numpy(float)
        fs_native = float(sample_rate_hz)
        t = np.arange(x.size) / fs_native

    return t, x, fs_native


def load_uploaded_segment(t: np.ndarray, x: np.ndarray, fs_native: float,
                          target_fs: int = TARGET_FS):
    """Resample + bandpass an uploaded recording into a core.Segment."""
    return loader.to_segment(t, x, fs=fs_native, target_fs=target_fs, bandpass=True)
