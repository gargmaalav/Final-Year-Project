"""
parse_uploaded_csv / load_uploaded_segment
============================================

Turns a user-uploaded CSV into a core.Segment the rest of the pipeline
(models/classify.py, models/fatigue_forecast.py) already knows how to consume,
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


def _read_csv(raw: bytes):
    """Read the CSV, auto-detecting whether it has a header row.

    pandas assumes header=0. A headerless numeric export (very common straight
    off a DAQ) would silently lose its first sample to the header, so sniff the
    first row and re-read with header=None when it parses as numbers.
    """
    try:
        first = pd.read_csv(io.BytesIO(raw), nrows=1, header=None)
    except Exception as e:
        raise UploadError(f"couldn't read that as a CSV ({e})") from e

    headerless = True
    for value in first.iloc[0].tolist():
        try:
            float(value)
        except (TypeError, ValueError):
            headerless = False
            break

    try:
        return pd.read_csv(io.BytesIO(raw), header=None if headerless else 0)
    except Exception as e:
        raise UploadError(f"couldn't read that as a CSV ({e})") from e


def _numeric_column(df, index: int, label: str) -> np.ndarray:
    """Coerce one column to float, with a message naming what went wrong.

    This is why non-EMG CSVs (dates, text, IDs) used to crash the app with a
    raw ValueError instead of a readable message.
    """
    col = pd.to_numeric(df.iloc[:, index], errors="coerce").to_numpy(float)
    if np.isnan(col).all():
        raise UploadError(
            f"the {label} column contains no numbers -- this does not look "
            "like an EMG recording. Expected either 'time_s, signal' or a "
            "single column of signal values.")
    if np.isnan(col).any():
        n_bad = int(np.isnan(col).sum())
        raise UploadError(
            f"the {label} column has {n_bad} value(s) that aren't numbers "
            "(blank rows, text, or units mixed into the data?)")
    return col


def parse_uploaded_csv(file, sample_rate_hz: float | None) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (t_native, x_native, fs_native) from an uploaded CSV file-like."""
    df = _read_csv(file.getvalue())

    if df.shape[1] == 0 or df.shape[0] < 2:
        raise UploadError("the file has no usable rows")

    if df.shape[1] >= 2:
        t = _numeric_column(df, 0, "time")
        x = _numeric_column(df, 1, "signal")
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
        x = _numeric_column(df, 0, "signal")
        fs_native = float(sample_rate_hz)
        t = np.arange(x.size) / fs_native

    if not np.isfinite(fs_native) or fs_native <= 0:
        raise UploadError(f"inferred an impossible sample rate ({fs_native} Hz)")

    # Snap a near-integer inferred rate to the integer. Inferring 1259.29 Hz
    # where the rig is 1259 Hz builds a slightly different resampling grid from
    # loader.load_biceps_segment's, which drifts ~13 samples over 218 s and
    # moved one MDF reading by 4 Hz between the two paths.
    if abs(fs_native - round(fs_native)) / fs_native < 0.01:
        fs_native = float(round(fs_native))

    return t, x, fs_native


def load_uploaded_segment(t: np.ndarray, x: np.ndarray, fs_native: float,
                          target_fs: int = TARGET_FS):
    """Resample + bandpass an uploaded recording into a core.Segment."""
    return loader.to_segment(t, x, fs=fs_native, target_fs=target_fs, bandpass=True)
