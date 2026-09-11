# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Physical-time sampling for DROID MP4s, whose playback FPS is not a capture clock."""

import json
import os
from functools import lru_cache

import h5py
import numpy as np


class ClockAlignmentError(ValueError):
    """Missing or inconsistent frame-to-observation clock metadata; do not silently retry."""


class NoPhysicalClipError(ValueError):
    """An otherwise valid stream has no clip satisfying the requested timing tolerances."""


def _nearest_indices(times, targets):
    right = np.searchsorted(times, targets).clip(0, len(times) - 1)
    left = np.maximum(right - 1, 0)
    # Treat sub-picosecond roundoff as a tie; prefer the earlier observation.
    return np.where(np.abs(times[left] - targets) <= np.abs(times[right] - targets) + 1e-12, left, right)


class PhysicalClipSampler:
    """Sample a uniform physical-time grid without synthesizing observations.

    Timestamps must be seconds in one clock domain. A clip is admissible only when
    every target is within tolerance, all frames are distinct, and no source gap
    exceeds max_gap_s anywhere in the clip (including between sampled targets).
    The anchor is an actual frame; all targets are relative to that same anchor.
    """

    def __init__(self, timestamps, frames_per_clip, fps, tolerance_s=0.05, max_gap_s=0.25):
        times = np.asarray(timestamps, dtype=np.float64)
        if times.ndim != 1 or len(times) < 1 or not np.isfinite(times).all():
            raise ClockAlignmentError("Frame timestamps must be a nonempty, finite vector in seconds")
        if np.any(np.diff(times) <= 0):
            raise ClockAlignmentError("Frame timestamps must be strictly increasing")
        if not isinstance(frames_per_clip, (int, np.integer)) or frames_per_clip < 2:
            raise ValueError("frames_per_clip must be an integer >= 2")
        if fps is None or not np.isfinite(fps) or fps <= 0:
            raise ValueError("Physical sampling requires a finite positive fps")
        if not np.isfinite(tolerance_s) or tolerance_s < 0:
            raise ValueError("tolerance_s must be finite and nonnegative")
        if not np.isfinite(max_gap_s) or max_gap_s <= 0:
            raise ValueError("max_gap_s must be finite and positive")
        self.times = times - times[0]
        self.offsets = np.arange(frames_per_clip) / fps
        # Require full physical coverage: never clamp an out-of-range target.
        starts = np.flatnonzero(self.times + self.offsets[-1] <= self.times[-1] + 1e-12)
        valid = []
        gaps = np.r_[0, np.cumsum(np.diff(self.times) > max_gap_s + 1e-12)]
        # Bound temporary memory even for unusually long trajectories.
        for block in np.array_split(starts, max(1, (len(starts) + 4095) // 4096)):
            targets = self.times[block, None] + self.offsets
            indices = _nearest_indices(self.times, targets)
            ok = np.all(np.abs(self.times[indices] - targets) <= tolerance_s + 1e-12, axis=1)
            ok &= np.all(np.diff(indices, axis=1) > 0, axis=1)
            ok &= gaps[indices[:, -1]] == gaps[indices[:, 0]]
            valid.append(block[ok])
        self.valid_starts = np.concatenate(valid)
        if not len(self.valid_starts):
            raise NoPhysicalClipError("No clip has full coverage, distinct frames, and acceptable clock gaps/errors")
        for array in (self.times, self.offsets, self.valid_starts):
            array.setflags(write=False)

    def indices(self, start):
        """Return the deterministic indices for an admissible frame anchor."""
        pos = np.searchsorted(self.valid_starts, start)
        if pos == len(self.valid_starts) or self.valid_starts[pos] != start:
            raise NoPhysicalClipError(f"Frame {start} is not an admissible physical-time anchor")
        return _nearest_indices(self.times, self.times[start] + self.offsets)

    def sample(self):
        """Use NumPy's worker-seeded RNG, as the original loader does."""
        return self.indices(self.valid_starts[np.random.randint(len(self.valid_starts))])


def _signature(path):
    stat = os.stat(path)
    return stat.st_size, stat.st_mtime_ns


def load_physical_clock(
    tpath, vpath, camera_name, frame_count, frames_per_clip, fps, alignment="sidecar", tolerance_s=0.05, max_gap_s=0.25
):
    """Return a cached sampler and its explicit video-frame -> HDF5-row mapping.

    Native DROID sidecars contain ZED IMAGE timestamps in integer milliseconds.
    Exact timestamp matching permits missing video observations without shifting
    robot labels. `row` is an explicit compatibility contract, not a validation:
    the caller asserts MP4 frame i is HDF5 observation i, with at most one omitted
    trailing observation. Frame counts alone cannot establish that assertion.
    """
    if alignment not in ("sidecar", "row"):
        raise ValueError("timestamp_alignment must be 'sidecar' or 'row'")
    if not isinstance(frame_count, (int, np.integer)) or frame_count < 1:
        raise ClockAlignmentError("Video must contain at least one frame")
    sidecar = os.path.splitext(vpath)[0] + "_timestamps.json"
    sidecar_signature = _signature(sidecar) if os.path.isfile(sidecar) else None
    if sidecar_signature is None and alignment == "sidecar":
        raise ClockAlignmentError(
            f"Missing {sidecar}. Export native per-frame IMAGE timestamps from the SVO; "
            "do not reconstruct them from MP4 FPS. Set timestamp_alignment='row' only if "
            "frame i -> observation i is independently established for this export."
        )
    return _load_physical_clock_cached(
        os.path.abspath(tpath),
        os.path.abspath(sidecar),
        camera_name,
        frame_count,
        frames_per_clip,
        fps,
        alignment,
        tolerance_s,
        max_gap_s,
        _signature(tpath),
        sidecar_signature,
    )


@lru_cache(maxsize=128)
def _load_physical_clock_cached(
    tpath,
    sidecar,
    camera_name,
    frame_count,
    frames_per_clip,
    fps,
    alignment,
    tolerance_s,
    max_gap_s,
    trajectory_signature,
    sidecar_signature,
):
    del trajectory_signature  # Included in the cache key to invalidate changed inputs.
    with h5py.File(tpath, "r") as trajectory:
        key = f"observation/timestamp/cameras/{camera_name}_frame_received"
        if key not in trajectory:
            raise ClockAlignmentError(f"Missing camera clock {key} in {tpath}")
        camera_ms = np.asarray(trajectory[key])
        nrows = len(trajectory["observation/robot_state/cartesian_position"])
    if camera_ms.ndim != 1 or len(camera_ms) != nrows or not np.issubdtype(camera_ms.dtype, np.integer):
        raise ClockAlignmentError("DROID camera timestamps must contain one integer millisecond value per robot row")
    if np.any(camera_ms[1:] <= camera_ms[:-1]):
        raise ClockAlignmentError("DROID camera timestamps must be strictly increasing")
    if sidecar_signature is not None:
        try:
            with open(sidecar) as stream:
                frame_ms = np.asarray(json.load(stream))
        except (ValueError, TypeError) as error:
            raise ClockAlignmentError(f"Invalid timestamp sidecar {sidecar}") from error
        if frame_ms.ndim != 1 or len(frame_ms) != frame_count or not np.issubdtype(frame_ms.dtype, np.integer):
            raise ClockAlignmentError("Native sidecar must have one integer millisecond timestamp per video frame")
        rows = np.searchsorted(camera_ms, frame_ms)
        if np.any(rows >= nrows) or not np.array_equal(camera_ms[rows], frame_ms):
            raise ClockAlignmentError("Video timestamps do not exactly match this camera's observation clock")
        if np.any(np.diff(rows) <= 0):
            raise ClockAlignmentError("Frame-to-row mapping must be unique and strictly increasing")
    else:
        if alignment != "row" or nrows not in (frame_count, frame_count + 1):
            raise ClockAlignmentError("Declared row alignment requires N or N+1 robot rows for N video frames")
        rows = np.arange(frame_count)
        frame_ms = camera_ms[rows]
    # Subtract integer epochs before conversion to avoid cancellation at Unix time scale.
    times = (frame_ms - frame_ms[0]).astype(np.float64) / 1000.0
    rows.setflags(write=False)
    return PhysicalClipSampler(times, frames_per_clip, fps, tolerance_s, max_gap_s), rows
