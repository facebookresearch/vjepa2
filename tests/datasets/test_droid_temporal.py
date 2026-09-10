# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json

import h5py
import numpy as np
import pytest

from app.vjepa_droid.temporal import ClockAlignmentError, NoPhysicalClipError, PhysicalClipSampler, load_physical_clock


def test_physical_span_is_not_container_span():
    times = np.arange(160) / 15
    sampler = PhysicalClipSampler(times, 8, 4)
    idx = sampler.indices(0)
    assert np.max(np.abs(times[idx] - np.arange(8) / 4)) <= 1 / 30 + 1e-12
    assert abs(times[idx[-1]] - 1.75) <= 1 / 30
    assert times[np.arange(8) * 15][-1] == 7.0  # 60 FPS container / 4 FPS request


def test_uniform_four_hz_exact_boundary():
    sampler = PhysicalClipSampler([0, 0.25, 0.5, 0.75], 4, 4)
    np.testing.assert_array_equal(sampler.valid_starts, [0])
    np.testing.assert_array_equal(sampler.sample(), [0, 1, 2, 3])


def test_irregular_grid_uses_absolute_anchor_and_earlier_tie():
    times = [0, 0.1, 0.2, 0.3, 0.51, 0.73, 0.81, 1.01]
    sampler = PhysicalClipSampler(times, 4, 4, tolerance_s=0.051)
    np.testing.assert_array_equal(sampler.indices(0), [0, 2, 4, 5])


def test_dropped_observation_is_not_interpolated():
    times = np.delete(np.arange(40) / 15, [5, 11])
    sampler = PhysicalClipSampler(times, 4, 4, tolerance_s=0.07)
    idx = sampler.indices(0)
    assert np.all(np.diff(idx) > 0)
    assert np.max(abs(np.asarray(times)[idx] - np.arange(4) / 4)) <= 0.07


def test_gap_between_targets_is_rejected_even_if_targets_are_exact():
    with pytest.raises(NoPhysicalClipError):
        PhysicalClipSampler([0, 0.25, 0.5, 0.75], 4, 4, max_gap_s=0.2)


def test_gap_does_not_discard_healthy_segment():
    times = np.r_[np.arange(20) / 15, 9 + np.arange(20) / 15]
    sampler = PhysicalClipSampler(times, 4, 4)
    for start in sampler.valid_starts:
        idx = sampler.indices(start)
        assert idx[-1] < 20 or idx[0] >= 20


@pytest.mark.parametrize("times", [[], [0, 0], [1, 0], [0, np.nan], [0, np.inf], [[0, 1]]])
def test_bad_clock_rejected(times):
    with pytest.raises(ClockAlignmentError):
        PhysicalClipSampler(times, 4, 4)


@pytest.mark.parametrize("fps", [None, 0, -1, np.nan, np.inf])
def test_bad_requested_rate_rejected(fps):
    with pytest.raises(ValueError):
        PhysicalClipSampler(np.arange(20) / 15, 4, fps)


def test_no_repeated_frame_or_out_of_range_targets():
    with pytest.raises(NoPhysicalClipError):
        PhysicalClipSampler([0, 0.1, 0.2], 4, 50, tolerance_s=0.1)
    with pytest.raises(NoPhysicalClipError):
        PhysicalClipSampler([0, 0.1, 0.2], 4, 4)


def test_worker_seed_reproducibility():
    sampler = PhysicalClipSampler(np.arange(200) / 15, 8, 4)
    np.random.seed(44)
    first = sampler.sample()
    np.random.seed(44)
    np.testing.assert_array_equal(first, sampler.sample())


def fixture_clock(tmp_path, rows=50):
    tpath = tmp_path / "trajectory.h5"
    camera_ms = 1686087812000 + np.rint(np.arange(rows) * 1000 / 15).astype(np.int64)
    with h5py.File(tpath, "w") as h:
        h.create_dataset("observation/timestamp/cameras/123_frame_received", data=camera_ms)
        h.create_dataset("observation/robot_state/cartesian_position", data=np.zeros((rows, 6)))
    return tpath, tmp_path / "123.mp4", camera_ms


def test_sidecar_maps_nonidentity_frame_indices_to_robot_rows(tmp_path):
    tpath, vpath, ms = fixture_clock(tmp_path)
    kept = np.delete(np.arange(len(ms)), [3, 8])
    vpath.with_name("123_timestamps.json").write_text(json.dumps(ms[kept].tolist()))
    sampler, rows = load_physical_clock(tpath, str(vpath), "123", len(kept), 4, 4, tolerance_s=0.07)
    np.testing.assert_array_equal(rows, kept)
    idx = sampler.indices(0)
    np.testing.assert_array_equal(ms[rows[idx]], ms[kept[idx]])
    assert not np.array_equal(rows, np.arange(len(rows)))


def test_missing_sidecar_fails_closed(tmp_path):
    tpath, vpath, ms = fixture_clock(tmp_path)
    with pytest.raises(ClockAlignmentError, match="Missing"):
        load_physical_clock(tpath, str(vpath), "123", len(ms) - 1, 4, 4)


def test_row_fallback_requires_explicit_contract_and_cardinality(tmp_path):
    tpath, vpath, ms = fixture_clock(tmp_path)
    _, rows = load_physical_clock(tpath, str(vpath), "123", len(ms) - 1, 4, 4, alignment="row")
    np.testing.assert_array_equal(rows, np.arange(len(ms) - 1))
    with pytest.raises(ClockAlignmentError):
        load_physical_clock(tpath, str(vpath), "123", len(ms) - 2, 4, 4, alignment="row")


@pytest.mark.parametrize("corruption", ["shift", "seconds", "duplicate", "reverse", "short"])
def test_sidecar_mismatch_rejected_even_in_row_mode(tmp_path, corruption):
    tpath, vpath, ms = fixture_clock(tmp_path)
    bad = ms.copy()
    if corruption == "shift":
        bad += 1
    elif corruption == "seconds":
        bad = bad / 1000
    elif corruption == "duplicate":
        bad[2] = bad[1]
    elif corruption == "reverse":
        bad = bad[::-1]
    else:
        bad = bad[:-1]
    vpath.with_name("123_timestamps.json").write_text(json.dumps(bad.tolist()))
    with pytest.raises(ClockAlignmentError):
        load_physical_clock(tpath, str(vpath), "123", len(ms), 4, 4, alignment="row")


def test_cache_reuses_plan_and_invalidates_modified_sidecar(tmp_path):
    tpath, vpath, ms = fixture_clock(tmp_path)
    sidecar = vpath.with_name("123_timestamps.json")
    sidecar.write_text(json.dumps(ms.tolist()))
    args = (tpath, str(vpath), "123", len(ms), 4, 4)
    first = load_physical_clock(*args)
    assert first is load_physical_clock(*args)
    sidecar.write_text(json.dumps((ms + 1).tolist()) + "\n")
    with pytest.raises(ClockAlignmentError):
        load_physical_clock(*args)


def test_two_camera_clocks_are_not_interchangeable(tmp_path):
    tpath, vpath, ms = fixture_clock(tmp_path)
    vpath.with_name("123_timestamps.json").write_text(json.dumps((ms + 5).tolist()))
    with pytest.raises(ClockAlignmentError):
        load_physical_clock(tpath, str(vpath), "123", len(ms), 4, 4)


def test_malformed_sidecar_has_clock_error_type(tmp_path):
    tpath, vpath, ms = fixture_clock(tmp_path)
    vpath.with_name("123_timestamps.json").write_text("{broken")
    with pytest.raises(ClockAlignmentError, match="Invalid timestamp sidecar"):
        load_physical_clock(tpath, str(vpath), "123", len(ms), 4, 4)


def test_zero_frame_video_rejected(tmp_path):
    tpath, vpath, _ = fixture_clock(tmp_path)
    with pytest.raises(ClockAlignmentError, match="at least one"):
        load_physical_clock(tpath, str(vpath), "123", 0, 4, 4, alignment="row")


def test_cached_plan_arrays_cannot_be_mutated():
    sampler = PhysicalClipSampler(np.arange(30) / 15, 4, 4)
    with pytest.raises(ValueError):
        sampler.times[1] = 99


def test_public_droid_clock_regression():
    from pathlib import Path

    fixture = Path(__file__).with_name("fixtures") / "droid_real_clock.json"
    ms = np.asarray(json.loads(fixture.read_text())["camera_ms"])
    seconds = (ms - ms[0]) / 1000.0
    sampler = PhysicalClipSampler(seconds, 8, 4)
    indices = sampler.indices(0)
    assert np.max(abs(seconds[indices] - np.arange(8) / 4)) <= 0.05
    assert seconds[np.arange(8) * 15][-1] > 7.0
    assert 1.70 <= seconds[indices][-1] <= 1.80


def test_exact_gap_boundary_survives_binary_float_roundoff():
    times = np.array([0, 7938, 8188, 8438, 8688]) / 1000.0
    sampler = PhysicalClipSampler(times, 4, 4, tolerance_s=0, max_gap_s=0.25)
    np.testing.assert_array_equal(sampler.indices(1), [1, 2, 3, 4])


def test_native_millisecond_tie_prefers_earlier_despite_float_roundoff():
    sampler = PhysicalClipSampler(np.array([0, 200, 417, 483, 700]) / 1000, 2, 4)
    np.testing.assert_array_equal(sampler.indices(1), [1, 2])
