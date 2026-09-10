# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json

import cv2
import h5py
import numpy as np
import pytest
import torch

from app.vjepa_droid.droid import DROIDVideoDataset, init_data
from app.vjepa_droid.temporal import ClockAlignmentError
from app.vjepa_droid.transforms import make_transforms


@pytest.fixture
def episode(tmp_path):
    folder = tmp_path / "episode"
    videos = folder / "recordings" / "MP4"
    videos.mkdir(parents=True)
    n = 140
    rows = np.delete(np.arange(n), [5, 18, 29])
    times = 1686087812000 + np.rint(np.arange(n) * 1000 / 15).astype(np.int64)
    poses = np.zeros((n, 6))
    poses[:, 0] = np.arange(n) / 1000
    with h5py.File(folder / "trajectory.h5", "w") as h:
        h.create_dataset("observation/robot_state/cartesian_position", data=poses)
        h.create_dataset("observation/robot_state/gripper_position", data=np.arange(n) / n)
        h.create_dataset("observation/camera_extrinsics/123_left", data=np.zeros((n, 6)))
        h.create_dataset("observation/timestamp/cameras/123_frame_received", data=times)
    (folder / "metadata.json").write_text(json.dumps({"left_mp4_path": "recordings/MP4/123.mp4"}))
    writer = cv2.VideoWriter(str(videos / "123.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 60, (32, 32))
    assert writer.isOpened()
    for i in range(len(rows)):
        writer.write(np.full((32, 32, 3), i, dtype=np.uint8))
    writer.release()
    sidecar = videos / "123_timestamps.json"
    sidecar.write_text(json.dumps(times[rows].tolist()))
    csv = tmp_path / "paths.csv"
    csv.write_text(str(folder) + "\n")
    return folder, csv, rows, times, sidecar


def dataset(episode, **kwargs):
    return DROIDVideoDataset(
        str(episode[1]), camera_views=["left_mp4_path"], frameskip=1, frames_per_clip=8, fps=4, **kwargs
    )


def test_real_decord_alignment_after_missing_video_frames(episode):
    d = dataset(episode, temporal_sampling="physical", timestamp_tolerance_s=0.07)
    video, actions, states, extrinsics, idx = d[0]
    rows = episode[2][idx]
    np.testing.assert_allclose(states[:, 0], rows / 1000)
    np.testing.assert_allclose(actions[:, 0], np.diff(rows) / 1000)
    np.testing.assert_allclose(actions[:, -1], np.diff(rows) / 140)
    assert video.shape == (8, 32, 32, 3)
    assert actions.shape == (7, 7)
    assert extrinsics.shape == (8, 6)
    assert np.max(abs(episode[3][rows] - episode[3][rows[0]] - np.arange(8) * 250)) <= 70


def test_native_clock_error_escapes_retry(episode):
    episode[4].unlink()
    with pytest.raises(ClockAlignmentError, match="Missing"):
        dataset(episode, temporal_sampling="physical")[0]


def test_legacy_indices_preserve_original_seeded_sampling(episode):
    n = len(episode[2])
    np.random.seed(23)
    sf = np.random.randint(120, n) - 120
    expected = sf + np.arange(8) * 15
    np.random.seed(23)
    sample = dataset(episode)[0]
    np.testing.assert_array_equal(sample[-1], expected)
    np.testing.assert_allclose(sample[2][:, 0], expected / 1000)


def test_config_to_collated_batch(episode):
    transform = make_transforms(
        random_horizontal_flip=False, random_resize_aspect_ratio=(1, 1), random_resize_scale=(1, 1), crop_size=16
    )
    loader, _ = init_data(
        str(episode[1]),
        batch_size=1,
        frames_per_clip=8,
        fps=4,
        tubelet_size=1,
        camera_views=["left_mp4_path"],
        temporal_sampling="physical",
        timestamp_tolerance_s=0.07,
        num_workers=0,
        transform=transform,
    )
    images, actions, states, extrinsics, indices = next(iter(loader))
    assert images.shape == (1, 3, 8, 16, 16)
    assert actions.shape == (1, 7, 7)
    assert states.shape == (1, 8, 7)
    assert extrinsics.shape == (1, 8, 6)
    assert indices.shape == (1, 8)
    torch.testing.assert_close(actions[..., :3], states[:, 1:, :3] - states[:, :-1, :3])


def test_retry_is_bounded(episode, monkeypatch):
    d = dataset(episode)
    calls = []

    def fail(path):
        calls.append(path)
        raise ValueError("corrupt fixture")

    monkeypatch.setattr(d, "loadvideo_decord", fail)
    with pytest.raises(RuntimeError, match="10 attempts"):
        d[0]
    assert len(calls) == 10


def test_physical_mode_rejects_unaligned_frameskip(episode):
    with pytest.raises(ValueError, match="frameskip=1"):
        DROIDVideoDataset(str(episode[1]), temporal_sampling="physical", frameskip=2)


@pytest.mark.parametrize(
    "options",
    [
        {"timestamp_alignment": "guess"},
        {"timestamp_tolerance_s": -0.01},
        {"timestamp_tolerance_s": float("nan")},
        {"max_observation_gap_s": 0},
        {"max_observation_gap_s": float("inf")},
    ],
)
def test_invalid_clock_policy_fails_before_reading_dataset(options):
    with pytest.raises(ValueError):
        DROIDVideoDataset("unused.csv", temporal_sampling="physical", frameskip=1, **options)
