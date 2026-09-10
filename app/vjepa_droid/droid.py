# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import os
from logging import getLogger
from math import ceil

import h5py
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from decord import VideoReader, cpu
from scipy.spatial.transform import Rotation

from app.vjepa_droid.temporal import ClockAlignmentError, load_physical_clock

_GLOBAL_SEED = 0
logger = getLogger()


def init_data(
    data_path,
    batch_size,
    frames_per_clip=16,
    fps=5,
    crop_size=224,
    rank=0,
    world_size=1,
    camera_views=0,
    stereo_view=False,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    persistent_workers=True,
    collator=None,
    transform=None,
    camera_frame=False,
    tubelet_size=2,
    temporal_sampling="legacy",
    timestamp_alignment="sidecar",
    timestamp_tolerance_s=0.05,
    max_observation_gap_s=0.25,
):
    dataset = DROIDVideoDataset(
        data_path=data_path,
        frames_per_clip=frames_per_clip,
        transform=transform,
        fps=fps,
        camera_views=camera_views,
        frameskip=tubelet_size,
        camera_frame=camera_frame,
        temporal_sampling=temporal_sampling,
        timestamp_alignment=timestamp_alignment,
        timestamp_tolerance_s=timestamp_tolerance_s,
        max_observation_gap_s=max_observation_gap_s,
    )

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
    )

    logger.info("VideoDataset unsupervised data loader created")

    return data_loader, dist_sampler


def get_json(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Error decoding JSON in file: {filename}")
            except Exception as e:
                print(f"An unexpected error occurred while processing {filename}: {e}")


class DROIDVideoDataset(torch.utils.data.Dataset):
    """Video classification dataset."""

    def __init__(
        self,
        data_path,
        camera_views=["left_mp4_path", "right_mp4_path"],
        frameskip=2,
        frames_per_clip=16,
        fps=5,
        transform=None,
        camera_frame=False,
        temporal_sampling="legacy",
        timestamp_alignment="sidecar",
        timestamp_tolerance_s=0.05,
        max_observation_gap_s=0.25,
    ):
        self.data_path = data_path
        self.frames_per_clip = frames_per_clip
        self.frameskip = frameskip
        self.fps = fps
        self.transform = transform
        self.camera_frame = camera_frame
        if temporal_sampling not in ("legacy", "physical"):
            raise ValueError("temporal_sampling must be 'legacy' or 'physical'")
        if temporal_sampling == "physical" and frameskip != 1:
            raise ValueError("Physical sampling requires frameskip=1 so every image has an aligned state")
        if temporal_sampling == "physical" and (fps is None or not np.isfinite(fps) or fps <= 0):
            raise ValueError("Physical sampling requires a finite positive fps")
        if temporal_sampling == "physical":
            if timestamp_alignment not in ("sidecar", "row"):
                raise ValueError("timestamp_alignment must be 'sidecar' or 'row'")
            if not np.isfinite(timestamp_tolerance_s) or timestamp_tolerance_s < 0:
                raise ValueError("timestamp_tolerance_s must be finite and nonnegative")
            if not np.isfinite(max_observation_gap_s) or max_observation_gap_s <= 0:
                raise ValueError("max_observation_gap_s must be finite and positive")
        else:
            logger.warning("Legacy DROID sampling uses MP4 playback time; fps is not a physical acquisition rate")
        self.temporal_sampling = temporal_sampling
        self.timestamp_alignment = timestamp_alignment
        self.timestamp_tolerance_s = timestamp_tolerance_s
        self.max_observation_gap_s = max_observation_gap_s
        if temporal_sampling == "physical" and timestamp_alignment == "row":
            logger.warning("DROID row alignment is a caller assertion; counts do not verify frame-to-row identity")
        if VideoReader is None:
            raise ImportError('Unable to import "decord" which is required to read videos.')

        # Camera views
        # ---
        # wrist camera view
        # left camera view
        # right camera view
        self.camera_views = camera_views
        self.h5_name = "trajectory.h5"

        samples = list(pd.read_csv(data_path, header=None, delimiter=" ").values[:, 0])
        self.samples = samples

    def __getitem__(self, index):
        path = self.samples[index]

        # Bound corrupt/too-short video retries; clock errors require fixing the input.
        for attempt in range(10):
            try:
                return self.loadvideo_decord(path)
            except ClockAlignmentError:
                raise
            except Exception as e:
                logger.info(f"Encountered exception when loading video {path=} {e=}")
                if attempt == 9:
                    raise RuntimeError("Unable to load a DROID clip after 10 attempts") from e
                index = np.random.randint(self.__len__())
                path = self.samples[index]

    def poses_to_diffs(self, poses):
        xyz = poses[:, :3]  # shape [T, 3]
        thetas = poses[:, 3:6]  # euler angles, shape [T, 3]
        matrices = [Rotation.from_euler("xyz", theta, degrees=False).as_matrix() for theta in thetas]
        xyz_diff = xyz[1:] - xyz[:-1]
        angle_diff = [matrices[t + 1] @ matrices[t].T for t in range(len(matrices) - 1)]
        angle_diff = [Rotation.from_matrix(mat).as_euler("xyz", degrees=False) for mat in angle_diff]
        angle_diff = np.stack([d for d in angle_diff], axis=0)
        closedness = poses[:, -1:]
        closedness_delta = closedness[1:] - closedness[:-1]
        return np.concatenate([xyz_diff, angle_diff, closedness_delta], axis=1)

    def transform_frame(self, poses, extrinsics):
        gripper = poses[:, -1:]
        poses = poses[:, :-1]

        def pose_to_transform(pose):
            trans = pose[:3]  # shape [3]
            theta = pose[3:6]  # euler angles, shape [3]
            Rot = Rotation.from_euler("xyz", theta, degrees=False).as_matrix()
            T = np.eye(4)
            T[:3, :3] = Rot
            T[:3, 3] = trans
            return T

        def transform_to_pose(transform):
            trans = transform[:3, 3]
            Rot = transform[:3, :3]
            angle = Rotation.from_matrix(Rot).as_euler("xyz", degrees=False)
            return np.concatenate([trans, angle], axis=0)

        new_pose = []
        for p, e in zip(poses, extrinsics):
            p_transform = pose_to_transform(p)
            e_transform = pose_to_transform(e)
            new_pose_transform = np.linalg.inv(e_transform) @ p_transform
            new_pose += [transform_to_pose(new_pose_transform)]
        new_pose = np.stack(new_pose, axis=0)

        return np.concatenate([new_pose, gripper], axis=1)

    def loadvideo_decord(self, path):
        # -- load metadata
        metadata = get_json(path)
        if metadata is None:
            raise Exception(f"No metadata for video {path=}")

        # -- load trajectory info
        tpath = os.path.join(path, self.h5_name)

        # -- randomly sample a camera view
        camera_view = self.camera_views[torch.randint(0, len(self.camera_views), (1,))]
        mp4_name = metadata[camera_view].split("recordings/MP4/")[-1]
        camera_name = mp4_name.split(".")[0]
        with h5py.File(tpath, "r") as trajectory:
            extrinsics = np.asarray(trajectory["observation"]["camera_extrinsics"][f"{camera_name}_left"])
            states = np.column_stack(
                [
                    trajectory["observation"]["robot_state"]["cartesian_position"],
                    trajectory["observation"]["robot_state"]["gripper_position"],
                ]
            )  # [T, 7]
        vpath = os.path.join(path, "recordings/MP4", mp4_name)
        vr = VideoReader(vpath, num_threads=-1, ctx=cpu(0))
        # --
        vlen = len(vr)
        if self.temporal_sampling == "physical":
            sampler, frame_rows = load_physical_clock(
                tpath,
                vpath,
                camera_name,
                vlen,
                self.frames_per_clip,
                self.fps,
                self.timestamp_alignment,
                self.timestamp_tolerance_s,
                self.max_observation_gap_s,
            )
            indices = sampler.sample()
            state_indices = frame_rows[indices]
            if len(extrinsics) != len(states):
                raise ClockAlignmentError("Extrinsics and robot states must have the same row count")
        else:
            # Explicit compatibility mode: retain playback-clock semantics and RNG support.
            vfps = vr.get_avg_fps()
            fpc = self.frames_per_clip
            fps = self.fps if self.fps is not None else vfps
            fstp = ceil(vfps / fps)
            nframes = int(fpc * fstp)
            if vlen < nframes:
                raise ValueError(f"Video is too short {vpath=}, {nframes=}, {vlen=}")
            sf = np.random.randint(vlen - nframes) if vlen > nframes else 0
            indices = np.arange(sf, sf + nframes, fstp).astype(np.int64)
            state_indices = indices
        states = states[state_indices, :][:: self.frameskip]
        extrinsics = extrinsics[state_indices, :][:: self.frameskip]
        if self.camera_frame:
            states = self.transform_frame(states, extrinsics)
        actions = self.poses_to_diffs(states)
        # --
        vr.seek(0)  # go to start of video before sampling frames
        buffer = vr.get_batch(indices).asnumpy()
        if self.transform is not None:
            buffer = self.transform(buffer)

        return buffer, actions, states, extrinsics, indices

    def __len__(self):
        return len(self.samples)
