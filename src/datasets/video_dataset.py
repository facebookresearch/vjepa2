# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import pathlib
import warnings
from logging import getLogger

import numpy as np
import pandas as pd
import torch
import torchvision
from decord import cpu, VideoReader

from src.datasets.utils.dataloader import (
    ConcatIndices,
    MonitoredDataset,
    NondeterministicDataLoader,
)
from src.datasets.utils.weighted_sampler import DistributedWeightedSampler

_GLOBAL_SEED = 0
logger = getLogger()


def make_videodataset(
    data_paths,
    batch_size,
    frames_per_clip=8,
    dataset_fpcs=None,
    frame_step=4,
    duration=None,
    fps=None,
    num_clips=1,
    random_clip_sampling=True,
    allow_clip_overlap=False,
    filter_short_videos=False,
    filter_long_videos=int(10**9),
    transform=None,
    shared_transform=None,
    rank=0,
    world_size=1,
    datasets_weights=None,
    collator=None,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    persistent_workers=True,
    deterministic=True,
    log_dir=None,
):
    dataset = VideoDataset(
        data_paths=data_paths,
        datasets_weights=datasets_weights,
        frames_per_clip=frames_per_clip,
        dataset_fpcs=dataset_fpcs,
        duration=duration,
        fps=fps,
        frame_step=frame_step,
        num_clips=num_clips,
        random_clip_sampling=random_clip_sampling,
        allow_clip_overlap=allow_clip_overlap,
        filter_short_videos=filter_short_videos,
        filter_long_videos=filter_long_videos,
        shared_transform=shared_transform,
        transform=transform,
    )

    log_dir = pathlib.Path(log_dir) if log_dir else None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        # Worker ID will replace '%w'
        resource_log_filename = log_dir / f"resource_file_{rank}_%w.csv"
        dataset = MonitoredDataset(
            dataset=dataset,
            log_filename=str(resource_log_filename),
            log_interval=10.0,
            monitor_interval=5.0,
        )

    logger.info("VideoDataset dataset created")
    if datasets_weights is not None:
        dist_sampler = DistributedWeightedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
    else:
        dist_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        )

    if deterministic:
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
    else:
        data_loader = NondeterministicDataLoader(
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

    return dataset, data_loader, dist_sampler


class VideoDataset(torch.utils.data.Dataset):
    """Video classification dataset."""

    def __init__(
        self,
        data_paths,
        datasets_weights=None,
        frames_per_clip=16,
        fps=None,
        dataset_fpcs=None,
        frame_step=4,
        num_clips=1,
        transform=None,
        shared_transform=None,
        random_clip_sampling=True,
        allow_clip_overlap=False,
        filter_short_videos=False,
        filter_long_videos=int(10**9),
        duration=None,  # duration in seconds
    ):
        self.data_paths = data_paths
        self.datasets_weights = datasets_weights
        self.frame_step = frame_step
        self.num_clips = num_clips
        self.transform = transform
        self.shared_transform = shared_transform
        self.random_clip_sampling = random_clip_sampling
        self.allow_clip_overlap = allow_clip_overlap
        self.filter_short_videos = filter_short_videos
        self.filter_long_videos = filter_long_videos
        self.duration = duration
        self.fps = fps

        if sum([v is not None for v in (fps, duration, frame_step)]) != 1:
            raise ValueError(
                f"Must specify exactly one of either {fps=}, {duration=}, or {frame_step=}."
            )

        if isinstance(data_paths, str):
            data_paths = [data_paths]

        if dataset_fpcs is None:
            self.dataset_fpcs = [frames_per_clip for _ in data_paths]
        else:
            if len(dataset_fpcs) != len(data_paths):
                raise ValueError(
                    "Frames per clip not properly specified for NFS data paths"
                )
            self.dataset_fpcs = dataset_fpcs

        if VideoReader is None:
            raise ImportError(
                'Unable to import "decord" which is required to read videos.'
            )

        # Load video paths and labels
        samples, labels = [], []
        self.num_samples_per_dataset = []
        for data_path in self.data_paths:

            if data_path[-4:] == ".csv":
                try:
                    data = pd.read_csv(data_path, header=None, delimiter=" ")
                except pd.errors.ParserError:
                    # In image captioning datasets where we have space, we use :: as delimiter.
                    data = pd.read_csv(data_path, header=None, delimiter="::")
                samples += list(data.values[:, 0])
                labels += list(data.values[:, 1])
                num_samples = len(data)
                self.num_samples_per_dataset.append(num_samples)

            elif data_path[-4:] == ".npy":
                data = np.load(data_path, allow_pickle=True)
                data = list(map(lambda x: repr(x)[1:-1], data))
                samples += data
                labels += [0] * len(data)
                num_samples = len(data)
                self.num_samples_per_dataset.append(len(data))

        self.per_dataset_indices = ConcatIndices(self.num_samples_per_dataset)

        # [Optional] Weights for each sample to be used by downstream
        # weighted video sampler
        self.sample_weights = None
        if self.datasets_weights is not None:
            self.sample_weights = []
            for dw, ns in zip(self.datasets_weights, self.num_samples_per_dataset):
                self.sample_weights += [dw / ns] * ns

        self.samples = samples
        self.labels = labels

    def __getitem__(self, index):
        sample = self.samples[index]
        loaded_sample = False
        # Keep trying to load videos until you find a valid sample
        while not loaded_sample:
            if not isinstance(sample, str):
                logger.warning("Invalid sample.")
            else:
                if sample.split(".")[-1].lower() in ("jpg", "png", "jpeg"):
                    loaded_sample = self.get_item_image(index)
                else:
                    loaded_sample = self.get_item_video(index)

            if not loaded_sample:
                index = np.random.randint(self.__len__())
                sample = self.samples[index]

        return loaded_sample

    def get_item_video(self, index):
        sample = self.samples[index]
        dataset_idx, _ = self.per_dataset_indices[index]
        frames_per_clip = self.dataset_fpcs[dataset_idx]

        buffer, clip_indices = self.loadvideo_decord(
            sample, frames_per_clip
        )  # [T H W 3]
        loaded_video = len(buffer) > 0
        if not loaded_video:
            return

        # Label/annotations for video
        label = self.labels[index]

        def split_into_clips(video):
            """Split video into a list of clips"""
            fpc = frames_per_clip
            nc = self.num_clips
            return [video[i * fpc : (i + 1) * fpc] for i in range(nc)]

        # Parse video into frames & apply data augmentations
        if self.shared_transform is not None:
            buffer = self.shared_transform(buffer)
        buffer = split_into_clips(buffer)
        if self.transform is not None:
            buffer = [self.transform(clip) for clip in buffer]

        return buffer, label, clip_indices

    def get_item_image(self, index):
        sample = self.samples[index]
        dataset_idx, _ = self.per_dataset_indices[index]
        fpc = self.dataset_fpcs[dataset_idx]

        try:
            image_tensor = torchvision.io.read_image(
                path=sample, mode=torchvision.io.ImageReadMode.RGB
            )
        except Exception:
            return
        label = self.labels[index]
        clip_indices = [np.arange(start=0, stop=fpc, dtype=np.int32)]

        # Expanding the input image [3, H, W] ==> [T, 3, H, W]
        buffer = image_tensor.unsqueeze(dim=0).repeat((fpc, 1, 1, 1))
        buffer = buffer.permute((0, 2, 3, 1))  # [T, 3, H, W] ==> [T H W 3]

        if self.shared_transform is not None:
            # Technically we can have only transform, doing this just for the sake of consistency with videos.
            buffer = self.shared_transform(buffer)

        if self.transform is not None:
            buffer = [self.transform(buffer)]

        return buffer, label, clip_indices

    def loadvideo_decord(self, sample, fpc):
        """Load video content using Decord"""

        fname = sample
        if not os.path.exists(fname):
            warnings.warn(f"video path not found {fname=}")
            return [], None

        _fsize = os.path.getsize(fname)
        if _fsize > self.filter_long_videos:
            warnings.warn(f"skipping long video of size {_fsize=} (bytes)")
            return [], None

        try:
            vr = VideoReader(fname, num_threads=-1, ctx=cpu(0))
        except Exception:
            return [], None

        fstp = self.frame_step
        if self.duration is not None or self.fps is not None:
            try:
                video_fps = math.ceil(vr.get_avg_fps())
            except Exception as e:
                logger.warning(e)

            if self.duration is not None:
                assert self.fps is None
                fstp = int(self.duration * video_fps / fpc)
            else:
                assert self.duration is None
                fstp = video_fps // self.fps

        assert fstp is not None and fstp > 0
        clip_len = int(fpc * fstp)

        if self.filter_short_videos and len(vr) < clip_len:
            warnings.warn(f"skipping video of length {len(vr)}")
            return [], None

        vr.seek(0)  # Go to start of video before sampling frames

        # Partition video into equal sized segments and sample each clip
        # from a different segment
        partition_len = len(vr) // self.num_clips

        all_indices, clip_indices = [], []
        for i in range(self.num_clips):

            if partition_len > clip_len:
                # If partition_len > clip len, then sample a random window of
                # clip_len frames within the segment
                end_indx = clip_len
                if self.random_clip_sampling:
                    end_indx = np.random.randint(clip_len, partition_len)
                start_indx = end_indx - clip_len
                indices = np.linspace(start_indx, end_indx, num=fpc)
                indices = np.clip(indices, start_indx, end_indx - 1).astype(np.int64)
                # --
                indices = indices + i * partition_len
            else:
                # If partition overlap not allowed and partition_len < clip_len
                # then repeatedly append the last frame in the segment until
                # we reach the desired clip length
                if not self.allow_clip_overlap:
                    indices = np.linspace(0, partition_len, num=partition_len // fstp)
                    indices = np.concatenate(
                        (
                            indices,
                            np.ones(fpc - partition_len // fstp) * partition_len,
                        )
                    )
                    indices = np.clip(indices, 0, partition_len - 1).astype(np.int64)
                    # --
                    indices = indices + i * partition_len

                # If partition overlap is allowed and partition_len < clip_len
                # then start_indx of segment i+1 will lie within segment i
                else:
                    max_idx = min(clip_len, len(vr)) - 1
                    indices = np.clip(
                        np.arange(fpc) * fstp, 0, max_idx
                    ).astype(np.int64)
                    # --
                    clip_step = 0
                    if len(vr) > clip_len:
                        clip_step = (len(vr) - clip_len) // (self.num_clips - 1)
                    indices = indices + i * clip_step

            clip_indices.append(indices)
            all_indices.extend(list(indices))

        buffer = vr.get_batch(all_indices).asnumpy()
        return buffer, clip_indices

    def __len__(self):
        return len(self.samples)
