# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from logging import getLogger

import numpy as np
import torch
import torchvision

_GLOBAL_SEED = 0
logger = getLogger()


class ImageNet(torchvision.datasets.ImageFolder):

    def __init__(
        self,
        root,
        tar_file="imagenet_full_size-061417.tar.gz",
        transform=None,
        train=True,
        job_id=None,
        local_rank=None,
        index_targets=False,
    ):
        """
        ImageNet

        Dataset wrapper

        :param root: root network directory for ImageNet data
        :param tar_file: zipped image_folder inside root network directory
        :param train: whether to load train data (or validation)
        :param job_id: scheduler job-id used to create dir on local machine
        :param index_targets: whether to index the id of each labeled image
        """

        suffix = "train/" if train else "val/"
        data_path = os.path.join(root, suffix)
        logger.info(f"data-path {data_path}")

        super(ImageNet, self).__init__(root=data_path, transform=transform)
        logger.info("Initialized ImageNet")

        if index_targets:
            self.targets = []
            for sample in self.samples:
                self.targets.append(sample[1])
            self.targets = np.array(self.targets)
            self.samples = np.array(self.samples)

            mint = None
            self.target_indices = []
            for t in range(len(self.classes)):
                indices = np.squeeze(np.argwhere(self.targets == t)).tolist()
                self.target_indices.append(indices)
                mint = len(indices) if mint is None else min(mint, len(indices))
                logger.debug(f"num-labeled target {t} {len(indices)}")
            logger.info(f"min. labeled indices {mint}")


class ImageNetSubset(object):

    def __init__(self, dataset, subset_file):
        """
        ImageNetSubset

        :param dataset: ImageNet dataset object
        :param subset_file: '.txt' file containing IDs of IN1K images to keep
        """
        self.dataset = dataset
        self.subset_file = subset_file
        self.filter_dataset_(subset_file)

    def filter_dataset_(self, subset_file):
        """Filter self.dataset to a subset"""
        root = self.dataset.root
        class_to_idx = self.dataset.class_to_idx
        # -- update samples to subset of IN1k targets/samples
        new_samples = []
        logger.info(f"Using {subset_file}")
        with open(subset_file, "r") as rfile:
            for line in rfile:
                class_name = line.split("_")[0]
                target = class_to_idx[class_name]
                img = line.split("\n")[0]
                new_samples.append((os.path.join(root, class_name, img), target))
        self.samples = new_samples

    @property
    def classes(self):
        return self.dataset.classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.dataset.loader(path)
        if self.dataset.transform is not None:
            img = self.dataset.transform(img)
        if self.dataset.target_transform is not None:
            target = self.dataset.target_transform(target)
        return img, target


def make_imagenet1k(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    training=True,
    drop_last=True,
    persistent_workers=False,
    subset_file=None,
):
    dataset = ImageNet(
        root=root_path,
        transform=transform,
        train=training,
        index_targets=False,
    )
    if subset_file is not None:
        dataset = ImageNetSubset(dataset, subset_file)
    logger.info("ImageNet dataset created")
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset, num_replicas=world_size, rank=rank
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )
    logger.info("ImageNet unsupervised data loader created")

    return dataset, data_loader, dist_sampler
