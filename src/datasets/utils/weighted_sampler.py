# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Iterator, Optional

import numpy as np
import torch
from torch.utils.data import DistributedSampler, RandomSampler

from src.utils.logging import get_logger

logger = get_logger("WeightedSampler")


class DistributedWeightedSampler(DistributedSampler):
    """
    This class implements a weighted sampler for distributed training.
    See https://pytorch.org/docs/stable/_modules/torch/utils/data/distributed.html#DistributedSampler for more details.

    It shares the same interface as `torch.utils.data.DistributedSampler`.
    The effective change is replacing `DistributedSampler`'s `torch.randperm` for generating the sequence
    of indices with `numpy.random.Generator.choice`, with replacement. This allows weighted sampling and
    avoiding issue with `torch.randperm` when the number of samples is larger than 2^24 samples.
    """

    def __init__(
        self,
        dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        logger.info(f"Using DistributedWeightedSampler with rank {rank} / {num_replicas}")
        assert hasattr(
            dataset, "sample_weights"
        ), "Dataset must have sample_weights property for using DistributedWeightedSampler"
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )

    @property
    def sample_probabilities(self) -> np.ndarray:
        sample_weights = self.dataset.sample_weights
        if isinstance(sample_weights, torch.Tensor):
            sample_weights = sample_weights.cpu().numpy()
        elif isinstance(sample_weights, list):
            sample_weights = np.array(sample_weights)
        assert isinstance(
            sample_weights, np.ndarray
        ), f"sample_weights must be a numpy array, torch.Tensor, or python list; got {type(sample_weights)}"
        return sample_weights / np.sum(sample_weights)

    def __iter__(self) -> Iterator:
        n = len(self.dataset)

        # deterministically shuffle based on epoch and seed
        rng = np.random.default_rng(self.seed + self.epoch)
        indices = rng.choice(
            range(0, n),
            size=self.total_size,
            p=self.sample_probabilities,
            replace=True,
        ).tolist()

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


class MemoryEfficientDistributedWeightedSampler(DistributedSampler):
    """
    This class implements a memory efficient version of `DistributedWeightedSampler`.
    It shares the same interface as `DistributedWeightedSampler`.
    The effective change is just-in-time sampling of the indices, instead of pre-computing them.
    """

    def __init__(
        self,
        dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
    ):
        logger.info(f"Using MemoryEfficientDistributedWeightedSampler with rank {rank} / {num_replicas}")
        assert hasattr(
            dataset, "dataset_weights"
        ), "Dataset must have dataset_weights property for using MemoryEfficientDistributedWeightedSampler"
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
        )

        self.dataset_weights = dataset.dataset_weights
        self.dataset_sizes = [len(d) for d in dataset.datasets]
        if len(self.dataset_sizes) != len(self.dataset_weights):
            raise ValueError(
                f"Number of datasets ({len(self.dataset_sizes)}) "
                f"does not match number of dataset weights ({len(self.dataset_weights)})"
            )

        if self.shuffle:
            self.rng = np.random.default_rng(self.seed + self.rank + self.epoch)
            total_weights = sum(self.dataset_weights)
            self.dataset_probablities = np.array([w / total_weights for w in self.dataset_weights])
        else:
            if any([not isinstance(w, int) for w in self.dataset_weights]):
                raise ValueError("Dataset weights must be integers when shuffle is False")

            self.dataset_orders = []
            for i, w in enumerate(self.dataset_weights):
                self.dataset_orders.extend([i] * w)

            self.drawn_samples = 0

    def __iter__(self) -> Iterator:
        return self

    def __next__(self) -> int:
        if self.shuffle:
            selected_dataset_idx = self.rng.choice(range(len(self.dataset_weights)), p=self.dataset_probablities)

            # In order to avoid sampling the same example multiple times between the ranks,
            # we limit each rank to a subset of the total number of samples in the dataset.
            # For example if our dataset is [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], and we have 2 ranks,
            # then rank 0 will ONLY sample from [0, 2, 4, 6, 8], and rank 1 from [1, 3, 5, 7, 9].
            # In each iteration we first produce `in_rank_sample` which is the sample index in the rank,
            # based on the size of the subset which that rank can sample from.
            # Then we compute `sample_idx_in_dataset` for the index of the sample in the whole dataset.
            # For the above example if we are sampling for rank 1, we have `self.rng.integers(5)`.
            # Let's assume the result is 2, then `in_rank_sample` is 2 (number "5" in the subset),
            # so the sample index in the whole dataset is
            # `in_rank_sample * self.num_replicas + self.rank`: 2 * 2 + 1 = 5.

            selected_dataset_size = self.dataset_sizes[selected_dataset_idx]
            # 1) Getting sample index in the rank.
            # NOTE: this may effectively drop the last batch,
            # but given the sample sizes that we use this sampler with, it should not be an issue.
            num_samples_in_rank = selected_dataset_size // self.num_replicas
            in_rank_sample = self.rng.integers(num_samples_in_rank)

            # 2) Getting sample index in the dataset.
            sample_idx_in_dataset = in_rank_sample * self.num_replicas + self.rank

        else:
            # Iterate through the dataset orders in a round-robin fashion, offset by the rank
            dataset_orders_idx = (self.rank + self.drawn_samples) % len(self.dataset_orders)
            selected_dataset_idx = self.dataset_orders[dataset_orders_idx]
            # Get the sample index in the selected dataset by skipping with the num_replicas * drawn_samples
            sample_idx_in_dataset = (self.drawn_samples * self.num_replicas + self.rank) % self.dataset_sizes[
                selected_dataset_idx
            ]
            self.drawn_samples += 1

        # Getting the index of the sample in the whole dataset
        # For example if the total dataset has 4 datasets with sizes [10, 20, 30, 5].
        # and our selected_dataset_idx=3 and sample_idx_in_dataset=5
        # then the index of the sample in the whole dataset is
        #   10 (for dataset 1) + 20 (for dataset 1) + 5 (for sample_idx_in_dataset) = 35
        # This is because the first 10 samples are from the first dataset, the next 20 are from the second dataset,
        # then we reach at the 3rd dataset which is the selected dataset, and the 5th sample in the 3rd dataset.
        sample_idx = 0
        for i, d in enumerate(self.dataset_sizes):
            if selected_dataset_idx == i:
                break
            sample_idx += d
        sample_idx += sample_idx_in_dataset

        return sample_idx


def safe_next(iterator):
    try:
        return next(iterator)
    except StopIteration:
        return None


class MemoryEfficientDistributedWeightedSamplerLessRepeat(DistributedSampler):
    """
    This class implements a memory efficient version of `DistributedWeightedSampler`.
    It shares the same interface as `DistributedWeightedSampler`.
    The effective change is pre-computing the permutations of indices over a subset of total indices.
    This subset is the selected with picking the indices in a dataset with steps sizes equal to the world size.
    For example, if world size is 12 and rank is 2, for a dataset of size N,
    this sampler only permutes the indices in range(2, n, 12)

    Compared with MemoryEfficientDistributedWeightedSampler, this will reduce the effective number of repeat.
    See discussions here: https://github.com/fairinternal/jepa-internal/pull/254
    """

    def __init__(
        self,
        dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
    ):
        logger.info(f"Using MemoryEfficientDistributedWeightedSamplerLessRepeat with rank {rank} / {num_replicas}")
        assert hasattr(
            dataset, "dataset_weights"
        ), "Dataset must have dataset_weights property for using MemoryEfficientDistributedWeightedSamplerLessRepeat"
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
        )

        self._generator = torch.Generator()
        self._generator.manual_seed(seed)

        self.dataset_weights = dataset.dataset_weights
        self.dataset_sizes = [len(d) for d in dataset.datasets]
        if len(self.dataset_sizes) != len(self.dataset_weights):
            raise ValueError(
                f"Number of datasets ({len(self.dataset_sizes)}) "
                f"does not match number of dataset weights ({len(self.dataset_weights)})"
            )

        if self.shuffle:
            self.rng = np.random.default_rng(self.seed + self.rank + self.epoch)
            total_weights = sum(self.dataset_weights)
            self.dataset_probablities = np.array([w / total_weights for w in self.dataset_weights])

            # For each dataset we generate a permutation of the indices that will be processed by that rank.
            # This is going to be the subset of indices, selected by the steps sizes of the world size.
            logger.info(f"Generating dataset indices for rank {self.rank} / {self.num_replicas}")

            # Getting a RandomSampler for indices assigned to each dataset.
            self.individual_dataset_sampler = []
            for ids, ds in enumerate(self.dataset_sizes):

                # NOTE: this may effectively drop the last batch,
                # but given the sample sizes that we use this sampler with, it should not be an issue.
                num_samples_in_rank = ds // self.num_replicas
                self.individual_dataset_sampler.append(self._new_sampler(num_samples_in_rank))

        else:
            if any([not isinstance(w, int) for w in self.dataset_weights]):
                raise ValueError("Dataset weights must be integers when shuffle is False")

            self.dataset_orders = []
            for i, w in enumerate(self.dataset_weights):
                self.dataset_orders.extend([i] * w)

            self.drawn_samples = 0

    def __iter__(self) -> Iterator:
        return self

    def _new_sampler(self, sample_size: int) -> RandomSampler:
        assert self.shuffle

        return iter(
            RandomSampler(
                range(sample_size),
                generator=self._generator,
            )
        )

    def _in_rank_next_index_for_dataset(self, dataset_idx: int) -> int:
        assert self.shuffle

        next_sampler_idx = safe_next(self.individual_dataset_sampler[dataset_idx])
        if next_sampler_idx is None:
            # We have reached the end of the dataset, we need to reset the sampler.
            num_samples_in_rank = self.dataset_sizes[dataset_idx] // self.num_replicas
            self.individual_dataset_sampler[dataset_idx] = self._new_sampler(num_samples_in_rank)
            next_sampler_idx = safe_next(self.individual_dataset_sampler[dataset_idx])
            assert next_sampler_idx is not None

        return next_sampler_idx

    def __next__(self) -> int:
        if self.shuffle:
            selected_dataset_idx = self.rng.choice(range(len(self.dataset_weights)), p=self.dataset_probablities)
            in_rank_sample = self._in_rank_next_index_for_dataset(selected_dataset_idx)

            # 2) Getting sample index in the dataset.
            sample_idx_in_dataset = in_rank_sample * self.num_replicas + self.rank

        else:
            # Iterate through the dataset orders in a round-robin fashion, offset by the rank
            dataset_orders_idx = (self.rank + self.drawn_samples) % len(self.dataset_orders)
            selected_dataset_idx = self.dataset_orders[dataset_orders_idx]
            # Get the sample index in the selected dataset by skipping with the num_replicas * drawn_samples
            sample_idx_in_dataset = (self.drawn_samples * self.num_replicas + self.rank) % self.dataset_sizes[
                selected_dataset_idx
            ]
            self.drawn_samples += 1

        # Getting the index of the sample in the whole dataset
        # For example if the total dataset has 4 datasets with sizes [10, 20, 30, 5].
        # and our selected_dataset_idx=3 and sample_idx_in_dataset=5
        # then the index of the sample in the whole dataset is
        #   10 (for dataset 1) + 20 (for dataset 1) + 5 (for sample_idx_in_dataset) = 35
        # This is because the first 10 samples are from the first dataset, the next 20 are from the second dataset,
        # then we reach at the 3rd dataset which is the selected dataset, and the 5th sample in the 3rd dataset.
        sample_idx = 0
        for i, d in enumerate(self.dataset_sizes):
            if selected_dataset_idx == i:
                break
            sample_idx += d
        sample_idx += sample_idx_in_dataset

        return sample_idx
