# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools

import pytest

from src.datasets.utils.weighted_sampler import (
    MemoryEfficientDistributedWeightedSampler,
    MemoryEfficientDistributedWeightedSamplerLessRepeat,
)


class Dataset:
    def __init__(self, sizes, weights):
        self.datasets = [range(size) for size in sizes]
        self.dataset_weights = weights

    def __len__(self):
        return sum(map(len, self.datasets))


@pytest.mark.parametrize(
    "sampler_cls", [MemoryEfficientDistributedWeightedSampler, MemoryEfficientDistributedWeightedSamplerLessRepeat]
)
@pytest.mark.parametrize("weights", [[1, 1], [1, 3], [2, 2]])
@pytest.mark.parametrize("replicas", [1, 2, 3])
def test_sequential_mixture_visits_every_rank_assigned_sample(sampler_cls, weights, replicas):
    sizes = [12, 18]
    dataset = Dataset(sizes, weights)
    draws = sum(weights) * max(size // replicas for size in sizes)
    for rank in range(replicas):
        sampler = sampler_cls(dataset, num_replicas=replicas, rank=rank, shuffle=False)
        actual = list(itertools.islice(sampler, draws))
        expected = set(range(rank, sizes[0], replicas))
        expected.update(sizes[0] + index for index in range(rank, sizes[1], replicas))
        assert set(actual) == expected
        # The dataset mixture order and its weights stay unchanged.
        dataset_ids = [int(index >= sizes[0]) for index in actual]
        order = [i for i, weight in enumerate(weights) for _ in range(weight)]
        assert dataset_ids == [order[(rank + step) % len(order)] for step in range(draws)]


@pytest.mark.parametrize(
    "sampler_cls", [MemoryEfficientDistributedWeightedSampler, MemoryEfficientDistributedWeightedSamplerLessRepeat]
)
def test_single_dataset_keeps_its_existing_sequential_order(sampler_cls):
    sampler = sampler_cls(Dataset([12], [1]), num_replicas=3, rank=1, shuffle=False)
    assert list(itertools.islice(sampler, 8)) == [1, 4, 7, 10, 1, 4, 7, 10]


@pytest.mark.parametrize(
    "sampler_cls", [MemoryEfficientDistributedWeightedSampler, MemoryEfficientDistributedWeightedSamplerLessRepeat]
)
@pytest.mark.parametrize("weights", [[1, 1], [1, 3], [2, 2]])
def test_each_dataset_traversal_restarts_on_epoch_change(sampler_cls, weights):
    dataset = Dataset([12, 18], weights)
    sampler = sampler_cls(dataset, num_replicas=3, rank=1, shuffle=False)
    expected = list(itertools.islice(sampler, 19))
    list(itertools.islice(sampler, 17))
    sampler.set_epoch(5)
    assert list(itertools.islice(sampler, 19)) == expected
    sampler.set_epoch(5)
    assert list(itertools.islice(sampler, 19)) == expected
