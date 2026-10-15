# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from src.datasets.utils.weighted_sampler import (
    MemoryEfficientDistributedWeightedSampler,
    MemoryEfficientDistributedWeightedSamplerLessRepeat,
)


class Dataset:
    def __init__(self):
        self.datasets = [range(120), range(240)]
        self.dataset_weights = [1, 2]

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)


SAMPLERS = (
    MemoryEfficientDistributedWeightedSampler,
    MemoryEfficientDistributedWeightedSamplerLessRepeat,
)


def make_sampler(cls, rank=0, shuffle=True):
    return cls(Dataset(), num_replicas=4, rank=rank, seed=17, shuffle=shuffle)


def draw(sampler, count=200):
    return [next(sampler) for _ in range(count)]


class TestWeightedSamplerEpoch(unittest.TestCase):
    def test_epoch_is_reproducible_from_fresh_sampler(self):
        for cls in SAMPLERS:
            with self.subTest(sampler=cls.__name__):
                sampler = make_sampler(cls)
                draw(sampler, 93)
                sampler.set_epoch(8)
                expected = draw(sampler)
                fresh = make_sampler(cls)
                fresh.set_epoch(8)
                self.assertEqual(draw(fresh), expected)
                sampler.set_epoch(8)
                self.assertEqual(draw(sampler), expected)
                sampler.set_epoch(9)
                self.assertNotEqual(draw(sampler), expected)

    def test_rank_partition_is_preserved_after_epoch_changes(self):
        for cls in SAMPLERS:
            for epoch in (0, 1, 17):
                for rank in range(4):
                    with self.subTest(sampler=cls.__name__, epoch=epoch, rank=rank):
                        sampler = make_sampler(cls, rank=rank)
                        sampler.set_epoch(epoch)
                        for index in draw(sampler):
                            self.assertEqual(index % 4, rank)
                            self.assertGreaterEqual(index, 0)
                            self.assertLess(index, len(Dataset()))

    def test_nonshuffled_epoch_restarts_ordered_iteration(self):
        for cls in SAMPLERS:
            with self.subTest(sampler=cls.__name__):
                sampler = make_sampler(cls, rank=3, shuffle=False)
                expected = draw(sampler, 31)
                draw(sampler, 71)
                sampler.set_epoch(4)
                self.assertEqual(draw(sampler, 31), expected)


if __name__ == '__main__':
    unittest.main()
