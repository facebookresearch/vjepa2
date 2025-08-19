# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import torch

from app.vjepa import transforms
from src.datasets.utils.video import functional
from src.datasets.utils.video.volume_transforms import ClipToTensor


class TestNormalize(unittest.TestCase):

    def setUp(self):
        self.g = torch.Generator()
        self.g.manual_seed(42)

    def test_approximation_equivalence(self):
        T, H, W, C = 16, 224, 224, 3
        shape = (T, H, W, C)
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        for i in range(10):
            X = torch.randint(low=0, high=255, size=shape, generator=self.g, dtype=torch.uint8)
            X_clone = X.clone().permute(3, 0, 1, 2)  # C, T, H, W

            X_norm = transforms.tensor_normalize(X, mean, std)
            X_norm_fast = transforms._tensor_normalize_inplace(X_clone, 255.0 * mean, 255.0 * std)
            self.assertTrue(torch.allclose(X_norm, X_norm_fast.permute(1, 2, 3, 0)))


class TestVideoTransformFunctionalCrop(unittest.TestCase):
    def test_tensor_numpy(self):
        T, C, H, W = 16, 3, 280, 320
        shape = (T, C, H, W)
        crop_size = (10, 10, 224, 224)
        video_tensor = torch.randint(low=0, high=255, size=shape, dtype=torch.uint8)
        video_numpy = video_tensor.numpy()

        cropped_tensor = functional.crop_clip(video_tensor, *crop_size)
        self.assertIsInstance(cropped_tensor[0], torch.Tensor)

        cropped_np_array = functional.crop_clip(video_numpy, *crop_size)
        self.assertIsInstance(cropped_np_array[0], np.ndarray)

        for clip_tensor, clip_np in zip(cropped_tensor, cropped_np_array):
            torch.testing.assert_close(clip_tensor, torch.Tensor(clip_np).to(dtype=torch.uint8))


class TestVideoTransformFunctionalResize(unittest.TestCase):
    def test_tensor_numpy(self):
        T, C, H, W = 16, 3, 280, 320
        shape = (T, C, H, W)
        resize_to = 256

        video_tensor = torch.randint(low=0, high=255, size=shape, dtype=torch.int16)
        # We permute the videos because our underlying numpy.array based transforms are expecting
        # image in (H, W, C) shape whereas our tensor transforms are mostly in (C, H, W)
        video_numpy = video_tensor.permute(0, 2, 3, 1).numpy()  # (T, C, H, W) -> (T, H, W, C)

        resized_tensor = functional.resize_clip(video_tensor, resize_to)
        self.assertIsInstance(resized_tensor[0], torch.Tensor)

        resized_np_array = functional.resize_clip(video_numpy, resize_to)
        self.assertIsInstance(resized_np_array[0], np.ndarray)

        for clip_tensor, clip_np in zip(resized_tensor, resized_np_array):
            clip_tensor = clip_tensor.permute(1, 2, 0)
            diff = torch.mean((torch.abs(clip_tensor - torch.Tensor(clip_np).to(torch.int16))) / (clip_tensor + 1))

            # Transformations can not exactly match because of their interpolation functions coming from
            # two different sources. Here we check for their relative differences.
            # See the discussion here: https://github.com/fairinternal/jepa-internal/pull/65#issuecomment-2101833959
            self.assertLess(diff, 0.05)


class TestVideoTransformClipToTensor(unittest.TestCase):
    def test_tensor_numpy(self):
        T, C, H, W = 16, 3, 280, 320
        shape = (T, C, H, W)
        transform = ClipToTensor()

        video_tensor = [clip for clip in torch.randint(low=0, high=255, size=shape, dtype=torch.int16)]
        # We permute the videos because our underlying numpy.array based transforms are expecting
        # image in (H, W, C) shape whereas our tensor transforms are mostly in (C, H, W)
        video_numpy = [clip.permute(1, 2, 0).numpy() for clip in video_tensor]
        torch.testing.assert_close(transform(video_tensor), transform(video_numpy))
