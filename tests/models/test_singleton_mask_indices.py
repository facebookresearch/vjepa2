# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from src.masks.multiseq_multiblock3d import _MaskGenerator


class TestSingletonMaskIndices(unittest.TestCase):
    def test_single_context_and_target_keep_batch_and_token_axes(self):
        generator = _MaskGenerator(
            crop_size=(16, 32), num_frames=2, spatial_patch_size=(16, 16), temporal_patch_size=2,
            spatial_pred_mask_scale=(0.5, 0.5), temporal_pred_mask_scale=(1.0, 1.0),
            aspect_ratio=(1.0, 1.0), npred=1,
        )
        context, target = generator(8)
        self.assertEqual(context.shape, (8, 1))
        self.assertEqual(target.shape, (8, 1))
        self.assertEqual(context.dtype, torch.int64)
        self.assertEqual(target.dtype, torch.int64)
        torch.testing.assert_close(context + target, torch.ones_like(context))

    def test_singleton_complements_and_inverse_blocks(self):
        for option in ({'full_complement': True}, {'pred_full_complement': True}, {'inv_block': True}):
            with self.subTest(option=option):
                generator = _MaskGenerator(
                    crop_size=(16, 32), num_frames=2, spatial_patch_size=(16, 16), temporal_patch_size=2,
                    spatial_pred_mask_scale=(0.5, 0.5), temporal_pred_mask_scale=(1.0, 1.0),
                    aspect_ratio=(1.0, 1.0), npred=1, **option,
                )
                context, target = generator(3)
                self.assertEqual(context.shape, (3, 1))
                self.assertEqual(target.shape, (3, 1))
                self.assertTrue(torch.all(context != target))

    def test_regular_masks_keep_complement_and_valid_indices(self):
        generator = _MaskGenerator(
            crop_size=(32, 32), num_frames=4, spatial_patch_size=(16, 16), temporal_patch_size=2,
            spatial_pred_mask_scale=(0.25, 0.25), temporal_pred_mask_scale=(0.5, 0.5),
            aspect_ratio=(1.0, 1.0), npred=1,
        )
        context, target = generator(4)
        self.assertEqual(context.shape, (4, 7))
        self.assertEqual(target.shape, (4, 1))
        for enc, pred in zip(context, target):
            self.assertEqual(set(enc.tolist()) | set(pred.tolist()), set(range(8)))
            self.assertFalse(set(enc.tolist()) & set(pred.tolist()))


if __name__ == '__main__':
    unittest.main()
