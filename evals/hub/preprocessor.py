# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


def _make_transforms(crop_size=256, preserve_border=False):
    from ..video_classification_frozen.utils import make_transforms

    return make_transforms(crop_size=crop_size, training=False, preserve_border=preserve_border)


def vjepa2_preprocessor(*, pretrained: bool = True, preserve_border: bool = False, **kwargs):
    crop_size = kwargs.get("crop_size", 256)
    return _make_transforms(crop_size=crop_size, preserve_border=preserve_border)
