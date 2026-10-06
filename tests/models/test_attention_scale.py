# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy

import pytest
import torch

from app.vjepa_2_1.models.utils import modules as vjepa21
from src.models.utils import modules as vjepa2


@pytest.mark.parametrize(
    "attention_cls",
    [vjepa2.Attention, vjepa2.RoPEAttention, vjepa2.ACRoPEAttention, vjepa21.Attention, vjepa21.RoPEAttention],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("scale", [None, 0.75, 0.001])
def test_attention_scale_matches_explicit_path_outputs_and_gradients(attention_cls, dtype, scale):
    torch.manual_seed(19)
    layer = attention_cls(dim=24, num_heads=2, qkv_bias=True, qk_scale=scale, use_sdpa=True).to(dtype=dtype)
    explicit = copy.deepcopy(layer)
    explicit.use_sdpa = False
    x = torch.randn(2, 8, 24, dtype=dtype, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    kwargs = {}
    if attention_cls in [vjepa2.RoPEAttention, vjepa21.RoPEAttention]:
        kwargs = dict(T=2, H_patches=2, W_patches=2)
    elif attention_cls is vjepa2.ACRoPEAttention:
        kwargs = dict(T=2, H=2, W=2)
    actual = layer(x, **kwargs)
    expected = explicit(reference_x, **kwargs)
    if attention_cls is vjepa21.RoPEAttention:
        actual, _ = actual
        expected, _ = expected
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)

    weights = torch.randn_like(actual)
    actual_grads = torch.autograd.grad((actual * weights).sum(), (x, *layer.parameters()))
    expected_grads = torch.autograd.grad((expected * weights).sum(), (reference_x, *explicit.parameters()))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=5e-5, atol=5e-6)
