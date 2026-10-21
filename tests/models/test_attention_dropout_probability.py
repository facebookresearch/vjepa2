# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy

import pytest
import torch

from app.vjepa_2_1.models.utils import modules as vjepa21
from src.models.utils import modules as vjepa2

ATTENTION_CLASSES = [
    vjepa2.Attention,
    vjepa2.RoPEAttention,
    vjepa2.ACRoPEAttention,
    vjepa21.Attention,
    vjepa21.RoPEAttention,
]


def run_attention(layer, inputs):
    kwargs = {}
    if isinstance(layer, (vjepa2.RoPEAttention, vjepa21.RoPEAttention)):
        kwargs = dict(T=2, H_patches=2, W_patches=2)
    elif isinstance(layer, vjepa2.ACRoPEAttention):
        kwargs = dict(T=2, H=2, W=2)
    output = layer(inputs, **kwargs)
    return output[0] if isinstance(layer, vjepa21.RoPEAttention) else output


@pytest.mark.parametrize("attention_cls", ATTENTION_CLASSES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("attn_drop,proj_drop", [(0.35, 0.0), (0.0, 0.45), (1.0, 0.0), (0.0, 0.0)])
def test_training_attention_dropout_matches_explicit_probabilities(attention_cls, dtype, attn_drop, proj_drop):
    torch.manual_seed(23)
    layer = attention_cls(
        dim=24, num_heads=2, qkv_bias=True, attn_drop=attn_drop, proj_drop=proj_drop, use_sdpa=True
    ).to(dtype=dtype)
    layer.train()
    explicit = copy.deepcopy(layer)
    explicit.use_sdpa = False
    inputs = torch.randn(2, 8, 24, dtype=dtype, requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_()

    # The CPU math kernel and explicit path draw their probability masks from
    # the same RNG stream. Projection dropout remains a separate later draw.
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        torch.manual_seed(41)
        actual = run_attention(layer, inputs)
        torch.manual_seed(41)
        expected = run_attention(explicit, reference_inputs)

    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
    if attn_drop == 1.0:
        # With every attention probability dropped, only the output bias remains.
        torch.testing.assert_close(actual, layer.proj.bias.expand_as(actual), rtol=0, atol=0)

    weights = torch.randn_like(actual)
    actual_grads = torch.autograd.grad((actual * weights).sum(), (inputs, *layer.parameters()))
    expected_grads = torch.autograd.grad((expected * weights).sum(), (reference_inputs, *explicit.parameters()))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=6e-5, atol=5e-6)
