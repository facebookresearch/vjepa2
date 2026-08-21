# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from app.vjepa_droid.utils import _migrate_predictor_optimizer_state, init_opt


def test_predictor_optimizer_excludes_frozen_encoder_parameters():
    predictor = torch.nn.Linear(4, 3)

    optimizer, _, _, _ = init_opt(
        predictor=predictor,
        iterations_per_epoch=2,
        start_lr=1e-4,
        ref_lr=1e-3,
        warmup=1,
        anneal=1,
        num_epochs=2,
    )

    optimized_parameters = {id(param) for group in optimizer.param_groups for param in group["params"]}
    assert optimized_parameters == {id(param) for param in predictor.parameters()}
    assert len(optimizer.param_groups) == 2


def test_migrate_predictor_optimizer_state_from_four_groups():
    optimizer_state = {
        "state": {index: {"step": index} for index in range(6)},
        "param_groups": [
            {"params": [0, 1]},
            {"params": [2, 3]},
            {"params": [4]},
            {"params": [5]},
        ],
    }

    migrated = _migrate_predictor_optimizer_state(optimizer_state, num_param_groups=2)

    assert [group["params"] for group in migrated["param_groups"]] == [[2, 3], [5]]
    assert set(migrated["state"]) == {2, 3, 5}


def test_migrated_state_loads_into_predictor_only_optimizer():
    old_encoder = torch.nn.Linear(4, 3)
    old_predictor = torch.nn.Linear(4, 3)
    old_optimizer = torch.optim.AdamW(
        [
            {"params": [old_encoder.weight]},
            {"params": [old_predictor.weight]},
            {"params": [old_encoder.bias], "weight_decay": 0},
            {"params": [old_predictor.bias], "weight_decay": 0},
        ]
    )
    for param in [*old_encoder.parameters(), *old_predictor.parameters()]:
        param.grad = torch.ones_like(param)
    old_optimizer.step()

    new_predictor = torch.nn.Linear(4, 3)
    new_optimizer, _, _, _ = init_opt(
        predictor=new_predictor,
        iterations_per_epoch=2,
        start_lr=1e-4,
        ref_lr=1e-3,
        warmup=1,
        anneal=1,
        num_epochs=2,
    )
    migrated = _migrate_predictor_optimizer_state(
        old_optimizer.state_dict(), num_param_groups=len(new_optimizer.param_groups)
    )

    new_optimizer.load_state_dict(migrated)

    assert len(new_optimizer.state) == len(list(new_predictor.parameters()))
