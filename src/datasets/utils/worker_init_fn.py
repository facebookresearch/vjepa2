# Copyright (c) Meta Platforms, Inc. and affiliates.

# Copyright The Lightning AI team.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This code originally comes from PyTorch Lighting with some light modifications:
# https://github.com/Lightning-AI/pytorch-lightning/blob/a944e7744e57a5a2c13f3c73b9735edf2f71e329/src/lightning/fabric/utilities/seed.py


import os
import random
from typing import Optional

import numpy as np
import torch

from src.utils.logging import get_logger

logger = get_logger("worker_init_fn")


def _generate_seed_sequence(base_seed: int, worker_id: int, global_rank: int, count: int) -> list[int]:
    """Generates a sequence of seeds from a base seed, worker id and rank using the linear congruential generator (LCG)
    algorithm."""
    # Combine base seed, worker id and rank into a unique 64-bit number
    combined_seed = (base_seed << 32) | (worker_id << 16) | global_rank
    seeds = []
    for _ in range(count):
        # x_(n+1) = (a * x_n + c) mod m. With c=1, m=2^64 and a is D. Knuth's constant
        combined_seed = (combined_seed * 6364136223846793005 + 1) & ((1 << 64) - 1)
        seeds.append(combined_seed)
    return seeds


def pl_worker_init_function(worker_id: int, rank: Optional[int] = None) -> None:  # pragma: no cover
    r"""The worker_init_fn that Lightning automatically adds to your dataloader if you previously set the seed with
    ``seed_everything(seed, workers=True)``.

    See also the PyTorch documentation on
    `randomness in DataLoaders <https://pytorch.org/docs/stable/notes/randomness.html#dataloader>`_.

    """
    # implementation notes: https://github.com/pytorch/pytorch/issues/5059#issuecomment-817392562
    if rank is None:
        procid = os.environ.get("SLURM_PROCID")
        if procid is None:
            logger.warning("SLURM_PROCID is not set, setting rank to 0")
            rank = 0
        else:
            rank = int(procid)

    process_seed = torch.initial_seed()
    # back out the base seed so we can use all the bits
    base_seed = process_seed - worker_id
    logger.debug(
        f"Initializing random number generators of process {rank} worker {worker_id} with base seed {base_seed}"
    )
    seed_sequence = _generate_seed_sequence(base_seed, worker_id, rank, count=4)
    torch.manual_seed(seed_sequence[0])  # torch takes a 64-bit seed
    random.seed((seed_sequence[1] << 32) | seed_sequence[2])  # combine two 64-bit seeds

    ss = np.random.SeedSequence([base_seed, worker_id, rank])
    np_rng_seed = ss.generate_state(4)

    np.random.seed(np_rng_seed)
