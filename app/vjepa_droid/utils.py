# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import sys

import torch

import src.models.ac_predictor as vit_ac_pred
import src.models.vision_transformer as video_vit
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.schedulers import CosineWDSchedule, WSDSchedule

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def _migrate_predictor_optimizer_state(optimizer_state, num_param_groups):
    """Drop the unused encoder groups from pre-fix V-JEPA2-AC checkpoints."""
    if num_param_groups != 2 or len(optimizer_state["param_groups"]) != 4:
        return optimizer_state

    predictor_groups = [optimizer_state["param_groups"][1], optimizer_state["param_groups"][3]]
    predictor_param_ids = {param_id for group in predictor_groups for param_id in group["params"]}
    return {
        **optimizer_state,
        "state": {
            param_id: state for param_id, state in optimizer_state["state"].items() if param_id in predictor_param_ids
        },
        "param_groups": predictor_groups,
    }


def load_pretrained(
    r_path,
    encoder=None,
    predictor=None,
    target_encoder=None,
    context_encoder_key="encoder",
    target_encoder_key="target_encoder",
    load_predictor=False,
    load_encoder=True,
):
    logger.info(f"Loading pretrained model from {r_path}")
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))

    epoch = checkpoint["epoch"]

    if load_encoder and encoder is not None:
        # -- loading encoder
        pretrained_dict = checkpoint[context_encoder_key]
        pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
        msg = encoder.load_state_dict(pretrained_dict, strict=False)
        logger.info(f"loaded pretrained encoder from epoch {epoch} with msg: {msg}")

    if load_predictor:
        # -- loading predictor
        pretrained_dict = checkpoint["predictor"]
        pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
        msg = predictor.load_state_dict(pretrained_dict, strict=False)
        logger.info(f"loaded pretrained predictor from epoch {epoch} with msg: {msg}")

    # -- loading target_encoder
    if load_encoder:
        if target_encoder is not None:
            print(list(checkpoint.keys()))
            pretrained_dict = checkpoint[target_encoder_key]
            pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
            msg = target_encoder.load_state_dict(pretrained_dict, strict=False)
            logger.info(f"loaded pretrained target encoder from epoch {epoch} with msg: {msg}")

    del checkpoint

    return (
        encoder,
        predictor,
        target_encoder,
    )


def load_checkpoint(
    r_path,
    encoder,
    predictor,
    target_encoder,
    opt=None,
    scaler=None,
    replace_kw=["backbone."],
):
    logger.info(f"Loading checkpoint from {r_path}")
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))

    epoch = checkpoint["epoch"]

    if encoder is not None:
        # -- loading encoder
        pretrained_dict = checkpoint["encoder"]
        for kw in replace_kw:
            pretrained_dict = {k.replace(kw, ""): v for k, v in pretrained_dict.items()}
        msg = encoder.load_state_dict(pretrained_dict, strict=False)
        logger.info(f"loaded pretrained encoder from epoch {epoch} with msg: {msg}")

    # -- loading predictor
    pretrained_dict = checkpoint["predictor"]
    for kw in replace_kw:
        pretrained_dict = {k.replace(kw, ""): v for k, v in pretrained_dict.items()}
    msg = predictor.load_state_dict(pretrained_dict, strict=False)
    logger.info(f"loaded pretrained predictor from epoch {epoch} with msg: {msg}")

    # -- loading target_encoder
    if target_encoder is not None:
        print(list(checkpoint.keys()))
        pretrained_dict = checkpoint["target_encoder"]
        for kw in replace_kw:
            pretrained_dict = {k.replace(kw, ""): v for k, v in pretrained_dict.items()}
        msg = target_encoder.load_state_dict(pretrained_dict, strict=False)
        logger.info(f"loaded pretrained target encoder from epoch {epoch} with msg: {msg}")

    # -- loading optimizer
    if opt is not None:
        optimizer_state = _migrate_predictor_optimizer_state(checkpoint["opt"], len(opt.param_groups))
        opt.load_state_dict(optimizer_state)

    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler"])

    logger.info(f"loaded optimizers from epoch {epoch}")
    logger.info(f"read-path: {r_path}")
    del checkpoint

    return (
        encoder,
        predictor,
        target_encoder,
        opt,
        scaler,
        epoch,
    )


def init_video_model(
    device,
    patch_size=16,
    max_num_frames=16,
    tubelet_size=2,
    model_name="vit_base",
    crop_size=224,
    pred_depth=6,
    pred_num_heads=None,
    pred_embed_dim=384,
    uniform_power=False,
    use_sdpa=False,
    use_rope=False,
    use_silu=False,
    use_pred_silu=False,
    wide_silu=False,
    pred_is_frame_causal=True,
    use_activation_checkpointing=False,
    return_all_tokens=False,
    action_embed_dim=7,
    use_extrinsics=False,
    old_pred=False,
):
    encoder = video_vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
        use_rope=use_rope,
    )

    predictor = vit_ac_pred.__dict__["vit_ac_predictor"](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=pred_embed_dim,
        action_embed_dim=action_embed_dim,
        depth=pred_depth,
        is_frame_causal=pred_is_frame_causal,
        num_heads=encoder.num_heads if pred_num_heads is None else pred_num_heads,
        uniform_power=uniform_power,
        use_rope=use_rope,
        use_sdpa=use_sdpa,
        use_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_extrinsics=use_extrinsics,
        use_activation_checkpointing=use_activation_checkpointing,
    )

    encoder.to(device)
    predictor.to(device)
    logger.info(encoder)
    logger.info(predictor)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Encoder number of parameters: {count_parameters(encoder)}")
    logger.info(f"Predictor number of parameters: {count_parameters(predictor)}")

    return encoder, predictor


def init_opt(
    predictor,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    anneal,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    mixed_precision=False,
    betas=(0.9, 0.999),
    eps=1e-8,
    zero_init_bias_wd=True,
):
    param_groups = [
        {
            "params": (p for n, p in predictor.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)),
        },
        {
            "params": (p for n, p in predictor.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
    ]

    optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)
    scheduler = WSDSchedule(
        optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        anneal_steps=int(anneal * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(num_epochs * iterations_per_epoch),
    )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(num_epochs * iterations_per_epoch),
    )
    scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
    return optimizer, scaler, scheduler, wd_scheduler
