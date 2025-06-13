# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

import logging
import math
import pprint

import numpy as np
import torch
import torch.multiprocessing as mp
import torchvision.transforms as transforms
from timm.data import create_transform as timm_make_transforms
from torch.nn.parallel import DistributedDataParallel

from evals.image_classification_frozen.models import init_module
from src.datasets.data_manager import init_data
from src.models.attentive_pooler import AttentiveClassifier
from src.models.utils.modules import Block, CrossAttentionBlock
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.distributed import AllReduce, init_distributed
from src.utils.logging import AverageMeter, CSVLogger

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

pp = pprint.PrettyPrinter(indent=4)


def main(args_eval, resume_preempt=False):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- VAL ONLY
    val_only = args_eval.get("val_only", False)
    if val_only:
        logger.info("VAL ONLY")

    # -- EXPERIMENT
    pretrain_folder = args_eval.get("folder", None)
    resume_checkpoint = args_eval.get("resume_checkpoint", False) or resume_preempt
    eval_tag = args_eval.get("tag", None)

    # -- PRETRAIN
    args_pretrain = args_eval.get("model_kwargs")
    checkpoint = args_pretrain.get("checkpoint")
    module_name = args_pretrain.get("module_name")
    args_model = args_pretrain.get("pretrain_kwargs")
    args_wrapper = args_pretrain.get("wrapper_kwargs")

    args_exp = args_eval.get("experiment")

    # -- CLASSIFIER
    args_classifier = args_exp.get("classifier")
    num_probe_blocks = args_classifier.get("num_probe_blocks", 1)
    num_heads = args_classifier.get("num_heads", 16)

    # -- DATA
    args_data = args_exp.get("data")
    dataset_name = args_data.get("dataset_name")
    num_classes = args_data.get("num_classes")
    root_path = args_data.get("root_path", None)
    image_folder = args_data.get("image_folder", None)
    resolution = args_data.get("resolution", 224)
    normalization = args_data.get("normalization", None)

    # -- OPTIMIZATION
    args_opt = args_exp.get("optimization")
    batch_size = args_opt.get("batch_size")
    num_epochs = args_opt.get("num_epochs")
    use_bfloat16 = args_opt.get("use_bfloat16")
    opt_kwargs = [
        dict(
            ref_wd=kwargs.get("weight_decay"),
            final_wd=kwargs.get("final_weight_decay"),
            start_lr=kwargs.get("start_lr"),
            ref_lr=kwargs.get("lr"),
            final_lr=kwargs.get("final_lr"),
            warmup=kwargs.get("warmup"),
        )
        for kwargs in args_opt.get("multihead_kwargs")
    ]
    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    # -- log/checkpointing paths
    folder = os.path.join(pretrain_folder, "image_classification_frozen/")
    if eval_tag is not None:
        folder = os.path.join(folder, eval_tag)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    log_file = os.path.join(folder, f"log_r{rank}.csv")
    latest_path = os.path.join(folder, "latest.pt")

    # -- make csv_logger
    if rank == 0:
        csv_logger = CSVLogger(log_file, ("%d", "epoch"), ("%.5f", "loss"), ("%.5f", "acc"))

    # -- init models
    encoder = init_module(
        module_name=module_name,
        resolution=resolution,
        checkpoint=checkpoint,
        model_kwargs=args_model,
        wrapper_kwargs=args_wrapper,
        device=device,
    )
    # -- init classifier
    classifiers = [
        AttentiveClassifier(
            embed_dim=encoder.embed_dim,
            num_heads=num_heads,
            depth=num_probe_blocks,
            num_classes=num_classes,
            use_activation_checkpointing=True,
        ).to(device)
        for _ in opt_kwargs
    ]
    classifiers = [DistributedDataParallel(c, static_graph=True) for c in classifiers]
    print(classifiers[0])

    train_loader, train_sampler = make_dataloader(
        dataset_name=dataset_name,
        root_path=root_path,
        img_size=resolution,
        image_folder=image_folder,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        training=True,
        normalization=normalization,
    )
    val_loader, _ = make_dataloader(
        dataset_name=dataset_name,
        root_path=root_path,
        img_size=resolution,
        image_folder=image_folder,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        training=False,
        normalization=normalization,
    )
    ipe = len(train_loader)
    logger.info(f"Dataloader created... iterations per epoch: {ipe}")

    # -- optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        classifiers=classifiers,
        opt_kwargs=opt_kwargs,
        iterations_per_epoch=ipe,
        num_epochs=num_epochs,
        use_bfloat16=use_bfloat16,
    )

    # -- load training checkpoint
    start_epoch = 0
    if resume_checkpoint and os.path.exists(latest_path):
        classifiers, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=latest_path,
            classifiers=classifiers,
            opt=optimizer,
            scaler=scaler,
            val_only=val_only,
        )
        for _ in range(start_epoch * ipe):
            [s.step() for s in scheduler]
            [wds.step() for wds in wd_scheduler]

    def save_checkpoint(epoch):
        all_classifier_dicts = [c.state_dict() for c in classifiers]
        all_opt_dicts = [o.state_dict() for o in optimizer]

        save_dict = {
            "classifiers": all_classifier_dicts,
            "opt": all_opt_dicts,
            "scaler": None if scaler is None else [s.state_dict() for s in scaler],
            "epoch": epoch,
            "batch_size": batch_size,
            "world_size": world_size,
        }
        if rank == 0:
            torch.save(save_dict, latest_path)

    # TRAIN LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info("Epoch %d" % (epoch + 1))
        train_sampler.set_epoch(epoch)
        if val_only:
            train_acc = -1.0
        else:
            train_acc = run_one_epoch(
                device=device,
                training=True,
                encoder=encoder,
                classifiers=classifiers,
                scaler=scaler,
                optimizer=optimizer,
                scheduler=scheduler,
                wd_scheduler=wd_scheduler,
                data_loader=train_loader,
                use_bfloat16=use_bfloat16,
            )

        val_acc = run_one_epoch(
            device=device,
            training=False,
            encoder=encoder,
            classifiers=classifiers,
            scaler=scaler,
            optimizer=optimizer,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            data_loader=val_loader,
            use_bfloat16=use_bfloat16,
        )

        logger.info("[%5d] train: %.3f%% test: %.3f%%" % (epoch + 1, train_acc, val_acc))
        if rank == 0:
            csv_logger.log(epoch + 1, train_acc, val_acc)

        if val_only:
            return

        save_checkpoint(epoch + 1)


def run_one_epoch(
    device,
    training,
    encoder,
    classifiers,
    scaler,
    optimizer,
    scheduler,
    wd_scheduler,
    data_loader,
    use_bfloat16,
):

    for c in classifiers:
        c.train(mode=training)

    criterion = torch.nn.CrossEntropyLoss()
    top1_meters = [AverageMeter() for _ in classifiers]
    for itr, data in enumerate(data_loader):
        if training:
            [s.step() for s in scheduler]
            [wds.step() for wds in wd_scheduler]

        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_bfloat16):
            imgs, labels = data[0].to(device), data[1].to(device)
            with torch.no_grad():
                outputs = encoder(imgs)
                if not training:
                    outputs = [c(outputs) for c in classifiers]
            if training:
                outputs = [c(outputs) for c in classifiers]

        losses = [criterion(o, labels) for o in outputs]
        top1_accs = [100.0 * o.max(dim=1).indices.eq(labels).sum() / len(imgs) for o in outputs]
        top1_accs = [float(AllReduce.apply(t)) for t in top1_accs]
        for t1m, t1a in zip(top1_meters, top1_accs):
            t1m.update(t1a)

        if training:
            if use_bfloat16:
                [s.scale(l).backward() for s, l in zip(scaler, losses)]
                [s.step(o) for s, o in zip(scaler, optimizer)]
                [s.update() for s in scaler]
            else:
                [loss.backward() for loss in losses]
                [o.step() for o in optimizer]
            [o.zero_grad() for o in optimizer]

        _agg_top1 = np.array([t1m.avg for t1m in top1_meters])
        if itr % 20 == 0:
            logger.info(
                "[%5d] %.3f%% [%.3f%% %.3f%%] [mem: %.2e]"
                % (
                    itr,
                    _agg_top1.max(),
                    _agg_top1.mean(),
                    _agg_top1.min(),
                    torch.cuda.max_memory_allocated() / 1024.0**2,
                )
            )

    return _agg_top1.max()


def load_checkpoint(device, r_path, classifiers, opt, scaler, val_only=False):
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))
    logger.info(f"read-path: {r_path}")

    # -- loading encoder
    msg = [c.load_state_dict(pd) for c, pd in zip(classifiers, checkpoint["classifiers"])]

    if val_only:
        logger.info(f"loaded pretrained classifier from epoch with msg: {msg}")
        return classifiers, opt, scaler, 0

    epoch = checkpoint["epoch"]
    logger.info(f"loaded pretrained classifier from epoch {epoch} with msg: {msg}")

    # -- loading optimizer
    [o.load_state_dict(c) for o, c in zip(opt, checkpoint["opt"])]

    if scaler is not None:
        [s.load_state_dict(c) for s, c in zip(scaler, checkpoint["scaler"])]
    logger.info(f"loaded optimizers from epoch {epoch}")

    return classifiers, opt, scaler, epoch


DEFAULT_NORMALIZATION = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def make_dataloader(
    dataset_name,
    root_path,
    image_folder,
    batch_size,
    world_size,
    rank,
    img_size=224,
    training=False,
    subset_file=None,
    normalization=None,
):
    if normalization is None:
        normalization = DEFAULT_NORMALIZATION

    if training:
        logger.info("implementing auto-augment strategy")
        transform = timm_make_transforms(
            input_size=img_size,
            is_training=training,
            auto_augment="original",
            interpolation="bicubic",
            re_prob=0.25,
            re_mode="pixel",
            re_count=1,
            mean=normalization[0],
            std=normalization[1],
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(size=int(img_size * 256 / 224)),
                transforms.CenterCrop(size=img_size),
                transforms.ToTensor(),
                transforms.Normalize(normalization[0], normalization[1]),
            ]
        )

    data_loader, data_sampler = init_data(
        data=dataset_name,
        transform=transform,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        root_path=root_path,
        image_folder=image_folder,
        training=training,
        drop_last=False,
        subset_file=subset_file,
    )
    return data_loader, data_sampler


def init_opt(classifiers, iterations_per_epoch, opt_kwargs, num_epochs, use_bfloat16=False):
    optimizers, schedulers, wd_schedulers, scalers = [], [], [], []
    for c, kwargs in zip(classifiers, opt_kwargs):
        param_groups = [
            {
                "params": (p for n, p in c.named_parameters()),
                "mc_warmup_steps": int(kwargs.get("warmup") * iterations_per_epoch),
                "mc_start_lr": kwargs.get("start_lr"),
                "mc_ref_lr": kwargs.get("ref_lr"),
                "mc_final_lr": kwargs.get("final_lr"),
                "mc_ref_wd": kwargs.get("ref_wd"),
                "mc_final_wd": kwargs.get("final_wd"),
            }
        ]
        logger.info("Using AdamW")
        optimizers += [torch.optim.AdamW(param_groups)]
        schedulers += [WarmupCosineLRSchedule(optimizers[-1], T_max=int(num_epochs * iterations_per_epoch))]
        wd_schedulers += [CosineWDSchedule(optimizers[-1], T_max=int(num_epochs * iterations_per_epoch))]
        scalers += [torch.cuda.amp.GradScaler() if use_bfloat16 else None]
    return optimizers, scalers, schedulers, wd_schedulers


class WarmupCosineLRSchedule(object):

    def __init__(self, optimizer, T_max, last_epoch=-1):
        self.optimizer = optimizer
        self.T_max = T_max
        self._step = 0.0

    def step(self):
        self._step += 1
        for group in self.optimizer.param_groups:
            ref_lr = group.get("mc_ref_lr")
            final_lr = group.get("mc_final_lr")
            start_lr = group.get("mc_start_lr")
            warmup_steps = group.get("mc_warmup_steps")
            T_max = self.T_max - warmup_steps
            if self._step < warmup_steps:
                progress = float(self._step) / float(max(1, warmup_steps))
                new_lr = start_lr + progress * (ref_lr - start_lr)
            else:
                # -- progress after warmup
                progress = float(self._step - warmup_steps) / float(max(1, T_max))
                new_lr = max(
                    final_lr,
                    final_lr + (ref_lr - final_lr) * 0.5 * (1.0 + math.cos(math.pi * progress)),
                )
            group["lr"] = new_lr


class CosineWDSchedule(object):

    def __init__(self, optimizer, T_max):
        self.optimizer = optimizer
        self.T_max = T_max
        self._step = 0.0

    def step(self):
        self._step += 1
        progress = self._step / self.T_max

        for group in self.optimizer.param_groups:
            ref_wd = group.get("mc_ref_wd")
            final_wd = group.get("mc_final_wd")
            new_wd = final_wd + (ref_wd - final_wd) * 0.5 * (1.0 + math.cos(math.pi * progress))
            if final_wd <= ref_wd:
                new_wd = max(final_wd, new_wd)
            else:
                new_wd = min(final_wd, new_wd)
            group["weight_decay"] = new_wd
