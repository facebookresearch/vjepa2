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
import pprint
import random
import time

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from evals.action_anticipation_frozen.dataloader import filter_annotations, init_data
from evals.action_anticipation_frozen.losses import sigmoid_focal_loss
from evals.action_anticipation_frozen.metrics import ClassMeanRecall
from evals.action_anticipation_frozen.models import init_classifier, init_module
from evals.action_anticipation_frozen.utils import init_opt
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter, CSVLogger

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
random.seed(_GLOBAL_SEED)
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.cuda.manual_seed(_GLOBAL_SEED)
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
    val_only = args_eval.get("val_only", False)
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
    num_heads = args_classifier.get("num_heads")

    # -- DATA
    args_data = args_exp.get("data")
    dataset = args_data.get("dataset")  # Name of dataset (e.g., "EK100")
    base_path = args_data.get("base_path")  # Root directory containing videos
    file_format = args_data.get("file_format", 1)
    num_workers = args_data.get("num_workers", 12)
    pin_mem = args_data.get("pin_memory", True)
    # -- / frame sampling hyper-params
    frames_per_clip = args_data.get("frames_per_clip")
    frames_per_second = args_data.get("frames_per_second")
    resolution = args_data.get("resolution", 224)
    # -- / anticipation details
    train_anticipation_time_sec = args_data.get("train_anticipation_time_sec")
    train_anticipation_point = args_data.get("train_anticipation_point")
    val_anticipation_point = args_data.get("val_anticipation_point", [0.0, 0.0])
    val_anticipation_time_sec = args_data.get("anticipation_time_sec")
    # -- / data augmentations
    auto_augment = args_data.get("auto_augment")
    motion_shift = args_data.get("motion_shift")
    reprob = args_data.get("reprob")
    random_resize_scale = args_data.get("random_resize_scale")
    # --
    train_annotations_path = args_data.get("dataset_train")
    val_annotations_path = args_data.get("dataset_val")
    train_data_path = base_path  # os.path.join(base_path, "train")
    val_data_path = base_path  # os.path.join(base_path, "test")

    # -- OPTIMIZATION
    args_opt = args_exp.get("optimization")
    batch_size = args_opt.get("batch_size")
    num_epochs = args_opt.get("num_epochs")
    use_bfloat16 = args_opt.get("use_bfloat16")
    use_focal_loss = args_opt.get("use_focal_loss", False)
    criterion = sigmoid_focal_loss if use_focal_loss else torch.nn.CrossEntropyLoss()
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

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    # -- log/checkpointing paths
    folder = os.path.join(pretrain_folder, "action_anticipation_frozen/")
    if eval_tag is not None:
        folder = os.path.join(folder, eval_tag)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    log_file = os.path.join(folder, f"log_r{rank}.csv")
    latest_path = os.path.join(folder, "latest.pt")

    action_is_verb_noun = True
    if dataset in ["COIN_anticipation"]:
        action_is_verb_noun = False

    # -- make csv_logger
    if rank == 0:
        if action_is_verb_noun:
            csv_logger = CSVLogger(
                log_file,
                ("%d", "epoch"),
                ("%.5f", "train-acc"),
                ("%.5f", "train-acc-verb"),
                ("%.5f", "train-acc-noun"),
                ("%.5f", "train-recall"),
                ("%.5f", "train-recall-verb"),
                ("%.5f", "train-recall-noun"),
                ("%.5f", "val-acc"),
                ("%.5f", "val-acc-verb"),
                ("%.5f", "val-acc-noun"),
                ("%.5f", "val-recall"),
                ("%.5f", "val-recall-verb"),
                ("%.5f", "val-recall-noun"),
            )
        else:
            csv_logger = CSVLogger(
                log_file,
                ("%d", "epoch"),
                ("%.5f", "train-acc"),
                ("%.5f", "train-recall"),
                ("%.5f", "val-acc"),
                ("%.5f", "val-recall"),
            )

    # -- process annotations to unify action class labels between train/val
    _annotations = filter_annotations(
        dataset,
        base_path,
        train_annotations_path,
        val_annotations_path,
        file_format=file_format,
    )
    action_classes = _annotations["actions"]
    verb_classes = {}
    noun_classes = {}
    if action_is_verb_noun:
        verb_classes = _annotations["verbs"]
        noun_classes = _annotations["nouns"]
    # --
    val_actions = _annotations["val_actions"]
    val_verbs = {}
    val_nouns = {}
    if action_is_verb_noun:
        val_verbs = _annotations["val_verbs"]
        val_nouns = _annotations["val_nouns"]
    # --
    train_annotations = _annotations["train"]
    val_annotations = _annotations["val"]

    # -- init models
    model = init_module(
        module_name=module_name,
        frames_per_clip=frames_per_clip,
        frames_per_second=frames_per_second,
        resolution=resolution,
        checkpoint=checkpoint,
        model_kwargs=args_model,
        wrapper_kwargs=args_wrapper,
        device=device,
    )
    classifiers = init_classifier(
        embed_dim=model.embed_dim,
        num_heads=num_heads,
        verb_classes=verb_classes,
        noun_classes=noun_classes,
        action_classes=action_classes,
        num_blocks=num_probe_blocks,
        device=device,
        num_classifiers=len(opt_kwargs),
    )

    # --init data
    train_set, train_loader, train_data_info = init_data(
        dataset=dataset,
        training=True,
        base_path=train_data_path,
        annotations_path=train_annotations,
        batch_size=batch_size,
        frames_per_clip=frames_per_clip,
        fps=frames_per_second,
        anticipation_time_sec=train_anticipation_time_sec,
        anticipation_point=train_anticipation_point,
        # --
        random_resize_scale=random_resize_scale,
        reprob=reprob,
        auto_augment=auto_augment,
        motion_shift=motion_shift,
        # --
        crop_size=resolution,
        world_size=world_size,
        rank=rank,
        num_workers=num_workers,
        pin_mem=pin_mem,
        persistent_workers=False,
    )
    ipe = train_loader.num_batches
    logger.info(f"Dataloader created... iterations per epoch: {ipe}")
    _, val_loader, _ = init_data(
        dataset=dataset,
        training=False,
        base_path=val_data_path,
        annotations_path=val_annotations,
        batch_size=batch_size,
        frames_per_clip=frames_per_clip,
        fps=frames_per_second,
        anticipation_time_sec=val_anticipation_time_sec,
        anticipation_point=val_anticipation_point,
        crop_size=resolution,
        world_size=world_size,
        rank=rank,
        num_workers=num_workers,
        pin_mem=pin_mem,
        persistent_workers=False,
    )
    val_ipe = val_loader.num_batches
    logger.info(f"Val dataloader created... iterations per epoch: {val_ipe}")

    # -- optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        classifiers=classifiers,
        opt_kwargs=opt_kwargs,
        iterations_per_epoch=ipe,
        num_epochs=num_epochs,
        use_bfloat16=use_bfloat16,
    )
    classifiers = [DistributedDataParallel(c, static_graph=True) for c in classifiers]

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
        if val_only:
            start_epoch = 0

    def save_checkpoint(epoch):
        save_dict = {
            "classifiers": [c.state_dict() for c in classifiers],
            "opt": [o.state_dict() for o in optimizer],
            "scaler": None if scaler is None else [s.state_dict() for s in scaler],
            "epoch": epoch,
            "batch_size": batch_size,
            "world_size": world_size,
        }
        if rank == 0:
            torch.save(save_dict, latest_path)

    # -- Train action recognition model
    for epoch in range(start_epoch, num_epochs):
        logging.info(f"Epoch {epoch}")

        train_data_info.set_epoch(epoch)

        # report train action recognition (AR)
        if not val_only:
            logging.info("Training...")
            # with torch.autograd.detect_anomaly(): # TODO Remove
            train_metrics = train_one_epoch(
                action_is_verb_noun=action_is_verb_noun,
                ipe=ipe,
                device=device,
                model=model,
                classifiers=classifiers,
                scaler=scaler,
                optimizer=optimizer,
                scheduler=scheduler,
                wd_scheduler=wd_scheduler,
                data_loader=train_loader,
                use_bfloat16=use_bfloat16,
                verb_classes=verb_classes,
                noun_classes=noun_classes,
                action_classes=action_classes,
                criterion=criterion,
            )

        # report val action anticipation (AA)
        val_metrics = validate(
            action_is_verb_noun=action_is_verb_noun,
            ipe=val_ipe,
            device=device,
            model=model,
            classifiers=classifiers,
            data_loader=val_loader,
            use_bfloat16=use_bfloat16,
            valid_verbs=val_verbs,
            valid_nouns=val_nouns,
            valid_actions=val_actions,
            verb_classes=verb_classes,
            noun_classes=noun_classes,
            action_classes=action_classes,
            criterion=criterion,
        )
        if val_only:
            logger.info(
                "val acc (v/n): %.1f%% (%.1f%% %.1f%%) "
                "val recall (v/n): %.1f%% (%.1f%% %.1f%%) "
                % (
                    val_metrics["action"]["accuracy"],
                    val_metrics["verb"]["accuracy"],
                    val_metrics["noun"]["accuracy"],
                    val_metrics["action"]["recall"],
                    val_metrics["verb"]["recall"],
                    val_metrics["noun"]["recall"],
                )
            )
            return

        if action_is_verb_noun:
            logger.info(
                "[%5d] "
                "train acc (v/n): %.1f%% (%.1f%% %.1f%%) "
                "train recall (v/n): %.1f%% (%.1f%% %.1f%%) "
                "val acc (v/n): %.1f%% (%.1f%% %.1f%%) "
                "val recall (v/n): %.1f%% (%.1f%% %.1f%%) "
                % (
                    epoch + 1,
                    train_metrics["action"]["accuracy"],
                    train_metrics["verb"]["accuracy"],
                    train_metrics["noun"]["accuracy"],
                    train_metrics["action"]["recall"],
                    train_metrics["verb"]["recall"],
                    train_metrics["noun"]["recall"],
                    val_metrics["action"]["accuracy"],
                    val_metrics["verb"]["accuracy"],
                    val_metrics["noun"]["accuracy"],
                    val_metrics["action"]["recall"],
                    val_metrics["verb"]["recall"],
                    val_metrics["noun"]["recall"],
                )
            )
            if rank == 0:
                csv_logger.log(
                    epoch + 1,
                    train_metrics["action"]["accuracy"],
                    train_metrics["verb"]["accuracy"],
                    train_metrics["noun"]["accuracy"],
                    train_metrics["action"]["recall"],
                    train_metrics["verb"]["recall"],
                    train_metrics["noun"]["recall"],
                    val_metrics["action"]["accuracy"],
                    val_metrics["verb"]["accuracy"],
                    val_metrics["noun"]["accuracy"],
                    val_metrics["action"]["recall"],
                    val_metrics["verb"]["recall"],
                    val_metrics["noun"]["recall"],
                )
        else:
            logger.info(
                "[%5d] "
                "train acc (v/n): %.1f%% "
                "train recall (v/n): %.1f%% "
                "val acc (v/n): %.1f%% "
                "val recall (v/n): %.1f%% "
                % (
                    epoch + 1,
                    train_metrics["action"]["accuracy"],
                    train_metrics["action"]["recall"],
                    val_metrics["action"]["accuracy"],
                    val_metrics["action"]["recall"],
                )
            )
            if rank == 0:
                csv_logger.log(
                    epoch + 1,
                    train_metrics["action"]["accuracy"],
                    train_metrics["action"]["recall"],
                    val_metrics["action"]["accuracy"],
                    val_metrics["action"]["recall"],
                )

        save_checkpoint(epoch + 1)


def train_one_epoch(
    action_is_verb_noun,
    ipe,
    device,
    model,
    classifiers,
    scaler,
    optimizer,
    scheduler,
    wd_scheduler,
    data_loader,
    use_bfloat16,
    noun_classes,
    verb_classes,
    action_classes,
    criterion,
):
    _data_loader = iter(data_loader)
    for c in classifiers:
        c.train(mode=True)
    if action_is_verb_noun:
        verb_metric_loggers = [ClassMeanRecall(num_classes=len(verb_classes), device=device, k=5) for _ in classifiers]
        noun_metric_loggers = [ClassMeanRecall(num_classes=len(noun_classes), device=device, k=5) for _ in classifiers]
    action_metric_loggers = [ClassMeanRecall(num_classes=len(action_classes), device=device, k=5) for _ in classifiers]
    data_elapsed_time_meter = AverageMeter()

    for itr in range(ipe):
        itr_start_time = time.time()

        try:
            udata = next(_data_loader)
        except Exception:
            _data_loader = iter(data_loader)
            udata = next(_data_loader)

        [s.step() for s in scheduler]
        [wds.step() for wds in wd_scheduler]

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):

            # Format of udata: ("video", "verb", "noun", "anticipation_time_sec")
            clips = udata[0].to(device)
            anticipation_times = udata[-1].to(device)  # [B]

            if action_is_verb_noun:
                # Map verb/nouns to unified class labels
                _verbs, _nouns = udata[1], udata[2]
                verb_labels, noun_labels, action_labels = [], [], []
                for v, n in zip(_verbs, _nouns):
                    verb_labels.append(verb_classes[int(v)])
                    noun_labels.append(noun_classes[int(n)])
                    action_labels.append(action_classes[(int(v), int(n))])
                verb_labels = torch.tensor(verb_labels).to(device).to(_verbs.dtype)
                noun_labels = torch.tensor(noun_labels).to(device).to(_verbs.dtype)
                action_labels = torch.tensor(action_labels).to(device).to(_verbs.dtype)
            else:
                _actions = udata[1]
                action_labels = [action_classes[str(int(a))] for a in _actions]
                action_labels = torch.tensor(action_labels).to(device).to(_actions.dtype)

            # --
            data_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0
            data_elapsed_time_meter.update(data_elapsed_time_ms)

            # Forward and prediction
            with torch.no_grad():
                outputs = model(clips, anticipation_times)
            outputs = [c(outputs) for c in classifiers]

        # Compute loss & update weights
        if action_is_verb_noun:
            verb_loss = [criterion(o["verb"], verb_labels) for o in outputs]
            noun_loss = [criterion(o["noun"], noun_labels) for o in outputs]
            action_loss = [criterion(o["action"], action_labels) for o in outputs]
            loss = [v + n + a for v, n, a in zip(verb_loss, noun_loss, action_loss)]
        else:
            loss = [criterion(o["action"], action_labels) for o in outputs]
        if use_bfloat16:
            [s.scale(l).backward() for s, l in zip(scaler, loss)]
            [s.step(o) for s, o in zip(scaler, optimizer)]
            [s.update() for s in scaler]
        else:
            [L.backward() for L in loss]
            [o.step() for o in optimizer]
        [o.zero_grad() for o in optimizer]

        # Compute metrics for logging (e.g., accuracy and mean class recall)
        with torch.no_grad():
            if action_is_verb_noun:
                verb_metrics = [m(o["verb"], verb_labels) for o, m in zip(outputs, verb_metric_loggers)]
                noun_metrics = [m(o["noun"], noun_labels) for o, m in zip(outputs, noun_metric_loggers)]
            action_metrics = [m(o["action"], action_labels) for o, m in zip(outputs, action_metric_loggers)]

        if itr % 10 == 0 or itr == ipe - 1:
            if action_is_verb_noun:
                logger.info(
                    "[%5d] "
                    "acc (v/n): %.1f%% (%.1f%% %.1f%%) "
                    "recall (v/n): %.1f%% (%.1f%% %.1f%%) "
                    "[mem: %.2e] "
                    "[data: %.1f ms]"
                    % (
                        itr,
                        max([a["accuracy"] for a in action_metrics]),
                        max([v["accuracy"] for v in verb_metrics]),
                        max([n["accuracy"] for n in noun_metrics]),
                        max([a["recall"] for a in action_metrics]),
                        max([v["recall"] for v in verb_metrics]),
                        max([n["recall"] for n in noun_metrics]),
                        torch.cuda.max_memory_allocated() / 1024.0**2,
                        data_elapsed_time_meter.avg,
                    )
                )
            else:
                logger.info(
                    "[%5d] "
                    "acc (v/n): %.1f%% "
                    "recall (v/n): %.1f%% "
                    "[mem: %.2e] "
                    "[data: %.1f ms]"
                    % (
                        itr,
                        max([a["accuracy"] for a in action_metrics]),
                        max([a["recall"] for a in action_metrics]),
                        torch.cuda.max_memory_allocated() / 1024.0**2,
                        data_elapsed_time_meter.avg,
                    )
                )

    del _data_loader
    ret = dict(
        action=dict(
            accuracy=max([a["accuracy"] for a in action_metrics]),
            recall=max([a["recall"] for a in action_metrics]),
        ),
    )
    if action_is_verb_noun:
        ret.update(
            dict(
                verb=dict(
                    accuracy=max([v["accuracy"] for v in verb_metrics]),
                    recall=max([v["recall"] for v in verb_metrics]),
                ),
                noun=dict(
                    accuracy=max([n["accuracy"] for n in noun_metrics]),
                    recall=max([n["recall"] for n in noun_metrics]),
                ),
            )
        )
    return ret


@torch.no_grad()
def validate(
    action_is_verb_noun,
    ipe,
    device,
    model,
    classifiers,
    data_loader,
    use_bfloat16,
    valid_nouns,
    valid_verbs,
    valid_actions,
    noun_classes,
    verb_classes,
    action_classes,
    criterion,
):
    logger.info("Running val...")
    _data_loader = iter(data_loader)
    for c in classifiers:
        c.train(mode=False)
    if action_is_verb_noun:
        verb_metric_loggers = [ClassMeanRecall(num_classes=len(verb_classes), device=device, k=5) for _ in classifiers]
        noun_metric_loggers = [ClassMeanRecall(num_classes=len(noun_classes), device=device, k=5) for _ in classifiers]
    action_metric_loggers = [ClassMeanRecall(num_classes=len(action_classes), device=device, k=5) for _ in classifiers]

    for itr in range(ipe):
        try:
            udata = next(_data_loader)
        except Exception:
            _data_loader = iter(data_loader)
            udata = next(_data_loader)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
            # Format of udata: ("video", "verb", "noun", "anticipation_time_sec")
            clips = udata[0].to(device)
            anticipation_times = udata[-1].to(device)  # [B]

            if action_is_verb_noun:
                # Map verb/nouns to unified class labels
                _verbs, _nouns = udata[1], udata[2]
                verb_labels, noun_labels, action_labels = [], [], []
                for v, n in zip(_verbs, _nouns):
                    verb_labels.append(verb_classes[int(v)])
                    noun_labels.append(noun_classes[int(n)])
                    action_labels.append(action_classes[(int(v), int(n))])
                verb_labels = torch.tensor(verb_labels).to(device).to(_verbs.dtype)
                noun_labels = torch.tensor(noun_labels).to(device).to(_verbs.dtype)
                action_labels = torch.tensor(action_labels).to(device).to(_verbs.dtype)
            else:
                _actions = udata[1]
                action_labels = [action_classes[str(int(a))] for a in _actions]
                action_labels = torch.tensor(action_labels).to(device).to(_actions.dtype)

            # Forward and prediction
            outputs = model(clips, anticipation_times)
            outputs = [c(outputs) for c in classifiers]

            if action_is_verb_noun:
                verb_loss = sum([criterion(o["verb"], verb_labels) for o in outputs])
                noun_loss = sum([criterion(o["noun"], noun_labels) for o in outputs])
                action_loss = sum([criterion(o["action"], action_labels) for o in outputs])
                loss = verb_loss + noun_loss + action_loss
            else:
                loss = sum([criterion(o["action"], action_labels) for o in outputs])

            action_metrics = [m(o["action"], action_labels) for o, m in zip(outputs, action_metric_loggers)]
            if action_is_verb_noun:
                verb_metrics = [m(o["verb"], verb_labels) for o, m in zip(outputs, verb_metric_loggers)]
                noun_metrics = [m(o["noun"], noun_labels) for o, m in zip(outputs, noun_metric_loggers)]

        if itr % 10 == 0 or itr == ipe - 1:
            if action_is_verb_noun:
                logger.info(
                    "[%5d] "
                    "acc (v/n): %.1f%% (%.1f%% %.1f%%) "
                    "recall (v/n): %.1f%% (%.1f%% %.1f%%) "
                    "loss (v/n): %.3f (%.3f %.3f) "
                    "[mem: %.2e] "
                    % (
                        itr,
                        max([a["accuracy"] for a in action_metrics]),
                        max([v["accuracy"] for v in verb_metrics]),
                        max([n["accuracy"] for n in noun_metrics]),
                        max([a["recall"] for a in action_metrics]),
                        max([v["recall"] for v in verb_metrics]),
                        max([n["recall"] for n in noun_metrics]),
                        loss,
                        verb_loss,
                        noun_loss,
                        torch.cuda.max_memory_allocated() / 1024.0**2,
                    )
                )
            else:
                logger.info(
                    "[%5d] "
                    "acc (v/n): %.1f%% "
                    "recall (v/n): %.1f%% "
                    "loss (v/n): %.3f "
                    "[mem: %.2e] "
                    % (
                        itr,
                        max([a["accuracy"] for a in action_metrics]),
                        max([a["recall"] for a in action_metrics]),
                        loss,
                        torch.cuda.max_memory_allocated() / 1024.0**2,
                    )
                )

    del _data_loader
    ret = dict(
        action=dict(
            accuracy=max([a["accuracy"] for a in action_metrics]),
            recall=max([a["recall"] for a in action_metrics]),
        ),
    )
    if action_is_verb_noun:
        ret.update(
            dict(
                verb=dict(
                    accuracy=max([v["accuracy"] for v in verb_metrics]),
                    recall=max([v["recall"] for v in verb_metrics]),
                ),
                noun=dict(
                    accuracy=max([n["accuracy"] for n in noun_metrics]),
                    recall=max([n["recall"] for n in noun_metrics]),
                ),
            )
        )
    return ret


def load_checkpoint(device, r_path, classifiers, opt, scaler, val_only=False):
    logger.info(f"read-path: {r_path}")
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))

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
        [s.load_state_dict(c) for s, c in zip(opt, checkpoint["scaler"])]
    logger.info(f"loaded optimizers from epoch {epoch}")

    return classifiers, opt, scaler, epoch
