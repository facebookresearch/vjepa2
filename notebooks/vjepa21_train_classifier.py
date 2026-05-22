# Copyright (c) Meta Platforms, Inc. and affiliates.
# Train a video classifier on top of a frozen V-JEPA 2.1 encoder.
#
# Expected dataset layout (standard folder-per-class):
#   <data_root>/
#     train/
#       class_a/  video1.mp4  video2.mp4 ...
#       class_b/  ...
#     val/
#       class_a/  ...
#       class_b/  ...
#
# Run:
#   python -m notebooks.vjepa21_train_classifier \
#       --data_root /path/to/dataset \
#       --checkpoint_path vjepa2_1_vitb_dist_vitG_384.pt \
#       --num_classes 10

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from decord import VideoReader
from torch.utils.data import DataLoader, Dataset

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms
from app.vjepa_2_1.models import vision_transformer as vit_encoder_21
from src.hub.backbones import _clean_backbone_key
from src.models.attentive_pooler import AttentiveClassifier

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Train a video classifier on top of V-JEPA 2.1"
    )

    # Paths
    p.add_argument(
        "--data_root",
        required=True,
        help="Root dir with train/ and val/ sub-folders (one sub-folder per class)",
    )
    p.add_argument(
        "--checkpoint_path",
        required=True,
        help="Path to the V-JEPA 2.1 .pt encoder checkpoint",
    )
    p.add_argument(
        "--output_dir",
        default="classifier_checkpoints",
        help="Where to save classifier checkpoints",
    )
    p.add_argument(
        "--resume", default=None, help="Path to a classifier checkpoint to resume from"
    )

    # Encoder
    p.add_argument(
        "--model_arch",
        default="vit_base",
        choices=["vit_base", "vit_large", "vit_giant_xformers_rope"],
        help="Encoder architecture",
    )
    p.add_argument("--img_size", type=int, default=384)
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument(
        "--encoder_num_frames",
        type=int,
        default=64,
        help="num_frames the encoder was built/trained with",
    )
    p.add_argument("--tubelet_size", type=int, default=2)
    p.add_argument(
        "--checkpoint_key",
        default="ema_encoder",
        help="Key inside the .pt file that holds the encoder weights",
    )

    # Video sampling
    p.add_argument(
        "--num_frames", type=int, default=8, help="Number of frames to sample per clip"
    )
    p.add_argument(
        "--num_indices_between_frames",
        type=int,
        default=4,
        help="Stride between consecutive sampled frames",
    )

    # Classifier head
    p.add_argument("--num_classes", type=int, required=True)
    p.add_argument("--classifier_depth", type=int, default=4)
    p.add_argument("--classifier_num_heads", type=int, default=16)

    # Training
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument(
        "--log_interval",
        type=int,
        default=20,
        help="Print training stats every N batches",
    )
    p.add_argument(
        "--save_every", type=int, default=5, help="Save a checkpoint every N epochs"
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


class RandomHorizontalFlipTensor:
    """Random horizontal flip for a (C, T, H, W) tensor (applied after ClipToTensor)."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        # clip shape: (C, T, H, W)
        if torch.rand(1).item() < self.p:
            return clip.flip(-1)  # flip last dim = W
        return clip


def build_train_transform(img_size: int):
    """Augmented transform for training clips.

    RandomHorizontalFlip is applied as a tensor op after ClipToTensor because
    the repo's video_transforms only support numpy/PIL inputs for that class.
    """
    short_side_size = int(256.0 / 224 * img_size)
    return video_transforms.Compose(
        [
            video_transforms.Resize(short_side_size, interpolation="bilinear"),
            video_transforms.CenterCrop(size=(img_size, img_size)),
            volume_transforms.ClipToTensor(),
            RandomHorizontalFlipTensor(p=0.5),
            video_transforms.Normalize(
                mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
            ),
        ]
    )


def build_val_transform(img_size: int):
    """Deterministic transform for validation clips."""
    short_side_size = int(256.0 / 224 * img_size)
    return video_transforms.Compose(
        [
            video_transforms.Resize(short_side_size, interpolation="bilinear"),
            video_transforms.CenterCrop(size=(img_size, img_size)),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(
                mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def _find_videos(root: Path):
    """Return sorted list of (video_path, class_index) from a folder-per-class tree."""
    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if not class_dirs:
        raise RuntimeError(f"No class sub-directories found under {root}")
    class_to_idx = {d.name: i for i, d in enumerate(class_dirs)}

    samples = []
    for class_dir in class_dirs:
        idx = class_to_idx[class_dir.name]
        for f in sorted(class_dir.iterdir()):
            if f.suffix.lower() in VIDEO_EXTENSIONS:
                samples.append((str(f), idx))

    if not samples:
        raise RuntimeError(f"No video files found under {root}")

    return samples, class_to_idx


class VideoClassificationDataset(Dataset):
    """
    Folder-per-class video classification dataset.

    Each call to __getitem__ samples a random clip from the video using
    the same read_video strategy as the inference demo:
      - random start frame
      - 1 frame every `num_indices_between_frames` steps
    """

    def __init__(
        self,
        root: str,
        transform,
        num_frames: int = 8,
        num_indices_between_frames: int = 4,
    ):
        self.transform = transform
        self.num_frames = num_frames
        self.num_indices_between_frames = num_indices_between_frames

        self.samples, self.class_to_idx = _find_videos(Path(root))
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        print(
            f"  {Path(root).name}: {len(self.samples)} videos, "
            f"{len(self.class_to_idx)} classes"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        clip = self._load_clip(path)  # (C, T, H, W)
        return clip, label

    def _load_clip(self, path: str) -> torch.Tensor:
        vr = VideoReader(path)
        total_frames = len(vr)

        clip_span = 1 + (self.num_frames - 1) * self.num_indices_between_frames

        if clip_span > total_frames:
            # Video is shorter than the requested span: reduce stride to fit
            stride = max(1, (total_frames - 1) // max(self.num_frames - 1, 1))
            clip_span = 1 + (self.num_frames - 1) * stride
        else:
            stride = self.num_indices_between_frames

        start = np.random.randint(0, max(1, total_frames - clip_span + 1))
        frame_idx = np.arange(start, start + (self.num_frames - 1) * stride + 1, stride)
        # Clamp to valid range (safety against rounding)
        frame_idx = np.clip(frame_idx, 0, total_frames - 1)

        frames = vr.get_batch(frame_idx).asnumpy()  # (T, H, W, C)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float()  # (T, C, H, W)
        return self.transform(frames)  # (C, T, H, W)


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


def build_encoder(args, device):
    encoder = vit_encoder_21.__dict__[args.model_arch](
        img_size=(args.img_size, args.img_size),
        patch_size=args.patch_size,
        num_frames=args.encoder_num_frames,
        tubelet_size=args.tubelet_size,
        use_sdpa=True,
        use_SiLU=False,
        wide_SiLU=True,
        uniform_power=False,
        use_rope=True,
        img_temporal_dim_size=1,  # sentinel: 1 frame → image mode, else video mode
        interpolate_rope=True,
        return_all_tokens=False,
        n_output_distillation=1,
    )

    checkpoint = torch.load(args.checkpoint_path, weights_only=True, map_location="cpu")
    pretrained_dict = _clean_backbone_key(checkpoint[args.checkpoint_key])
    msg = encoder.load_state_dict(pretrained_dict, strict=True)
    print(f"Encoder loaded from {args.checkpoint_path}")
    print(f"  missing: {msg.missing_keys}  unexpected: {msg.unexpected_keys}")

    encoder.to(device).eval()

    # Freeze all encoder parameters
    for param in encoder.parameters():
        param.requires_grad_(False)

    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"  Encoder params (frozen): {n_params / 1e6:.1f}M")
    return encoder


def build_classifier(
    embed_dim: int, num_heads: int, depth: int, num_classes: int, device
) -> AttentiveClassifier:
    classifier = AttentiveClassifier(
        embed_dim=embed_dim,
        num_heads=num_heads,
        depth=depth,
        num_classes=num_classes,
    ).to(device)

    n_params = sum(p.numel() for p in classifier.parameters())
    print(f"  Classifier params (trainable): {n_params / 1e6:.2f}M")
    return classifier


# ---------------------------------------------------------------------------
# Training / validation
# ---------------------------------------------------------------------------


def run_one_epoch(
    encoder,
    classifier,
    loader,
    criterion,
    optimizer,
    device,
    is_train: bool,
    log_interval: int,
    epoch: int,
):
    classifier.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    t0 = time.time()

    ctx = torch.enable_grad() if is_train else torch.inference_mode()
    with ctx:
        for batch_idx, (clips, labels) in enumerate(loader):
            # clips: (B, C, T, H, W)  labels: (B,)
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # ── Frozen encoder forward ──────────────────────────────────────
            # Use no_grad (not inference_mode): inference_mode produces tensors
            # that cannot be consumed by the classifier's autograd graph.
            with torch.no_grad():
                features = encoder(clips)  # (B, N_tokens, D)

            # ── Classifier forward ──────────────────────────────────────────
            if is_train:
                logits = classifier(features)  # (B, num_classes)
                loss = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                logits = classifier(features)
                loss = criterion(logits, labels)

            # ── Metrics ─────────────────────────────────────────────────────
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size

            if is_train and (batch_idx + 1) % log_interval == 0:
                elapsed = time.time() - t0
                avg_loss = total_loss / total_samples
                avg_acc = total_correct / total_samples * 100
                print(
                    f"  Epoch {epoch} [{batch_idx + 1}/{len(loader)}]  "
                    f"loss={avg_loss:.4f}  acc={avg_acc:.1f}%  "
                    f"({elapsed:.0f}s elapsed)"
                )

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples * 100
    return avg_loss, avg_acc


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def save_checkpoint(
    output_dir: str, epoch: int, classifier, optimizer, best_acc: float
):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"classifier_epoch{epoch:03d}.pt")
    torch.save(
        {
            "epoch": epoch,
            "classifier": classifier.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_acc": best_acc,
        },
        path,
    )
    print(f"  Saved checkpoint → {path}")
    return path


def load_checkpoint(path: str, classifier, optimizer=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    classifier.load_state_dict(ckpt["classifier"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    epoch = ckpt.get("epoch", 0)
    best_acc = ckpt.get("best_val_acc", 0.0)
    print(f"  Resumed from {path}  (epoch {epoch}, best val acc {best_acc:.1f}%)")
    return epoch, best_acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Transforms ──────────────────────────────────────────────────────────
    train_transform = build_train_transform(args.img_size)
    val_transform = build_val_transform(args.img_size)

    # ── Datasets & loaders ──────────────────────────────────────────────────
    print("Loading datasets...")
    train_dataset = VideoClassificationDataset(
        root=os.path.join(args.data_root),
        transform=train_transform,
        num_frames=args.num_frames,
        num_indices_between_frames=args.num_indices_between_frames,
    )
    val_dataset = VideoClassificationDataset(
        root=os.path.join(args.data_root),
        transform=val_transform,
        num_frames=args.num_frames,
        num_indices_between_frames=args.num_indices_between_frames,
    )

    # Verify class sets match
    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        print("WARNING: train and val class sets differ!")
    print(f"  Class mapping: {train_dataset.class_to_idx}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device == "cuda",
    )

    # ── Models ──────────────────────────────────────────────────────────────
    print("Building encoder (frozen)...")
    encoder = build_encoder(args, device)

    print("Building classifier...")
    classifier = build_classifier(
        embed_dim=encoder.embed_dim,
        num_heads=args.classifier_num_heads,
        depth=args.classifier_depth,
        num_classes=args.num_classes,
        device=device,
    )

    # ── Optimizer & loss ────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    # ── Optional resume ─────────────────────────────────────────────────────
    start_epoch = 0
    best_val_acc = 0.0
    if args.resume:
        start_epoch, best_val_acc = load_checkpoint(args.resume, classifier, optimizer)
        start_epoch += 1  # continue from the next epoch

    # ── Training loop ────────────────────────────────────────────────────────
    print(f"\nStarting training for {args.epochs} epochs")
    print(f"  num_frames={args.num_frames}  stride={args.num_indices_between_frames}")
    print(f"  batch_size={args.batch_size}  lr={args.lr}  wd={args.weight_decay}\n")

    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss, train_acc = run_one_epoch(
            encoder,
            classifier,
            train_loader,
            criterion,
            optimizer,
            device,
            is_train=True,
            log_interval=args.log_interval,
            epoch=epoch,
        )

        # Validate
        val_loss, val_acc = run_one_epoch(
            encoder,
            classifier,
            val_loader,
            criterion,
            optimizer,
            device,
            is_train=False,
            log_interval=args.log_interval,
            epoch=epoch,
        )

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc

        print(
            f"Epoch {epoch:3d} | "
            f"train loss={train_loss:.4f} acc={train_acc:.1f}% | "
            f"val loss={val_loss:.4f} acc={val_acc:.1f}%"
            + (" ← best" if is_best else "")
        )

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 or is_best:
            tag = "best" if is_best else f"epoch{epoch:03d}"
            os.makedirs(args.output_dir, exist_ok=True)
            path = os.path.join(args.output_dir, f"classifier_{tag}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "classifier": classifier.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_val_acc": best_val_acc,
                    "class_to_idx": train_dataset.class_to_idx,
                    "args": vars(args),
                },
                path,
            )
            print(f"  Saved → {path}")

    print(f"\nTraining complete. Best val acc: {best_val_acc:.1f}%")


if __name__ == "__main__":
    # python -m notebooks.vjepa21_train_classifier --data_root /path/to/data \
    #   --checkpoint_path vjepa2_1_vitb_dist_vitG_384.pt --num_classes 10
    main()
