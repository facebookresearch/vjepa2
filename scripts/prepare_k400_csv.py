#!/usr/bin/env python3
"""
Generate V-JEPA 2 compatible CSV manifest for Kinetics-400 validation set.

Supports two directory layouts:
  A) Flat: ~/data/k400/val/{youtube_id}_{start}_{end}.mp4
     Requires --annotations pointing to the K400 val.csv annotations file.
  B) Class dirs: ~/data/k400/val/{class_name}/{video}.mp4
     Labels assigned alphabetically (0-indexed).

Output CSV format (space-delimited, no header):
    /home/ubuntu/data/k400/val/video1.mp4 0
    /home/ubuntu/data/k400/val/video2.mp4 1

Usage:
    # Flat layout (CVDF tars):
    python scripts/prepare_k400_csv.py \
        --val_dir ~/data/k400/val \
        --annotations ~/data/k400/val.csv \
        --output ~/data/k400/k400_val_paths.csv

    # Class directory layout:
    python scripts/prepare_k400_csv.py \
        --val_dir ~/data/k400/val \
        --output ~/data/k400/k400_val_paths.csv
"""

import argparse
import csv
from pathlib import Path


def build_from_annotations(val_dir, annotations_path):
    """Build manifest from flat video dir + K400 annotations CSV."""
    # Read annotations: label,youtube_id,time_start,time_end,split,is_cc
    label_names = set()
    video_info = []
    with open(annotations_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_names.add(row["label"])
            video_info.append(row)

    # Alphabetical label -> integer mapping (standard K400 ordering)
    sorted_labels = sorted(label_names)
    label_to_id = {name: i for i, name in enumerate(sorted_labels)}

    # Build filename -> (label_name, label_id) lookup
    # Filename pattern: {youtube_id}_{time_start:06d}_{time_end:06d}.mp4
    file_lookup = {}
    for row in video_info:
        yt_id = row["youtube_id"]
        t_start = int(row["time_start"])
        t_end = int(row["time_end"])
        fname = f"{yt_id}_{t_start:06d}_{t_end:06d}.mp4"
        file_lookup[fname] = (row["label"], label_to_id[row["label"]])

    # Scan val directory for actual video files
    video_extensions = {".mp4", ".avi", ".mkv", ".webm"}
    entries = []
    matched = 0
    unmatched = 0
    for f in sorted(val_dir.iterdir()):
        if not f.is_file() or f.suffix.lower() not in video_extensions:
            continue
        if f.name in file_lookup:
            _, label_id = file_lookup[f.name]
            entries.append((str(f), label_id))
            matched += 1
        else:
            unmatched += 1

    print(f"Annotations: {len(video_info)} entries, {len(sorted_labels)} classes")
    print(f"Videos matched: {matched}, unmatched: {unmatched}")
    return entries, sorted_labels, label_to_id


def build_from_class_dirs(val_dir):
    """Build manifest from class-directory layout."""
    class_dirs = sorted([d for d in val_dir.iterdir() if d.is_dir()])
    if len(class_dirs) == 0:
        raise ValueError(f"No class directories found in {val_dir}")

    label_to_id = {d.name: i for i, d in enumerate(class_dirs)}
    sorted_labels = [d.name for d in class_dirs]

    video_extensions = {".mp4", ".avi", ".mkv", ".webm"}
    entries = []
    for class_dir in class_dirs:
        label_id = label_to_id[class_dir.name]
        videos = sorted([
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in video_extensions
        ])
        for video in videos:
            entries.append((str(video), label_id))

    print(f"Classes: {len(class_dirs)}")
    return entries, sorted_labels, label_to_id


def main():
    parser = argparse.ArgumentParser(description="Generate K400 CSV manifest for V-JEPA 2")
    parser.add_argument("--val_dir", type=str, required=True, help="Path to extracted K400 val directory")
    parser.add_argument("--annotations", type=str, default=None, help="Path to K400 val.csv annotations (for flat layout)")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    args = parser.parse_args()

    val_dir = Path(args.val_dir).resolve()
    if not val_dir.exists():
        raise FileNotFoundError(f"Val directory not found: {val_dir}")

    # Detect layout: check if subdirectories exist
    has_subdirs = any(d.is_dir() for d in val_dir.iterdir())

    if has_subdirs and args.annotations is None:
        print("Detected class-directory layout.")
        entries, sorted_labels, label_to_id = build_from_class_dirs(val_dir)
    elif args.annotations is not None:
        print("Using annotations CSV for flat layout.")
        entries, sorted_labels, label_to_id = build_from_annotations(val_dir, args.annotations)
    else:
        raise ValueError(
            "Flat video directory detected but no --annotations provided.\n"
            "Download annotations: wget https://s3.amazonaws.com/kinetics/400/annotations/val.csv\n"
            "Then run: python prepare_k400_csv.py --val_dir ... --annotations val.csv --output ..."
        )

    # Write CSV
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for video_path, label in entries:
            f.write(f"{video_path} {label}\n")

    print(f"Total videos: {len(entries)}")
    print(f"CSV written to: {output_path}")

    # Write label map for reference
    label_map_path = output_path.parent / "k400_label_map.txt"
    with open(label_map_path, "w") as f:
        for name, label_id in sorted(label_to_id.items(), key=lambda x: x[1]):
            f.write(f"{label_id} {name}\n")
    print(f"Label map written to: {label_map_path}")


if __name__ == "__main__":
    main()
