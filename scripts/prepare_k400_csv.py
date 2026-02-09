#!/usr/bin/env python3
"""
Generate V-JEPA 2 compatible CSV manifest for Kinetics-400 validation set.

Expected directory structure after extracting K400 val tars:
    ~/data/k400/val/
        abseiling/
            video1.mp4
            video2.mp4
        air_drumming/
            video1.mp4
            ...

Output CSV format (space-delimited, no header):
    /home/ubuntu/data/k400/val/abseiling/video1.mp4 0
    /home/ubuntu/data/k400/val/air_drumming/video2.mp4 1
    ...

Usage:
    python scripts/prepare_k400_csv.py --val_dir ~/data/k400/val --output ~/data/k400/k400_val_paths.csv
"""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Generate K400 CSV manifest for V-JEPA 2")
    parser.add_argument("--val_dir", type=str, required=True, help="Path to extracted K400 val directory")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    args = parser.parse_args()

    val_dir = Path(args.val_dir).resolve()
    if not val_dir.exists():
        raise FileNotFoundError(f"Val directory not found: {val_dir}")

    # Discover all class directories and sort alphabetically for consistent label assignment
    class_dirs = sorted([d for d in val_dir.iterdir() if d.is_dir()])
    if len(class_dirs) == 0:
        raise ValueError(f"No class directories found in {val_dir}")

    # Map class name -> integer label (alphabetical order, 0-indexed)
    class_to_label = {d.name: i for i, d in enumerate(class_dirs)}

    # Collect all video files
    video_extensions = {".mp4", ".avi", ".mkv", ".webm"}
    entries = []
    missing_classes = 0
    for class_dir in class_dirs:
        label = class_to_label[class_dir.name]
        videos = [
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in video_extensions
        ]
        if len(videos) == 0:
            missing_classes += 1
            continue
        for video in sorted(videos):
            entries.append((str(video), label))

    # Write CSV
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for video_path, label in entries:
            f.write(f"{video_path} {label}\n")

    print(f"Classes found: {len(class_dirs)}")
    print(f"Classes with no videos: {missing_classes}")
    print(f"Total videos: {len(entries)}")
    print(f"CSV written to: {output_path}")

    # Also write label map for reference
    label_map_path = output_path.parent / "k400_label_map.txt"
    with open(label_map_path, "w") as f:
        for class_name, label in sorted(class_to_label.items(), key=lambda x: x[1]):
            f.write(f"{label} {class_name}\n")
    print(f"Label map written to: {label_map_path}")


if __name__ == "__main__":
    main()
