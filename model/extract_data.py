#!/usr/bin/env python3
"""
Extract keypoints from ASL video files using the new converter pipeline.

Output: directory of .npy files organized by sign class, each with shape
(35, 258) — ready for model training.

Usage
-----
    # Process all videos in a directory
    python extract_data.py /path/to/videos /path/to/output

    # Process a single sign class (resume-friendly)
    python extract_data.py /path/to/videos /path/to/output --class AND

Expected input layout
---------------------
    /path/to/videos/
        AND/
            video1.mp4
            video2.mp4
            ...
        APPLE/
            video1.mp4
            ...
        ...

Output layout
-------------
    /path/to/output/
        AND/
            0.npy
            1.npy
            ...
        APPLE/
            0.npy
            ...
"""

import sys
import os
import argparse
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Path setup — import converter from sibling translationService/
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
_TRANSLATION_SERVICE = os.path.join(_PROJECT, "translationService")
if _TRANSLATION_SERVICE not in sys.path:
    sys.path.insert(0, _TRANSLATION_SERVICE)

from converter import Converter, FEATURE_DIM, WINDOW_SIZE
from normalize import normalize_frames

# ---------------------------------------------------------------------------
# Supported video extensions
# ---------------------------------------------------------------------------
_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")

# ---------------------------------------------------------------------------
# Target ASL vocabulary (31 signs)
# ---------------------------------------------------------------------------
_TARGET_CLASSES = frozenset({
    "AND", "ANGRY", "APPLE", "ASK", "BAD", "DRINK", "EAT", "FRIEND",
    "GO", "GOOD", "HAPPY", "HE", "HOME", "LIKE", "MILK", "MOTHER",
    "MY", "NAME", "NO", "PLEASE", "SEE", "THEY", "TIRED", "WANT",
    "WATER", "WE", "WHAT", "WHERE", "YES", "YOU", "ME",
})


def extract_video(video_path: str, converter: Converter) -> np.ndarray | None:
    """Run MediaPipe on every frame of *video_path*.

    Returns
    -------
    np.ndarray of shape (T, FEATURE_DIM) where T = number of frames, or
    None if the video could not be opened or yielded no keypoints.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ⚠️  Could not open: {video_path}")
        return None

    # Reset bounded-persistence state for a fresh video
    converter.reset_state()

    frames: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        try:
            kp = converter.extract_from_frame(frame)
        except Exception as e:
            print(f"  ⚠️  Frame error in {video_path}: {e}")
            continue
        frames.append(kp)

    cap.release()

    if not frames:
        print(f"  ⚠️  No frames extracted from {video_path}")
        return None

    return np.array(frames, dtype=np.float32)


def process_videos(
    input_dir: str,
    output_dir: str,
    converter: Converter,
    *,
    target_class: str | None = None,
    resume: str | None = None,
    min_frames: int = 2,
) -> None:
    """Walk *input_dir*/*class*/*.mp4, extract keypoints, save to *output_dir*.

    Parameters
    ----------
    input_dir : str
        Root directory containing one subdirectory per sign class.
    output_dir : str
        Where to write the .npy files (mirrors the class layout).
    converter : Converter
        Initialised MediaPipe converter instance.
    target_class : str or None
        If set, only process this one class (e.g. ``"AND"``).
    resume : str or None
        Class name to resume from (skips earlier classes).
    min_frames : int
        Minimum number of successfully extracted frames required.
    """
    sign_classes = sorted(
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
    )

    if target_class:
        sign_classes = [c for c in sign_classes if c.upper() == target_class.upper()]

    if resume:
        try:
            idx = [c.upper() for c in sign_classes].index(resume.upper())
            sign_classes = sign_classes[idx:]
        except ValueError:
            print(f"  ⚠️  Resume class '{resume}' not found, starting from beginning.")

    total_saved = 0
    total_skipped = 0

    for sign_name in sign_classes:
        if sign_name.upper() not in _TARGET_CLASSES:
            print(f"\nSkipping '{sign_name}' — not in target vocabulary")
            continue

        sign_input = os.path.join(input_dir, sign_name)
        sign_output = os.path.join(output_dir, sign_name)
        os.makedirs(sign_output, exist_ok=True)

        video_files = sorted(
            v for v in os.listdir(sign_input)
            if v.lower().endswith(_VIDEO_EXTS)
        )

        if not video_files:
            print(f"\n'{sign_name}' — no video files found")
            continue

        print(f"\n{'='*60}")
        print(f"  {sign_name}  ({len(video_files)} videos)")
        print(f"{'='*60}")

        saved = 0
        skipped = 0

        for i, video_name in enumerate(video_files):
            video_path = os.path.join(sign_input, video_name)
            seq = extract_video(video_path, converter)

            if seq is None:
                skipped += 1
                continue

            if len(seq) < min_frames:
                print(f"  ⚠️  '{video_name}' too short ({len(seq)} frames, need ≥{min_frames})")
                skipped += 1
                continue

            # Normalise variable-length sequence to WINDOW_SIZE (35) frames
            seq_list = normalize_frames(seq.tolist(), WINDOW_SIZE)
            normalized = np.array(seq_list, dtype=np.float32)

            # Validate shape
            assert normalized.shape == (WINDOW_SIZE, FEATURE_DIM), (
                f"Expected ({WINDOW_SIZE}, {FEATURE_DIM}), got {normalized.shape}"
            )

            out_path = os.path.join(sign_output, f"{i}.npy")
            np.save(out_path, normalized)
            saved += 1

            if saved % 10 == 0 or saved == len(video_files):
                print(f"  ✓ {saved}/{len(video_files)} saved"
                      f"{'  (last: ' + video_name + ')' if saved % 10 == 0 else ''}")

        total_saved += saved
        total_skipped += skipped
        print(f"  ──> {saved} saved, {skipped} skipped")

    print(f"\n{'='*60}")
    print(f"  Done: {total_saved} files saved, {total_skipped} skipped")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")


def count_videos(input_dir: str) -> None:
    """Dry-run: count videos per class without processing."""
    sign_classes = sorted(
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
    )
    total = 0
    for sign_name in sign_classes:
        if sign_name.upper() not in _TARGET_CLASSES:
            continue
        sign_input = os.path.join(input_dir, sign_name)
        videos = [
            v for v in os.listdir(sign_input)
            if v.lower().endswith(_VIDEO_EXTS)
        ]
        print(f"  {sign_name:10s}  {len(videos):3d} videos")
        total += len(videos)
    print(f"  {'─'*20}")
    print(f"  {'TOTAL':10s}  {total:3d} videos")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract (35, 258) keypoints from ASL video files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /data/videos /data/keypoints
  %(prog)s /data/videos /data/keypoints --class AND
  %(prog)s /data/videos /data/keypoints --resume HAPPY
  %(prog)s /data/videos /data/keypoints --dry-run
  %(prog)s /data/videos /data/keypoints --min-frames 5
        """,
    )
    parser.add_argument("input_dir", help="Root dir with subdirs per sign class")
    parser.add_argument("output_dir", help="Where to write .npy files")
    parser.add_argument(
        "--class", dest="target_class",
        help="Process only this sign class (case-insensitive)",
    )
    parser.add_argument(
        "--resume",
        help="Resume from this class (skips earlier ones)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only count videos, don't extract",
    )
    parser.add_argument(
        "--min-frames", type=int, default=2,
        help="Minimum extracted frames to keep a sample (default: 2)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: input_dir '{args.input_dir}' does not exist.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.dry_run:
        print(f"\nVideo count in {args.input_dir}:\n")
        count_videos(args.input_dir)
        return

    print("Initialising MediaPipe Holistic converter...")
    converter = Converter()
    print("Ready.\n")

    process_videos(
        args.input_dir,
        args.output_dir,
        converter,
        target_class=args.target_class,
        resume=args.resume,
        min_frames=args.min_frames,
    )


if __name__ == "__main__":
    main()
