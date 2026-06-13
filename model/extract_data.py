#!/usr/bin/env python3
"""
Extract keypoints from ASL video files for model training.

Uses the stable ``mp.solutions.holistic`` API (not the Tasks API) to avoid
a known GPU/driver crash on Intel Mesa hardware.

Output: directory of .npy files organized by sign class, each with shape
(35, 258) — ready for model training.

Usage
-----
    # Activate translationService venv first:
    source ../translationService/.venv/bin/activate

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
# Path setup — import normalize from sibling translationService/
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
_TRANSLATION_SERVICE = os.path.join(_PROJECT, "translationService")
if _TRANSLATION_SERVICE not in sys.path:
    sys.path.insert(0, _TRANSLATION_SERVICE)

from normalize import normalize_frames

# Feature layout: lh(63) + rh(63) + pose(132) = 258
WINDOW_SIZE = 35
FEATURE_DIM = 258
_LH_DIM = 63
_RH_DIM = 63
_POSE_DIM = 132
PERSIST_WINDOW = 5  # match converter.py bounded persistence duration

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

# ---------------------------------------------------------------------------
# MediaPipe holistic (old stable API — avoids Tasks API GPU crash)
# ---------------------------------------------------------------------------
import mediapipe as mp

_holistic_model = None


def _get_holistic():
    """Lazy-init the old ``mp.solutions.holistic.Holistic`` pipeline."""
    global _holistic_model
    if _holistic_model is None:
        _holistic_model = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.7,
        )
    return _holistic_model


class BoundedPersistenceExtractor:
    """Mirrors production converter.py's bounded persistence logic.

    Maintains per-video state (lost-frame counters, last-known hand
    positions) so that training data matches inference exactly:
    - Hand detected → use real position, update cache, reset counter
    - Hand lost < PERSIST_WINDOW frames → use last-known position
    - Hand lost >= PERSIST_WINDOW frames → use zeros (genuinely absent)
    - Pose always from current frame (not persisted)
    - Nose-normalised after persistence is applied
    """

    def __init__(self):
        self.holistic = _get_holistic()
        self.reset()

    def reset(self):
        self._last_lh = np.zeros(_LH_DIM, dtype=np.float32)
        self._last_rh = np.zeros(_RH_DIM, dtype=np.float32)
        self._lh_lost = 0
        self._rh_lost = 0

    @property
    def is_idle_frame(self) -> bool:
        """True if the MOST RECENTLY PROCESSED frame had pose visible but no hands.

        *Both* hands must have been absent for >= PERSIST_WINDOW frames so
        that bounded persistence has already decayed to zeros.
        """
        return (self._lh_lost >= PERSIST_WINDOW
                and self._rh_lost >= PERSIST_WINDOW)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Run MediaPipe + bounded persistence + nose normalisation.

        Returns (258,) keypoint vector identical to converter.py output.
        """
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.holistic.process(image_rgb)
        image_rgb.flags.writeable = True

        # ---- Raw extraction (no normalisation yet) ----
        lh = np.zeros(_LH_DIM, dtype=np.float32)
        if results.left_hand_landmarks:
            for i, lm in enumerate(results.left_hand_landmarks.landmark):
                idx = i * 3
                lh[idx] = lm.x; lh[idx + 1] = lm.y; lh[idx + 2] = lm.z
            self._last_lh = lh.copy()
            self._lh_lost = 0
        else:
            self._lh_lost += 1

        rh = np.zeros(_RH_DIM, dtype=np.float32)
        if results.right_hand_landmarks:
            for i, lm in enumerate(results.right_hand_landmarks.landmark):
                idx = i * 3
                rh[idx] = lm.x; rh[idx + 1] = lm.y; rh[idx + 2] = lm.z
            self._last_rh = rh.copy()
            self._rh_lost = 0
        else:
            self._rh_lost += 1

        pose = np.zeros(_POSE_DIM, dtype=np.float32)
        pose_detected = results.pose_landmarks is not None
        if pose_detected:
            for i, lm in enumerate(results.pose_landmarks.landmark):
                idx = i * 4
                pose[idx] = lm.x
                pose[idx + 1] = lm.y
                pose[idx + 2] = lm.z
                pose[idx + 3] = getattr(lm, "visibility", 0.0)

        # ---- Bounded persistence for hands ----
        if self._lh_lost >= PERSIST_WINDOW:
            lh = np.zeros(_LH_DIM, dtype=np.float32)
        else:
            lh = self._last_lh.copy()

        if self._rh_lost >= PERSIST_WINDOW:
            rh = np.zeros(_RH_DIM, dtype=np.float32)
        else:
            rh = self._last_rh.copy()

        # ---- Nose normalisation (same as converter.py) ----
        nose_xyz = pose[0:3] if pose_detected else np.zeros(3, dtype=np.float32)
        lh = (lh.reshape(-1, 3) - nose_xyz).flatten()
        rh = (rh.reshape(-1, 3) - nose_xyz).flatten()
        p = pose.copy().reshape(-1, 4)
        p[:, :3] -= nose_xyz
        pose = p.flatten()

        return np.concatenate([lh, rh, pose]).astype(np.float32)


def extract_video(video_path: str, extractor: BoundedPersistenceExtractor,
                  background_segments: list[np.ndarray] | None = None) -> np.ndarray | None:
    """Run MediaPipe on every frame of *video_path* with bounded persistence.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    extractor : BoundedPersistenceExtractor
        Reusable extractor (will be reset internally).
    background_segments : list or None
        If provided, idle segments (pose visible, no hands for >= PERSIST_WINDOW
        frames) of at least WINDOW_SIZE consecutive frames are appended here as
        BACKGROUND training samples.  These use the exact same -nose_xyz hand
        pattern that inference produces.

    Returns
    -------
    np.ndarray of shape (T, FEATURE_DIM) where T = number of frames, or
    None if the video could not be opened or yielded no keypoints.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ⚠️  Could not open: {video_path}")
        return None

    extractor.reset()
    frames: list[np.ndarray] = []

    # Idle-segment tracking for background collection
    _bg_buf: list[np.ndarray] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        try:
            kp = extractor.process_frame(frame)
        except Exception as e:
            print(f"  ⚠️  Frame error in {video_path}: {e}")
            continue
        frames.append(kp)

        # Collect idle frames as potential BACKGROUND samples
        if background_segments is not None:
            if extractor.is_idle_frame:
                _bg_buf.append(kp)
            else:
                # Idle segment ended — save if long enough
                if len(_bg_buf) >= WINDOW_SIZE:
                    background_segments.append(np.array(_bg_buf, dtype=np.float32))
                _bg_buf.clear()

    cap.release()

    # Flush remaining idle segment at end of video
    if background_segments is not None and len(_bg_buf) >= WINDOW_SIZE:
        background_segments.append(np.array(_bg_buf, dtype=np.float32))

    if not frames:
        print(f"  ⚠️  No frames extracted from {video_path}")
        return None

    return np.array(frames, dtype=np.float32)


def _generate_background_synthetic(bg_dir: str) -> int:
    """Generate synthetic BACKGROUND samples that don't need video input.

    These cover the "empty frame" cases (no person visible).  The
    "person visible, not signing" case is handled by idle segments
    collected during the main extraction (see ``background_segments``).

    Types:
      1. All-zero keypoints — empty frame, no person visible
      2. All-zero + small Gaussian noise — camera noise

    Returns
    -------
    int  Number of background samples saved.
    """
    os.makedirs(bg_dir, exist_ok=True)
    rng = np.random.default_rng(seed=42)
    saved = 0

    zero_base = np.zeros((WINDOW_SIZE, FEATURE_DIM), dtype=np.float32)

    # All-zero samples
    for i in range(50):
        np.save(os.path.join(bg_dir, f"zero_{i}.npy"), zero_base)
        saved += 1

    # All-zero + noise
    for i in range(50):
        noise = rng.normal(0, 0.005, (WINDOW_SIZE, FEATURE_DIM)).astype(np.float32)
        np.save(os.path.join(bg_dir, f"zero_noise_{i}.npy"), zero_base + noise)
        saved += 1

    return saved


def process_videos(
    input_dir: str,
    output_dir: str,
    *,
    target_class: str | None = None,
    resume: str | None = None,
    min_frames: int = 2,
    generate_bg: bool = False,
) -> None:
    """Walk *input_dir*/*class*/*.mp4, extract keypoints, save to *output_dir*.

    Parameters
    ----------
    input_dir : str
        Root directory containing one subdirectory per sign class.
    output_dir : str
        Where to write the .npy files (mirrors the class layout).
    target_class : str or None
        If set, only process this one class (e.g. ``"AND"``).
    resume : str or None
        Class name to resume from (skips earlier classes).
    min_frames : int
        Minimum number of successfully extracted frames required.
    generate_bg : bool
        If True, generate synthetic BACKGROUND class samples after extraction.
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

    extractor = BoundedPersistenceExtractor()
    total_saved = 0
    total_skipped = 0

    # Collect idle segments from videos as BACKGROUND samples
    # (pose visible, no hands for >= PERSIST_WINDOW — produces correct -nose_xyz)
    background_segments: list[np.ndarray] = []

    for sign_name in sign_classes:
        sign_input = os.path.join(input_dir, sign_name)
        if not os.path.isdir(sign_input):
            print(f"\n  ⚠️  '{sign_name}' is not a directory, skipping")
            continue

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
            # Pass background_segments when --generate-background is active
            bg_list = background_segments if generate_bg else None
            seq = extract_video(video_path, extractor, background_segments=bg_list)

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
    print(f"  Extraction done: {total_saved} files saved, {total_skipped} skipped")
    print(f"{'='*60}")

    # Background generation (runs after all sign extraction)
    if generate_bg:
        print("\nGenerating BACKGROUND class samples...")

        # 1. Save idle segments collected during extraction
        bg_dir = os.path.join(output_dir, "BACKGROUND")
        os.makedirs(bg_dir, exist_ok=True)
        bg_from_video = 0
        for idx, seg in enumerate(background_segments):
            # Truncate or split long segments into WINDOW_SIZE chunks
            for chunk_i in range(0, len(seg), WINDOW_SIZE):
                chunk = seg[chunk_i:chunk_i + WINDOW_SIZE]
                if len(chunk) < WINDOW_SIZE:
                    continue
                # Normalise to fixed length (usually already WINDOW_SIZE)
                chunk_list = normalize_frames(chunk.tolist(), WINDOW_SIZE)
                chunk = np.array(chunk_list, dtype=np.float32)
                if chunk.shape == (WINDOW_SIZE, FEATURE_DIM):
                    np.save(os.path.join(bg_dir, f"idle_{idx}_{chunk_i // WINDOW_SIZE}.npy"), chunk)
                    bg_from_video += 1

        print(f"  Idle segments from videos → {bg_from_video} samples")

        # 2. Synthetic samples (all-zero and zero+noise only —
        #    pose-only with zeroed hands was incorrect; idle segments above
        #    give the correct -nose_xyz pattern)
        synthetic_saved = _generate_background_synthetic(bg_dir)
        total_bg = bg_from_video + synthetic_saved
        print(f"  Synthetic → {synthetic_saved} samples")
        print(f"\n{'='*60}")
        print(f"  Total: {total_saved} sign + {total_bg} background")
        print(f"{'='*60}")

        saved = 0
        skipped = 0

        for i, video_name in enumerate(video_files):
            video_path = os.path.join(sign_input, video_name)
            seq = extract_video(video_path, extractor)

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
    print(f"  Extraction done: {total_saved} files saved, {total_skipped} skipped")
    print(f"{'='*60}")

    # Background generation (runs after all sign extraction)
    if generate_bg:
        print("\nGenerating BACKGROUND class samples...")
        bg_saved = generate_background(output_dir, output_dir)
        total_saved += bg_saved
        print(f"\n{'='*60}")
        print(f"  Total: {total_saved} files (including {bg_saved} background)")
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
    parser.add_argument(
        "--generate-background", action="store_true",
        help="Generate synthetic BACKGROUND class samples after extraction",
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

    print("Initialising MediaPipe Holistic (old stable API)...")
    _ = _get_holistic()
    print("Ready.\n")

    process_videos(
        args.input_dir,
        args.output_dir,
        target_class=args.target_class,
        resume=args.resume,
        min_frames=args.min_frames,
        generate_bg=args.generate_background,
    )


if __name__ == "__main__":
    main()
