#!/usr/bin/env python3
"""
Camera test script for Sign Segmentation Service.

Run with the project virtual environment:
    .venv/bin/python test_camera.py
    .venv/bin/python test_camera.py <camera_id>

Captures frames from your webcam, runs MediaPipe keypoint extraction
and motion-based sign boundary detection in real-time, and displays
a rich visual overlay so you can verify segmentation is working.

Controls
--------
  q / ESC  – quit
  r        – reset the motion detector (start fresh)
  p        – pause / resume

Visual indicators
-----------------
  ● Border color:
      GREEN  → actively signing (motion above low threshold)
      YELLOW → motion in hysteresis band (near-still)
      RED    → motion stopped → sign boundary detected
      CYAN   → motion detector just reset

  ● Left panel shows:
      total segment count (SEG #), current sign frame length,
      motion score, adaptive low/high thresholds, still counter, FPS

  ● Segment timeline bar at bottom:
      each completed sign = colored block with its frame count
      (helps you verify at a glance if the right number of segments
      was detected)

  ● MediaPipe holistic mesh overlaid on the camera feed
      (face dots in yellow, left hand in pink, right hand in purple)
      Pose skeleton is hidden to reduce visual clutter.
      Watch for flickering — if landmarks jitter when you're still,
      that's why the still counter doesn't increment.
"""

from __future__ import annotations

import sys
import time
from collections import deque

import cv2 as cv
import numpy as np

# Local modules – assumes running from signSegmentationService/
from converter import Converter, FEATURE_DIM
from motion_detector import MotionDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FPSMeter:
    """Simple sliding-window FPS counter."""
    def __init__(self, window: int = 30):
        self._times = deque(maxlen=window)

    def tick(self) -> float:
        self._times.append(time.perf_counter())
        if len(self._times) < 2:
            return 0.0
        return len(self._times) / (self._times[-1] - self._times[0])


def draw_info_panel(
    frame: np.ndarray,
    sign_count: int,
    sign_length: int,
    motion: float,
    low_th: float,
    high_th: float,
    still_counter: int,
    still_required: int,
    fps: float,
    status: str,
    segment_log: list,
) -> None:
    """Draw a semi-transparent info panel on the left side."""
    h, w = frame.shape[:2]
    panel_w = 320

    # ---- overlay background ----
    overlay = frame.copy()
    cv.rectangle(overlay, (0, 0), (panel_w, h), (20, 20, 30), -1)
    cv.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    lines = [
        ("SIGN SEGMENTATION TEST", (255, 255, 255), 0.7),
        ("─" * 28, (180, 180, 180), 0.5),
        (f"Status          : {status}", (255, 255, 255), 0.55),
        (f"Motion score    : {motion:.4f}", (200, 200, 200), 0.5),
        (f"Low threshold   : {low_th:.4f}", (100, 200, 255), 0.5),
        (f"High threshold  : {high_th:.4f}", (100, 255, 200), 0.5),
        (f"Still counter   : {still_counter}/{still_required}", (255, 200, 100), 0.5),
        (f"FPS             : {fps:.1f}", (200, 200, 200), 0.5),
        (f"Frame size      : {w}x{h}", (150, 150, 150), 0.45),
        ("─" * 28, (180, 180, 180), 0.5),
    ]

    y = 30
    for text, color, scale in lines:
        cv.putText(frame, text, (15, y), cv.FONT_HERSHEY_SIMPLEX,
                   scale, color, 1, cv.LINE_AA)
        y += 26

    # ---- segment count (large & prominent) ----
    seg_text = f"SEGMENTS: {sign_count}"
    (tw, _), _ = cv.getTextSize(seg_text, cv.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv.putText(frame, seg_text,
               (panel_w - tw - 15, y - 8),  # right-aligned in panel
               cv.FONT_HERSHEY_SIMPLEX, 0.65,
               (100, 255, 100), 2, cv.LINE_AA)

    # ---- mini segment timeline ----
    if segment_log:
        total_frames = sum(s["frames"] for s in segment_log) or 1
        timeline_x = 15
        timeline_y = y + 10
        bar_h = 14
        max_bar_w = panel_w - 30

        cv.putText(frame, "Segments:", (timeline_x, timeline_y - 2),
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv.LINE_AA)
        timeline_y += 4

        for s in segment_log:
            seg_w = max(int((s["frames"] / total_frames) * max_bar_w), 6)
            # pick color by index
            colors = [(100, 200, 255), (100, 255, 200),
                      (255, 200, 100), (200, 100, 255),
                      (255, 100, 200), (100, 255, 255)]
            c = colors[s["idx"] % len(colors)]
            cv.rectangle(frame,
                         (timeline_x, timeline_y),
                         (timeline_x + seg_w, timeline_y + bar_h),
                         c, -1)
            cv.rectangle(frame,
                         (timeline_x, timeline_y),
                         (timeline_x + seg_w, timeline_y + bar_h),
                         (255, 255, 255), 1)
            # frame count label inside bar
            label = str(s["frames"])
            (lw, lh), _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            if lw < seg_w - 4:
                cv.putText(frame, label,
                           (timeline_x + (seg_w - lw) // 2, timeline_y + bar_h - 3),
                           cv.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv.LINE_AA)
            timeline_x += seg_w + 2


def draw_motion_meter(
    frame: np.ndarray,
    motion: float,
    low_th: float,
    high_th: float,
) -> None:
    """Draw a horizontal bar showing current motion vs thresholds."""
    h, w = frame.shape[:2]
    bar_x, bar_y = w - 220, h - 40
    bar_w, bar_h = 200, 20

    # Scale motion to bar width (cap at 3x high_th for display)
    max_display = max(high_th * 3, 0.1)
    fill = min(int((motion / max_display) * bar_w), bar_w)

    # Background
    cv.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                 (60, 60, 60), -1)

    # Low threshold mark
    low_x = int((low_th / max_display) * bar_w)
    cv.line(frame, (bar_x + low_x, bar_y - 4), (bar_x + low_x, bar_y + bar_h + 4),
            (100, 200, 255), 2)

    # High threshold mark
    high_x = int((high_th / max_display) * bar_w)
    cv.line(frame, (bar_x + high_x, bar_y - 4), (bar_x + high_x, bar_y + bar_h + 4),
            (100, 255, 200), 2)

    # Fill bar (green when below low_th → red)
    bar_color = (
        min(255, int((motion / max_display) * 400)),
        max(0, 255 - int((motion / max_display) * 400)),
        50
    )
    cv.rectangle(frame, (bar_x + 1, bar_y + 1),
                 (bar_x + fill - 1, bar_y + bar_h - 1), bar_color, -1)

    # Border
    cv.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                 (150, 150, 150), 1)

    cv.putText(frame, "MOTION", (bar_x - 70, bar_y + 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv.LINE_AA)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ---- open camera ----
    camera_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    cap = cv.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera #{camera_id}")
        sys.exit(1)

    # Set reasonable resolution
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv.CAP_PROP_FPS, 30)

    # ---- components ----
    converter = Converter()
    detector = MotionDetector(
        low_factor=0.5,
        high_factor=2.0,
        still_frames_required=8,
        min_sign_duration=5,
        history_size=30,
        feature_dim=FEATURE_DIM,
        smoothing_alpha=0.4,   # dampens MediaPipe frame-to-frame jitter
    )

    fps_meter = FPSMeter()

    # ---- state ----
    sign_count = 0
    segment_log: list[dict] = []  # track each completed segment
    paused = False
    frame_idx = 0

    print("=" * 55)
    print("  Sign Segmentation – Camera Test")
    print("=" * 55)
    print(f"  Camera    : #{camera_id}")
    print(f"  Feature dim: {FEATURE_DIM}")
    print(f"  Controls  : [r]eset  [p]ause  [q]uit")
    print("=" * 55)
    print()
    print("  Waiting for motion to start...")
    print()

    window_name = "Sign Segmentation Test"
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(window_name, 1280, 720)

    while True:
        # ---- handle key presses ----
        key = cv.pollKey()
        if key in (ord('q'), 27):  # q or ESC
            break
        elif key == ord('r'):
            detector.reset()
            sign_count = 0
            segment_log.clear()
            print("  [RESET] Motion detector and segment log cleared.")
            status_color = (255, 255, 0)  # CYAN flash
        elif key == ord('p'):
            paused = not paused
            print(f"  {'[PAUSED]' if paused else '[RESUMED]'}")

        if paused:
            cv.imshow(window_name, frame)
            continue

        # ---- capture ----
        ret, raw = cap.read()
        if not ret:
            print("[WARN] Failed to capture frame, retrying...")
            continue

        frame_idx += 1
        current_fps = fps_meter.tick()

        # ---- process ----
        try:
            # Encode frame to JPEG bytes (Converter expects raw bytes)
            ret_jpg, jpg_buf = cv.imencode('.jpg', raw, [cv.IMWRITE_JPEG_QUALITY, 85])
            if not ret_jpg:
                continue
            image_bytes = jpg_buf.tobytes()

            # Run MediaPipe
            keypoints = converter.point_detection(image_bytes)
            _ = converter.process_new_frame(keypoints)

            # Run motion detection
            sign_ended, completed = detector.update(keypoints)

        except Exception as e:
            print(f"[ERROR] {e}")
            continue

        # ---- compute display values ----
        motion = 0.0
        low_th = 0.02
        high_th = 0.08
        still_counter = detector.still_counter
        still_required = detector.still_frames_required
        sign_length = detector.get_current_sign_length()

        if len(detector.motion_history) > 0:
            motion = detector.motion_history[-1]
        if len(detector.motion_history) >= 10:
            base = float(np.median(list(detector.motion_history)))
            low_th = base * detector.low_factor
            high_th = base * detector.high_factor

        # ---- status & border color ----
        status = "SIGNING"
        status_color = (100, 255, 100)  # GREEN
        if sign_ended and completed is not None:
            est_duration = len(completed) / current_fps if current_fps > 0 else 0
            sign_count += 1  # increment before printing
            segment_log.append({
                "idx": sign_count,
                "frames": len(completed),
                "duration": est_duration,
            })
            status = "SIGN ENDED!"
            status_color = (50, 50, 255)  # RED

            # Build a segment-summary string for console
            seg_summary = " ".join(
                f"[#{s['idx']}:{s['frames']}fr]"
                for s in segment_log
            )
            print()
            print(f"  ╔══════════════════════════════════════════╗")
            print(f"  ║       ★ SIGN #{sign_count} DETECTED ★        ║")
            print(f"  ╠══════════════════════════════════════════╣")
            print(f"  ║  Frames       : {len(completed):>4}                      ║")
            print(f"  ║  Duration     : {est_duration:>6.2f}s (est)             ║")
            print(f"  ║  Keypoints    : {completed[0].shape[0]} dims per frame    ║")
            print(f"  ╠══════════════════════════════════════════╣")
            print(f"  ║  SEGMENTS: {sign_count} total")
            print(f"  ║  {seg_summary}")
            print(f"  ╚══════════════════════════════════════════╝")
            print()
        elif still_counter >= still_required:
            status = f"STILL ({still_counter}/{still_required})"
            status_color = (50, 50, 255)  # RED

        if motion < low_th:
            if status != "SIGN ENDED!" and status != f"STILL ({still_counter}/{still_required})":
                status = "STILL"
                status_color = (50, 150, 255)  # ORANGE
        elif low_th <= motion <= high_th:
            if not status.startswith("STILL"):
                status = f"HYSTERESIS ({still_counter}/{still_required})"
                status_color = (50, 200, 255)  # YELLOW

        # ---- draw on frame ----
        frame = raw.copy()
        h, w = frame.shape[:2]

        # Border
        border_w = 25
        cv.rectangle(frame, (0, 0), (w - 1, h - 1), status_color, border_w)

        # Draw MediaPipe holistic mesh on the frame
        converter.draw_landmarks(frame)

        # Info panel
        draw_info_panel(frame, sign_count, sign_length, motion,
                        low_th, high_th, still_counter, still_required,
                        current_fps, status, segment_log)

        # Motion meter
        draw_motion_meter(frame, motion, low_th, high_th)

        # ---- show ----
        cv.imshow(window_name, frame)

    # ---- cleanup ----
    cap.release()
    cv.destroyAllWindows()
    print()
    print("=" * 55)
    print(f"  Session complete – {sign_count} segment(s) detected")
    for s in segment_log:
        print(f"    #{s['idx']}: {s['frames']} frames ({s['duration']:.2f}s)")
    print("=" * 55)


if __name__ == "__main__":
    main()
