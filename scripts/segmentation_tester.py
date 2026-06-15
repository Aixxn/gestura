#!/usr/bin/env python3
"""
Interactive segmentation tuning tool.

Shows real-time motion graphs, adaptive thresholds, and sign boundaries
so you can dial in motion detector parameters without ML inference.

Usage
-----
    source translationService/.venv/bin/activate
    python scripts/segmentation_tester.py

Controls
--------
    +/-       Stillness floor ±0.05
    [/]       Still frames required ±1
    ,/.       Low factor / high factor ±0.1
    ;/'       Min sign duration ±1
    1-4       Load presets (1=lenient, 2=default, 3=strict, 4=fast)
    I         Toggle idle gate bypass
    R         Reset detector
    S         Print current settings to terminal
    D         Toggle debug overlay
    Q / ESC   Quit
"""

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
_TS = os.path.join(_PROJECT, "translationService")
if _TS not in sys.path:
    sys.path.insert(0, _TS)

from converter import Converter, FEATURE_DIM
from motion_detector import MotionDetector

import cv2 as cv
import numpy as np
import argparse
import time
from collections import deque

# ------------------------------------------------------------------ #
#  Defaults                                                           #
# ------------------------------------------------------------------ #

DEFAULT_LOW_FACTOR = 0.5
DEFAULT_HIGH_FACTOR = 4.0
DEFAULT_STILL_FRAMES = 8
DEFAULT_MIN_DURATION = 5
DEFAULT_HISTORY_SIZE = 30
DEFAULT_MOTION_SMOOTHING = 0.6
DEFAULT_STILLNESS_FLOOR = 0.5

PRESETS = {
    1: ("Lenient", dict(stillness_floor=0.3, low_factor=0.4, still_frames=12)),
    2: ("Default", dict(stillness_floor=0.5, low_factor=0.5, still_frames=8)),
    3: ("Strict",  dict(stillness_floor=0.7, low_factor=0.8, still_frames=5)),
    4: ("Fast",    dict(stillness_floor=0.4, low_factor=0.5, still_frames=4)),
}

GRAPH_MAX_FRAMES = 200
PANEL_W = 400
FONT = cv.FONT_HERSHEY_SIMPLEX

C_RAW = (70, 70, 70)
C_SMOOTH = (0, 255, 200)
C_LOW = (255, 200, 0)
C_HIGH = (0, 100, 255)
C_FLOOR = (200, 100, 0)
C_BOUNDARY = (255, 255, 255)
C_GREEN = (0, 255, 100)
C_RED = (0, 50, 200)
C_DIM = (120, 120, 120)
C_BRIGHT = (230, 230, 230)
C_BG = (18, 18, 18)
C_PANEL_BORDER = (40, 40, 40)


def parse_args():
    p = argparse.ArgumentParser(description="Interactive segmentation tuning")
    p.add_argument("--low-factor", type=float, default=DEFAULT_LOW_FACTOR)
    p.add_argument("--high-factor", type=float, default=DEFAULT_HIGH_FACTOR)
    p.add_argument("--still-frames", type=int, default=DEFAULT_STILL_FRAMES)
    p.add_argument("--min-duration", type=int, default=DEFAULT_MIN_DURATION)
    p.add_argument("--history-size", type=int, default=DEFAULT_HISTORY_SIZE)
    p.add_argument("--motion-smoothing", type=float, default=DEFAULT_MOTION_SMOOTHING)
    p.add_argument("--stillness-floor", type=float, default=DEFAULT_STILLNESS_FLOOR)
    return p.parse_args()


def _open_webcam():
    for idx in range(3):
        cap = cv.VideoCapture(idx, cv.CAP_V4L2)
        if cap.isOpened():
            cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc("M", "J", "P", "G"))
            cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
            return cap
    return None


# ------------------------------------------------------------------ #
#  Graph drawing                                                      #
# ------------------------------------------------------------------ #

def draw_motion_graph(canvas, motion_data, boundary_indices,
                      still_counter, still_frames_req, stillness_floor):
    h, w = canvas.shape[:2]

    if not motion_data:
        cv.putText(canvas, "Waiting for motion...", (10, h // 2),
                   FONT, 0.5, C_DIM, 1)
        return

    raw_vals = [d[0] for d in motion_data]
    sm_vals = [d[1] for d in motion_data]
    low_vals = [d[2] for d in motion_data]
    high_vals = [d[3] for d in motion_data]

    visible = max(min(len(motion_data), w - 20), 2)
    start = len(motion_data) - visible

    # y-range
    all_vals = raw_vals + sm_vals + low_vals + high_vals + [stillness_floor]
    pos = [v for v in all_vals if v > 0]
    y_max = max(pos) * 1.3 if pos else max(stillness_floor * 1.5, 0.5)
    y_max = max(y_max, 0.3)
    y_min = 0.0

    def y2p(val):
        return int(h - 12 - (val - y_min) / (y_max - y_min) * (h - 24))

    def x2p(idx):
        return int(10 + (idx - start) / (visible - 1) * (w - 20))

    # Grid
    for level in np.linspace(y_min, y_max, 6):
        yy = y2p(level)
        cv.line(canvas, (10, yy), (w - 10, yy), (30, 30, 30), 1)
        cv.putText(canvas, f"{level:.2f}", (w - 52, yy - 2),
                   FONT, 0.3, (60, 60, 60), 1)

    # Threshold lines (latest values)
    if low_vals:
        ly = y2p(low_vals[-1])
        cv.line(canvas, (10, ly), (w - 10, ly), C_LOW, 1, cv.LINE_AA)
        cv.putText(canvas, f"low={low_vals[-1]:.3f}", (10, ly - 3),
                   FONT, 0.3, C_LOW, 1)
    if high_vals:
        hy = y2p(high_vals[-1])
        cv.line(canvas, (10, hy), (w - 10, hy), C_HIGH, 1, cv.LINE_AA)
        cv.putText(canvas, f"high={high_vals[-1]:.3f}", (10, hy - 3),
                   FONT, 0.3, C_HIGH, 1)

    # Floor line
    fy = y2p(stillness_floor)
    cv.line(canvas, (10, fy), (w - 10, fy), C_FLOOR, 1, cv.LINE_AA)
    cv.putText(canvas, f"floor={stillness_floor:.2f}", (w // 2 + 20, fy - 3),
               FONT, 0.3, C_FLOOR, 1)

    # Boundary markers
    for bi in boundary_indices:
        if start <= bi < start + visible:
            bx = x2p(bi)
            cv.line(canvas, (bx, 4), (bx, h - 4), C_BOUNDARY, 1, cv.LINE_AA)

    # Raw motion polyline
    pts_raw = []
    for i in range(start, len(motion_data)):
        pts_raw.append((x2p(i), y2p(raw_vals[i])))
    if len(pts_raw) > 1:
        for i in range(len(pts_raw) - 1):
            cv.line(canvas, pts_raw[i], pts_raw[i + 1], C_RAW, 1, cv.LINE_AA)

    # Smoothed motion polyline
    pts_sm = []
    for i in range(start, len(motion_data)):
        pts_sm.append((x2p(i), y2p(sm_vals[i])))
    if len(pts_sm) > 1:
        for i in range(len(pts_sm) - 1):
            cv.line(canvas, pts_sm[i], pts_sm[i + 1], C_SMOOTH, 2, cv.LINE_AA)

    # Still counter bar (bottom-right)
    bar_x = w - 80
    bar_y = h - 18
    bar_w = 66
    bar_h = 10
    fill = int(bar_w * min(still_counter / max(still_frames_req, 1), 1.0))
    cv.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (35, 35, 35), -1)
    bar_color = C_GREEN if still_counter >= still_frames_req else (80, 80, 80)
    if fill > 0:
        cv.rectangle(canvas, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), bar_color, -1)
    cv.putText(canvas, f"still {still_counter}/{still_frames_req}",
               (bar_x, bar_y - 3), FONT, 0.3, C_DIM, 1)


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

def main():
    args = parse_args()

    print(f"\n{'=' * 60}")
    print(f"  SEGMENTATION TESTER")
    print(f"  Tune motion detector parameters in real-time")
    print(f"{'=' * 60}")
    print(f"\n  Initial: stillness_floor={args.stillness_floor}  "
          f"still_frames={args.still_frames}")
    print(f"  low_factor={args.low_factor}  high_factor={args.high_factor}  "
          f"min_dur={args.min_duration}  smoothing={args.motion_smoothing}\n")

    print("Initialising MediaPipe Holistic...")
    converter = Converter()

    md = MotionDetector(
        low_factor=args.low_factor,
        high_factor=args.high_factor,
        still_frames_required=args.still_frames,
        min_sign_duration=args.min_duration,
        history_size=args.history_size,
        feature_dim=FEATURE_DIM,
        motion_smoothing=args.motion_smoothing,
        stillness_floor=args.stillness_floor,
    )

    cap = _open_webcam()
    if cap is None:
        print("ERROR: No webcam found.")
        sys.exit(1)

    win_w = 640 + PANEL_W
    win_h = 480
    cv.namedWindow("Segmentation Tester", cv.WINDOW_NORMAL)
    cv.resizeWindow("Segmentation Tester", win_w, win_h)

    # Motion history
    motion_data = deque(maxlen=GRAPH_MAX_FRAMES)
    boundary_markers = deque(maxlen=50)

    show_debug = True
    bypass_idle = False
    sign_count = 0
    prev_time = time.time()
    frame_idx = 0
    last_sign_info = ""

    # Live params (start from CLI)
    slf = args.stillness_floor
    sfr = args.still_frames
    lf = args.low_factor
    hf = args.high_factor
    mind = args.min_duration

    def update_md_params():
        md.stillness_floor = slf
        md.still_frames_required = sfr
        md.low_factor = lf
        md.high_factor = hf
        md.min_sign_duration = mind

    _controls_help = [
        "+/- floor  [/] still  ,/. low/high",
        ";' min_dur  I idle  R reset  S print  D debug",
    ]

    print("\n  Controls:")
    for line in _controls_help:
        print(f"    {line}")
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv.flip(frame, 1)
        frame_idx += 1

        # Build canvas: camera | panel
        canvas = np.zeros((480, win_w, 3), dtype=np.uint8)
        canvas[:, :640] = frame

        # --- Pipeline ---
        try:
            kp = converter.extract_from_frame(frame)
        except Exception as e:
            print(f"  Extraction error: {e}")
            continue

        idle = converter.is_idle

        if bypass_idle or not idle:
            sign_ended, completed_sign = md.update(kp)

            # Record motion data for graphing
            raw_m = list(md.raw_motion_history)[-1] if md.raw_motion_history else 0.0
            sm = md.smoothed_motion
            if len(md.raw_motion_history) >= 10:
                ab = float(np.median(list(md.raw_motion_history)))
                lt = ab * md.low_factor
                ht = ab * md.high_factor
            else:
                lt = 0.1
                ht = 0.4
            motion_data.append((raw_m, sm, lt, ht))

            if sign_ended:
                sign_count += 1
                dur = len(completed_sign) / 30.0
                raw_last = raw_m
                sm_last = sm
                print(f"  SIGN #{sign_count}  "
                      f"frames={len(completed_sign)}  dur={dur:.2f}s  "
                      f"motion(raw={raw_last:.4f}  sm={sm_last:.4f})  "
                      f"idle={idle}")
                last_sign_info = f"SIGN #{sign_count}  {len(completed_sign)}fr  {dur:.2f}s"
                boundary_markers.append(len(motion_data) - 1)
        else:
            # Idle: record zero-motion data so the graph doesn't freeze
            motion_data.append((0.0, 0.0, 0.0, 0.0))

        converter.draw_landmarks(frame)
        hand_ok = converter._lh_lost_counter < 5 or converter._rh_lost_counter < 5

        # --- Right Panel ---
        panel = canvas[:, 640:]
        panel[:] = C_BG

        y = 12
        cv.putText(panel, "MOTION GRAPH", (12, y + 10), FONT, 0.5, C_GREEN, 2)
        cv.line(panel, (12, y + 18), (PANEL_W - 12, y + 18), C_PANEL_BORDER, 1)

        graph_canvas = panel[y + 24:y + 24 + 200, 10:PANEL_W - 10]
        if graph_canvas.size > 0:
            draw_motion_graph(graph_canvas, motion_data, boundary_markers,
                              md.still_counter, sfr, slf)

        # --- Parameter panel ---
        yy = y + 24 + 200 + 10
        cv.line(panel, (12, yy), (PANEL_W - 12, yy), C_PANEL_BORDER, 1)
        yy += 4
        cv.putText(panel, "PARAMETERS", (12, yy + 10), FONT, 0.45, C_GREEN, 1)

        params = [
            (f"stillness_floor={slf:.2f}",  "+/-"),
            (f"still_frames={sfr}",          "[/]"),
            (f"low_factor={lf:.2f}",         ","),
            (f"high_factor={hf:.2f}",        "."),
            (f"min_duration={mind}",         ";'"),
        ]
        yy += 24
        for label, key in params:
            cv.putText(panel, label, (16, yy), FONT, 0.4, C_BRIGHT, 1)
            cv.putText(panel, key, (PANEL_W - 50, yy), FONT, 0.35, C_DIM, 1)
            yy += 20

        # --- Status panel ---
        yy += 4
        cv.line(panel, (12, yy), (PANEL_W - 12, yy), C_PANEL_BORDER, 1)
        yy += 4
        cv.putText(panel, "STATUS", (12, yy + 10), FONT, 0.45, C_GREEN, 1)
        yy += 22

        status_lines = [
            (f"Idle: {idle}  {'(bypass)' if bypass_idle else ''}",
             C_DIM if idle else C_GREEN),
            (f"Hands: LH={'Y' if converter._corrected_lh else 'N'}  "
             f"RH={'Y' if converter._corrected_rh else 'N'}",
             C_GREEN if hand_ok else C_RED),
            (f"Lost: LH={converter._lh_lost_counter}  "
             f"RH={converter._rh_lost_counter}",
             C_DIM),
            (f"Still: {md.still_counter}/{sfr}  Sign: {md.sign_frames}fr",
             C_GREEN if md.still_counter >= sfr else C_DIM),
            (f"Motion: raw={motion_data[-1][0]:.4f}  sm={motion_data[-1][1]:.4f}"
             if motion_data else "Motion: --",
             C_BRIGHT),
        ]
        for text, color in status_lines:
            cv.putText(panel, text, (16, yy), FONT, 0.38, color, 1)
            yy += 18

        # --- Last sign info ---
        yy += 4
        cv.line(panel, (12, yy), (PANEL_W - 12, yy), C_PANEL_BORDER, 1)
        yy += 4
        cv.putText(panel, "SIGN LOG", (12, yy + 10), FONT, 0.45, C_GREEN, 1)
        yy += 22
        if last_sign_info:
            cv.putText(panel, last_sign_info, (16, yy), FONT, 0.42, C_BRIGHT, 1)
            yy += 20
            c = sign_count
            hz = frame_idx / max(time.time() - prev_time + 1e-6, 1)
            cv.putText(panel, f"Total: {sign_count} signs  "
                       f"FPS: {hz:.0f}", (16, yy), FONT, 0.38, C_DIM, 1)
            yy += 18
            cv.putText(panel, f"Frame: {frame_idx}", (16, yy), FONT, 0.35, C_DIM, 1)

        # Controls help at bottom of panel
        yy = 440
        cv.line(panel, (12, yy), (PANEL_W - 12, yy), C_PANEL_BORDER, 1)
        for i, line in enumerate(_controls_help):
            cv.putText(panel, line, (12, yy + 14 + i * 16),
                       FONT, 0.32, C_DIM, 1)

        # Preset indicator
        cv.putText(panel, "Presets: 1=Len  2=Def  3=Strict  4=Fast",
                   (12, 478), FONT, 0.3, C_DIM, 1)

        cv.imshow("Segmentation Tester", canvas)
        key = cv.waitKey(1) & 0xFF

        # --- Key handling ---
        if key == ord("q") or key == 27:
            break
        elif key == ord("r"):
            md.reset()
            motion_data.clear()
            boundary_markers.clear()
            sign_count = 0
            last_sign_info = ""
            print("  [R] Detector reset.\n")
        elif key == ord("d"):
            show_debug = not show_debug
            print(f"  [D] Debug overlay {'ON' if show_debug else 'OFF'}\n")
        elif key == ord("i"):
            bypass_idle = not bypass_idle
            print(f"  [I] Idle gate bypass {'ON' if bypass_idle else 'OFF'}\n")
        elif key == ord("s"):
            print(f"\n  {'=' * 50}")
            print(f"  CURRENT SETTINGS — copy these:")
            print(f"  {'=' * 50}")
            print(f"    stillness_floor={slf:.2f}")
            print(f"    still_frames_required={sfr}")
            print(f"    low_factor={lf:.2f}")
            print(f"    high_factor={hf:.2f}")
            print(f"    min_sign_duration={mind}")
            print(f"    motion_smoothing={md.motion_smoothing:.2f}")
            print(f"    idle_threshold check: "
                  f"idle_threshold > still_frames_required => "
                  f"{'OK' if 15 > sfr else 'WARN: idle may preempt signs'}")
            print(f"\n  CLI:")
            print(f"    --stillness-floor {slf:.2f} --still-frames {sfr} "
                  f"--low-factor {lf:.2f} --high-factor {hf:.2f} "
                  f"--min-duration {mind}")
            print(f"\n  MotionDetector constructor:")
            print(f"    MotionDetector(")
            print(f"        low_factor={lf:.2f},")
            print(f"        high_factor={hf:.2f},")
            print(f"        still_frames_required={sfr},")
            print(f"        min_sign_duration={mind},")
            print(f"        history_size={md.history_size},")
            print(f"        feature_dim={FEATURE_DIM},")
            print(f"        motion_smoothing={md.motion_smoothing:.2f},")
            print(f"        stillness_floor={slf:.2f},")
            print(f"    )")
            print(f"  {'=' * 50}\n")
        elif key == ord("=") or key == ord("+"):
            slf = min(slf + 0.05, 2.0)
            update_md_params()
            print(f"  stillness_floor={slf:.2f}")
        elif key == ord("-") or key == ord("_"):
            slf = max(slf - 0.05, 0.0)
            update_md_params()
            print(f"  stillness_floor={slf:.2f}")
        elif key == ord("["):
            sfr = max(sfr - 1, 1)
            update_md_params()
            print(f"  still_frames_required={sfr}")
        elif key == ord("]"):
            sfr = min(sfr + 1, 50)
            update_md_params()
            print(f"  still_frames_required={sfr}")
        elif key == ord(","):
            lf = max(lf - 0.1, 0.1)
            update_md_params()
            print(f"  low_factor={lf:.2f}")
        elif key == ord("."):
            lf = min(lf + 0.1, 5.0)
            update_md_params()
            print(f"  low_factor={lf:.2f}")
        elif key == ord(";"):
            mind = max(mind - 1, 1)
            update_md_params()
            print(f"  min_sign_duration={mind}")
        elif key == ord("'"):
            mind = min(mind + 1, 50)
            update_md_params()
            print(f"  min_sign_duration={mind}")
        elif key >= ord("1") and key <= ord("4"):
            preset_num = key - ord("0")
            if preset_num in PRESETS:
                name, p = PRESETS[preset_num]
                slf = p["stillness_floor"]
                lf = p["low_factor"]
                sfr = p["still_frames"]
                update_md_params()
                md.reset()
                motion_data.clear()
                boundary_markers.clear()
                sign_count = 0
                last_sign_info = ""
                print(f"  [{preset_num}] Preset '{name}': "
                      f"floor={slf:.2f}  low={lf:.2f}  still={sfr}\n")

    cap.release()
    cv.destroyAllWindows()
    print("Stopped.\n")


if __name__ == "__main__":
    main()
