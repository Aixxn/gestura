#!/usr/bin/env python3
"""
Full pipeline integration test: extraction → segmentation → ML → grammar.

Runs the same components as the production translation service in a single
process with live webcam feed. No HTTP server required.

Usage
-----
    # Activate the translation service venv first:
    source translationService/.venv/bin/activate

    python scripts/test_pipeline.py

Controls
--------
    Q / ESC  — Stop session, run Groq grammar correction, print result
    C        — Clear accumulated words and restart the session

Output
------
    Real-time:  detected words printed to terminal + shown on frame
    On stop:    ASL gloss → corrected English printed to terminal
"""

import sys
import os

# Force CPU-only BEFORE any TensorFlow/Keras import — avoids XLA libdevice
# error on machines without NVIDIA CUDA toolkit installed.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Quick check: is the translationService venv activated?
try:
    import keras
except ImportError:
    print("\n  ❌ ERROR: 'keras' not found. You need to activate the translationService venv:")
    print("     source translationService/.venv/bin/activate")
    print("     python scripts/test_pipeline.py\n")
    sys.exit(1)

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
_TS = os.path.join(_PROJECT, "translationService")
if _TS not in sys.path:
    sys.path.insert(0, _TS)

from converter import Converter, WINDOW_SIZE, FEATURE_DIM
from motion_detector import MotionDetector
from normalize import normalize_frames

import cv2 as cv
import numpy as np
import keras
import time
from groq import Groq
from dotenv import load_dotenv

# Tunable via env var (overrides motion_detector default of 0.5)
_MD_STILLNESS_FLOOR = float(os.getenv("MD_STILLNESS_FLOOR", "0.5"))
_MD_STILL_FRAMES = int(os.getenv("MD_STILL_FRAMES", "8"))

# ------------------------------------------------------------------ #
#  Model + class map                                                  #
# ------------------------------------------------------------------ #

_MODEL_PATH = os.path.join(_TS, "best_model.keras")
_CLASSES_PATH = os.path.join(_TS, "sign_classes.npy")
CONFIDENCE_THRESH = 0.5

model = None
word_mapping: list[str] = []
_loud = "  !!! "

print(f"\n{'='*60}")
print(f"  Looking for model at: {_MODEL_PATH}")
print(f"  File exists:          {os.path.isfile(_MODEL_PATH)}")
print(f"{'='*60}\n")

print("Loading model...")
try:
    model = keras.models.load_model(_MODEL_PATH)
    num_classes = model.output_shape[-1]
    print(f"  ✅ Model loaded: {model.input_shape} -> {model.output_shape}")
except Exception as e:
    import traceback
    print(f"  ❌ FAILED to load model: {e}")
    traceback.print_exc()
    print(f"  {_loud}FALLBACK: will print 'sign_detected' for all boundaries")

if model is not None:
    try:
        word_mapping = list(np.load(_CLASSES_PATH, allow_pickle=True))
        if len(word_mapping) != num_classes:
            print(f"  ⚠️  sign_classes ({len(word_mapping)}) "
                  f"!= model ({num_classes}), using fallback labels")
            word_mapping = [f"word_{i}" for i in range(num_classes)]
        else:
            print(f"  ✅ {len(word_mapping)} classes loaded: {word_mapping[:5]}...")
    except Exception as e:
        print(f"  ❌ Could not load sign_classes.npy ({e}), using fallback labels")
        word_mapping = [f"word_{i}" for i in range(num_classes)]

# ------------------------------------------------------------------ #
#  Groq grammar fixer                                                 #
# ------------------------------------------------------------------ #

load_dotenv(os.path.join(_TS, ".env"))
groq_client = None
if os.getenv("GROQ_API_KEY"):
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    print("  Groq LLM ready for grammar correction")
else:
    print("  No GROQ_API_KEY — grammar correction disabled")

_GROQ_SYSTEM_PROMPT = (
    "You are an expert in American Sign Language (ASL) grammar conversion. "
    "Convert ASL gloss text (space-separated signs) into natural, "
    "grammatically correct English."
)


def fix_grammar(asl_gloss: str) -> str:
    if not groq_client or not asl_gloss.strip():
        return asl_gloss
    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": _GROQ_SYSTEM_PROMPT},
                {"role": "user", "content": f"Convert to English: {asl_gloss}"},
            ],
            temperature=0.3,
            max_tokens=100,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  Groq error: {e}")
        return asl_gloss


# ------------------------------------------------------------------ #
#  Predict helper                                                      #
# ------------------------------------------------------------------ #

def predict_word(keypoints_sequence: list) -> tuple[str, float]:
    if model is None:
        return "sign_detected", 0.0
    normalized = normalize_frames(keypoints_sequence, WINDOW_SIZE)
    inp = np.array(normalized, dtype=np.float32)[np.newaxis, ...]
    probs = model.predict(inp, verbose=0)[0]
    idx = int(np.argmax(probs))
    return word_mapping[idx], float(probs[idx])


# ------------------------------------------------------------------ #
#  Webcam helpers                                                      #
# ------------------------------------------------------------------ #

def _open_webcam() -> cv.VideoCapture | None:
    for idx in range(3):
        cap = cv.VideoCapture(idx, cv.CAP_V4L2)
        if cap.isOpened():
            cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc("M", "J", "P", "G"))
            cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
            return cap
    return None


# ------------------------------------------------------------------ #
#  Main loop                                                           #
# ------------------------------------------------------------------ #

def run():
    print("\nInitialising MediaPipe Holistic...")
    converter = Converter()
    print(f"  MotionDetector: stillness_floor={_MD_STILLNESS_FLOOR}, still_frames={_MD_STILL_FRAMES}")
    print(f"  Set MD_STILLNESS_FLOOR env var to tune (default 0.5).")
    print(f"  Tip: try MD_STILLNESS_FLOOR=0.3 (old default) or 0.8 if too many/too few detections.")
    md = MotionDetector(
        low_factor=0.5,
        high_factor=4.0,
        still_frames_required=_MD_STILL_FRAMES,
        min_sign_duration=5,
        history_size=30,
        feature_dim=FEATURE_DIM,
        motion_smoothing=0.6,
        stillness_floor=_MD_STILLNESS_FLOOR,
    )

    cap = _open_webcam()
    if cap is None:
        print("ERROR: No webcam found.")
        sys.exit(1)

    cv.namedWindow("Gestura — Pipeline Test", cv.WINDOW_NORMAL)
    cv.resizeWindow("Gestura — Pipeline Test", 960, 480)

    words: list[str] = []
    last_word = ""
    last_conf = 0.0
    word_ttl = 0
    prev_time = time.time()

    # Final model status check
    if model is None:
        print(f"  {_loud}WARNING: model is None — all predictions will be 'sign_detected'!")
        print(f"  {_loud}Check the error above for why model loading failed.")
    else:
        print(f"  ✅ model is loaded and ready ({model.input_shape} -> {model.output_shape})")

    print("\n" + "=" * 60)
    print("  Gestura Pipeline Tester  [DIAGNOSTIC MODE]")
    print("  Sign in front of the webcam.")
    print("  Controls:  Q = stop & translate     C = clear")
    print("  D = toggle debug overlay (default ON)")
    print("=" * 60 + "\n")

    show_debug = True
    frame_count = 0
    debug_terminal_interval = 30  # print debug to terminal every N frames

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv.flip(frame, 1)
        frame_count += 1

        # ---- Pipeline: extraction + segmentation ----
        try:
            kp = converter.extract_from_frame(frame)
        except Exception as e:
            print(f"  Extraction error: {e}")
            continue

        # TODO: Re-add idle gate once tuned (see converter.is_idle / IDLE_THRESHOLD)
        # if converter.is_idle:
        #     md.reset()
        # else:
        sign_ended, completed_sign = md.update(kp)

        if sign_ended and completed_sign is not None:
            kp_list = [k.tolist() for k in completed_sign]
            word, conf = predict_word(kp_list)
            words.append(word)
            last_word = word
            last_conf = conf
            word_ttl = 60

            model_status = "MODEL_LOADED" if model is not None else "MODEL_NONE"
            print(f"  -> Sign #{len(words)-1}: {word}  ({conf:.0%})  [{model_status}]")
            print(f"     Gloss so far: {' '.join(words)}")

        # ---- Draw landmarks ----
        converter.draw_landmarks(frame)

        # ---- HUD overlay ----
        h, w = frame.shape[:2]

        # Last detected word (top-left)
        if word_ttl > 0:
            word_ttl -= 1
            label = f"{last_word}  ({last_conf:.0%})"
            tw = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0][0]
            cv.rectangle(frame, (10, 10), (10 + tw + 20, 50), (10, 10, 10), -1)
            cv.putText(frame, label, (20, 40), cv.FONT_HERSHEY_SIMPLEX,
                       1.0, (0, 255, 180), 2)

        # Sign count + recent words
        cv.putText(frame, f"Signs: {len(words)}", (10, 80),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        if words:
            recent = " ".join(words[-8:])
            cv.putText(frame, recent, (10, 100),
                       cv.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

        # Accumulated gloss (bottom bar)
        gloss = " ".join(words)
        if gloss:
            cv.rectangle(frame, (0, h - 50), (w, h), (10, 10, 10), -1)
            cv.putText(frame, gloss, (10, h - 15),
                       cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 180), 2)

        # Hand detection status (bottom-left)
        lh_detected = converter._corrected_lh is not None
        rh_detected = converter._corrected_rh is not None
        pose_detected = converter._last_result is not None and converter._last_result.pose_landmarks is not None
        hand_status = f"LH={'Y' if lh_detected else 'N'}  RH={'Y' if rh_detected else 'N'}  Pose={'Y' if pose_detected else 'N'}"
        cv.putText(frame, hand_status, (10, h - 55),
                   cv.FONT_HERSHEY_SIMPLEX, 0.4,
                   (0, 255, 0) if (lh_detected or rh_detected) else (100, 100, 100), 1)

        # FPS
        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv.putText(frame, f"FPS: {fps:.0f}", (w - 100, 25),
                   cv.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)

        # ---- Debug overlay (motion detector internals) ----
        if show_debug:
            # Compute adaptive thresholds for display
            if len(md.raw_motion_history) >= 10:
                adaptive_base = float(np.median(list(md.raw_motion_history)))
                low_th = adaptive_base * md.low_factor
                high_th = adaptive_base * md.high_factor
            else:
                adaptive_base = 0.0
                low_th = 0.1
                high_th = 0.4

            lines = [
                f"still_ct:  {md.still_counter}/{md.still_frames_required}",
                f"sign_fr:   {md.sign_frames}",
                f"smoothed:  {md.smoothed_motion:.4f}",
                f"stillness: {md.stillness_floor:.2f}  <-- floor",
                f"low_th:    {low_th:.4f}",
                f"high_th:   {high_th:.4f}",
                f"base_med:  {adaptive_base:.4f}",
                f"lh_lost:   {converter._lh_lost_counter:2d}  rh_lost: {converter._rh_lost_counter:2d}",
                f"idle:      {converter.is_idle}",
            ]
            # Draw semi-transparent debug panel (top-right)
            panel_x = w - 260
            panel_w = 250
            panel_h = len(lines) * 22 + 20
            overlay = frame.copy()
            cv.rectangle(overlay, (panel_x, 10), (panel_x + panel_w, 10 + panel_h),
                         (20, 20, 20), -1)
            cv.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            for i, line in enumerate(lines):
                color = (0, 255, 100) if i == 0 and md.still_counter >= md.still_frames_required else \
                        (255, 200, 0) if i == 0 else (200, 200, 200)
                cv.putText(frame, line, (panel_x + 10, 30 + i * 22),
                           cv.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Terminal debug (periodic)
            if frame_count % debug_terminal_interval == 0:
                print(f"  [dbg fr={frame_count:4d}] still_ct={md.still_counter:2d}  "
                      f"smoothed={md.smoothed_motion:.4f}  floor={md.stillness_floor:.2f}  "
                      f"low={low_th:.4f}  high={high_th:.4f}  "
                      f"lh_lost={converter._lh_lost_counter:2d}  rh_lost={converter._rh_lost_counter:2d}  "
                      f"is_idle={converter.is_idle}  signs={len(words)}")

        cv.imshow("Gestura — Pipeline Test", frame)
        key = cv.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break
        elif key == ord("c"):
            words.clear()
            last_word = ""
            last_conf = 0.0
            word_ttl = 0
            md.reset()
            converter.reset_state()
            print("\n  [C] Session cleared. Starting fresh.\n")
        elif key == ord("d"):
            show_debug = not show_debug
            print(f"\n  [D] Debug overlay {'ON' if show_debug else 'OFF'}\n")

    # ---- Cleanup ----
    cap.release()
    cv.destroyAllWindows()

    # ---- Grammar correction ----
    asl_gloss = " ".join(words)
    print("\n" + "=" * 60)
    print(f"  ASL Gloss: {asl_gloss}")

    if asl_gloss.strip():
        print("  Fixing grammar... ", end="", flush=True)
        english = fix_grammar(asl_gloss)
        print(f"\n  English:   {english}")
    else:
        print("  No signs detected.")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    run()
