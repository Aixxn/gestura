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
    Q / ESC  — Stop session, run Flan-T5 grammar correction, print result
    C        — Clear accumulated words and restart the session

Output
------
    Real-time:  detected words printed to terminal + shown on frame
    On stop:    ASL gloss → corrected English printed to terminal
"""

import sys
import os
from pathlib import Path

# ---- CUDA / XLA setup: find libdevice.10.bc so XLA can compile GPU kernels ----
# This must happen before any keras/tensorflow import.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
_CUDA_CANDIDATES = [
    os.environ.get("CUDA_HOME"),
    os.environ.get("CUDA_ROOT"),
    os.environ.get("CUDA_TOOLKIT_ROOT_DIR"),
    "/usr/local/cuda",
    "/opt/cuda",
    "/usr/lib/cuda",
    "/usr/local/cuda-12",
    "/usr/local/cuda-11",
]
# Auto-detect: check for libdevice under each candidate, plus wildcard search
_found_cuda = None
for cand in _CUDA_CANDIDATES:
    if cand and os.path.isfile(os.path.join(cand, "nvvm", "libdevice", "libdevice.10.bc")):
        _found_cuda = cand
        break
if not _found_cuda:
    # Fallback: try locating via nvidia-smi or ldconfig
    import subprocess  # noqa: E402
    try:
        _nsmi = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
        if _nsmi.returncode == 0:
            # Look in typical locations relative to driver
            for _root in ["/usr/lib/cuda", "/usr/local/cuda"]:
                if os.path.isfile(os.path.join(_root, "nvvm", "libdevice", "libdevice.10.bc")):
                    _found_cuda = _root
                    break
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    # Final fallback: try nvidia-cuda-nvvm pip package
    if not _found_cuda:
        try:
            import importlib.resources as _res
            # The nvidia-cuda-nvvm package bundles libdevice
            _nvvm = __import__("nvidia.cuda_nvvm", fromlist=[""])
            _pkg_path = os.path.dirname(_nvvm.__file__)
            _candidate = os.path.join(_pkg_path, "nvvm", "libdevice")
            if os.path.isfile(os.path.join(_candidate, "libdevice.10.bc")):
                # Set CUDA data dir to the parent of nvvm/
                _found_cuda = os.path.dirname(_candidate)
        except ImportError:
            pass

if _found_cuda:
    os.environ.setdefault("XLA_FLAGS", f"--xla_gpu_cuda_data_dir={_found_cuda}")
    print(f"[CUDA] Found at {_found_cuda}, set XLA_FLAGS")
else:
    print("[CUDA] libdevice.10.bc not found in common locations.")
    print("[CUDA] If loading fails, set XLA_FLAGS manually, e.g.:")
    print('[CUDA]   export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"')
# ---------------------------------------------------------------------------

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

from converter import Converter, WINDOW_SIZE, FEATURE_DIM, ExponentialMovingAverage
from motion_detector import MotionDetector
from normalize import normalize_frames

import cv2 as cv
import numpy as np
import keras
import time
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
#  Flan-T5 grammar fixer                                               #
# ------------------------------------------------------------------ #

load_dotenv(os.path.join(_TS, ".env"))


class ASLGrammarFixer:
    def __init__(self, model_name: str | None = None, max_new_tokens: int = 100):
        local_model_path = Path(_TS) / "models" / "flan-t5-asl-mini"
        default_model = str(local_model_path) if local_model_path.exists() else "google/flan-t5-small"
        self.model_name = model_name or os.getenv("FLAN_T5_MODEL", default_model)
        self.max_new_tokens = max_new_tokens
        self._tokenizer = None
        self._model = None
        self.prompt_template = "translate ASL gloss to English: {asl_gloss}"

    def _load_model(self):
        if self._tokenizer is None or self._model is None:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self._model.eval()
        return self._tokenizer, self._model

    def fix_grammar(self, asl_gloss: str) -> str:
        cleaned_gloss = self._clean_gloss(asl_gloss)
        if not cleaned_gloss:
            return ""

        try:
            tokenizer, model = self._load_model()
            prompt = self.prompt_template.format(asl_gloss=cleaned_gloss)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            if hasattr(inputs, "to"):
                inputs = inputs.to(model.device)
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=4,
                do_sample=False,
            )
            translated = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            translated = self._clean_output(translated)
            if not self._is_acceptable_polish(cleaned_gloss, translated):
                raise ValueError(f"FLAN-T5 returned invalid English output: {translated}")
            return translated
        except Exception as e:
            print(f"  FLAN-T5 error: {e}")
            raise e

    def _clean_output(self, text: str) -> str:
        return text.strip().strip('"').strip("'").strip()

    def _clean_gloss(self, asl_gloss: str) -> str:
        tokens = self._collapse_repeated_tokens(self._tokenize_gloss(asl_gloss))
        return " ".join(tokens)

    def _tokenize_gloss(self, asl_gloss: str) -> list[str]:
        return [
            token.strip(" ,.!?;:\"'()[]{}").upper()
            for token in asl_gloss.split()
            if token.strip(" ,.!?;:\"'()[]{}")
        ]

    def _collapse_repeated_tokens(self, tokens: list[str]) -> list[str]:
        collapsed: list[str] = []
        for token in tokens:
            if not collapsed or collapsed[-1] != token:
                collapsed.append(token)
        return collapsed

    def _normalize_for_compare(self, text: str) -> list[str]:
        cleaned = text.replace("?", " ").replace(".", " ").replace(",", " ")
        return [token.upper() for token in cleaned.split()]

    def _is_echo(self, asl_gloss: str, translated: str) -> bool:
        return self._normalize_for_compare(asl_gloss) == self._normalize_for_compare(translated)

    def _is_acceptable_polish(self, asl_gloss: str, translated: str) -> bool:
        if not translated:
            return False
        lowered = translated.lower()
        if "asl gloss:" in lowered or "structured english:" in lowered:
            return False
        if translated == translated.upper() and any(char.isalpha() for char in translated):
            return False
        return True


grammar_fixer = ASLGrammarFixer()


def fix_grammar(asl_gloss: str) -> str:
    return grammar_fixer.fix_grammar(asl_gloss)


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

    smoother = ExponentialMovingAverage(alpha=0.4)

    cap = _open_webcam()
    if cap is None:
        print("ERROR: No webcam found.")
        sys.exit(1)

    cv.namedWindow("Gestura — Pipeline Test", cv.WINDOW_NORMAL)
    cv.resizeWindow("Gestura — Pipeline Test", 1000, 480)

    words: list[str] = []
    last_word = ""
    last_conf = 0.0
    word_ttl = 0
    sign_records: list[dict] = []
    background_count = 0
    _session_start = time.time()
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
    print("  Controls:  Q = stop & translate     C = clear     P = perf")
    print("  D = toggle debug overlay (default ON)")
    print("=" * 60 + "\n")

    show_debug = True
    show_perf = False
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
        smoothed_kp = smoother.update(kp)

        if not converter.is_idle:
            sign_ended, completed_sign = md.update(kp, store_kp=smoothed_kp)

            if sign_ended and completed_sign is not None:
                kp_list = [k.tolist() for k in completed_sign]
                word, conf = predict_word(kp_list)
                dur = len(completed_sign) / 30.0
                raw_m = list(md.raw_motion_history)[-1] if md.raw_motion_history else 0.0
                sm_m = md.smoothed_motion

                # Record metrics for performance summary
                # (record before BACKGROUND check so we track all boundaries)
                rec = dict(word=word, conf=conf, frames=len(completed_sign),
                           dur=dur, raw_motion=raw_m, smoothed_motion=sm_m)
                sign_records.append(rec)

                # Skip BACKGROUND predictions — they are not real signs
                if word == "BACKGROUND":
                    background_count += 1
                    print(f"  ━━ BG #{background_count} ━━  "
                          f"frames={len(completed_sign)}  dur={dur:.2f}s  motion={sm_m:.4f}")
                    continue

                words.append(word)
                last_word = word
                last_conf = conf
                word_ttl = 60

                model_status = "MODEL_LOADED" if model is not None else "MODEL_NONE"
                print(f"  ━━ SIGN #{len(words)} ━━  "
                      f'word="{word}"  conf={conf:.0%}  '
                      f"frames={len(completed_sign)}  dur={dur:.2f}s  "
                      f"motion={sm_m:.4f}  [{model_status}]")
                print(f"     Gloss: {' '.join(words)}")
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

    # ---- Segmentation performance summary ----
    if sign_records or background_count > 0:
        elapsed = time.time() - _session_start
        total_valid = sum(1 for r in sign_records if r["word"] != "BACKGROUND")
        total_all = len(sign_records)
        print("\n" + "=" * 60)
        print("  SEGMENTATION PERFORMANCE")
        print("=" * 60)
        print(f"  Valid signs:       {total_valid}")
        print(f"  Skipped (BG):      {background_count}")
        print(f"  Total boundaries:  {total_all}")
        if total_valid > 0:
            valid = [r for r in sign_records if r["word"] != "BACKGROUND"]
            durs = [r["dur"] for r in valid]
            confs = [r["conf"] for r in valid]
            motions = [r["smoothed_motion"] for r in valid]
            print(f"  Duration:          avg={sum(durs)/len(durs):.2f}s  "
                  f"min={min(durs):.2f}s  max={max(durs):.2f}s")
            print(f"  Confidence:        avg={sum(confs)/len(confs):.0%}  "
                  f"min={min(confs):.0%}  max={max(confs):.0%}")
            print(f"  Motion@boundary:   avg={sum(motions)/len(motions):.4f}")
            spm = total_valid / max(elapsed / 60.0, 0.001)
            print(f"  Signs/minute:      {spm:.1f}")
            print(f"  Session duration:  {elapsed:.1f}s")
        print("=" * 60 + "\n")

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
