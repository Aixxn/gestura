#!/usr/bin/env python3
"""
Live ASL inference with bounded persistence.

Uses the same nose-normalised, bounded-persistence keypoint extraction as the
production converter (translationService/converter.py) so that live webcam
inference matches training data format exactly.

Bounded persistence: when MediaPipe briefly loses a hand (flicker), the
last-known landmark positions are reused for up to PERSIST_WINDOW frames before
decaying to zeros.  This avoids zero-vector spikes that would corrupt the
sliding window buffer and trigger false "no hand" resets.

Usage
-----
    python model/ModelTester.py
"""

import sys
import os

# Path setup — import normalize_frames from sibling translationService/
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
_TRANSLATION_SERVICE = os.path.join(_PROJECT, "translationService")
if _TRANSLATION_SERVICE not in sys.path:
    sys.path.insert(0, _TRANSLATION_SERVICE)

from normalize import normalize_frames

import cv2 as cv
import numpy as np
import tensorflow as tf
import collections
import time
import mediapipe as mp

# ------------------------------------------------------------------ #
#  Configuration                                                       #
# ------------------------------------------------------------------ #

MODEL_PATH        = "best_model.keras"
CLASSES_PATH      = "sign_classes.npy"
CONFIDENCE_THRESH = 0.70
SEQ_LENGTH        = 35
FEATURE_DIM       = 258
SMOOTH_WINDOW     = 5          # majority-vote over last N predictions
PERSIST_WINDOW    = 5          # bounded persistence grace period (frames)

# Feature sub-dimensions
_LH_DIM   = 21 * 3      # 63
_RH_DIM   = 21 * 3      # 63
_POSE_DIM = 33 * 4      # 132

assert _LH_DIM + _RH_DIM + _POSE_DIM == FEATURE_DIM

# Visual
FONT         = cv.FONT_HERSHEY_SIMPLEX
COLOR_GREEN  = (0, 255, 180)
COLOR_DIM    = (80, 80, 80)
COLOR_WARN   = (0, 140, 255)
COLOR_WHITE  = (230, 230, 230)
BAR_COLOR    = (0, 200, 140)
BAR_BG       = (40, 40, 40)

mp_draw   = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


# ------------------------------------------------------------------ #
#  Nose-normalised keypoint extraction                                 #
# ------------------------------------------------------------------ #

def _extract_258_raw(results) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, bool]:
    """Extract raw (un-normalised) keypoint components from MediaPipe result.

    Returns
    -------
    lh, rh, pose : np.ndarray  Raw coordinate arrays (not nose-normalised).
    lh_detected, rh_detected : bool  Whether each hand was found by MediaPipe.
    """
    lh = np.zeros(_LH_DIM, dtype=np.float32)
    lh_detected = False
    if results.left_hand_landmarks:
        lh_detected = True
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            idx = i * 3
            lh[idx]     = lm.x
            lh[idx + 1] = lm.y
            lh[idx + 2] = lm.z

    rh = np.zeros(_RH_DIM, dtype=np.float32)
    rh_detected = False
    if results.right_hand_landmarks:
        rh_detected = True
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            idx = i * 3
            rh[idx]     = lm.x
            rh[idx + 1] = lm.y
            rh[idx + 2] = lm.z

    pose = np.zeros(_POSE_DIM, dtype=np.float32)
    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark):
            idx = i * 4
            pose[idx]     = lm.x
            pose[idx + 1] = lm.y
            pose[idx + 2] = lm.z
            pose[idx + 3] = getattr(lm, "visibility", 0.0)

    return lh, rh, pose, lh_detected, rh_detected


def _normalize(lh: np.ndarray, rh: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Apply nose-normalisation to raw component arrays.

    Returns a unified (258,) keypoint vector matching the training format.
    """
    nose_xyz = pose[0:3] if np.any(pose != 0) else np.zeros(3, dtype=np.float32)
    # Normalise all components relative to the nose
    lh = (lh.reshape(-1, 3) - nose_xyz).flatten()
    rh = (rh.reshape(-1, 3) - nose_xyz).flatten()
    p = pose.copy().reshape(-1, 4)
    p[:, :3] -= nose_xyz
    pose = p.flatten()
    return np.concatenate([lh, rh, pose]).astype(np.float32)


# ------------------------------------------------------------------ #
#  Bounded persistence state                                           #
# ------------------------------------------------------------------ #

class BoundedPersistence:
    """Last-known hand positions fill in during brief MediaPipe flicker.

    Operates on RAW (un-normalised) coordinates.  Nose-normalisation is
    applied AFTER persistence, matching the production converter pipeline.

    Each hand has its own lost-frame counter:
    - Hand detected       → update cache, reset counter to 0.
    - Hand lost (< limit) → return cached (persisted) position.
    - Hand lost (≥ limit) → return zeros (genuinely absent).

    The ``lh_detected`` / ``rh_detected`` flags come directly from
    MediaPipe's detection result, NOT from a threshold on the normalised
    values (which can falsely indicate a hand is present when a person
    is visible but not signing — see nose-normalisation of zero hands).
    """

    def __init__(self, persist_window: int = PERSIST_WINDOW):
        self.persist_window = persist_window
        self.reset()

    def reset(self):
        self._last_lh = np.zeros(_LH_DIM, dtype=np.float32)
        self._last_rh = np.zeros(_RH_DIM, dtype=np.float32)
        self._lh_lost = 0
        self._rh_lost = 0

    def update(self, lh: np.ndarray, rh: np.ndarray, pose: np.ndarray,
               lh_detected: bool, rh_detected: bool) -> np.ndarray:
        """Apply bounded persistence to raw landmark arrays.

        Parameters
        ----------
        lh, rh, pose : np.ndarray
            Raw coordinate arrays from ``_extract_258_raw()``.
        lh_detected, rh_detected : bool
            Whether MediaPipe found each hand in this frame.

        Returns
        -------
        Unified keypoint with persisted hands, shape (258,).  Caller
        should then apply ``_normalize()`` before feeding to the model.
        """
        # Left hand
        if lh_detected:
            self._last_lh = lh.copy()
            self._lh_lost = 0
        else:
            self._lh_lost += 1

        # Right hand
        if rh_detected:
            self._last_rh = rh.copy()
            self._rh_lost = 0
        else:
            self._rh_lost += 1

        # Build unified vector with bounded persistence
        if self._lh_lost < self.persist_window:
            lh_out = self._last_lh.copy()
        else:
            lh_out = np.zeros(_LH_DIM, dtype=np.float32)

        if self._rh_lost < self.persist_window:
            rh_out = self._last_rh.copy()
        else:
            rh_out = np.zeros(_RH_DIM, dtype=np.float32)

        return np.concatenate([lh_out, rh_out, pose]).astype(np.float32)

    @property
    def any_hand_alive(self) -> bool:
        """At least one hand is freshly detected or within the persistence window."""
        return self._lh_lost < self.persist_window or self._rh_lost < self.persist_window


# ------------------------------------------------------------------ #
#  Drawing helpers                                                     #
# ------------------------------------------------------------------ #

def draw_progress_bar(img, x, y, w, h, value, bar_color=BAR_COLOR, show_threshold=False):
    cv.rectangle(img, (x, y), (x + w, y + h), BAR_BG, -1)
    fill = int(w * min(max(value, 0.0), 1.0))
    if fill > 0:
        cv.rectangle(img, (x, y), (x + fill, y + h), bar_color, -1)
    if show_threshold:
        thresh_x = x + int(w * CONFIDENCE_THRESH)
        cv.line(img, (thresh_x, y - 2), (thresh_x, y + h + 2), (255, 255, 255), 1)


def put_text(img, text, pos, scale, color, thickness=1):
    x, y = pos
    cv.putText(img, text, (x + 1, y + 1), FONT, scale,
               (0, 0, 0), thickness + 1, cv.LINE_AA)
    cv.putText(img, text, pos, FONT, scale, color, thickness, cv.LINE_AA)


def draw_landmarks(img, results):
    """Draw MediaPipe landmarks using the old-API result object (raw)."""
    if results.left_hand_landmarks:
        mp_draw.draw_landmarks(
            img, results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_draw.DrawingSpec(color=(0, 200, 255), thickness=1, circle_radius=2),
            mp_draw.DrawingSpec(color=(0, 120, 200), thickness=1))

    if results.right_hand_landmarks:
        mp_draw.draw_landmarks(
            img, results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_draw.DrawingSpec(color=(0, 255, 180), thickness=1, circle_radius=2),
            mp_draw.DrawingSpec(color=(0, 160, 100), thickness=1))

    if results.pose_landmarks:
        mp_draw.draw_landmarks(
            img, results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_draw.DrawingSpec(color=(60, 60, 60), thickness=1, circle_radius=1),
            mp_draw.DrawingSpec(color=(40, 40, 40), thickness=1))


# ------------------------------------------------------------------ #
#  HUD                                                                 #
# ------------------------------------------------------------------ #

def draw_hud(frame, label, confidence, top_preds, buf_len,
             smoothed, fps, hand_ok):
    h, w = frame.shape[:2]
    panel_w = 290
    px      = w - panel_w

    overlay = frame.copy()
    cv.rectangle(overlay, (px, 0), (w, h), (10, 10, 10), -1)
    cv.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    put_text(frame, "GESTURA", (px + 12, 34), 0.7, COLOR_GREEN, 2)
    cv.line(frame, (px + 12, 44), (w - 12, 44), COLOR_DIM, 1)

    put_text(frame, "BUFFER", (px + 12, 68), 0.4, COLOR_DIM)
    draw_progress_bar(frame, px + 12, 75, panel_w - 24, 7,
                      buf_len / SEQ_LENGTH, bar_color=(80, 160, 255))
    put_text(frame, f"{buf_len}/{SEQ_LENGTH}", (px + 12, 95), 0.38, COLOR_DIM)

    sc = COLOR_GREEN if hand_ok else COLOR_WARN
    put_text(frame, "HAND DETECTED" if hand_ok else "NO HAND",
             (px + 12, 120), 0.42, sc)

    cv.line(frame, (px + 12, 132), (w - 12, 132), COLOR_DIM, 1)

    put_text(frame, "PREDICTION", (px + 12, 155), 0.42, COLOR_DIM)
    if confidence >= CONFIDENCE_THRESH:
        put_text(frame, label,                    (px + 12, 190), 0.9,  COLOR_GREEN, 2)
        put_text(frame, f"{int(confidence*100)}%", (px + 12, 212), 0.55, COLOR_WHITE)
        draw_progress_bar(frame, px + 12, 218, panel_w - 24, 6,
                          confidence, show_threshold=True)
    elif confidence > 0:
        put_text(frame, label,                         (px + 12, 190), 0.9,  COLOR_DIM, 2)
        put_text(frame, f"{int(confidence*100)}% (low)", (px + 12, 212), 0.42, COLOR_WARN)
        draw_progress_bar(frame, px + 12, 218, panel_w - 24, 6,
                          confidence, bar_color=COLOR_WARN, show_threshold=True)
    else:
        put_text(frame, "---", (px + 12, 190), 0.9, COLOR_DIM, 2)

    cv.line(frame, (px + 12, 242), (w - 12, 242), COLOR_DIM, 1)

    put_text(frame, "STABLE", (px + 12, 264), 0.42, COLOR_DIM)
    put_text(frame, smoothed or "---", (px + 12, 294),
             0.75, COLOR_GREEN if smoothed else COLOR_DIM, 2)

    cv.line(frame, (px + 12, 308), (w - 12, 308), COLOR_DIM, 1)

    put_text(frame, "TOP 3", (px + 12, 330), 0.42, COLOR_DIM)
    for i, (name, prob) in enumerate(top_preds[:3]):
        y = 352 + i * 46
        col = COLOR_GREEN if i == 0 else (0, 140, 100)
        put_text(frame, name, (px + 12, y), 0.48,
                 COLOR_WHITE if i == 0 else COLOR_DIM)
        draw_progress_bar(frame, px + 12, y + 5, panel_w - 65, 5,
                          prob, bar_color=col)
        put_text(frame, f"{int(prob*100)}%", (w - 50, y), 0.4, COLOR_DIM)

    put_text(frame, f"FPS {fps:.0f}", (px + 12, h - 14), 0.38, COLOR_DIM)

    if confidence >= CONFIDENCE_THRESH:
        bg_w = max(len(label) * 26 + 20, 100)
        cv.rectangle(frame, (10, h - 78), (10 + bg_w, h - 18), (10, 10, 10), -1)
        put_text(frame, label, (20, h - 26), 1.4, COLOR_GREEN, 3)

    meter_y = h - 100
    put_text(frame, f"CONF  {int(confidence*100)}%", (12, meter_y), 0.45, COLOR_WHITE)
    draw_progress_bar(frame, 12, meter_y + 6, 160, 6,
                      confidence,
                      bar_color=COLOR_GREEN if confidence >= CONFIDENCE_THRESH else COLOR_WARN,
                      show_threshold=True)
    put_text(frame, f"THRESH {int(CONFIDENCE_THRESH*100)}%", (12, meter_y + 22), 0.38, COLOR_DIM)


# ------------------------------------------------------------------ #
#  Main loop                                                           #
# ------------------------------------------------------------------ #

def run():
    cap = cv.VideoCapture(0, cv.CAP_V4L2)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam. Try changing index to 1 or 2.")
        return

    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    cv.namedWindow("Gestura — ASL Recognition", cv.WINDOW_NORMAL)
    cv.resizeWindow("Gestura — ASL Recognition", 960, 480)

    frame_buffer = collections.deque(maxlen=SEQ_LENGTH)
    pred_history = collections.deque(maxlen=SMOOTH_WINDOW)

    label      = ""
    confidence = 0.0
    top_preds  = []
    smoothed   = ""
    prev_time  = time.time()

    # Track hand state transitions — clear buffer only when hands REAPPEAR
    # (start of a new session), not when they disappear (let buffer drain naturally)
    _prev_hand_ok = True

    # Bounded persistence state — survives brief MediaPipe flicker
    persistence = BoundedPersistence(persist_window=PERSIST_WINDOW)

    print("Inference running. Press Q to quit.")

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame read failed.")
                break

            frame = cv.flip(frame, 1)

            # --- Detect landmarks (old stable API) ---
            image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = holistic.process(image_rgb)
            image_rgb.flags.writeable = True

            draw_landmarks(frame, results)

            # --- Extract raw keypoints, persist, THEN nose-normalise ---
            # Order matches production converter.py: persist raw → normalise,
            # so undetected hands decay to zeros before nose-offset is applied.
            lh, rh, pose, lh_det, rh_det = _extract_258_raw(results)
            persisted = persistence.update(lh, rh, pose, lh_det, rh_det)
            normalised_kp = _normalize(persisted[:_LH_DIM],
                                       persisted[_LH_DIM:_LH_DIM + _RH_DIM],
                                       persisted[_LH_DIM + _RH_DIM:])
            frame_buffer.append(normalised_kp)

            hand_ok = persistence.any_hand_alive

            # Clear buffer only when hands REAPPEAR (new signing session),
            # not when they disappear (let buffer drain naturally so the
            # model transitions to BACKGROUND on zero/persisted input).
            if hand_ok and not _prev_hand_ok:
                frame_buffer.clear()
                pred_history.clear()
                persistence.reset()
                label      = ""
                confidence = 0.0
                top_preds  = []
                smoothed   = ""
            _prev_hand_ok = hand_ok

            if len(frame_buffer) == SEQ_LENGTH:
                seq = np.array(frame_buffer, dtype=np.float32)          # (35, 258)
                # Normalise variable-length → fixed-length (35)
                seq_list = normalize_frames(seq.tolist(), SEQ_LENGTH)
                seq = np.array(seq_list, dtype=np.float32)
                inp = seq[np.newaxis]                                    # (1, 35, 258)

                preds     = model.predict(inp, verbose=0)[0]
                top_idx   = np.argsort(preds)[::-1]
                top_preds = [(sign_classes[i], float(preds[i])) for i in top_idx[:3]]

                best       = top_idx[0]
                confidence = float(preds[best])
                label      = sign_classes[best]

                # BACKGROUND means no sign happening — don't treat it as a word
                if label == "BACKGROUND":
                    smoothed = ""
                elif confidence >= CONFIDENCE_THRESH:
                    pred_history.append(label)
                    if len(pred_history) == SMOOTH_WINDOW:
                        smoothed = collections.Counter(pred_history).most_common(1)[0][0]
                else:
                    pred_history.clear()
                    smoothed = ""

            now       = time.time()
            fps       = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            draw_hud(frame, label, confidence, top_preds,
                     len(frame_buffer), smoothed, fps, hand_ok)

            cv.imshow("Gestura — ASL Recognition", frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv.destroyAllWindows()
    print("Stopped.")


# ------------------------------------------------------------------ #
#  Load model and classes                                              #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    if os.path.exists(CLASSES_PATH):
        sign_classes = np.load(CLASSES_PATH, allow_pickle=True)
    else:
        DATA_DIR = os.path.join(_HERE, "keypoint_data_augmented")
        sign_classes = sorted([
            d for d in os.listdir(DATA_DIR)
            if os.path.isdir(os.path.join(DATA_DIR, d))
        ])

    print(f"Loaded {len(sign_classes)} classes: {list(sign_classes)}")
    run()
