import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2 as cv
import numpy as np
import os

# Defaults – can be overridden via env vars
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "35"))
FEATURE_DIM = int(os.getenv("FEATURE_DIM", "1662"))

_MODEL_DIR = os.path.dirname(__file__)
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "holistic_landmarker/holistic_landmarker/float16/latest/"
    "holistic_landmarker.task"
)
_MODEL_PATH = os.path.join(_MODEL_DIR, "holistic_landmarker.task")


def _ensure_model() -> str:
    """Download the holistic landmarker model if not present."""
    if os.path.exists(_MODEL_PATH):
        return _MODEL_PATH
    print(f"[Converter] Downloading holistic_landmarker.task (~13 MB)...")
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    print(f"[Converter] Model saved to {_MODEL_PATH}")
    return _MODEL_PATH


class Converter:
    """
    Converts raw image bytes → MediaPipe Holistic keypoints → sliding window.

    Usage
    -----
        conv = Converter()
        kp = conv.point_detection(raw_image_bytes)       # 1662-dim vector
        window = conv.process_new_frame(kp)               # (WINDOW_SIZE, FEATURE_DIM)
        final = conv.stop()                                # current window snapshot
    """

    def __init__(self):
        model_path = _ensure_model()
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HolisticLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_hand_landmarks_confidence=0.7,  # reduce jittery hand detections
        )
        self.landmarker = vision.HolisticLandmarker.create_from_options(options)

        # Fixed-size sliding window buffer
        self.window = np.zeros((WINDOW_SIZE, FEATURE_DIM), dtype=np.float32)
        self.current_length = 0  # how many real frames have been inserted

        self._debug_image = None
        self._last_result = None

        # Drawing persistence cache — stores normalized (x, y) coordinates
        # for each landmark group so draw_landmarks() always has something
        # to render, even when MediaPipe briefly loses detection.
        self._draw_face: list[tuple[float, float]] = []
        self._draw_lh: list[tuple[float, float]] = []
        self._draw_rh: list[tuple[float, float]] = []

        # Landmark persistence cache — when MediaPipe briefly loses a hand
        # (detection flicker), reuse the last-known keypoints instead of
        # falling back to zeros. This prevents huge motion spikes when the
        # hand "pops" back into view.
        self._last_face = np.zeros(468 * 3, dtype=np.float32)
        self._last_lh = np.zeros(21 * 3, dtype=np.float32)
        self._last_rh = np.zeros(21 * 3, dtype=np.float32)
        self._last_pose = np.zeros(33 * 4, dtype=np.float32)

    def __del__(self):
        if hasattr(self, 'landmarker'):
            self.landmarker.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def point_detection(self, image_bytes: bytes) -> np.ndarray:
        """
        Decode raw image bytes and run MediaPipe Holistic.

        Returns
        -------
        keypoints : np.ndarray, shape (FEATURE_DIM,)
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        cv_image = cv.imdecode(nparr, cv.IMREAD_COLOR)
        if cv_image is None:
            raise ValueError("Could not decode image bytes (corrupt data).")

        image_rgb = cv.cvtColor(cv_image, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        detection_result = self.landmarker.detect(mp_image)
        self._last_result = detection_result
        keypoints = self._extract_keypoints(detection_result)

        self._debug_image = cv.cvtColor(image_rgb, cv.COLOR_RGB2BGR)

        if keypoints.shape != (FEATURE_DIM,):
            raise ValueError(
                f"MediaPipe returned {keypoints.shape[0]}-dim keypoints, "
                f"expected {FEATURE_DIM}. Check FEATURE_DIM env var."
            )

        return keypoints

    def process_new_frame(self, frame_vector: np.ndarray) -> np.ndarray:
        """
        Insert a new frame into the sliding window buffer.

        Returns the current window state: (WINDOW_SIZE, FEATURE_DIM).
        Before the buffer is full, the right side stays zero-padded.
        """
        if self.current_length < WINDOW_SIZE:
            self.window[self.current_length] = frame_vector
            self.current_length += 1
        else:
            # Slide window left, append at end
            self.window[:-1] = self.window[1:]
            self.window[-1] = frame_vector

        return self.window

    def stop(self) -> np.ndarray:
        """Return the current window snapshot (right-padded with zeros)."""
        return self.window

    def draw_landmarks(self, frame: np.ndarray) -> None:
        """Draw the MediaPipe holistic landmark mesh onto *frame* in-place.

        Uses cached coordinates when MediaPipe briefly loses a landmark
        group, so the overlay stays stable instead of flickering.
        """
        if self._last_result is None:
            return
        h, w = frame.shape[:2]
        result = self._last_result

        def _to_px(nx: float, ny: float) -> tuple[int, int]:
            return int(nx * w), int(ny * h)

        # --- Face landmarks (drawn as small dots) ---
        if result.face_landmarks:
            self._draw_face = [(lm.x, lm.y) for lm in result.face_landmarks]
        for nx, ny in self._draw_face:
            cv.circle(frame, _to_px(nx, ny), 1, (200, 200, 100), -1)

        # (Pose landmarks intentionally omitted — reduces visual clutter)

        hand_conn = [(i, i + 1) for i in range(20)]

        # --- Left hand landmarks ---
        if result.left_hand_landmarks:
            self._draw_lh = [(lm.x, lm.y) for lm in result.left_hand_landmarks]
        if self._draw_lh:
            pts = [_to_px(nx, ny) for nx, ny in self._draw_lh]
            for a, b in hand_conn:
                cv.line(frame, pts[a], pts[b], (255, 0, 100), 2)
            for pt in pts:
                cv.circle(frame, pt, 4, (255, 50, 150), -1)

        # --- Right hand landmarks ---
        if result.right_hand_landmarks:
            self._draw_rh = [(lm.x, lm.y) for lm in result.right_hand_landmarks]
        if self._draw_rh:
            pts = [_to_px(nx, ny) for nx, ny in self._draw_rh]
            for a, b in hand_conn:
                cv.line(frame, pts[a], pts[b], (100, 0, 255), 2)
            for pt in pts:
                cv.circle(frame, pt, 4, (150, 50, 255), -1)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_keypoints(self, result) -> np.ndarray:
        """Flatten MediaPipe Holistic landmarks into a single vector.

        When a landmark group (e.g. left hand) is not detected in the current
        frame, the **last known** keypoints are reused instead of zeros.
        This prevents huge motion spikes from detection flicker.
        """
        # The new holistic landmarker model returns 478 face landmarks,
        # but the existing pipeline (ML model, translation service) expects
        # 468 (= 1662 total). Truncate to maintain compatibility.
        MAX_FACE = 468
        FACE_DIM = MAX_FACE * 3  # 1404

        # --- Face landmarks ---
        face = np.zeros(FACE_DIM, dtype=np.float32)
        if result.face_landmarks:
            for i, lm in enumerate(result.face_landmarks[:MAX_FACE]):
                idx = i * 3
                face[idx] = lm.x
                face[idx + 1] = lm.y
                face[idx + 2] = lm.z
            self._last_face = face.copy()
        else:
            face = self._last_face.copy()

        # --- Left hand landmarks (21 × 3 = 63) ---
        # Most prone to flicker — persistence is critical here.
        lh = np.zeros(21 * 3, dtype=np.float32)
        if result.left_hand_landmarks:
            for i, lm in enumerate(result.left_hand_landmarks):
                idx = i * 3
                lh[idx] = lm.x
                lh[idx + 1] = lm.y
                lh[idx + 2] = lm.z
            self._last_lh = lh.copy()
        else:
            lh = self._last_lh.copy()

        # --- Right hand landmarks (21 × 3 = 63) ---
        rh = np.zeros(21 * 3, dtype=np.float32)
        if result.right_hand_landmarks:
            for i, lm in enumerate(result.right_hand_landmarks):
                idx = i * 3
                rh[idx] = lm.x
                rh[idx + 1] = lm.y
                rh[idx + 2] = lm.z
            self._last_rh = rh.copy()
        else:
            rh = self._last_rh.copy()

        # --- Pose landmarks (33 × 4 = 132; includes visibility) ---
        pose = np.zeros(33 * 4, dtype=np.float32)
        if result.pose_landmarks:
            for i, lm in enumerate(result.pose_landmarks):
                idx = i * 4
                pose[idx] = lm.x
                pose[idx + 1] = lm.y
                pose[idx + 2] = lm.z
                pose[idx + 3] = lm.visibility if hasattr(lm, 'visibility') else 0.0
            self._last_pose = pose.copy()
        else:
            pose = self._last_pose.copy()

        concat = np.concatenate([face, lh, rh, pose])

        # Normalise by subtracting the first landmark (translation invariance)
        return (concat - concat[0]).astype(np.float32)
