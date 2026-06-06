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
        )
        self.landmarker = vision.HolisticLandmarker.create_from_options(options)

        # Fixed-size sliding window buffer
        self.window = np.zeros((WINDOW_SIZE, FEATURE_DIM), dtype=np.float32)
        self.current_length = 0  # how many real frames have been inserted

        self._debug_image = None

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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_keypoints(result) -> np.ndarray:
        """Flatten MediaPipe Holistic landmarks into a single vector."""

        # The new holistic landmarker model returns 478 face landmarks,
        # but the existing pipeline (ML model, translation service) expects
        # 468 (= 1662 total). Truncate to maintain compatibility.
        MAX_FACE = 468
        FACE_DIM = MAX_FACE * 3  # 1404

        face = np.zeros(FACE_DIM, dtype=np.float32)
        if result.face_landmarks:
            for i, lm in enumerate(result.face_landmarks[:MAX_FACE]):
                idx = i * 3
                face[idx] = lm.x
                face[idx + 1] = lm.y
                face[idx + 2] = lm.z

        # Left hand landmarks (21 × 3 = 63)
        lh = np.zeros(21 * 3, dtype=np.float32)
        if result.left_hand_landmarks:
            for i, lm in enumerate(result.left_hand_landmarks):
                idx = i * 3
                lh[idx] = lm.x
                lh[idx + 1] = lm.y
                lh[idx + 2] = lm.z

        # Right hand landmarks (21 × 3 = 63)
        rh = np.zeros(21 * 3, dtype=np.float32)
        if result.right_hand_landmarks:
            for i, lm in enumerate(result.right_hand_landmarks):
                idx = i * 3
                rh[idx] = lm.x
                rh[idx + 1] = lm.y
                rh[idx + 2] = lm.z

        # Pose landmarks (33 × 4 = 132; includes visibility)
        pose = np.zeros(33 * 4, dtype=np.float32)
        if result.pose_landmarks:
            for i, lm in enumerate(result.pose_landmarks):
                idx = i * 4
                pose[idx] = lm.x
                pose[idx + 1] = lm.y
                pose[idx + 2] = lm.z
                pose[idx + 3] = lm.visibility if hasattr(lm, 'visibility') else 0.0

        concat = np.concatenate([face, lh, rh, pose])

        # Normalise by subtracting the first landmark (translation invariance)
        return (concat - concat[0]).astype(np.float32)
