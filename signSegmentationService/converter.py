import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2 as cv
import numpy as np
import os

# Defaults – can be overridden via env vars
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "35"))
FEATURE_DIM = int(os.getenv("FEATURE_DIM", "258"))

_LH_DIM = 21 * 3      # 63
_RH_DIM = 21 * 3      # 63
_POSE_DIM = 33 * 4    # 132
assert _LH_DIM + _RH_DIM + _POSE_DIM == FEATURE_DIM, (
    f"FEATURE_DIM {FEATURE_DIM} != {_LH_DIM} (lh) + {_RH_DIM} (rh) + {_POSE_DIM} (pose)"
)

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

    Face landmarks are intentionally excluded (the current ML model only
    uses hands + pose).  All coordinates are normalised relative to the
    **pose nose** (landmark 0) for translation invariance.

    Usage
    -----
        conv = Converter()
        kp = conv.point_detection(raw_image_bytes)       # 258-dim vector
        window = conv.process_new_frame(kp)               # (WINDOW_SIZE, FEATURE_DIM)
        final = conv.stop()                                # current window snapshot
    """

    def __init__(self):
        model_path = _ensure_model()
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HolisticLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_hand_landmarks_confidence=0.7,
        )
        self.landmarker = vision.HolisticLandmarker.create_from_options(options)

        # Fixed-size sliding window buffer
        self.window = np.zeros((WINDOW_SIZE, FEATURE_DIM), dtype=np.float32)
        self.current_length = 0

        self._last_result = None

        # Corrected hand references (set by _fix_handedness → _extract_keypoints)
        self._corrected_lh = None
        self._corrected_rh = None

        # Drawing persistence cache — stores normalized (x, y) coordinates
        # so draw_landmarks() always has something to render.
        self._draw_lh: list[tuple[float, float]] = []
        self._draw_rh: list[tuple[float, float]] = []

        # Landmark persistence cache — last-known hand positions prevent
        # zero-to-real motion spikes during detection flicker.
        self._last_lh = np.zeros(_LH_DIM, dtype=np.float32)
        self._last_rh = np.zeros(_RH_DIM, dtype=np.float32)
        self._last_pose = np.zeros(_POSE_DIM, dtype=np.float32)

        # Persisted keypoint vector (built by point_detection for the
        # motion detector — uses last-known hand positions to avoid spikes).
        self._persisted_kp = np.zeros(FEATURE_DIM, dtype=np.float32)

        # Unnormalised raw components from the most recent frame; used by
        # _build_persisted_kp() to reconstruct a motion-stable vector.
        self._raw_components: tuple[np.ndarray, ...] | None = None

    def __del__(self):
        if hasattr(self, 'landmarker'):
            self.landmarker.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def point_detection(self, image_bytes: bytes) -> np.ndarray:
        """
        Decode raw image bytes and run MediaPipe Holistic.

        Returns **raw** keypoints (zeros for undetected hand groups) — these
        match the training data distribution of the ML model.

        Use :meth:`get_persisted_keypoints` to obtain a motion-stable version
        for the motion detector (last-known positions fill in for flicker).
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        cv_image = cv.imdecode(nparr, cv.IMREAD_COLOR)
        if cv_image is None:
            raise ValueError("Could not decode image bytes (corrupt data).")

        image_rgb = cv.cvtColor(cv_image, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        detection_result = self.landmarker.detect(mp_image)
        self._last_result = detection_result
        raw_kp = self._extract_keypoints(detection_result)

        self._persisted_kp = self._build_persisted_kp()

        if raw_kp.shape != (FEATURE_DIM,):
            raise ValueError(
                f"MediaPipe returned {raw_kp.shape[0]}-dim keypoints, "
                f"expected {FEATURE_DIM}. Check FEATURE_DIM env var."
            )

        return raw_kp

    def get_persisted_keypoints(self) -> np.ndarray:
        """Return the motion-stable keypoint vector for the current frame.

        Uses last-known hand positions when MediaPipe briefly loses detection,
        so the motion detector sees smooth transitions instead of zero-to-real
        spikes.  **Not** suitable for the ML model — the model was trained on
        zeros for missing hands.
        """
        return self._persisted_kp

    def process_new_frame(self, frame_vector: np.ndarray) -> np.ndarray:
        """Insert a new frame into the sliding window buffer.

        Returns the current window state: (WINDOW_SIZE, FEATURE_DIM).
        Before the buffer is full, the right side stays zero-padded.
        """
        if self.current_length < WINDOW_SIZE:
            self.window[self.current_length] = frame_vector
            self.current_length += 1
        else:
            self.window[:-1] = self.window[1:]
            self.window[-1] = frame_vector
        return self.window

    def stop(self) -> np.ndarray:
        """Return the current window snapshot (right-padded with zeros)."""
        return self.window

    def draw_landmarks(self, frame: np.ndarray) -> None:
        """Draw hand landmarks onto *frame* in-place.

        Uses cached coordinates when MediaPipe briefly loses a hand,
        so the overlay stays stable instead of flickering.
        """
        if not hasattr(self, '_corrected_lh'):
            return
        h, w = frame.shape[:2]

        def _to_px(nx: float, ny: float) -> tuple[int, int]:
            return int(nx * w), int(ny * h)

        hand_conn = [(i, i + 1) for i in range(20)]

        # Left hand
        if self._corrected_lh:
            self._draw_lh = [(lm.x, lm.y) for lm in self._corrected_lh]
        if self._draw_lh:
            pts = [_to_px(nx, ny) for nx, ny in self._draw_lh]
            for a, b in hand_conn:
                cv.line(frame, pts[a], pts[b], (255, 0, 100), 2)
            for pt in pts:
                cv.circle(frame, pt, 4, (255, 50, 150), -1)

        # Right hand
        if self._corrected_rh:
            self._draw_rh = [(lm.x, lm.y) for lm in self._corrected_rh]
        if self._draw_rh:
            pts = [_to_px(nx, ny) for nx, ny in self._draw_rh]
            for a, b in hand_conn:
                cv.line(frame, pts[a], pts[b], (100, 0, 255), 2)
            for pt in pts:
                cv.circle(frame, pt, 4, (150, 50, 255), -1)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fix_handedness(self, result):
        """Correct MediaPipe left/right hand label swaps when only one hand is visible.

        Uses the **pose nose** X coordinate (not face) as the midline,
        since face landmarks are no longer extracted.
        """
        left = result.left_hand_landmarks
        right = result.right_hand_landmarks

        if result.pose_landmarks and (left is None) != (right is None):
            nose_x = result.pose_landmarks[0].x
            if left is not None:
                if left[0].x > nose_x:    # labeled "left" but on right side
                    right = left
                    left = None
            elif right is not None:
                if right[0].x < nose_x:   # labeled "right" but on left side
                    left = right
                    right = None

        return left, right

    def _nose_anchor(self, pose: np.ndarray) -> np.ndarray:
        """Return the (x, y, z) of the nose from an unnormalised pose vector.

        Pose landmark 0 = nose.  Returns zeros if pose was not detected.
        """
        return pose[0:3]  # (x, y, z)

    def _normalize(self, lh: np.ndarray, rh: np.ndarray,
                   pose: np.ndarray, nose_xyz: np.ndarray) -> np.ndarray:
        """Concatenate lh, rh, pose and normalise relative to *nose_xyz*."""
        # Hands: each landmark (x, y, z) shifted by -nose_xyz
        if _LH_DIM:
            lh = (lh.reshape(-1, 3) - nose_xyz).flatten()
        if _RH_DIM:
            rh = (rh.reshape(-1, 3) - nose_xyz).flatten()
        # Pose: subtract nose_xyz from (x, y, z) of every landmark;
        # visibility is left untouched.
        if _POSE_DIM:
            p = pose.copy().reshape(-1, 4)
            p[:, :3] -= nose_xyz
            pose = p.flatten()
        return np.concatenate([lh, rh, pose]).astype(np.float32)

    def _extract_keypoints(self, result) -> np.ndarray:
        """Extract **raw** keypoints — zeros for undetected landmark groups.

        Normalised relative to the pose nose.  The persistence cache is
        updated here but NOT used in the returned vector.
        """
        lh_data, rh_data = self._fix_handedness(result)

        # Store corrected references for draw_landmarks
        self._corrected_lh = lh_data
        self._corrected_rh = rh_data

        # --- Left hand (21 × 3 = 63) ---
        lh = np.zeros(_LH_DIM, dtype=np.float32)
        if lh_data:
            for i, lm in enumerate(lh_data):
                idx = i * 3
                lh[idx] = lm.x
                lh[idx + 1] = lm.y
                lh[idx + 2] = lm.z
            self._last_lh = lh.copy()

        # --- Right hand (21 × 3 = 63) ---
        rh = np.zeros(_RH_DIM, dtype=np.float32)
        if rh_data:
            for i, lm in enumerate(rh_data):
                idx = i * 3
                rh[idx] = lm.x
                rh[idx + 1] = lm.y
                rh[idx + 2] = lm.z
            self._last_rh = rh.copy()

        # --- Pose (33 × 4 = 132; includes visibility) ---
        pose = np.zeros(_POSE_DIM, dtype=np.float32)
        if result.pose_landmarks:
            for i, lm in enumerate(result.pose_landmarks):
                idx = i * 4
                pose[idx] = lm.x
                pose[idx + 1] = lm.y
                pose[idx + 2] = lm.z
                pose[idx + 3] = lm.visibility if hasattr(lm, 'visibility') else 0.0
            self._last_pose = pose.copy()

        # Store unnormalised copies for _build_persisted_kp
        self._raw_components = (lh.copy(), rh.copy(), pose.copy())

        # Normalise all components relative to the nose
        nose_xyz = self._nose_anchor(pose)
        return self._normalize(lh, rh, pose, nose_xyz)

    def _build_persisted_kp(self) -> np.ndarray:
        """Build a motion-stable keypoint vector using last-known hand positions.

        Uses the current frame's pose but substitutes **last-known** hand
        keypoints when MediaPipe briefly loses a hand.  Normalised relative
        to the nose from the current raw pose.
        """
        if self._raw_components is None:
            return np.zeros(FEATURE_DIM, dtype=np.float32)

        _, _, raw_pose = self._raw_components  # discard raw lh/rh
        nose_xyz = self._nose_anchor(raw_pose)

        # Use cached hand keypoints (last-known positions)
        lh = self._last_lh.copy()
        rh = self._last_rh.copy()
        pose = raw_pose.copy()

        return self._normalize(lh, rh, pose, nose_xyz)
