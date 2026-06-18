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
PERSIST_WINDOW = int(os.getenv("PERSIST_WINDOW", "5"))
# Number of consecutive frames both hands must be absent before we consider
# the user idle.  Must be **larger** than the motion detector's
# ``still_frames_required`` (default 8) so a natural sign-ending stillness
# triggers the boundary *before* the idle gate resets the pipeline.
IDLE_THRESHOLD = int(os.getenv("IDLE_THRESHOLD", "15"))

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


class ExponentialMovingAverage:
    """Smooths out webcam micro-jitter to prevent static signs from looking like motion.

    Applied AFTER bounded persistence and nose-normalisation, the EMA blends
    each new keypoint frame with previous ones, killing high-frequency noise
    while preserving the overall gesture trajectory.

    ``alpha=1.0`` = no smoothing (pass-through).
    ``alpha=0.4`` = moderate (default).
    ``alpha=0.1`` = heavy smoothing.
    """

    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha
        self.state: np.ndarray | None = None

    def reset(self):
        self.state = None

    def update(self, current_val: np.ndarray) -> np.ndarray:
        if self.state is None:
            self.state = current_val.copy()
            return current_val.copy()
        self.state = self.alpha * current_val + (1.0 - self.alpha) * self.state
        return self.state


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
    Converts raw image bytes → MediaPipe Holistic keypoints.

    Face landmarks are intentionally excluded (the current ML model only
    uses hands + pose).  All coordinates are normalised relative to the
    **pose nose** (landmark 0) for translation invariance.

    Hand keypoints use **bounded persistence**: last-known positions fill in
    during brief detection flicker (up to *PERSIST_WINDOW* frames), then
    decay to zeros for genuinely absent hands.  A single unified keypoint
    vector is returned for both the motion detector and the ML model.
    """

    def __init__(self, use_gpu: bool = False):
        model_path = _ensure_model()
        delegate = python.BaseOptions.Delegate.GPU if use_gpu else python.BaseOptions.Delegate.CPU
        base_options = python.BaseOptions(
            model_asset_path=model_path,
            delegate=delegate,
        )
        options = vision.HolisticLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_hand_landmarks_confidence=0.7,
        )
        self.landmarker = vision.HolisticLandmarker.create_from_options(options)

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

        # Lost-frame counters for bounded persistence.
        # Incremented each frame a hand is undetected; reset on detection.
        # Start at IDLE_THRESHOLD so is_idle is True at init — prevents
        # the motion detector from firing false boundaries on the first
        # ~15 frames before the user has started signing.
        self._lh_lost_counter = IDLE_THRESHOLD
        self._rh_lost_counter = IDLE_THRESHOLD

        # Unnormalised raw components from the most recent frame; used by
        # _build_unified_kp() to reconstruct a bounded-persistence vector.
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

        Returns a **unified** keypoint vector with bounded persistence:
        last-known hand positions fill in during brief detection flicker
        (up to *PERSIST_WINDOW* frames), then decay to zeros for genuinely
        absent hands.  Suitable for **both** the motion detector and the
        ML inference pipeline.
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        cv_image = cv.imdecode(nparr, cv.IMREAD_COLOR)
        if cv_image is None:
            raise ValueError("Could not decode image bytes (corrupt data).")

        image_rgb = cv.cvtColor(cv_image, cv.COLOR_BGR2RGB)
        # Ensure contiguous memory layout — mp.Image can choke on
        # non-contiguous views returned by cv.cvtColor.
        image_rgb = np.ascontiguousarray(image_rgb)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        detection_result = self.landmarker.detect(mp_image)
        self._last_result = detection_result
        kp = self._extract_keypoints(detection_result)

        if kp.shape != (FEATURE_DIM,):
            raise ValueError(
                f"MediaPipe returned {kp.shape[0]}-dim keypoints, "
                f"expected {FEATURE_DIM}. Check FEATURE_DIM env var."
            )

        return kp

    def extract_from_frame(self, cv_image: np.ndarray) -> np.ndarray:
        """Process a cv2 BGR frame directly (skips JPEG encode/decode).

        Same result as :meth:`point_detection` but avoids the encode-decode
        round trip.  Useful for offline data extraction from video files.
        """
        image_rgb = cv.cvtColor(cv_image, cv.COLOR_BGR2RGB)
        # Ensure contiguous memory layout — mp.Image can choke on
        # non-contiguous views returned by cv.cvtColor.
        image_rgb = np.ascontiguousarray(image_rgb)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        detection_result = self.landmarker.detect(mp_image)
        self._last_result = detection_result
        kp = self._extract_keypoints(detection_result)

        if kp.shape != (FEATURE_DIM,):
            raise ValueError(
                f"MediaPipe returned {kp.shape[0]}-dim keypoints, "
                f"expected {FEATURE_DIM}. Check FEATURE_DIM env var."
            )
        return kp

    def reset_state(self):
        """Reset per-video state (counters, caches, buffer).

        Call between videos during batch extraction so bounded-persistence
        counters don't carry over from one video to the next.
        """
        self._lh_lost_counter = IDLE_THRESHOLD
        self._rh_lost_counter = IDLE_THRESHOLD
        self._last_lh = np.zeros(_LH_DIM, dtype=np.float32)
        self._last_rh = np.zeros(_RH_DIM, dtype=np.float32)
        self._last_pose = np.zeros(_POSE_DIM, dtype=np.float32)
        self._raw_components = None

    @property
    def is_idle(self) -> bool:
        """True when **both** hands have been absent for >= IDLE_THRESHOLD frames.

        During idle the motion detector receives only persisted or zero
        keypoints and would otherwise trigger false sign boundaries.
        One-handed ASL signs are fine — the signing hand keeps its own
        counter reset even when the other hand is not visible.

        Note
        ----
        IDLE_THRESHOLD must be **larger** than the motion detector's
        ``still_frames_required`` (default 8) so that a natural end-of-sign
        stillness triggers the boundary before the idle gate preempts it.
        """
        return (self._lh_lost_counter >= IDLE_THRESHOLD
                and self._rh_lost_counter >= IDLE_THRESHOLD)

    def get_raw_keypoints(self) -> np.ndarray:
        """Return the raw (non-persisted) keypoint from the current frame.

        Hands that were not detected in this frame are set to **zeros**,
        matching the ML model's training distribution.  Use this as the
        ``store_kp`` argument to :meth:`~motion_detector.MotionDetector.update`
        so the sign buffer never contains persisted (last-known) positions.
        """
        if self._raw_components is None:
            return np.zeros(FEATURE_DIM, dtype=np.float32)
        raw_lh, raw_rh, raw_pose = self._raw_components
        nose_xyz = self._nose_anchor(raw_pose)
        return self._normalize(raw_lh, raw_rh, raw_pose, nose_xyz)

    def get_persisted_keypoints(self) -> np.ndarray:
        """Return the current unified keypoint vector (bounded persistence).

        This is now identical to what :meth:`point_detection` returns.
        Retained for backward compatibility.
        """
        return self._build_unified_kp()

    def draw_landmarks(self, frame: np.ndarray) -> None:
        """Draw hand landmarks onto *frame* in-place.

        Uses cached coordinates when MediaPipe briefly loses a hand
        (up to PERSIST_WINDOW frames), then clears the cache so the
        overlay disappears when both hands are genuinely absent.
        """
        if not hasattr(self, '_corrected_lh'):
            return
        h, w = frame.shape[:2]

        def _to_px(nx: float, ny: float) -> tuple[int, int]:
            return int(nx * w), int(ny * h)

        hand_conn = [(i, i + 1) for i in range(20)]

        # Left hand — update cache when detected; draw only if still alive
        if self._corrected_lh:
            self._draw_lh = [(lm.x, lm.y) for lm in self._corrected_lh]
        elif self._lh_lost_counter >= PERSIST_WINDOW:
            self._draw_lh.clear()
        if self._draw_lh:
            pts = [_to_px(nx, ny) for nx, ny in self._draw_lh]
            for a, b in hand_conn:
                cv.line(frame, pts[a], pts[b], (255, 0, 100), 2)
            for pt in pts:
                cv.circle(frame, pt, 4, (255, 50, 150), -1)

        # Right hand
        if self._corrected_rh:
            self._draw_rh = [(lm.x, lm.y) for lm in self._corrected_rh]
        elif self._rh_lost_counter >= PERSIST_WINDOW:
            self._draw_rh.clear()
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
        """Extract keypoints with bounded persistence.

        1. Extracts raw landmark data from the MediaPipe result.
        2. Updates the persistence cache (_last_lh / _last_rh / _last_pose).
        3. Tracks lost-frame counters for each hand.
        4. Builds and returns a unified vector with bounded persistence.
        """
        lh_data, rh_data = self._fix_handedness(result)

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
            self._lh_lost_counter = 0
        else:
            self._lh_lost_counter += 1

        # --- Right hand (21 × 3 = 63) ---
        rh = np.zeros(_RH_DIM, dtype=np.float32)
        if rh_data:
            for i, lm in enumerate(rh_data):
                idx = i * 3
                rh[idx] = lm.x
                rh[idx + 1] = lm.y
                rh[idx + 2] = lm.z
            self._last_rh = rh.copy()
            self._rh_lost_counter = 0
        else:
            self._rh_lost_counter += 1

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

        self._raw_components = (lh.copy(), rh.copy(), pose.copy())

        return self._build_unified_kp()

    def _build_unified_kp(self) -> np.ndarray:
        """Build a unified keypoint vector with bounded persistence.

        Uses the current frame's raw pose.  For each hand:
        - **Detected** or lost for < PERSIST_WINDOW frames → last-known position.
        - Lost for ≥ PERSIST_WINDOW frames → zeros (genuinely absent).

        Normalised relative to the pose nose.
        """
        if self._raw_components is None:
            return np.zeros(FEATURE_DIM, dtype=np.float32)

        _, _, raw_pose = self._raw_components
        nose_xyz = self._nose_anchor(raw_pose)

        if self._lh_lost_counter >= PERSIST_WINDOW:
            lh = np.zeros(_LH_DIM, dtype=np.float32)
        else:
            lh = self._last_lh.copy()

        if self._rh_lost_counter >= PERSIST_WINDOW:
            rh = np.zeros(_RH_DIM, dtype=np.float32)
        else:
            rh = self._last_rh.copy()

        pose = raw_pose.copy()
        return self._normalize(lh, rh, pose, nose_xyz)
