import numpy as np
from typing import List, Tuple, Optional
from collections import deque


class MotionDetector:
    """
    Detects when hand motion stops to determine sign boundaries.

    Strategy
    --------
    1. Raw motion is computed from raw keypoints (no smoothing — preserves
       the true timing of gestures).
    2. Motion magnitude is smoothed with an EMA to suppress jitter spikes.
    3. A hard **stillness_floor** guarantees that tiny jitter is always
       treated as stillness, breaking the noise-feedback loop.
    4. Hysteresis thresholds adapt to the median of recent motion.

    Flow for each frame:
      raw_motion = ||keypoints[t] - keypoints[t-1]||
      smoothed_motion = EMA(raw_motion)
      if smoothed_motion < stillness_floor → still_counter++
      elif motion < low_threshold       → still_counter++
      elif motion > high_threshold      → still_counter = 0
      else                              → hysteresis (hold state)

    When *still_frames_required* consecutive frames stay still, the
    current sign is considered complete.
    """

    def __init__(self,
                 low_factor: float = 0.5,
                 high_factor: float = 2.0,
                 still_frames_required: int = 8,
                 min_sign_duration: int = 5,
                 history_size: int = 30,
                 feature_dim: int = 1662,
                 motion_smoothing: float = 0.6,
                 stillness_floor: float = 0.3):
        """
        Parameters
        ----------
        low_factor : float
            Multiplier for the adaptive low threshold.
        high_factor : float
            Multiplier for the adaptive high threshold.
        still_frames_required : int
            Consecutive frames still to end a sign.
        min_sign_duration : int
            Minimum frames for a valid sign (noise filter).
        history_size : int
            Recent frames for the adaptive threshold median.
        feature_dim : int
            Expected dimensionality of the keypoint vector.
        motion_smoothing : float
            EMA factor for the motion magnitude (0=no smoothing, 0.9=heavy).
            Smoothes out jitter spikes while keeping motion timing intact.
        stillness_floor : float
            Absolute motion value below which we ALWAYS count as still,
            regardless of adaptive thresholds. Prevents jitter from
            blocking segmentation.
        """
        self.low_factor = low_factor
        self.high_factor = high_factor
        self.still_frames_required = still_frames_required
        self.min_sign_duration = min_sign_duration
        self.history_size = history_size
        self.feature_dim = feature_dim
        self.motion_smoothing = motion_smoothing
        self.stillness_floor = stillness_floor

        # Runtime state
        self.raw_motion_history = deque(maxlen=history_size)
        self.smoothed_motion = 0.0
        self.still_counter = 0
        self.sign_frames = 0
        self.previous_keypoints: Optional[np.ndarray] = None
        self.current_sign_keypoints: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self,
               motion_kp: np.ndarray,
               store_kp: Optional[np.ndarray] = None) -> Tuple[bool, Optional[List[np.ndarray]]]:
        """
        Process a single frame.

        Parameters
        ----------
        motion_kp : np.ndarray
            Used for motion calculation (persisted — last-known hand
            positions fill in for detection flicker).
        store_kp : np.ndarray or None
            Stored in the sign buffer (raw — zeros for missing hands,
            matching the ML model's training distribution).
            When ``None`` (default), *motion_kp* is used for both roles
            (backward-compatible with existing tests).

        Returns
        -------
        (sign_ended, completed_sign_keypoints)
            sign_ended : True if a sign was just completed.
            completed_sign_keypoints : list of per-frame keypoint vectors
                                       for the just-completed sign.
        """
        if store_kp is None:
            store_kp = motion_kp

        # --- input validation ---
        self._validate_keypoints(motion_kp)

        # --- first frame ---
        if self.previous_keypoints is None:
            self.previous_keypoints = motion_kp.copy()
            self.current_sign_keypoints.append(store_kp.copy())
            self.sign_frames = 1
            self.smoothed_motion = 0.0
            return False, None

        # --- raw motion (from persisted motion_kp — no zero-to-real spikes) ---
        raw_motion = float(np.linalg.norm(motion_kp - self.previous_keypoints))
        self.previous_keypoints = motion_kp.copy()

        # --- smooth the motion magnitude (not the keypoints) ---
        a = self.motion_smoothing
        if self.sign_frames > 1:
            self.smoothed_motion = a * self.smoothed_motion + (1.0 - a) * raw_motion
        else:
            self.smoothed_motion = raw_motion

        self.raw_motion_history.append(raw_motion)

        # --- adaptive thresholds ---
        if len(self.raw_motion_history) >= 10:
            adaptive_base = float(np.median(list(self.raw_motion_history)))
            low_th = adaptive_base * self.low_factor
            high_th = adaptive_base * self.high_factor
        else:
            low_th = 0.1
            high_th = 0.4

        motion = self.smoothed_motion

        # --- hysteresis state machine ---
        if motion < self.stillness_floor:
            self.still_counter += 1
        elif motion < low_th:
            self.still_counter += 1
        elif motion > high_th:
            if len(self.raw_motion_history) >= 10:
                self.still_counter = 0

        # --- accumulate raw keypoints (matching training data) ---
        self.current_sign_keypoints.append(store_kp.copy())
        self.sign_frames += 1

        # --- sign boundary check ---
        if (self.still_counter >= self.still_frames_required
                and self.sign_frames >= self.min_sign_duration):
            completed = [kp.copy() for kp in self.current_sign_keypoints]
            self.still_counter = 0
            self.sign_frames = 0
            self.current_sign_keypoints.clear()
            return True, completed

        return False, None

    def reset(self):
        """Return the detector to its initial state."""
        self.raw_motion_history.clear()
        self.smoothed_motion = 0.0
        self.still_counter = 0
        self.sign_frames = 0
        self.previous_keypoints = None
        self.current_sign_keypoints.clear()

    def get_current_sign_length(self) -> int:
        return len(self.current_sign_keypoints)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_keypoints(self, kp: np.ndarray):
        if kp.ndim != 1:
            raise ValueError(
                f"Expected 1-D keypoint array, got {kp.ndim}D "
                f"(shape {kp.shape})"
            )
        if kp.shape[0] != self.feature_dim:
            raise ValueError(
                f"Keypoint dimension mismatch: got {kp.shape[0]}, "
                f"expected {self.feature_dim}"
            )
