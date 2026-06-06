import numpy as np
from typing import List, Tuple, Optional
from collections import deque


class MotionDetector:
    """
    Detects when hand motion stops to determine sign boundaries.

    Uses hysteresis with adaptive thresholding for robustness:
      - motion < low_threshold   → still_counter increments
      - motion > high_threshold  → still_counter resets
      - low ≤ motion ≤ high      → hysteresis band (holds previous state)

    When *still_frames_required* consecutive frames stay below the low
    threshold, the current sign is considered complete.
    """

    def __init__(self,
                 low_factor: float = 0.5,
                 high_factor: float = 2.0,
                 still_frames_required: int = 15,
                 min_sign_duration: int = 5,
                 history_size: int = 30,
                 feature_dim: int = 1662,
                 smoothing_alpha: float = 0.0):
        """
        Parameters
        ----------
        low_factor : float
            Multiplier for the adaptive low threshold.
        high_factor : float
            Multiplier for the adaptive high threshold.
        still_frames_required : int
            Consecutive frames below low threshold to end a sign.
        min_sign_duration : int
            Minimum frames required for a valid sign (filters noise bursts).
        history_size : int
            Number of recent frames to use for the adaptive threshold median.
        feature_dim : int
            Expected dimensionality of the keypoint vector.
        smoothing_alpha : float
            EMA smoothing factor for keypoints (0=no smoothing, 1=fully smooth).
            Helps suppress MediaPipe jitter. 0.4 is a good default.
        """
        self.low_factor = low_factor
        self.high_factor = high_factor
        self.still_frames_required = still_frames_required
        self.min_sign_duration = min_sign_duration
        self.history_size = history_size
        self.feature_dim = feature_dim
        self.smoothing_alpha = smoothing_alpha

        # Runtime state
        self.motion_history = deque(maxlen=history_size)
        self.still_counter = 0
        self.sign_frames = 0
        self.previous_keypoints: Optional[np.ndarray] = None
        self.current_sign_keypoints: List[np.ndarray] = []
        self.smoothed_keypoints: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, keypoints: np.ndarray) -> Tuple[bool, Optional[List[np.ndarray]]]:
        """
        Process a single frame's keypoints.

        Returns
        -------
        (sign_ended, completed_sign_keypoints)
            sign_ended : True if a sign was just completed.
            completed_sign_keypoints : list of per-frame keypoint vectors
                                       for the just-completed sign, or None.
        """
        # --- input validation ---
        self._validate_keypoints(keypoints)

        # --- optionally apply EMA smoothing to suppress jitter ---
        if self.smoothing_alpha > 0 and self.smoothed_keypoints is not None:
            # smoothing_alpha=0.4 → 40% old + 60% new (moderate)
            a = self.smoothing_alpha
            self.smoothed_keypoints = (
                a * self.smoothed_keypoints + (1.0 - a) * keypoints
            )
        else:
            # No smoothing, or first frame — use raw keypoints
            self.smoothed_keypoints = keypoints.copy()

        ready_kp = self.smoothed_keypoints

        # --- first frame ---
        if self.previous_keypoints is None:
            self.previous_keypoints = ready_kp.copy()
            self.current_sign_keypoints.append(keypoints.copy())  # raw
            self.sign_frames = 1
            return False, None

        # --- motion computation (on smoothed keypoints) ---
        motion = float(np.linalg.norm(ready_kp - self.previous_keypoints))
        self.motion_history.append(motion)
        self.previous_keypoints = ready_kp.copy()

        # --- adaptive thresholds ---
        if len(self.motion_history) >= 10:
            adaptive_base = float(np.median(list(self.motion_history)))
            low_th = adaptive_base * self.low_factor
            high_th = adaptive_base * self.high_factor
        else:
            low_th = 0.02
            high_th = 0.08

        # --- hysteresis state machine ---
        if motion < low_th:
            self.still_counter += 1
        elif motion > high_th:
            self.still_counter = 0
        # hysteresis band: hold previous still_counter

        # --- accumulate ---
        self.current_sign_keypoints.append(keypoints.copy())
        self.sign_frames += 1

        # --- sign boundary check ---
        if (self.still_counter >= self.still_frames_required
                and self.sign_frames >= self.min_sign_duration):
            completed = [kp.copy() for kp in self.current_sign_keypoints]
            # reset for next sign
            self.still_counter = 0
            self.sign_frames = 0
            self.current_sign_keypoints.clear()
            return True, completed

        return False, None

    def reset(self):
        """Return the detector to its initial state."""
        self.motion_history.clear()
        self.still_counter = 0
        self.sign_frames = 0
        self.previous_keypoints = None
        self.current_sign_keypoints.clear()
        self.smoothed_keypoints = None

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
