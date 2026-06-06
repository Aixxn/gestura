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

        # Corrected hand references (set by _extract_keypoints after
        # handedness fix). Initialised here for draw_landmarks safety.
        self._corrected_face = None
        self._corrected_lh = None
        self._corrected_rh = None

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

        # Persisted keypoint vector (built by point_detection for the
        # motion detector — uses last-known hand positions to avoid spikes).
        self._persisted_kp = np.zeros(FEATURE_DIM, dtype=np.float32)

        # Unnormalized raw components from the most recent frame; used by
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

        # Build persisted version for motion detector (swap in last-known
        # hand positions so detection flicker doesn't spike the motion signal).
        self._persisted_kp = self._build_persisted_kp()

        self._debug_image = cv.cvtColor(image_rgb, cv.COLOR_RGB2BGR)

        if raw_kp.shape != (FEATURE_DIM,):
            raise ValueError(
                f"MediaPipe returned {raw_kp.shape[0]}-dim keypoints, "
                f"expected {FEATURE_DIM}. Check FEATURE_DIM env var."
            )

        return raw_kp

    def get_persisted_keypoints(self) -> np.ndarray:
        """Return the motion-stable keypoint vector for the current frame.

        This uses last-known hand positions when MediaPipe briefly loses
        detection, so the motion detector sees smooth transitions instead
        of zero-to-real spikes.  **Not** suitable for the ML model — the
        model was trained on zeros for missing hands.
        """
        return self._persisted_kp

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
        if not hasattr(self, '_corrected_face') or self._corrected_face is None:
            return
        h, w = frame.shape[:2]

        def _to_px(nx: float, ny: float) -> tuple[int, int]:
            return int(nx * w), int(ny * h)

        # Uses corrected references from _fix_handedness (runs in
        # _extract_keypoints) so hand labels are not swapped.

        # --- Face landmarks (drawn as small dots) ---
        if self._corrected_face:
            self._draw_face = [(lm.x, lm.y) for lm in self._corrected_face]
        for nx, ny in self._draw_face:
            cv.circle(frame, _to_px(nx, ny), 1, (200, 200, 100), -1)

        # (Pose landmarks intentionally omitted — reduces visual clutter)

        hand_conn = [(i, i + 1) for i in range(20)]

        # --- Left hand landmarks ---
        if self._corrected_lh:
            self._draw_lh = [(lm.x, lm.y) for lm in self._corrected_lh]
        if self._draw_lh:
            pts = [_to_px(nx, ny) for nx, ny in self._draw_lh]
            for a, b in hand_conn:
                cv.line(frame, pts[a], pts[b], (255, 0, 100), 2)
            for pt in pts:
                cv.circle(frame, pt, 4, (255, 50, 150), -1)

        # --- Right hand landmarks ---
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

        MediaPipe Holistic sometimes mislabels a hand when only one is in frame
        (e.g. the right hand is detected but labeled as "left"). This method
        checks wrist position relative to the face center and swaps if needed.

        Returns
        -------
        (left_hand_data, right_hand_data)
            Corrected landmark references (or None for the missing hand).
        """
        left = result.left_hand_landmarks
        right = result.right_hand_landmarks

        if result.face_landmarks and (left is None) != (right is None):
            face_cx = float(np.mean([lm.x for lm in result.face_landmarks]))
            if left is not None:
                if left[0].x > face_cx:   # labeled "left" but on right side
                    right = left
                    left = None
            elif right is not None:
                if right[0].x < face_cx:  # labeled "right" but on left side
                    left = right
                    right = None

        return left, right

    def _extract_keypoints(self, result) -> np.ndarray:
        """Extract **raw** keypoints — zeros for undetected landmark groups.

        This matches the ML model's training distribution (missing hands = 0).
        The landmark persistence cache is still *updated* here, but the
        returned vector does NOT use it — callers that need motion-stable
        keypoints should use :meth:`get_persisted_keypoints` instead.
        """
        MAX_FACE = 468
        FACE_DIM = MAX_FACE * 3  # 1404

        # Correct handedness before extraction
        lh_data, rh_data = self._fix_handedness(result)

        # Store corrected references for draw_landmarks
        self._corrected_face = result.face_landmarks
        self._corrected_lh = lh_data
        self._corrected_rh = rh_data

        # --- Face landmarks ---
        face = np.zeros(FACE_DIM, dtype=np.float32)
        if result.face_landmarks:
            for i, lm in enumerate(result.face_landmarks[:MAX_FACE]):
                idx = i * 3
                face[idx] = lm.x
                face[idx + 1] = lm.y
                face[idx + 2] = lm.z
            self._last_face = face.copy()
        # else: stays zeros (raw — matches training data)

        # --- Left hand landmarks (21 × 3 = 63) ---
        lh = np.zeros(21 * 3, dtype=np.float32)
        if lh_data:
            for i, lm in enumerate(lh_data):
                idx = i * 3
                lh[idx] = lm.x
                lh[idx + 1] = lm.y
                lh[idx + 2] = lm.z
            self._last_lh = lh.copy()
        # else: stays zeros (raw — matches training data)

        # --- Right hand landmarks (21 × 3 = 63) ---
        rh = np.zeros(21 * 3, dtype=np.float32)
        if rh_data:
            for i, lm in enumerate(rh_data):
                idx = i * 3
                rh[idx] = lm.x
                rh[idx + 1] = lm.y
                rh[idx + 2] = lm.z
            self._last_rh = rh.copy()
        # else: stays zeros (raw — matches training data)

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
        # else: stays zeros

        # Store unnormalised components so _build_persisted_kp can
        # reconstruct a motion-stable version.
        self._raw_components = (face, lh, rh, pose)

        concat = np.concatenate([face, lh, rh, pose])

        # Normalise by subtracting the first landmark (translation invariance)
        return (concat - concat[0]).astype(np.float32)

    def _build_persisted_kp(self) -> np.ndarray:
        """Build a motion-stable keypoint vector using last-known hand positions.

        The returned vector uses the current frame's face & pose (nearly always
        detected) but substitutes **last-known** hand keypoints when MediaPipe
        briefly loses a hand.  This is fed to the motion detector so that brief
        detection flicker doesn't create zero-to-real motion spikes.
        """
        if self._raw_components is None:
            return np.zeros(FEATURE_DIM, dtype=np.float32)

        face, _, _, pose = self._raw_components
        # Swap in cached hand keypoints (last-known positions)
        concat = np.concatenate([face, self._last_lh, self._last_rh, pose])
        return (concat - concat[0]).astype(np.float32)
