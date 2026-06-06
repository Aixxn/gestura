import mediapipe as mp
import cv2 as cv
import numpy as np
import os

# Defaults – can be overridden via env vars
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "35"))
FEATURE_DIM = int(os.getenv("FEATURE_DIM", "1662"))


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
        self.mp_model = mp.solutions.holistic.Holistic()

        # Fixed-size sliding window buffer
        self.window = np.zeros((WINDOW_SIZE, FEATURE_DIM), dtype=np.float32)
        self.current_length = 0  # how many real frames have been inserted

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

        image = cv.cvtColor(cv_image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = self.mp_model.process(image)
        keypoints = self._extract_keypoints(results)

        image.flags.writeable = True
        self._debug_image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

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
    def _extract_keypoints(results):
        """Flatten MediaPipe Holistic landmarks into a single vector."""
        lh = np.array([[res.x, res.y, res.z]
                       for res in results.left_hand_landmarks.landmark]).flatten() \
             if results.left_hand_landmarks else np.zeros(21 * 3)

        rh = np.array([[res.x, res.y, res.z]
                       for res in results.right_hand_landmarks.landmark]).flatten() \
             if results.right_hand_landmarks else np.zeros(21 * 3)

        face = np.array([[res.x, res.y, res.z]
                         for res in results.face_landmarks.landmark]).flatten() \
               if results.face_landmarks else np.zeros(468 * 3)

        pose = np.array([[res.x, res.y, res.z, res.visibility]
                         for res in results.pose_landmarks.landmark]).flatten() \
                if results.pose_landmarks else np.zeros(33 * 4)

        concat = np.concatenate([face, lh, rh, pose])

        # Normalise by subtracting the first landmark (translation invariance)
        return (concat - concat[0]).astype(np.float32)
