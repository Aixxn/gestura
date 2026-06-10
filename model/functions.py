import mediapipe as mp
import numpy as np
import os
import cv2 as cv

class Gestura:
    mp_holistic = mp.solutions.holistic
    mp_draw = mp.solutions.drawing_utils

    DATA_PATH = os.path.join(os.path.dirname(__file__), "keypoint_data_selected")

    # FIX: Read sign classes from DATA_PATH (processed keypoints),
    # not from the raw video root. Falls back to empty array if not yet generated.
    try:
        sign = np.array([
            s for s in os.listdir(DATA_PATH)
            if os.path.isdir(os.path.join(DATA_PATH, s))
        ])
    except Exception:
        sign = np.array([])

    seq_length  = 35
    # FIX: Corrected comment — face landmarks were removed from the pipeline.
    # Feature layout: lh(63) + rh(63) + pose_coords(99) + pose_vis(33) = 258
    FEATURE_DIM = 258

    @staticmethod
    def preprocess_landmark_sequence(sequence: np.ndarray) -> np.ndarray:
        target_length = Gestura.seq_length
        original_length = sequence.shape[0]

        if original_length == target_length:
            return sequence.astype(np.float32)
        elif original_length > target_length:
            indices = np.linspace(0, original_length - 1, target_length, dtype=np.int32)
            sequence = sequence[indices]
        else:
            padding_frames = target_length - original_length
            pad_shape = (padding_frames, *sequence.shape[1:])
            pad_array = np.zeros(pad_shape, dtype=sequence.dtype)
            sequence = np.concatenate([sequence, pad_array], axis=0)

        return sequence.astype(np.float32)

    @staticmethod
    def point_detection(image, model):
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = model.process(image)
        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        return image, results

    @staticmethod
    def _normalize_coords(arr):
        # Subtract centroid of a flat (N*3,) coordinate array.
        # If all zeros (landmark absent), return as-is.
        if not np.any(arr):
            return arr
        coords = arr.reshape(-1, 3)
        centroid = coords.mean(axis=0)
        return (coords - centroid).flatten().astype(np.float32)

    @staticmethod
    def extract_keypoints(results):
        # Left hand
        lh = (
            np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
            if results.left_hand_landmarks else np.zeros(21 * 3)
        )
        lh = Gestura._normalize_coords(lh)

        # Right hand
        rh = (
            np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
            if results.right_hand_landmarks else np.zeros(21 * 3)
        )
        rh = Gestura._normalize_coords(rh)

        # Pose — coords normalized separately, visibility kept raw
        if results.pose_landmarks:
            pose_coords = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten()
            pose_coords = Gestura._normalize_coords(pose_coords)
            pose_vis = np.array([res.visibility for res in results.pose_landmarks.landmark], dtype=np.float32)
        else:
            pose_coords = np.zeros(33 * 3)
            pose_vis    = np.zeros(33)

        # lh(63) + rh(63) + pose_coords(99) + pose_vis(33) = 258
        return np.concatenate([lh, rh, pose_coords, pose_vis]).astype(np.float32)

if __name__ == "__main__":
    print(f"Gestura module loaded.")
    print(f"Feature dim per frame: {Gestura.FEATURE_DIM}")