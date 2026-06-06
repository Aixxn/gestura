import mediapipe as mp
import cv2 as cv
import numpy as np

WINDOW_SIZE = 35
FEATURE_DIM = 1662


class Converter:
    def __init__(self):
        self.mp_model = mp.solutions.holistic.Holistic()

        # Fixed window buffer (always length WINDOW_SIZE)
        self.window = np.zeros((WINDOW_SIZE, FEATURE_DIM), dtype=np.float32)

        # How many real frames have been inserted so far
        self.current_length = 0

    def point_detection(self, image_byte):
        """
        Runs Mediapipe holistic and returns a keypoint vector of shape (1662,)
        """
        nparr = np.frombuffer(image_byte, np.uint8)
        cv_image = cv.imdecode(nparr, cv.IMREAD_COLOR)
        image = cv.cvtColor(cv_image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = self.mp_model.process(image)
        keypoints = self.extract_keypoints(results)

        image.flags.writeable = True
        self.image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        return keypoints

    def extract_keypoints(self, results):
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

        a = np.concatenate([face, lh, rh, pose])

        # Normalize by subtracting origin
        base = a[0]
        lm_list = (a - base).astype(np.float32)

        return lm_list

    def process_new_frame(self, frame_vector):
        """
        Always returns a (WINDOW_SIZE, FEATURE_DIM) array.

        - Before full: fills left → right
        - After full: slides window left and appends new frame at the end
        """

        if self.current_length < WINDOW_SIZE:
            # Fill from left to right
            self.window[self.current_length] = frame_vector
            self.current_length += 1
        else:
            # Slide window left, append at end
            self.window[:-1] = self.window[1:]
            self.window[-1] = frame_vector

        return self.window

    def stop(self):
        """
        Return the current window. Already padded on the right.
        """
        return self.window

