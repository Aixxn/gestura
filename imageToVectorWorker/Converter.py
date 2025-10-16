import mediapipe as mp
import cv2 as cv
import numpy as np


class Converter:
    def __init__(self):
        self.mp_model = mp.solutions.holistic.Holistic()

    def point_detection(self, image_byte):
        '''
        args:
            image_byte: image buffer (byte)

        This method peform two processes, it fist converts image buffer (byte)
        into a numpy array then, converts the numpy array into an image using
        cv.imdecode(), essentially doing image byte to 2d image. It will then
        add landmarks to the image using the mediapipe model's process() method
        '''
        nparr = np.frombuffer(image_byte, np.uint8)
        cv_image = cv.imdecode(nparr, cv.IMREAD_COLOR)
        image = cv.cvtColor(cv_image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.mp_model.process(image)
        image.flags.writeable = True
        self.image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        return results

    def extract_keypoints(self, results):
        '''
        args:
            result: Image with keypoints

        This method extracts the coordinates of each keypoints and turn it into 
        a flatten numpy array.
        '''
        # list comprehension to loop over results and get needed data, then arranged to np.array. flattened to turn it into one array. else is to make a placeholder
        lh = np.array([[res.x, res.y, res.z] for res in results
                       .left_hand_landmarks
                       .landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)

        rh = np.array([[res.x, res.y, res.z] for res in results
                       .right_hand_landmarks
                       .landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)

        face = np.array([[res.x, res.y, res.z] for res in results
                         .face_landmarks
                         .landmark]).flatten() if results.face_landmarks else np.zeros(468*3)

        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results
                         .pose_landmarks
                         .landmark]).flatten() if results.pose_landmarks else np.zeros(132)

        # puts all data into one array
        a = np.concatenate([face, lh, rh, pose])

        lm_list = []

        for lm in a:
            base = a[0]
            lm_list.append(lm - base)
        lm_list = np.array(lm_list, dtype=np.float32).tolist()

        return lm_list
