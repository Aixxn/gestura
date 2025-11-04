import mediapipe as mp
import cv2 as cv
import numpy as np
from collections import deque
import time

# PARAMETERS (tune these)
WINDOW_SIZE = 80
STRIDE = 8
FEATURE_DIM = 1662
CONF_THRESH = 0.80
STABILITY_COUNT = 3
SMOOTHING_WINDOW = 5         # used if smoothing_type == 'mean'
EMA_ALPHA = 0.4              # used if smoothing_type == 'ema'
COOLDOWN_STEPS = 3           # number of predictions to skip after detection (not frames)
smoothing_type = 'mean'       # 'ema' or 'mean'


class Converter:
    def __init__(self):
        # state
        self.frame_buffer = deque(maxlen=WINDOW_SIZE)   # stores per-frame keypoints (1662,)
        self.pred_history = deque(maxlen=SMOOTHING_WINDOW)  # stores recent softmax vectors
        self.stable_counter = 0
        self.last_emitted_label = None
        self.cooldown_counter = 0
        self.ema_state = None  # for EMA smoothing (np.array of shape (num_classes,))
        self.mp_model = mp.solutions.holistic.Holistic()

    def get_asl_grammar(self):
        return split(self.pred_history)

    def post_process_keypoints(self, pred):
        # store raw probs history for mean smoothing if needed
        self.pred_history.append(pred)

        # compute smoothed probs
        smoothed = self.update_smoothing(pred)

        # decide predicted label and confidence
        predicted_label = int(np.argmax(smoothed))
        predicted_conf = float(smoothed[predicted_label])

        # cooldown handling
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return None

        # stability check
        if predicted_conf >= CONF_THRESH:
            # check if last_emitted_label is same as predicted_label
            if self.last_emitted_label == predicted_label:
                # if we already emitted same label earlier, avoid re-emitting until cooldown
                # or you can increase stable_counter to require sustained detection
                # reset stable_counter to STABILITY_COUNT to avoid immediate re-fire
                self.stable_counter = STABILITY_COUNT
            else:
                # we are seeing a candidate label
                self.stable_counter += 1
                if self.stable_counter >= STABILITY_COUNT:
                    # confirmed detection
                    self.last_emitted_label = predicted_label
                    self.cooldown_counter = COOLDOWN_STEPS
                    self.stable_counter = 0
                    # return the label (or process further e.g. append to sentence)
                    return predicted_label, predicted_conf
        else:
            self.stable_counter = 0

        return None

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
        keypoints = self.extract_keypoints(results)
        self.frame_buffer.append(keypoints)
        image.flags.writeable = True
        self.image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        return keypoints

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

    def stop(self):
        pred_list = list(self.pred_history)
        window = pred_list[-min(len(pred_list), WINDOW_SIZE):]
        smoothed_pred = np.mean(window, axis=0)

    def update_smoothing(self, new_probs):
        if smoothing_type == 'ema':
            if self.ema_state is None:
                self.ema_state = new_probs.copy()
            else:
                self.ema_state = EMA_ALPHA * new_probs + (1 - EMA_ALPHA) * self.ema_state
            return self.ema_state
        else:
            # mean smoothing using pred_history
            print('HISTORY:', self.pred_history)
            print('LEN OF HISTORY:', len(self.pred_history))
            arr = np.stack(list(self.pred_history) + [new_probs]) if len(self.pred_history) > 0 else new_probs
            return arr.mean(axis=0)

    def process_new_frame(self, frame):
        """
        Called for each incoming raw frame (OpenCV BGR).
        Steps:
          - run mediapipe detection to get results
          - extract keypoints vector (1662,)
          - append to frame_buffer
          - every STRIDE frames, if we have WINDOW_SIZE frames, run inference + smoothing + detection
        """
        if len(self.frame_buffer) < WINDOW_SIZE:
            return None

        self.frame_counter = getattr(self, 'frame_counter', 0) + 1
        if self.frame_counter % STRIDE != 0:
            return None

        return np.stack(self.frame_buffer)


