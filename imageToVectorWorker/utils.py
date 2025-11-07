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
smoothing_type = 'ema'       # 'ema' or 'mean'

# state
self.frame_buffer = deque(maxlen=WINDOW_SIZE)   # stores per-frame keypoints (1664,)
self.pred_history = deque(maxlen=SMOOTHING_WINDOW)  # stores recent softmax vectors
self.stable_counter = 0
self.last_emitted_label = None
self.cooldown_counter = 0
self.ema_state = None  # for EMA smoothing (np.array of shape (num_classes,))

# helpers (you'll replace these with your actual functions)
# - Gestura.extract_keypoints(results) -> (1662,)
# - Gestura.preprocess_landmark_sequence(seq) -> (WINDOW_SIZE, 1662)
# - model.predict(np.expand_dims(window, axis=0)) -> (1, num_classes)
# - label_names = list of class names in the same order as model outputs

def update_smoothing(new_probs):
    global self.ema_state
    if smoothing_type == 'ema':
        if self.ema_state is None:
            self.ema_state = new_probs.copy()
        else:
            self.ema_state = EMA_ALPHA * new_probs + (1 - EMA_ALPHA) * self.ema_state
        return self.ema_state
    else:
        # mean smoothing using pred_history
        arr = np.stack(list(self.pred_history) + [new_probs]) if len(self.pred_history) > 0 else new_probs
        return arr.mean(axis=0)

def run_inference_on_window(window_np):
    # window_np shape -> (WINDOW_SIZE, FEATURE_DIM)
    # Ensure dtype float32
    inp = window_np.astype(np.float32)[np.newaxis, ...]  # (1, W, D)
    probs = model.predict(inp, verbose=0)  # (1, num_classes)
    return probs[0]  # (num_classes,)

def process_new_frame(frame):
    """
    Called for each incoming raw frame (OpenCV BGR).
    Steps:
      - run mediapipe detection to get results
      - extract keypoints vector (1662,)
      - append to frame_buffer
      - every STRIDE frames, if we have WINDOW_SIZE frames, run inference + smoothing + detection
    """
    global self.stable_counter, self.last_emitted_label, self.cooldown_counter

    # if not enough frames yet, just wait
    if len(self.frame_buffer) < WINDOW_SIZE:
        return None

    # evaluate only every STRIDE frames (to reduce compute)
    # we use a simple frame counter approach; you can use timestamps instead
    process_new_frame.counter = getattr(process_new_frame, 'counter', 0) + 1
    if process_new_frame.counter % STRIDE != 0:
        return None

    # prepare window (stack into (W, D))
    window = np.stack(self.frame_buffer)  # newest 80 frames

    # optional: you might center crop or randomly jitter in production for robustness
    probs = run_inference_on_window(window)  # (num_classes,)

    # store raw probs history for mean smoothing if needed
    self.pred_history.append(probs)

    # compute smoothed probs
    smoothed = update_smoothing(probs)

    # decide predicted label and confidence
    predicted_label = int(np.argmax(smoothed))
    predicted_conf = float(smoothed[predicted_label])

    # cooldown handling
    if self.cooldown_counter > 0:
        cooldown_counter -= 1
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
                last_emitted_label = predicted_label
                cooldown_counter = COOLDOWN_STEPS
                self.stable_counter = 0
                # return the label (or process further e.g. append to sentence)
                return predicted_label, predicted_conf
    else:
        self.stable_counter = 0

    return None

