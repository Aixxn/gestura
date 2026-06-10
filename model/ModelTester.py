import cv2 as cv
import numpy as np
import tensorflow as tf
import collections
import time
import os
from functions import Gestura

# ------------------------------------------------------------------ #
#  Configuration                                                       #
# ------------------------------------------------------------------ #

MODEL_PATH        = "best_model.keras"
CLASSES_PATH      = "sign_classes.npy"
CONFIDENCE_THRESH = 0.70
SEQ_LENGTH        = 35
FEATURE_DIM       = 258
SMOOTH_WINDOW     = 5          # majority-vote over last N predictions

# Visual
FONT         = cv.FONT_HERSHEY_SIMPLEX
COLOR_GREEN  = (0, 255, 180)
COLOR_DIM    = (80, 80, 80)
COLOR_WARN   = (0, 140, 255)
COLOR_WHITE  = (230, 230, 230)
BAR_COLOR    = (0, 200, 140)
BAR_BG       = (40, 40, 40)


# ------------------------------------------------------------------ #
#  Load model and classes                                              #
# ------------------------------------------------------------------ #

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# Robust class loading: use npy or fallback to scanning dataset directory
if os.path.exists(CLASSES_PATH):
    sign_classes = np.load(CLASSES_PATH, allow_pickle=True)
else:
    DATA_DIR = '/home/jiyusss/gestura/model/Keypoint_Data_Augmented'
    sign_classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])

print(f"Loaded {len(sign_classes)} classes: {list(sign_classes)}")


# ------------------------------------------------------------------ #
#  Drawing helpers                                                     #
# ------------------------------------------------------------------ #

def draw_progress_bar(img, x, y, w, h, value, bar_color=BAR_COLOR, show_threshold=False):
    cv.rectangle(img, (x, y), (x + w, y + h), BAR_BG, -1)
    fill = int(w * min(max(value, 0.0), 1.0))
    if fill > 0:
        cv.rectangle(img, (x, y), (x + fill, y + h), bar_color, -1)
    # Draw threshold marker line on the bar
    if show_threshold:
        thresh_x = x + int(w * CONFIDENCE_THRESH)
        cv.line(img, (thresh_x, y - 2), (thresh_x, y + h + 2), (255, 255, 255), 1)


def put_text(img, text, pos, scale, color, thickness=1):
    x, y = pos
    cv.putText(img, text, (x + 1, y + 1), FONT, scale,
               (0, 0, 0), thickness + 1, cv.LINE_AA)
    cv.putText(img, text, pos, FONT, scale, color, thickness, cv.LINE_AA)


def draw_landmarks(img, results):
    if results.left_hand_landmarks:
        Gestura.mp_draw.draw_landmarks(
            img, results.left_hand_landmarks,
            Gestura.mp_holistic.HAND_CONNECTIONS,
            Gestura.mp_draw.DrawingSpec(color=(0, 200, 255), thickness=1, circle_radius=2),
            Gestura.mp_draw.DrawingSpec(color=(0, 120, 200), thickness=1))

    if results.right_hand_landmarks:
        Gestura.mp_draw.draw_landmarks(
            img, results.right_hand_landmarks,
            Gestura.mp_holistic.HAND_CONNECTIONS,
            Gestura.mp_draw.DrawingSpec(color=(0, 255, 180), thickness=1, circle_radius=2),
            Gestura.mp_draw.DrawingSpec(color=(0, 160, 100), thickness=1))

    if results.pose_landmarks:
        Gestura.mp_draw.draw_landmarks(
            img, results.pose_landmarks,
            Gestura.mp_holistic.POSE_CONNECTIONS,
            Gestura.mp_draw.DrawingSpec(color=(60, 60, 60), thickness=1, circle_radius=1),
            Gestura.mp_draw.DrawingSpec(color=(40, 40, 40), thickness=1))


def hands_present(results) -> bool:
    return (results.left_hand_landmarks is not None or
            results.right_hand_landmarks is not None)


# ------------------------------------------------------------------ #
#  HUD                                                                 #
# ------------------------------------------------------------------ #

def draw_hud(frame, label, confidence, top_preds, buf_len,
             smoothed, fps, hand_ok):
    h, w = frame.shape[:2]
    panel_w = 290
    px      = w - panel_w

    # semi-transparent panel
    overlay = frame.copy()
    cv.rectangle(overlay, (px, 0), (w, h), (10, 10, 10), -1)
    cv.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # title
    put_text(frame, "GESTURA", (px + 12, 34), 0.7, COLOR_GREEN, 2)
    cv.line(frame, (px + 12, 44), (w - 12, 44), COLOR_DIM, 1)

    # buffer bar
    put_text(frame, "BUFFER", (px + 12, 68), 0.4, COLOR_DIM)
    draw_progress_bar(frame, px + 12, 75, panel_w - 24, 7,
                      buf_len / SEQ_LENGTH, bar_color=(80, 160, 255))
    put_text(frame, f"{buf_len}/{SEQ_LENGTH}", (px + 12, 95), 0.38, COLOR_DIM)

    # hand status
    sc = COLOR_GREEN if hand_ok else COLOR_WARN
    put_text(frame, "HAND DETECTED" if hand_ok else "NO HAND",
             (px + 12, 120), 0.42, sc)

    cv.line(frame, (px + 12, 132), (w - 12, 132), COLOR_DIM, 1)

    # prediction
    put_text(frame, "PREDICTION", (px + 12, 155), 0.42, COLOR_DIM)
    if confidence >= CONFIDENCE_THRESH:
        put_text(frame, label,                    (px + 12, 190), 0.9,  COLOR_GREEN, 2)
        put_text(frame, f"{int(confidence*100)}%", (px + 12, 212), 0.55, COLOR_WHITE)
        draw_progress_bar(frame, px + 12, 218, panel_w - 24, 6,
                          confidence, show_threshold=True)
    elif confidence > 0:
        put_text(frame, label,                         (px + 12, 190), 0.9,  COLOR_DIM, 2)
        put_text(frame, f"{int(confidence*100)}% (low)", (px + 12, 212), 0.42, COLOR_WARN)
        draw_progress_bar(frame, px + 12, 218, panel_w - 24, 6,
                          confidence, bar_color=COLOR_WARN, show_threshold=True)
    else:
        put_text(frame, "---", (px + 12, 190), 0.9, COLOR_DIM, 2)

    # threshold label shown below confidence bar
    thresh_bar_x = px + 12 + int((panel_w - 24) * CONFIDENCE_THRESH)
    put_text(frame, f"threshold: {int(CONFIDENCE_THRESH*100)}%",
             (px + 12, 234), 0.35, COLOR_DIM)
    cv.line(frame, (px + 12, 242), (w - 12, 242), COLOR_DIM, 1)

    # stable label
    put_text(frame, "STABLE", (px + 12, 264), 0.42, COLOR_DIM)
    put_text(frame, smoothed or "---", (px + 12, 294),
             0.75, COLOR_GREEN if smoothed else COLOR_DIM, 2)

    cv.line(frame, (px + 12, 308), (w - 12, 308), COLOR_DIM, 1)

    # top-3
    put_text(frame, "TOP 3", (px + 12, 330), 0.42, COLOR_DIM)
    for i, (name, prob) in enumerate(top_preds[:3]):
        y = 352 + i * 46
        col = COLOR_GREEN if i == 0 else (0, 140, 100)
        put_text(frame, name, (px + 12, y), 0.48,
                 COLOR_WHITE if i == 0 else COLOR_DIM)
        draw_progress_bar(frame, px + 12, y + 5, panel_w - 65, 5,
                          prob, bar_color=col)
        put_text(frame, f"{int(prob*100)}%", (w - 50, y), 0.4, COLOR_DIM)

    # fps
    put_text(frame, f"FPS {fps:.0f}", (px + 12, h - 14), 0.38, COLOR_DIM)

    # large label overlay on video (bottom-left)
    if confidence >= CONFIDENCE_THRESH:
        bg_w = max(len(label) * 26 + 20, 100)
        cv.rectangle(frame, (10, h - 78), (10 + bg_w, h - 18), (10, 10, 10), -1)
        put_text(frame, label, (20, h - 26), 1.4, COLOR_GREEN, 3)

    # confidence meter bottom-left (always visible)
    meter_y = h - 100
    put_text(frame, f"CONF  {int(confidence*100)}%", (12, meter_y), 0.45, COLOR_WHITE)
    draw_progress_bar(frame, 12, meter_y + 6, 160, 6,
                      confidence,
                      bar_color=COLOR_GREEN if confidence >= CONFIDENCE_THRESH else COLOR_WARN,
                      show_threshold=True)
    put_text(frame, f"THRESH {int(CONFIDENCE_THRESH*100)}%", (12, meter_y + 22), 0.38, COLOR_DIM)


# ------------------------------------------------------------------ #
#  Main loop                                                           #
# ------------------------------------------------------------------ #

def run():
    cap = cv.VideoCapture(0, cv.CAP_V4L2)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam. Try changing index to 1 or 2.")
        return

    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    cv.namedWindow("Gestura — ASL Recognition", cv.WINDOW_NORMAL)
    cv.resizeWindow("Gestura — ASL Recognition", 960, 480)

    frame_buffer = collections.deque(maxlen=SEQ_LENGTH)
    pred_history = collections.deque(maxlen=SMOOTH_WINDOW)

    label      = ""
    confidence = 0.0
    top_preds  = []
    smoothed   = ""
    prev_time  = time.time()

    print("Inference running. Press Q to quit.")

    with Gestura.mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame read failed.")
                break

            frame = cv.flip(frame, 1)

            # detect landmarks
            _, results = Gestura.point_detection(frame, holistic)
            draw_landmarks(frame, results)

            # extract and buffer
            kp = Gestura.extract_keypoints(results)
            frame_buffer.append(kp)

            hand_ok = hands_present(results)

            if len(frame_buffer) == SEQ_LENGTH and hand_ok:
                seq      = np.array(frame_buffer, dtype=np.float32)   # (35, 258)
                seq      = Gestura.preprocess_landmark_sequence(seq)
                inp      = seq[np.newaxis]                             # (1, 35, 258)

                preds     = model.predict(inp, verbose=0)[0]
                top_idx   = np.argsort(preds)[::-1]
                top_preds = [(sign_classes[i], float(preds[i])) for i in top_idx[:3]]

                best       = top_idx[0]
                confidence = float(preds[best])
                label      = sign_classes[best]

                if confidence >= CONFIDENCE_THRESH:
                    pred_history.append(label)
                    if len(pred_history) == SMOOTH_WINDOW:
                        smoothed = collections.Counter(pred_history).most_common(1)[0][0]
                else:
                    pred_history.clear()
                    smoothed = ""

            elif not hand_ok:
                frame_buffer.clear()
                pred_history.clear()
                label      = ""
                confidence = 0.0
                top_preds  = []
                smoothed   = ""

            now       = time.time()
            fps       = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            draw_hud(frame, label, confidence, top_preds,
                     len(frame_buffer), smoothed, fps, hand_ok)

            cv.imshow("Gestura — ASL Recognition", frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv.destroyAllWindows()
    print("Stopped.")


if __name__ == "__main__":
    run()
