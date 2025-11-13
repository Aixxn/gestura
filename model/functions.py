import mediapipe as mp
import numpy as np
import time
import os
import keras as k
import cv2 as cv

class Gestura: 
    mp_holistic = mp.solutions.holistic
    mp_draw = mp.solutions.drawing_utils
    DATA_PATH = os.path.join("./Keypoint_Data_Selected")

    # Path to where raw ASL videos are stored on this machine. Listing the
    # directory at import time can raise FileNotFoundError or block execution
    # when running on a different machine. Wrap in try/except and default to
    # an empty array so importing this module is safe.
    _ASL_ROOT = r"C:\Users\yus\Desktop\ASL_Citizen\videos"
    try:
        sign = np.array([s for s in os.listdir(_ASL_ROOT) if os.path.isdir(os.path.join(_ASL_ROOT, s))])
    except Exception:
        sign = np.array([])
    seq_length = 35  # number of frames to be used per video

    @staticmethod
    def preprocess_landmark_sequence(sequence: np.ndarray) -> np.ndarray:
        """
        Preprocess a sequence of landmarks to have a fixed length.

        Parameters:
        - sequence: np.ndarray of shape (T, N, 3) 
            where T = frames, N = landmarks, 3 = x,y,z

        Returns:
        - np.ndarray of shape (seq_length, N, 3)
        """
        target_length = Gestura.seq_length
        original_length = sequence.shape[0]

        if original_length == target_length:
            return sequence.astype(np.float32)

        elif original_length > target_length:
            # Resample by picking evenly spaced frame indices
            indices = np.linspace(0, original_length - 1, target_length, dtype=np.int32)
            sequence = sequence[indices]

        else:
            # Pad with zeros at the end
            padding_frames = target_length - original_length
            pad_shape = (padding_frames, *sequence.shape[1:])
            pad_array = np.zeros(pad_shape, dtype=sequence.dtype)
            sequence = np.concatenate([sequence, pad_array], axis=0)

        return sequence.astype(np.float32) 

    def build_folder():
        for a in Gestura.sign:
            try:
                os.makedirs(os.path.join(Gestura.DATA_PATH, a))
            except:
                pass

    def point_detection(image, model): #image from cv; holistic model from mp
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB) #process color and flip
        image.flags.writeable = False
        results = model.process(image)
        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        return image, results

    def draw_styled_points(image, results):
        Gestura.mp_draw.draw_landmarks(image, results.right_hand_landmarks, Gestura.mp_holistic.HAND_CONNECTIONS,
                                Gestura.mp_draw.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=3),#landmark color
                                Gestura.mp_draw.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=1))#connection color

        Gestura.mp_draw.draw_landmarks(image, results.left_hand_landmarks, Gestura.mp_holistic.HAND_CONNECTIONS,
                                Gestura.mp_draw.DrawingSpec(color=(84, 44, 44), thickness=2, circle_radius=3),
                                Gestura.mp_draw.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=1))
        
        Gestura.mp_draw.draw_landmarks(image, results.face_landmarks, Gestura.mp_holistic.FACEMESH_TESSELATION,
                                Gestura.mp_draw.DrawingSpec(color=(255,170,170), thickness=1, circle_radius=1),
                                Gestura.mp_draw.DrawingSpec(color=(255,255,255), thickness=1, circle_radius=1))
        Gestura.mp_draw.draw_landmarks(image, results.pose_landmarks, Gestura.mp_holistic.POSE_CONNECTIONS,
                             Gestura.mp_draw.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             Gestura.mp_draw.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             )

    def extract_keypoints(results):
        # list comprehension to loop over results and get needed data, then arranged to np.array. flattened to turn it into one array. else is to make a placeholder
        lh = np.array([[res.x,res.y,res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3) #might be a bug
        rh = np.array([[res.x,res.y,res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3) #might be a bug
        face = np.array([[res.x,res.y,res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        # puts all data into one array
        a = np.concatenate([face, lh, rh, pose])

        lm_list = []

        for lm in a:
            base = a[0]
            lm_list.append(lm - base) 
        lm_list = np.array(lm_list, dtype=np.float32)

        return lm_list

if __name__ == "__main__":
    seq = []
    cap = cv.VideoCapture(0)  # Open the default camera
    holistic = Gestura.mp_holistic()
    sign_dir, video = "example_sign", "example_video"  # Replace with actual sign directory and video name
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        for frame_num in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            img, results = Gestura.point_detection(frame, holistic)
            img = cv.flip(img, 1)
            keypoints = Gestura.extract_keypoints(results)
            seq.append(keypoints)
            print(f'<--- Sign {sign_dir}, Video {video}, frame {frame_num} --->')
        break  # Exit the while loop after processing frames

    if len(seq) > 0:
        res_data = Gestura.preprocess_landmark_sequence(np.stack(seq))
        print(res_data.shape)
        # Save or process res_data as needed

    cap.release()