from functions import Gestura
import os
import cv2 as cv
import numpy as np


def main():
    # Define your target ASL vocabulary in uppercase
    asl_mini_words = {
        'AND', 'ANGRY', 'APPLE', 'ASK', 'BAD', 'DRINK', 'EAT', 'FRIEND', 'GO', 'GOOD',
        'HAPPY', 'HE', 'HOME','LIKE', 'MILK', 'MOTHER', 'MY', 'NAME', 'NO', 'PLEASE',
        'SEE', 'THEY', 'TIRED', 'WANT', 'WATER', 'WE', 'WHAT', 'WHERE', 'YES', 'YOU', 'ME'
    }

    # Initialize MediaPipe Holistic model
    with Gestura.mp_holistic.Holistic(
        min_detection_confidence=0.6,
        min_tracking_confidence=0.7
    ) as holistic:
        
        stopped_at = ''  # Leave blank unless resuming from a specific folder
        video_root = r'C:\Users\yus\Desktop\ASL_Citizen\videos'

        # List only directories in the root folder
        all_sign_dirs = [d for d in os.listdir(video_root) if os.path.isdir(os.path.join(video_root, d))]
        all_sign_dirs.sort()

        # Find resume index (if applicable)
        index_stopped = all_sign_dirs.index(stopped_at) if stopped_at in all_sign_dirs else 0

        for sign_dir in all_sign_dirs[index_stopped:]:
            normalized_name = sign_dir.strip().upper()

            # Skip signs not in target vocabulary
            if normalized_name not in asl_mini_words:
                print(f"Skipping '{sign_dir}' — not in target vocabulary")
                continue

            sign_path = os.path.join(video_root, sign_dir)
            videos = [v for v in os.listdir(sign_path) if v.lower().endswith(('.mp4', '.avi', '.mov'))]

            for video_num, video_file in enumerate(videos, start=1):
                video_path = os.path.join(sign_path, video_file)
                cap = cv.VideoCapture(video_path)

                if not cap.isOpened():
                    print(f"⚠️ Could not open video: {video_path}")
                    continue

                frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
                seq = []

                for frame_num in range(frame_count):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    img, results = Gestura.point_detection(frame, holistic)
                    img = cv.flip(img, 1)
                    keypoints = Gestura.extract_keypoints(results)
                    seq.append(keypoints)
                    print(f'<--- Sign {sign_dir}, Video {video_num}, Frame {frame_num} --->')

                if seq:
                    data = np.stack(seq)
                    print(f"Raw data shape: {data.shape}")

                    res_data = Gestura.preprocess_landmark_sequence(data)
                    print(f"Processed data shape: {res_data.shape}")
                    npy_path = os.path.join(Gestura.DATA_PATH, sign_dir, str(video_num))
                    os.makedirs(os.path.dirname(npy_path), exist_ok=True)
                    np.save(npy_path, res_data)
                    print(f"✅ Saved processed data to: {npy_path}.npy")

                cap.release()

        print("✅ Processing complete.")


if __name__ == "__main__":
    main()
