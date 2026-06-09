from functions import Gestura
import os
import shutil
import cv2 as cv
import numpy as np

# Configuration
VIDEO_ROOT = r"/home/jiyusss/Downloads/ASL_Citizen/videos"

ASL_MINI_WORDS = {
    'AND', 'ANGRY', 'APPLE', 'ASK', 'BAD', 'DRINK', 'EAT', 'FRIEND', 'GO', 'GOOD',
    'HAPPY', 'HE', 'HOME', 'LIKE', 'MILK', 'MOTHER', 'MY', 'NAME', 'NO', 'PLEASE',
    'SEE', 'THEY', 'TIRED', 'WANT', 'WATER', 'WE', 'WHAT', 'WHERE', 'YES', 'YOU', 'ME'
}

def organize_and_filter():
    """Moves files into categorized folders based on labels and ignores non-target words."""
    for filename in os.listdir(VIDEO_ROOT):
        filepath = os.path.join(VIDEO_ROOT, filename)
        if not os.path.isfile(filepath):
            continue

        parts = os.path.splitext(filename)[0].rsplit('-', 1)
        if len(parts) != 2:
            continue

        label = parts[1].strip().upper()
        if label not in ASL_MINI_WORDS:
            continue

        label_dir = os.path.join(VIDEO_ROOT, label)
        os.makedirs(label_dir, exist_ok=True)
        shutil.move(filepath, os.path.join(label_dir, filename))

def extract_keypoints_pipeline():
    """
    Processes organized video folders into .npy keypoint files.

    # FIX: Corrected shape docstring — face landmarks are NOT extracted.
    Each saved file has shape (35, 258):
      - 35  = fixed sequence length (padded / downsampled)
      - 258 = feature dim: lh(63) + rh(63) + pose_coords(99) + pose_vis(33)
    """
    stopped_at = ''  # Leave blank unless resuming from a specific folder

    with Gestura.mp_holistic.Holistic(
        min_detection_confidence=0.6,
        min_tracking_confidence=0.7
    ) as holistic:

        all_sign_dirs = [d for d in os.listdir(VIDEO_ROOT) if os.path.isdir(os.path.join(VIDEO_ROOT, d))]
        all_sign_dirs.sort()

        # Find resume index (if applicable)
        index_stopped = all_sign_dirs.index(stopped_at) if stopped_at in all_sign_dirs else 0

        for sign_dir in all_sign_dirs[index_stopped:]:
            normalized_name = sign_dir.strip().upper()

            # Skip signs not in target vocabulary
            if normalized_name not in ASL_MINI_WORDS:
                print(f"Skipping '{sign_dir}' — not in target vocabulary")
                continue

            sign_path = os.path.join(VIDEO_ROOT, sign_dir)
            save_path = os.path.join(Gestura.DATA_PATH, sign_dir)
            os.makedirs(save_path, exist_ok=True)

            videos = [v for v in os.listdir(sign_path) if v.lower().endswith(('.mp4', '.avi', '.mov'))]

            for video_num, video_file in enumerate(videos, start=1):
                video_path = os.path.join(sign_path, video_file)
                cap = cv.VideoCapture(video_path)

                if not cap.isOpened():
                    print(f"  ⚠️  Could not open video: {video_path}")
                    continue

                frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
                seq = []

                for frame_num in range(frame_count):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    _, results = Gestura.point_detection(frame, holistic)
                    keypoints = Gestura.extract_keypoints(results)  # (258,)
                    seq.append(keypoints)
                    print(f'<--- Sign {sign_dir}, Video {video_num}, Frame {frame_num} --->')

                cap.release()

                if len(seq) == 0:
                    print(f"  ⚠️  No frames in {sign_dir}/{video_file}, skipping.")
                    continue

                # Stack → (T, 258), then resample/pad to (35, 258)
                data = np.stack(seq)                                        # (T, 258)
                print(f"Raw data shape: {data.shape}")
                res_data = Gestura.preprocess_landmark_sequence(data)       # (35, 258)
                print(f"Processed data shape: {res_data.shape}")

                assert res_data.shape == (35, Gestura.FEATURE_DIM), \
                    f"Unexpected shape {res_data.shape} for {video_file}"

                npy_path = os.path.join(save_path, str(video_num))
                os.makedirs(os.path.dirname(npy_path), exist_ok=True)
                np.save(npy_path, res_data)
                print(f"  ✅ Saved: {npy_path}.npy")

        print("✅ Processing complete.")

if __name__ == "__main__":
    print("Organizing dataset...")
    organize_and_filter()
    print("\nExtracting landmarks...")
    extract_keypoints_pipeline()
    print("\nDone!")