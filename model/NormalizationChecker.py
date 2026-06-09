import numpy as np
import os

BASE_FOLDER = r"C:\Users\yus\Desktop\gestura\model\keypoint_data_normalized"

print(f"Checking if nose (landmark 0) is at origin in all frames")
print(f"Base folder: {BASE_FOLDER}\n")

total_files = 0
nose_ok_count = 0
error_count = 0

for root, _, files in os.walk(BASE_FOLDER):
    npy_files = [f for f in files if f.endswith(".npy")]
    if not npy_files:
        continue

    rel_folder = os.path.relpath(root, BASE_FOLDER)
    print(f"Folder: {rel_folder}")

    for filename in sorted(npy_files):
        file_path = os.path.join(root, filename)
        total_files += 1

        try:
            seq = np.load(file_path, allow_pickle=True)
        except Exception as e:
            print(f"   Error loading {filename}: {e}")
            error_count += 1
            continue

        # Ensure shape is (T, N, 3)
        if seq.ndim == 2:
            if seq.shape[1] % 3 != 0:
                print(f"   Warning: {filename} has invalid 2D shape (feature length not divisible by 3)")
                continue
            seq = seq.reshape(seq.shape[0], seq.shape[1] // 3, 3)
        elif seq.ndim != 3 or seq.shape[2] != 3:
            print(f"   Warning: {filename} has unexpected shape {seq.shape}")
            continue

        # Extract nose positions across all frames: (T, 3)
        nose_points = seq[:, 0, :]
        nose_distances = np.linalg.norm(nose_points, axis=1)
        max_nose_offset = np.max(nose_distances)

        # Tolerance: 1e-3 (0.001) — reasonable for float32 after normalization
        nose_ok = max_nose_offset < 1e-3

        if nose_ok:
            status = "Nose at origin"
            nose_ok_count += 1
        else:
            status = f"Nose offset! max={max_nose_offset:.6f}"

        print(f"   {filename:15} | {status}")

    print()

# Final summary
print("NOSE-ORIGIN CHECK SUMMARY")
print("-" * 40)
print(f"Total files checked: {total_files}")
print(f"Files with nose at origin: {nose_ok_count}")
print(f"Files with nose offset: {total_files - nose_ok_count - error_count}")
print(f"Load errors: {error_count}")