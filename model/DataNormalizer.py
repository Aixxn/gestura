import numpy as np
import os

class Normalizer:
    def __init__(self,
                 input_root=r"C:\Users\yus\Desktop\gestura\model\keypoint_data",
                 output_root=r"C:\Users\yus\Desktop\gestura\model\keypoint_data_normalized"):
        self.input_root = input_root
        self.output_root = output_root
        os.makedirs(self.output_root, exist_ok=True)

    def normalize_all(self):
        """Apply centered normalization to all .npy files in subfolders."""
        for folder in os.listdir(self.input_root):
            folder_in = os.path.join(self.input_root, folder)
            folder_out = os.path.join(self.output_root, folder)

            if not os.path.isdir(folder_in):
                continue

            os.makedirs(folder_out, exist_ok=True)
            print(f"\nProcessing folder: {folder}")

            for file in os.listdir(folder_in):
                if not file.endswith(".npy"):
                    continue

                path_in = os.path.join(folder_in, file)
                path_out = os.path.join(folder_out, file)

                try:
                    seq = np.load(path_in)
                except Exception as e:
                    print(f" Error loading {file}: {e}")
                    continue
                seq = self._reshape_to_landmarks(seq)
                # Ensure shape is (T, N, 3)
                seq = self._ensure_landmark_shape(seq, file)
                if seq is None:
                    continue

                # Apply normalization pipeline
                seq = self._center_on_nose(seq)

                # Save result
                np.save(path_out, seq.astype(np.float32))


        print("\nNormalization complete.")

    def _reshape_to_landmarks(self, seq):
        if seq.shape[1] % 3 != 0:
            seq = seq[:, :-1]
        num_landmarks = seq.shape[1] // 3
        return seq.reshape(seq.shape[0], num_landmarks, 3)
    
    def _ensure_landmark_shape(self, seq, filename):
        """Reshape to (T, N, 3) if needed; skip if invalid."""
        if seq.ndim == 2:
            feature_len = seq.shape[1]
            if feature_len % 3 != 0:
                print(f" {filename}: feature length {feature_len} not divisible by 3, skipping.")
                return None
            seq = seq.reshape(seq.shape[0], feature_len // 3, 3)
        elif seq.ndim != 3 or seq.shape[2] != 3:
            print(f" {filename}: unexpected shape {seq.shape}, skipping.")
            return None
        return seq

    def _center_on_nose(self, seq):
        """Translate so nose (landmark 0) is at origin."""
        nose0 = seq[:, 0:1, :].copy()
        return seq - nose0
    

if __name__ == "__main__":
    normalizer = Normalizer()
    normalizer.normalize_all()