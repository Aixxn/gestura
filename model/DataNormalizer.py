import numpy as np
import os

class Normalizer:
    def __init__(self,
                 input_root=r"C:\Users\yus\Desktop\repo\gestura\model\Keypoint_Data_Selected",
                 output_root=r"C:\Users\yus\Desktop\repo\gestura\model\keypoint_data_normalized"):
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

                # Apply normalization pipeline
                seq = self._center_on_nose(seq)

                # Save result
                np.save(path_out, seq.astype(np.float32))


        print("\nNormalization complete.")

    def _center_on_nose(self, seq):
        """Translate so nose (landmark 0) is at origin."""
        nose0 = seq[ 0:1, :].copy()
        return seq - nose0
    

if __name__ == "__main__":
    normalizer = Normalizer()
    normalizer.normalize_all()