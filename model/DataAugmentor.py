import numpy as np
import random
import os


class DataAugmentor:
    def __init__(self,
                 jitter_std=0.005,          # Reduced: less noise for subtle gestures
                 scale_range=(0.95, 1.05),   # Tighter scale
                 rotation_range=(-8, 8),     # Smaller rotation (front-facing camera)
                 translation_range=(-0.05, 0.05),  # Smaller translation
                 drop_frame_prob=0.05,       # Mild frame dropping
                 mirror_prob=0.5,
                 crop_min_frames=24):        # Adjust based on your typical sequence length
        self.jitter_std = jitter_std
        self.scale_range = scale_range
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.drop_frame_prob = drop_frame_prob
        self.mirror_prob = mirror_prob
        self.crop_min_frames = crop_min_frames

    def augment(self, sequence):
        """
        sequence: np.ndarray of shape (T, N, 3), dtype float32
        Returns: Augmented sequence (np.ndarray, same dtype)
        """
        sequence = sequence.copy()
        sequence = self._jitter(sequence)
        sequence = self._scale(sequence)
        sequence = self._rotate(sequence)
        sequence = self._translate(sequence)
        sequence = self._maybe_mirror(sequence)
        sequence = self._maybe_crop(sequence)
        sequence = self._maybe_drop_frames(sequence)
        return sequence

    def _jitter(self, seq):
        noise = np.random.normal(0, self.jitter_std, seq.shape).astype(seq.dtype)
        return seq + noise

    def _scale(self, seq):
        scale = np.random.uniform(*self.scale_range)
        return seq * scale

    def _rotate(self, seq):
        angle = np.radians(np.random.uniform(*self.rotation_range))
        cos_val, sin_val = np.cos(angle), np.sin(angle)
        rot_matrix = np.array([
            [cos_val, -sin_val],
            [sin_val,  cos_val]
        ], dtype=seq.dtype)
        # Rotate only x, y (leave z unchanged)
        xy = seq[:, :, :2]  # Shape: (T, N, 2)
        # Reshape to (T*N, 2) for matrix multiply, then back
        xy_rot = np.dot(xy.reshape(-1, 2), rot_matrix).reshape(xy.shape)
        seq[:, :, :2] = xy_rot
        return seq

    def _translate(self, seq):
        translation = np.random.uniform(
            *self.translation_range, size=(1, 1, 3)
        ).astype(seq.dtype)
        return seq + translation

    def _maybe_mirror(self, seq):
        if random.random() < self.mirror_prob:
            seq[:, :, 0] *= -1  # flip x-axis
        return seq

    def _maybe_crop(self, seq):
        if len(seq) <= self.crop_min_frames:
            return seq
        max_start = len(seq) - self.crop_min_frames
        start = random.randint(0, max_start)
        return seq[start:start + self.crop_min_frames]

    def _maybe_drop_frames(self, seq):
        if self.drop_frame_prob <= 0 or len(seq) <= 1:
            return seq
        keep = [i for i in range(len(seq)) if random.random() > self.drop_frame_prob]
        if not keep:
            keep = [random.randint(0, len(seq) - 1)]
        return seq[keep]


if __name__ == "__main__":
    INPUT_ROOT = r"C:\Users\yus\Desktop\gestura\model\keypoint_data_normalized"
    OUTPUT_ROOT = r"C:\Users\yus\Desktop\gestura\model\keypoint_data_augmented"
    NUM_AUG_PER_FILE = 3  # Generate  n augmented versions per file
    COPY_ORIGINAL = True  # Also copy original (non-augmented) files

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    augmentor = DataAugmentor()

    for folder_name in os.listdir(INPUT_ROOT):
        input_folder = os.path.join(INPUT_ROOT, folder_name)
        output_folder = os.path.join(OUTPUT_ROOT, folder_name)
        if not os.path.isdir(input_folder):
            continue
        os.makedirs(output_folder, exist_ok=True)
        print(f"\nAugmenting folder: {folder_name}")

        for filename in os.listdir(input_folder):
            if not filename.endswith(".npy"):
                continue

            input_path = os.path.join(input_folder, filename)
            try:
                seq = np.load(input_path).astype(np.float32)  # Load as float32 early
            except Exception as e:
                print(f"   Failed to load {filename}: {e}")
                continue

            if COPY_ORIGINAL:
                orig_out = os.path.join(output_folder, filename)
                np.save(orig_out, seq)

            # Generate augmented copies
            for i in range(NUM_AUG_PER_FILE):
                aug_seq = augmentor.augment(seq)
                # Safety check
                assert aug_seq.shape[2] == 3, "Keypoint dim must be 3"
                assert aug_seq.dtype == np.float32, "Must be float32"

                base_name = os.path.splitext(filename)[0]
                out_path = os.path.join(output_folder, f"{base_name}_aug{i}.npy")
                np.save(out_path, aug_seq)

            print(f"   {filename} -> {NUM_AUG_PER_FILE} augmented versions saved")

    print("\n Augmentation complete!")
    print(f" Output saved to: {OUTPUT_ROOT}")