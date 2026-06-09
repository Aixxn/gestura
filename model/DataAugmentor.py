import numpy as np
import random
import os

# FIX: Corrected comment — face landmarks are NOT in the feature vector.
# Feature layout for shape (35, 258):
#   [0:63]   = left hand coords   (21 * 3)
#   [63:126] = right hand coords  (21 * 3)
#   [126:225]= pose coords        (33 * 3)
#   [225:258]= pose visibility    (33 * 1) ← do NOT transform these
FEATURE_DIM    = 258
SEQ_LENGTH     = 35
POSE_COORD_END = 225  # everything before pose visibility
# Total coord landmarks: 21 (lh) + 21 (rh) + 33 (pose) = 75
N_LH   = 21
N_RH   = 21
N_POSE = 33

class DataAugmentor:
    def __init__(self,
                 jitter_std=0.004,
                 scale_range=(0.93, 1.07),
                 rotation_range=(-7, 7),
                 translation_range=(-0.05, 0.05),
                 mirror_prob=0.5):
        self.jitter_std        = jitter_std
        self.scale_range       = scale_range
        self.rotation_range    = rotation_range
        self.translation_range = translation_range
        self.mirror_prob       = mirror_prob

    def augment(self, sequence):
        """
        sequence: np.ndarray of shape (35, 258), dtype float32
        Returns:  augmented sequence of shape (35, 258), dtype float32

        Only coordinate columns [0:225] are transformed.
        Pose visibility [225:258] is carried through unchanged.
        """
        assert sequence.shape == (SEQ_LENGTH, FEATURE_DIM), \
            f"Expected ({SEQ_LENGTH}, {FEATURE_DIM}), got {sequence.shape}"

        T = sequence.shape[0]

        # Split coords from visibility
        coords = sequence[:, :POSE_COORD_END].copy()  # (35, 225)
        vis    = sequence[:, POSE_COORD_END:].copy()   # (35, 33)

        # Reshape coords into spatial tensor: (35, 75, 3)
        # 75 = 21 (lh) + 21 (rh) + 33 (pose) landmarks
        spatial = coords.reshape(T, -1, 3)  # (35, 75, 3)

        spatial = self._jitter(spatial)
        spatial = self._scale(spatial)
        spatial = self._rotate(spatial)
        spatial = self._translate(spatial)
        spatial = self._maybe_mirror(spatial)

        # Flatten coords back and re-attach visibility
        coords_aug = spatial.reshape(T, POSE_COORD_END)        # (35, 225)
        result     = np.concatenate([coords_aug, vis], axis=1) # (35, 258)

        assert result.shape == (SEQ_LENGTH, FEATURE_DIM)
        assert result.dtype == np.float32
        return result

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
        xy     = seq[:, :, :2]                                          # (T, 75, 2)
        xy_rot = np.dot(xy.reshape(-1, 2), rot_matrix).reshape(xy.shape)
        seq[:, :, :2] = xy_rot
        return seq

    def _translate(self, seq):
        translation = np.random.uniform(
            *self.translation_range, size=(1, 1, 3)
        ).astype(seq.dtype)
        return seq + translation

    def _maybe_mirror(self, seq):
        if random.random() >= self.mirror_prob:
            return seq

        # Flip x-axis for all landmarks
        seq[:, :, 0] *= -1

        # Swap lh ↔ rh blocks so semantics stay correct after mirroring
        # In the 75-landmark axis: lh=0:21, rh=21:42, pose=42:75
        lh_block = seq[:, :N_LH,           :].copy()
        rh_block = seq[:, N_LH:N_LH+N_RH,  :].copy()
        seq[:, :N_LH,          :] = rh_block
        seq[:, N_LH:N_LH+N_RH, :] = lh_block

        return seq

if __name__ == "__main__":
    INPUT_ROOT       = r"/home/jiyusss/gestura/model/Keypoint_Data_Selected"
    OUTPUT_ROOT      = r"./Keypoint_Data_Augmented"
    NUM_AUG_PER_FILE = 9
    COPY_ORIGINAL    = True

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    augmentor = DataAugmentor()

    for folder_name in os.listdir(INPUT_ROOT):
        input_folder  = os.path.join(INPUT_ROOT, folder_name)
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
                seq = np.load(input_path).astype(np.float32)
            except Exception as e:
                print(f"   Failed to load {filename}: {e}")
                continue

            if seq.shape != (SEQ_LENGTH, FEATURE_DIM):
                print(f"  ⚠️  Skipping {filename}: unexpected shape {seq.shape}")
                continue

            if COPY_ORIGINAL:
                np.save(os.path.join(output_folder, filename), seq)

            for i in range(NUM_AUG_PER_FILE):
                aug_seq = augmentor.augment(seq)
                assert aug_seq.ndim == 2,                            "Must be 2D"
                assert aug_seq.shape == (SEQ_LENGTH, FEATURE_DIM),  f"Shape must be ({SEQ_LENGTH}, {FEATURE_DIM})"
                assert aug_seq.dtype == np.float32,                  "Must be float32"

                base_name = os.path.splitext(filename)[0]
                out_path  = os.path.join(output_folder, f"{base_name}_aug{i}.npy")
                np.save(out_path, aug_seq)

            print(f"   {filename} -> {NUM_AUG_PER_FILE} augmented versions saved")

    print("\n✅ Augmentation complete!")
    print(f"   Output saved to: {OUTPUT_ROOT}")