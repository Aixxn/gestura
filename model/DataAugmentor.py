import numpy as np
import random

class DataAugmentor:
    def __init__(self,
                 jitter_std=0.01,
                 scale_range=(0.9, 1.1),
                 rotation_range=(-10, 10),  # degrees
                 translation_range=(-0.1, 0.1),
                 drop_frame_prob=0.1,
                 mirror_prob=0.5,
                 crop_min_frames=20):
        self.jitter_std = jitter_std
        self.scale_range = scale_range
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.drop_frame_prob = drop_frame_prob
        self.mirror_prob = mirror_prob
        self.crop_min_frames = crop_min_frames

    def augment(self, sequence):
        """
        sequence: np.ndarray of shape (T, N, 3)
        Returns: Augmented sequence (np.ndarray)
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
        noise = np.random.normal(0, self.jitter_std, seq.shape)
        return seq + noise

    def _scale(self, seq):
        scale = np.random.uniform(*self.scale_range)
        return seq * scale

    def _rotate(self, seq):
        angle = np.radians(np.random.uniform(*self.rotation_range))
        cos_val, sin_val = np.cos(angle), np.sin(angle)
        rot_matrix = np.array([
            [cos_val, -sin_val],
            [sin_val, cos_val]
        ])
        seq[:, :, :2] = np.dot(seq[:, :, :2], rot_matrix)  # rotate x,y
        return seq

    def _translate(self, seq):
        translation = np.random.uniform(*self.translation_range, size=(1, 1, 3))
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
        end = start + self.crop_min_frames
        return seq[start:end]

    def _maybe_drop_frames(self, seq):
        if self.drop_frame_prob <= 0:
            return seq
        keep_indices = [i for i in range(len(seq)) if random.random() > self.drop_frame_prob]
        if not keep_indices:
            keep_indices = [random.randint(0, len(seq) - 1)]  # keep at least 1
        return seq[keep_indices]

