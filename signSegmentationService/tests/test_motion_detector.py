"""
Unit tests for the MotionDetector class.

The MotionDetector uses hysteresis-based thresholding with adaptive thresholds
to detect sign boundaries in keypoint sequences.
"""

import numpy as np
import pytest
from motion_detector import MotionDetector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def detector():
    """Return a MotionDetector with small values for fast tests."""
    return MotionDetector(
        low_factor=0.5,
        high_factor=2.0,
        still_frames_required=3,   # low so tests aren't sluggish
        min_sign_duration=2,       # minimum 2 frames before sign can end
        history_size=10,
        feature_dim=258,
        motion_smoothing=0.0,      # no smoothing for deterministic tests
        stillness_floor=0.0,       # no floor for deterministic tests
    )


@pytest.fixture
def random_keypoints():
    """Return a realistic 258-dim keypoint vector (noise)."""
    rng = np.random.default_rng(42)
    return rng.random(258).astype(np.float32)


@pytest.fixture
def zero_keypoints():
    """Return a zero keypoint vector with the expected shape."""
    return np.zeros(258, dtype=np.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def nearly_equal(a: np.ndarray, b: np.ndarray, rtol=1e-5) -> bool:
    return np.allclose(a, b, rtol=rtol)


# ===================================================================
# First-frame behaviour
# ===================================================================

class TestFirstFrame:

    def test_first_frame_returns_not_ended(self, detector, random_keypoints):
        ended, sign = detector.update(random_keypoints)
        assert ended is False
        assert sign is None

    def test_first_frame_starts_accumulation(self, detector, random_keypoints):
        detector.update(random_keypoints)
        assert detector.sign_frames == 1
        assert detector.get_current_sign_length() == 1

    def test_first_frame_sets_previous(self, detector, random_keypoints):
        detector.update(random_keypoints)
        assert detector.previous_keypoints is not None
        assert nearly_equal(detector.previous_keypoints, random_keypoints)

    def test_first_frame_with_zero_keypoints(self, detector, zero_keypoints):
        ended, sign = detector.update(zero_keypoints)
        assert ended is False
        assert sign is None


# ===================================================================
# Motion thresholds
# ===================================================================

class TestMotionThresholds:

    def test_low_motion_increments_still_counter(self, detector, random_keypoints):
        detector.update(random_keypoints)
        still_kp = random_keypoints + 1e-6
        detector.update(still_kp)
        assert detector.still_counter >= 1

    def test_high_motion_resets_still_counter(self, random_keypoints):
        # Use high still_frames_required so the detector doesn't keep
        # ending signs during the low-motion setup phase.
        d = MotionDetector(
            low_factor=0.5,
            high_factor=2.0,
            still_frames_required=30,  # high → no sign endings during setup
            min_sign_duration=2,
            history_size=10,
            feature_dim=258,
            motion_smoothing=0.0,
            stillness_floor=0.0,
        )
        # Seed enough frames to pass the fallback period (≥10 history entries)
        d.update(random_keypoints)
        for _ in range(12):
            d.update(random_keypoints + 1e-6)
        assert d.still_counter >= 1

        # Now send very high motion — should reset the counter
        high_kp = random_keypoints + 10.0
        d.update(high_kp)
        assert d.still_counter == 0

    def test_hysteresis_band_holds_state(self, detector, random_keypoints):
        """
        Motion within the hysteresis band (low < motion < high) should NOT
        change the still counter.
        """
        detector.update(random_keypoints)
        mid_kp = random_keypoints.copy()
        mid_kp[0] += 0.2  # raw_motion = 0.2, within band [0.1, 0.4]
        before = detector.still_counter
        detector.update(mid_kp)
        assert detector.still_counter == before, \
            "Still counter should not change in the hysteresis band"


# ===================================================================
# Sign ending
# ===================================================================

class TestSignEnding:

    def test_sign_ends_after_enough_still_frames(self, detector, random_keypoints):
        detector.update(random_keypoints)
        for i in range(5):
            kp = random_keypoints + 1e-6 * (i + 1)
            ended, sign = detector.update(kp)
            if ended:
                assert sign is not None
                assert len(sign) >= detector.min_sign_duration
                return
        pytest.fail("Sign did not end after enough still frames")

    def test_sign_not_ended_with_insufficient_frames(self, random_keypoints):
        """Even with low motion, sign should not end before min_sign_duration."""
        detector = MotionDetector(
            still_frames_required=1,
            min_sign_duration=100,  # very high
            feature_dim=258,
        )
        detector.update(random_keypoints)
        kp = random_keypoints + 1e-6
        ended, _ = detector.update(kp)
        assert ended is False, "Sign should not end before min_sign_duration met"

    def test_consecutive_signs_can_be_detected(self, detector, random_keypoints):
        detector.update(random_keypoints)
        for i in range(3):
            detector.update(random_keypoints + (i + 1) * 0.1)
        for i in range(5):
            ended, sign = detector.update(random_keypoints + 1e-6)
            if ended:
                break
        else:
            pytest.fail("First sign never ended")

        detector.update(random_keypoints + 5.0)
        assert detector.sign_frames >= 1
        assert detector.get_current_sign_length() >= 1

    def test_reset_clears_state(self, detector, random_keypoints):
        detector.update(random_keypoints)
        detector.update(random_keypoints + 0.1)
        detector.reset()
        assert detector.still_counter == 0
        assert detector.sign_frames == 0
        assert detector.previous_keypoints is None
        assert len(detector.current_sign_keypoints) == 0
        assert len(detector.raw_motion_history) == 0
        assert detector.get_current_sign_length() == 0


# ===================================================================
# Adaptive thresholding
# ===================================================================

class TestAdaptiveThresholds:

    def test_thresholds_adapt_to_high_motion(self, detector, random_keypoints):
        detector.update(random_keypoints)
        for i in range(12):
            detector.update(random_keypoints + 5.0 + i * 0.1)
        median = float(np.median(list(detector.raw_motion_history)))
        assert median > 1.0, f"Expected median > 1.0, got {median}"

    def test_fallback_thresholds_without_history(self, detector, random_keypoints):
        detector.update(random_keypoints)
        kp = random_keypoints + 100.0
        detector.update(kp)
        assert detector.still_counter == 0


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:

    def test_exact_same_frame_no_motion(self, detector, random_keypoints):
        detector.update(random_keypoints)
        detector.update(random_keypoints.copy())
        assert detector.still_counter >= 1

    def test_zero_keypoints_no_crash(self, detector, zero_keypoints):
        """Identical zero-keypoint frames produce zero motion, triggering
        sign-ending after still_frames_required frames (which resets the
        accumulator)."""
        for _ in range(5):
            try:
                detector.update(zero_keypoints.copy())
            except Exception as e:
                pytest.fail(f"MotionDetector crashed on zero keypoints: {e}")
        assert detector.sign_frames >= 1

    def test_negative_keypoint_values(self, detector):
        neg = np.full(258, -0.5, dtype=np.float32)
        detector.update(neg)
        detector.update(np.full(258, -0.3, dtype=np.float32))
        assert detector.sign_frames == 2

    def test_nan_keypoints(self, detector, random_keypoints):
        nan_kp = np.full(258, np.nan, dtype=np.float32)
        detector.update(random_keypoints)
        detector.update(nan_kp)
        assert detector.sign_frames == 2

    def test_inf_keypoints(self, detector, random_keypoints):
        inf_kp = np.full(258, np.inf, dtype=np.float32)
        detector.update(random_keypoints)
        detector.update(inf_kp)
        assert detector.sign_frames == 2

    def test_wrong_shape_keypoints_raises_value_error(self, detector):
        with pytest.raises(ValueError, match="Keypoint dimension mismatch"):
            detector.update(np.array([1.0, 2.0, 3.0]))

    def test_2d_keypoints_raises_value_error(self, detector):
        with pytest.raises(ValueError, match="Expected 1-D"):
            detector.update(np.ones((1, 258), dtype=np.float32))

    def test_get_current_sign_length_after_reset(self, detector, random_keypoints):
        detector.update(random_keypoints)
        detector.update(random_keypoints + 0.1)
        detector.reset()
        assert detector.get_current_sign_length() == 0
        detector.update(random_keypoints)
        assert detector.get_current_sign_length() == 1


# ===================================================================
# Parameters
# ===================================================================

class TestParameters:

    def test_custom_parameters(self):
        d = MotionDetector(
            low_factor=0.3,
            high_factor=1.5,
            still_frames_required=20,
            min_sign_duration=10,
            history_size=50,
            feature_dim=1024,
            motion_smoothing=0.0,
            stillness_floor=0.0,
        )
        assert d.low_factor == 0.3
        assert d.high_factor == 1.5
        assert d.still_frames_required == 20
        assert d.min_sign_duration == 10
        assert d.history_size == 50
        assert d.feature_dim == 1024
        assert d.raw_motion_history.maxlen == 50

    def test_default_parameters(self):
        d = MotionDetector()
        assert d.low_factor == 0.5
        assert d.high_factor == 4.0
        assert d.still_frames_required == 8
        assert d.min_sign_duration == 5
        assert d.history_size == 30
        assert d.feature_dim == 258
        assert d.motion_smoothing == 0.6
        assert d.stillness_floor == 0.3


# ===================================================================
# Integration-style: multi-sign sequence
# ===================================================================

class TestFullSequence:

    def test_two_sign_sequence(self, detector, random_keypoints):
        rng = np.random.default_rng(1234)
        signs_detected = 0

        # Sign 1
        for i in range(6):
            detector.update(rng.random(258).astype(np.float32))

        # Pause to end sign 1
        last_kp = random_keypoints.copy()
        for i in range(6):
            ended, sign = detector.update(last_kp + 1e-6)
            if ended:
                signs_detected += 1
                break

        # Sign 2
        for i in range(6):
            detector.update(rng.random(258).astype(np.float32))

        # Pause to end sign 2
        last_kp = random_keypoints.copy()
        for i in range(6):
            ended, sign = detector.update(last_kp + 1e-6)
            if ended:
                signs_detected += 1
                break

        assert signs_detected == 2, f"Expected 2 signs, got {signs_detected}"
