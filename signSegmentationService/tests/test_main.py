"""
Integration tests for the Sign Segmentation Service API endpoints.

Uses FastAPI TestClient with a mocked converter module (to avoid the
heavy MediaPipe dependency).
"""

import base64
import sys
import types

import numpy as np
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Mock the ``converter`` module BEFORE importing ``main``
#
# ``main`` does ``from converter import Converter, FEATURE_DIM, WINDOW_SIZE``
# at import time.  By inserting a fake module into sys.modules we avoid
# loading the real converter.py (which requires mediapipe).
# ---------------------------------------------------------------------------

DEFAULT_WINDOW = np.zeros((35, 1662), dtype=np.float32)

_mock_converter_module = types.ModuleType("converter")
_mock_converter_module.FEATURE_DIM = 1662
_mock_converter_module.WINDOW_SIZE = 35


class _MockConverter:
    """Stand-in that returns dummy keypoints / window without MediaPipe."""

    def __init__(self):
        self.window = DEFAULT_WINDOW.copy()
        self.current_length = 0

    def point_detection(self, image_bytes: bytes) -> np.ndarray:
        """Return a dummy 1662-dim keypoint vector (ignores input)."""
        return np.zeros(1662, dtype=np.float32)

    def process_new_frame(self, frame_vector: np.ndarray) -> np.ndarray:
        """Slide the window and return its current state."""
        if self.current_length < 35:
            self.window[self.current_length] = frame_vector
            self.current_length += 1
        else:
            self.window[:-1] = self.window[1:]
            self.window[-1] = frame_vector
        return self.window

    def stop(self) -> np.ndarray:
        return self.window


_mock_converter_module.Converter = _MockConverter

# Inject before main.py is imported
sys.modules["converter"] = _mock_converter_module

import main as svc

client = TestClient(svc.app)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_state():
    """Reset session state and motion detector between tests."""
    svc.session_states.clear()
    svc.motion_detector.reset()
    # Reset the mock converter's window too
    svc.converter.window = DEFAULT_WINDOW.copy()
    svc.converter.current_length = 0
    yield


@pytest.fixture
def valid_jpeg_bytes() -> bytes:
    """Generate a tiny valid JPEG using OpenCV (guaranteed decodable)."""
    import cv2
    gray = np.ones((2, 2), dtype=np.uint8) * 128
    ok, buf = cv2.imencode(".jpg", gray)
    assert ok
    return buf.tobytes()


@pytest.fixture
def frame_req(valid_jpeg_bytes) -> dict:
    return {
        "uuid": "test-session-1",
        "image_bytes": base64.b64encode(valid_jpeg_bytes).decode("utf-8"),
        "timestamp_ms": 0,
    }


# ===================================================================
# Health
# ===================================================================

class TestHealthEndpoint:

    def test_health_ok(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "healthy", "service": "sign-segmentation"}


# ===================================================================
# /process-frame
# ===================================================================

class TestProcessFrame:

    # -- Happy path ------------------------------------------------------

    def test_first_frame_returns_processing(self, frame_req):
        """The very first frame should return 'processing'."""
        with patch.object(svc.motion_detector, 'update',
                          return_value=(False, None)):
            r = client.post("/process-frame", json=frame_req)
        assert r.status_code == 200
        assert r.json() == {"status": "processing", "frame_processed": True}

    def test_sign_ended_returns_full_result(self, frame_req):
        """When a sign ends the endpoint returns SignResult with window."""
        fake_sign = [np.zeros(1662, dtype=np.float32) for _ in range(5)]

        with patch.object(svc.motion_detector, 'update',
                          return_value=(True, fake_sign)):
            r = client.post("/process-frame", json=frame_req)

        assert r.status_code == 200
        data = r.json()
        assert data["sign_index"] == 0
        assert len(data["keypoints_sequence"]) == 5
        assert "window" in data
        assert isinstance(data["window"], list)
        assert len(data["window"]) == 35  # WINDOW_SIZE
        for frame in data["window"]:
            assert len(frame) == 1662    # FEATURE_DIM

    def test_sign_index_increments(self, frame_req):
        fake_sign = [np.zeros(1662, dtype=np.float32) for _ in range(3)]

        for idx in range(2):
            with patch.object(svc.motion_detector, 'update',
                              return_value=(True, fake_sign)):
                r = client.post("/process-frame", json=frame_req)
            assert r.status_code == 200
            assert r.json()["sign_index"] == idx

    def test_converter_process_new_frame_called(self, frame_req):
        """Verify that converter.process_new_frame is wired up."""
        original = svc.converter.process_new_frame
        called = False

        def tracking_fn(kp):
            nonlocal called
            called = True
            return original(kp)

        svc.converter.process_new_frame = tracking_fn
        with patch.object(svc.motion_detector, 'update',
                          return_value=(False, None)):
            client.post("/process-frame", json=frame_req)
        assert called, "converter.process_new_frame() was never called"

    # -- Error paths -----------------------------------------------------

    def test_invalid_base64_returns_500(self):
        """Base64 decode failure is caught and returns 500."""
        r = client.post("/process-frame", json={
            "uuid": "t", "image_bytes": "!!!not-base64!!!",
        })
        assert r.status_code == 500

    def test_missing_uuid_returns_422(self, valid_jpeg_bytes):
        r = client.post("/process-frame", json={
            "image_bytes": base64.b64encode(valid_jpeg_bytes).decode(),
        })
        assert r.status_code == 422

    def test_internal_error_returns_500(self, frame_req):
        with patch.object(svc.motion_detector, 'update',
                          side_effect=RuntimeError("boom")):
            r = client.post("/process-frame", json=frame_req)
        assert r.status_code == 500

    # -- Session isolation -----------------------------------------------

    def test_multiple_sessions_independent(self, valid_jpeg_bytes):
        fake_sign = [np.zeros(1662, dtype=np.float32) for _ in range(3)]
        img = base64.b64encode(valid_jpeg_bytes).decode()

        with patch.object(svc.motion_detector, 'update',
                          return_value=(True, fake_sign)):
            r1 = client.post("/process-frame", json={"uuid": "A", "image_bytes": img})
            r2 = client.post("/process-frame", json={"uuid": "B", "image_bytes": img})

        assert r1.json()["sign_index"] == 0
        assert r2.json()["sign_index"] == 0

    def test_frame_count_accumulates(self, frame_req):
        with patch.object(svc.motion_detector, 'update',
                          return_value=(False, None)):
            client.post("/process-frame", json=frame_req)
            client.post("/process-frame", json=frame_req)

        assert svc.session_states["test-session-1"]["total_frames"] == 2


# ===================================================================
# /end-sequence
# ===================================================================

class TestEndSequence:

    def test_end_active_session(self, valid_jpeg_bytes):
        img = base64.b64encode(valid_jpeg_bytes).decode()
        with patch.object(svc.motion_detector, 'update',
                          return_value=(False, None)):
            client.post("/process-frame", json={"uuid": "s1", "image_bytes": img})

        r = client.post("/end-sequence", params={"uuid": "s1"})
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "sequence ended"
        assert data["uuid"] == "s1"
        assert "final_window" in data
        assert "s1" not in svc.session_states

    def test_end_nonexistent_session(self):
        r = client.post("/end-sequence", params={"uuid": "ghost"})
        assert r.status_code == 200
        assert r.json() == {"status": "no active session", "uuid": "ghost"}
