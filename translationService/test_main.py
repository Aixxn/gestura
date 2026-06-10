"""
Tests for the Gesture Translation Service (merged segmentation + ML + Groq).

Run with: pytest test_main.py -v
"""

import sys
import types
import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, PropertyMock
import os

# ---------------------------------------------------------------------------
# Mock keras.models.load_model BEFORE importing ``main`` so model stays None
# ---------------------------------------------------------------------------
import keras  # noqa: E402
_original_load = keras.models.load_model
keras.models.load_model = lambda path: None

# ---------------------------------------------------------------------------
# Mock the ``converter`` module BEFORE importing ``main``
# ---------------------------------------------------------------------------

DEFAULT_WINDOW = np.zeros((35, 258), dtype=np.float32)

_mock_converter_module = types.ModuleType("converter")
_mock_converter_module.FEATURE_DIM = 258
_mock_converter_module.WINDOW_SIZE = 35


class _MockConverter:
    def __init__(self):
        self.window = DEFAULT_WINDOW.copy()
        self.current_length = 0
        self._lh_lost_counter = 0
        self._rh_lost_counter = 0

    def point_detection(self, image_bytes: bytes) -> np.ndarray:
        return np.zeros(258, dtype=np.float32)

    def process_new_frame(self, frame_vector: np.ndarray) -> np.ndarray:
        if self.current_length < 35:
            self.window[self.current_length] = frame_vector
            self.current_length += 1
        else:
            self.window[:-1] = self.window[1:]
            self.window[-1] = frame_vector
        return self.window

    def stop(self) -> np.ndarray:
        return self.window

    def get_persisted_keypoints(self) -> np.ndarray:
        return np.zeros(258, dtype=np.float32)

    def _build_unified_kp(self) -> np.ndarray:
        return np.zeros(258, dtype=np.float32)


_mock_converter_module.Converter = _MockConverter
sys.modules["converter"] = _mock_converter_module

import main as svc

client = TestClient(svc.app)

# ---------------------------------------------------------------------------
# Global fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_state():
    svc.session_states.clear()
    svc.motion_detector.reset()
    svc.converter.window = DEFAULT_WINDOW.copy()
    svc.converter.current_length = 0
    yield


@pytest.fixture
def mock_image_bytes() -> str:
    """Return a base64-encoded fake JPEG."""
    import base64
    return base64.b64encode(b"fake-jpeg-bytes").decode("utf-8")


# ===================================================================
# normalize_frames
# ===================================================================

class TestNormalizeFrames:
    def test_empty_list(self):
        from normalize import normalize_frames
        assert normalize_frames([], 35) == []

    def test_exact_match(self):
        from normalize import normalize_frames
        frames = [[1.0]] * 35
        result = normalize_frames(frames, 35)
        assert result == frames

    def test_padding_short_to_target(self):
        from normalize import normalize_frames
        frames = [[1.0], [2.0], [3.0]]
        result = normalize_frames(frames, 5)
        assert len(result) == 5
        assert result[3] == [3.0]
        assert result[4] == [3.0]

    def test_downsampling_long_to_target(self):
        from normalize import normalize_frames
        frames = [[float(i)] for i in range(100)]
        result = normalize_frames(frames, 10)
        assert len(result) == 10
        assert result[0] == [0.0]
        assert result[9] == [99.0]

    def test_preserves_feature_dimension(self):
        from normalize import normalize_frames
        frames = [list(range(258)) for _ in range(5)]
        result = normalize_frames(frames, 35)
        assert len(result) == 35
        for frame in result:
            assert len(frame) == 258


# ===================================================================
# ASL Grammar Fixer
# ===================================================================

class TestASLGrammarFixer:
    @patch('main.Groq')
    def test_init_with_api_key(self, mock_groq):
        fixer = svc.ASLGrammarFixer(api_key="test_key")
        mock_groq.assert_called_once_with(api_key="test_key")

    @patch.dict(os.environ, {'GROQ_API_KEY': 'env_key'})
    @patch('main.Groq')
    def test_init_with_env_var(self, mock_groq):
        fixer = svc.ASLGrammarFixer()
        mock_groq.assert_called_once_with(api_key="env_key")

    @patch('main.Groq')
    def test_fix_grammar_success(self, mock_groq):
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "I am hungry."
        mock_client.chat.completions.create.return_value = mock_response
        mock_groq.return_value = mock_client
        fixer = svc.ASLGrammarFixer(api_key="test_key")
        result = fixer.fix_grammar("ME HUNGRY")
        assert result == "I am hungry."


# ===================================================================
# Health
# ===================================================================

class TestHealth:
    def test_health_ok(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"


# ===================================================================
# /process-frame
# ===================================================================

class TestProcessFrame:
    def _make_session_with_mock_md(self, uuid: str, return_value, side_effect=None):
        """Pre-populate session state with a mocked motion detector."""
        md = Mock()
        if side_effect:
            md.update.side_effect = side_effect
        else:
            md.update.return_value = return_value
        svc.session_states[uuid] = {
            "motion_detector": md,
            "predicted_words": [],
            "sign_count": 0,
        }

    def test_processing_response(self, mock_image_bytes):
        uuid = "proc-test-uuid"
        self._make_session_with_mock_md(uuid, (False, None))
        r = client.post("/process-frame", json={
            "uuid": uuid,
            "image_bytes": mock_image_bytes,
        })
        assert r.status_code == 200
        assert r.json()["status"] == "processing"

    def test_word_detected_response(self, mock_image_bytes):
        uuid = "word-test-uuid"
        fake_sign = [np.zeros(258, dtype=np.float32) for _ in range(5)]
        self._make_session_with_mock_md(uuid, (True, fake_sign))
        r = client.post("/process-frame", json={
            "uuid": uuid,
            "image_bytes": mock_image_bytes,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "word_detected"
        assert data["word"] == "sign_detected"
        assert data["sign_index"] == 0

    def test_sign_index_increments(self, mock_image_bytes):
        uuid = "inc-test-uuid"
        fake_sign = [np.zeros(258, dtype=np.float32) for _ in range(3)]
        self._make_session_with_mock_md(uuid, (True, fake_sign))
        for i in range(3):
            r = client.post("/process-frame", json={
                "uuid": uuid,
                "image_bytes": mock_image_bytes,
            })
            assert r.json()["sign_index"] == i

    def test_accumulates_words(self, mock_image_bytes):
        uuid = "acc-test-uuid"
        fake_sign = [np.zeros(258, dtype=np.float32) for _ in range(3)]
        self._make_session_with_mock_md(uuid, (True, fake_sign))
        for _ in range(2):
            client.post("/process-frame", json={
                "uuid": uuid,
                "image_bytes": mock_image_bytes,
            })
        session = svc.session_states.get(uuid)
        assert session is not None
        assert len(session["predicted_words"]) == 2

    def test_error_returns_500(self, mock_image_bytes):
        uuid = "err-test-uuid"
        self._make_session_with_mock_md(uuid, None,
                                        side_effect=RuntimeError("boom"))
        r = client.post("/process-frame", json={
            "uuid": uuid,
            "image_bytes": mock_image_bytes,
        })
        assert r.status_code == 500


# ===================================================================
# /stop
# ===================================================================

class TestStop:
    def test_no_session_returns_404(self):
        r = client.post("/stop", json={"uuid": "nonexistent"})
        assert r.status_code == 404

    def test_no_words_returns_empty(self):
        svc.session_states["empty-uuid"] = {
            "motion_detector": Mock(),
            "predicted_words": [],
            "sign_count": 0,
        }
        r = client.post("/stop", json={"uuid": "empty-uuid"})
        assert r.status_code == 200
        assert r.json()["words"] == []
        assert r.json()["asl_gloss"] == ""

    @patch.object(svc, 'grammar_fixer')
    def test_returns_translation(self, mock_fixer):
        mock_fixer.fix_grammar.return_value = "Hello world"
        svc.session_states["full-uuid"] = {
            "motion_detector": Mock(),
            "predicted_words": ["hello", "world"],
            "sign_count": 2,
        }
        r = client.post("/stop", json={"uuid": "full-uuid"})
        assert r.status_code == 200
        data = r.json()
        assert data["asl_gloss"] == "hello world"
        assert data["english"] == "Hello world"
        assert data["words"] == ["hello", "world"]
        assert data["success"] is True

    @patch.object(svc, 'grammar_fixer')
    def test_grammar_failure_fallback(self, mock_fixer):
        mock_fixer.fix_grammar.side_effect = Exception("LLM down")
        svc.session_states["fail-uuid"] = {
            "motion_detector": Mock(),
            "predicted_words": ["hello"],
            "sign_count": 1,
        }
        r = client.post("/stop", json={"uuid": "fail-uuid"})
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is False
        assert data["english"] == "hello"


# ===================================================================
# /convert-sentence (standalone grammar fix)
# ===================================================================

class TestConvertSentence:
    @patch('main.grammar_fixer.fix_grammar')
    def test_success(self, mock_fix):
        mock_fix.return_value = "I am hungry."
        r = client.post("/convert-sentence", json={"asl_gloss": "ME HUNGRY"})
        assert r.status_code == 200
        assert r.json()["translated"] == "I am hungry."

    @patch('main.grammar_fixer.fix_grammar')
    def test_error_fallback(self, mock_fix):
        mock_fix.side_effect = Exception("LLM Error")
        r = client.post("/convert-sentence", json={"asl_gloss": "ME HUNGRY"})
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is False
        assert data["translated"] == "ME HUNGRY"

    def test_invalid_request_returns_422(self):
        r = client.post("/convert-sentence", json={"wrong": "data"})
        assert r.status_code == 422
