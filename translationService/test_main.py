"""
Tests for the Gesture Translation Service (merged segmentation + ML + FLAN-T5).

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

_mock_converter_module = types.ModuleType("converter")
_mock_converter_module.FEATURE_DIM = 258
_mock_converter_module.WINDOW_SIZE = 35


class _MockConverter:
    def __init__(self):
        self._lh_lost_counter = 0
        self._rh_lost_counter = 0

    @property
    def is_idle(self) -> bool:
        """Mirrors the real Converter.is_idle: both hands absent for >= IDLE_THRESHOLD (15) frames."""
        return False  # default to not idle so tests reach the motion detector

    def point_detection(self, image_bytes: bytes) -> np.ndarray:
        return np.zeros(258, dtype=np.float32)

    def get_persisted_keypoints(self) -> np.ndarray:
        return np.zeros(258, dtype=np.float32)

    def get_raw_keypoints(self) -> np.ndarray:
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
    def test_init_uses_flan_t5_default(self):
        fixer = svc.ASLGrammarFixer()
        assert fixer.model_name in {"google/flan-t5-small", "models/flan-t5-asl-mini"}

    def test_init_uses_model_name_from_env(self, monkeypatch):
        monkeypatch.setenv("FLAN_T5_MODEL", "google/flan-t5-base")
        fixer = svc.ASLGrammarFixer()
        assert fixer.model_name == "google/flan-t5-base"

    def test_fix_grammar_success(self, monkeypatch):
        class FakeInputs(dict):
            def to(self, _device):
                return self

        class FakeTokenizer:
            def __call__(self, prompt, **kwargs):
                assert "translate ASL gloss to English:" in prompt
                assert "ME HUNGRY" in prompt
                assert kwargs["return_tensors"] == "pt"
                return FakeInputs({"input_ids": [1]})

            def decode(self, output_ids, skip_special_tokens=True):
                assert skip_special_tokens is True
                assert output_ids == [101]
                return '"I am hungry."'

        class FakeModel:
            device = "cpu"

            def generate(self, **kwargs):
                assert kwargs["input_ids"] == [1]
                assert kwargs["max_new_tokens"] == 100
                return [[101]]

        fixer = svc.ASLGrammarFixer()
        monkeypatch.setattr(
            fixer,
            "_load_model",
            lambda: (FakeTokenizer(), FakeModel()),
        )
        result = fixer.fix_grammar("ME HUNGRY")
        assert result == "I am hungry."

    def test_fix_grammar_rejects_echoed_simple_sentence(self, monkeypatch):
        class FakeInputs(dict):
            def to(self, _device):
                return self

        class FakeTokenizer:
            def __call__(self, *_args, **_kwargs):
                return FakeInputs({"input_ids": [1]})

            def decode(self, *_args, **_kwargs):
                return "I DRINK WATER"

        class FakeModel:
            device = "cpu"

            def generate(self, **_kwargs):
                return [[101]]

        fixer = svc.ASLGrammarFixer()
        monkeypatch.setattr(
            fixer,
            "_load_model",
            lambda: (FakeTokenizer(), FakeModel()),
        )

        with pytest.raises(ValueError, match="invalid English output"):
            fixer.fix_grammar("I DRINK WATER")

    def test_fix_grammar_accepts_model_structured_sentence(self, monkeypatch):
        class FakeInputs(dict):
            def to(self, _device):
                return self

        class FakeTokenizer:
            def __call__(self, *_args, **_kwargs):
                return FakeInputs({"input_ids": [1]})

            def decode(self, *_args, **_kwargs):
                return "I eat an apple."

        class FakeModel:
            device = "cpu"

            def generate(self, **_kwargs):
                return [[101]]

        fixer = svc.ASLGrammarFixer()
        monkeypatch.setattr(
            fixer,
            "_load_model",
            lambda: (FakeTokenizer(), FakeModel()),
        )

        assert fixer.fix_grammar("I EAT APPLE") == "I eat an apple."

    def test_fix_grammar_cleans_repeated_gloss_before_model(self, monkeypatch):
        class FakeInputs(dict):
            def to(self, _device):
                return self

        class FakeTokenizer:
            def __call__(self, prompt, **_kwargs):
                assert "PLEASE DRINK HOME MILK" in prompt
                assert "PLEASE PLEASE" not in prompt
                assert "MILK MILK" not in prompt
                return FakeInputs({"input_ids": [1]})

            def decode(self, *_args, **_kwargs):
                return "Please drink milk at home."

        class FakeModel:
            device = "cpu"

            def generate(self, **_kwargs):
                return [[101]]

        fixer = svc.ASLGrammarFixer()
        monkeypatch.setattr(
            fixer,
            "_load_model",
            lambda: (FakeTokenizer(), FakeModel()),
        )

        result = fixer.fix_grammar("PLEASE PLEASE DRINK HOME MILK MILK")
        assert result == "Please drink milk at home."

    def test_fix_grammar_rejects_gloss_like_model_output(self, monkeypatch):
        class FakeInputs(dict):
            def to(self, _device):
                return self

        class FakeTokenizer:
            def __call__(self, *_args, **_kwargs):
                return FakeInputs({"input_ids": [1]})

            def decode(self, *_args, **_kwargs):
                return "PLEASE DRINK HOME MILK"

        class FakeModel:
            device = "cpu"

            def generate(self, **_kwargs):
                return [[101]]

        fixer = svc.ASLGrammarFixer()
        monkeypatch.setattr(
            fixer,
            "_load_model",
            lambda: (FakeTokenizer(), FakeModel()),
        )

        with pytest.raises(ValueError, match="invalid English output"):
            fixer.fix_grammar("PLEASE PLEASE DRINK HOME MILK MILK")

    def test_fix_grammar_rejects_prompt_artifact(self, monkeypatch):
        class FakeInputs(dict):
            def to(self, _device):
                return self

        class FakeTokenizer:
            def __call__(self, *_args, **_kwargs):
                return FakeInputs({"input_ids": [1]})

            def decode(self, *_args, **_kwargs):
                return "ASL gloss: I DRINK WATER Structured English: I drink water."

        class FakeModel:
            device = "cpu"

            def generate(self, **_kwargs):
                return [[101]]

        fixer = svc.ASLGrammarFixer()
        monkeypatch.setattr(
            fixer,
            "_load_model",
            lambda: (FakeTokenizer(), FakeModel()),
        )

        with pytest.raises(ValueError, match="invalid English output"):
            fixer.fix_grammar("I DRINK WATER")


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
        words_iter = iter(["HELLO", "WORLD"])
        with patch("main._predict_word", side_effect=lambda kp: next(words_iter)):
            for _ in range(2):
                client.post("/process-frame", json={
                    "uuid": uuid,
                    "image_bytes": mock_image_bytes,
                })
        session = svc.session_states.get(uuid)
        assert session is not None
        assert len(session["predicted_words"]) == 2
        assert session["predicted_words"] == ["HELLO", "WORLD"]

    def test_deduplicates_consecutive_words(self, mock_image_bytes):
        uuid = "dedup-test-uuid"
        fake_sign = [np.zeros(258, dtype=np.float32) for _ in range(3)]
        self._make_session_with_mock_md(uuid, (True, fake_sign))
        with patch("main._predict_word", return_value="HELLO"):
            for _ in range(2):
                client.post("/process-frame", json={
                    "uuid": uuid,
                    "image_bytes": mock_image_bytes,
                })
        session = svc.session_states.get(uuid)
        assert session is not None
        assert len(session["predicted_words"]) == 1
        assert session["predicted_words"] == ["HELLO"]

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
