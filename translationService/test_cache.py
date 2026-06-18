"""
Tests for the Semantic Cache mechanism.

Unit tests: SemanticCache class in isolation.
Integration tests: Cache + ASLGrammarFixer + HTTP endpoints.

Run with: pytest test_cache.py -v
"""

import sys
import types
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

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

    def point_detection(self, image_bytes: bytes) -> np.ndarray:
        return np.zeros(258, dtype=np.float32)

    def get_persisted_keypoints(self) -> np.ndarray:
        return np.zeros(258, dtype=np.float32)

    def _build_unified_kp(self) -> np.ndarray:
        return np.zeros(258, dtype=np.float32)


_mock_converter_module.Converter = _MockConverter
sys.modules["converter"] = _mock_converter_module

import main as svc

client = TestClient(svc.app)


# ===================================================================
# Helpers
# ===================================================================

def _make_embedder(embeddings: list[np.ndarray]):
    """Create a mock embedder that returns pre-defined embeddings in sequence."""
    mock = type("MockEmbedder", (), {})()
    mock.encode = lambda text: embeddings.pop(0) if embeddings else np.zeros(384)
    return mock


# ===================================================================
# Unit Tests — SemanticCache in isolation
# ===================================================================

class TestSemanticCacheUnit:
    """Pure unit tests for the SemanticCache class with mocked embedder."""

    # --- Initialization ---

    def test_init_default_threshold(self):
        cache = svc.SemanticCache()
        assert cache.threshold == 0.85
        assert cache.size == 0
        assert cache.available is True

    def test_init_custom_threshold(self):
        cache = svc.SemanticCache(threshold=0.95)
        assert cache.threshold == 0.95

    def test_init_with_env_threshold(self, monkeypatch):
        monkeypatch.setenv("CACHE_SIMILARITY_THRESHOLD", "0.90")
        fixer = svc.ASLGrammarFixer()
        assert fixer.cache.threshold == 0.90

    # --- Empty cache ---

    def test_lookup_empty_cache_returns_none(self):
        cache = svc.SemanticCache()
        assert cache.lookup("HELLO HOW YOU") is None

    def test_lookup_none_text_returns_none(self):
        cache = svc.SemanticCache()
        assert cache.lookup("") is None

    # --- Store and exact match ---

    def test_store_then_exact_match_hits(self):
        cache = svc.SemanticCache(threshold=0.8)
        emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        with patch.object(cache, "_get_embedder") as mock_emb:
            mock_emb.return_value.encode.return_value = emb
            cache.store("HELLO HOW YOU", "Hello, how are you?")
            mock_emb.return_value.encode.return_value = emb
            result = cache.lookup("HELLO HOW YOU")
            assert result == "Hello, how are you?"

    def test_store_empty_text_does_not_raise(self):
        cache = svc.SemanticCache()
        with patch.object(cache, "_get_embedder") as mock_emb:
            mock_emb.return_value.encode.return_value = np.zeros(384)
            cache.store("", "")  # should not raise

    # --- Semantic matching ---

    def test_semantic_near_match_hits(self):
        cache = svc.SemanticCache(threshold=0.8)
        emb_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        emb_b = np.array([0.9, 0.1, 0.0], dtype=np.float32)  # ~0.994 cosine sim
        with patch.object(cache, "_get_embedder") as mock_emb:
            mock_emb.return_value.encode.side_effect = [emb_a, emb_b]
            cache.store("HELLO HOW YOU", "Hello, how are you?")
            result = cache.lookup("HI HOW ARE YOU")
            assert result == "Hello, how are you?"

    def test_different_input_misses(self):
        cache = svc.SemanticCache(threshold=0.9)
        emb_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        emb_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # orthogonal → 0 sim
        with patch.object(cache, "_get_embedder") as mock_emb:
            mock_emb.return_value.encode.side_effect = [emb_a, emb_b]
            cache.store("HELLO HOW YOU", "Hello, how are you?")
            result = cache.lookup("I WANT WATER")
            assert result is None

    # --- Threshold boundary ---

    def test_exactly_at_threshold_hits(self):
        cache = svc.SemanticCache(threshold=0.90)
        emb_a = np.array([1.0, 0.0], dtype=np.float32)
        emb_b = np.array([0.9, np.sqrt(1 - 0.9**2)], dtype=np.float32)  # exactly 0.90
        with patch.object(cache, "_get_embedder") as mock_emb:
            mock_emb.return_value.encode.side_effect = [emb_a, emb_b]
            cache.store("HELLO", "Hello")
            result = cache.lookup("HI")
            assert result == "Hello"

    def test_barely_below_threshold_misses(self):
        cache = svc.SemanticCache(threshold=0.91)
        emb_a = np.array([1.0, 0.0], dtype=np.float32)
        emb_b = np.array([0.9, np.sqrt(1 - 0.9**2)], dtype=np.float32)  # exactly 0.90
        with patch.object(cache, "_get_embedder") as mock_emb:
            mock_emb.return_value.encode.side_effect = [emb_a, emb_b]
            cache.store("HELLO", "Hello")
            result = cache.lookup("HI")
            assert result is None

    # --- Best match selection ---

    def test_picks_best_match_among_multiple(self):
        cache = svc.SemanticCache(threshold=0.5)
        emb_query = np.array([0.0, 1.0], dtype=np.float32)
        emb_far = np.array([1.0, 0.0], dtype=np.float32)    # sim = 0.0
        emb_close = np.array([0.1, 0.99], dtype=np.float32)  # sim = ~0.99
        with patch.object(cache, "_get_embedder") as mock_emb:
            mock_emb.return_value.encode.side_effect = [
                emb_far, emb_close,  # store calls
                emb_query,            # lookup call
            ]
            cache.store("I WANT WATER", "I want water.")
            cache.store("HI", "Hi there!")
            result = cache.lookup("HELLO")
            assert result == "Hi there!"

    # --- Fallback behavior ---

    def test_not_available_returns_none_gracefully(self):
        cache = svc.SemanticCache()
        cache.available = False
        assert cache.lookup("HELLO") is None
        cache.store("HELLO", "Hi")  # should not raise

    def test_embedder_failure_on_lookup_returns_none(self):
        cache = svc.SemanticCache(threshold=0.8)
        emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        with patch.object(cache, "_get_embedder") as mock_emb:
            mock_emb.return_value.encode.side_effect = [emb, RuntimeError("fail")]
            cache.store("HELLO", "Hi")
            result = cache.lookup("HELLO")
            assert result is None

    def test_embedder_failure_on_store_does_not_raise(self):
        cache = svc.SemanticCache()
        with patch.object(cache, "_get_embedder") as mock_emb:
            mock_emb.return_value.encode.side_effect = RuntimeError("fail")
            cache.store("HELLO", "Hi")

    # --- Size tracking ---

    def test_size_tracks_entries_correctly(self):
        cache = svc.SemanticCache(threshold=0.8)
        emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        with patch.object(cache, "_get_embedder") as mock_emb:
            mock_emb.return_value.encode.return_value = emb
            assert cache.size == 0
            cache.store("A", "a")
            assert cache.size == 1
            cache.store("B", "b")
            assert cache.size == 2
            cache.store("C", "c")
            assert cache.size == 3

    def test_size_is_thread_safe(self):
        cache = svc.SemanticCache(threshold=0.8)
        emb = np.array([1.0, 0.0], dtype=np.float32)
        with patch.object(cache, "_get_embedder") as mock_emb:
            mock_emb.return_value.encode.return_value = emb

            def store_item(i):
                cache.store(f"WORD{i}", f"word{i}")

            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(store_item, i) for i in range(50)]
                for f in as_completed(futures):
                    f.result()

            assert cache.size == 50

    # --- State integrity ---

    def test_lookup_does_not_modify_cache(self):
        cache = svc.SemanticCache(threshold=0.8)
        emb = np.array([1.0, 0.0], dtype=np.float32)
        with patch.object(cache, "_get_embedder") as mock_emb:
            mock_emb.return_value.encode.return_value = emb
            cache.store("HELLO", "Hello")
            assert cache.size == 1
            cache.lookup("HELLO")
            cache.lookup("HI")
            cache.lookup("BYE")
            assert cache.size == 1  # still 1, not modified by lookups


# ===================================================================
# Integration Tests — Cache + ASLGrammarFixer
# ===================================================================

class TestCacheGrammarFixerIntegration:
    """Tests the cache integrated with ASLGrammarFixer (mocked Flan-T5)."""

    @pytest.fixture
    def mock_flan5(self):
        """Fixture providing fake tokenizer + model for Flan-T5."""
        class FakeInputs(dict):
            def to(self, _device):
                return self

        class FakeTokenizer:
            def __call__(self, prompt, **kwargs):
                assert "translate ASL gloss to English:" in prompt
                assert kwargs.get("return_tensors") == "pt"
                return FakeInputs({"input_ids": [1]})

            def decode(self, output_ids, skip_special_tokens=True):
                return "I am hungry."

        class FakeModel:
            device = "cpu"
            def generate(self, **kwargs):
                assert kwargs.get("max_new_tokens") == 100
                return [[101]]

        return FakeTokenizer(), FakeModel()

    # --- Cache hit behavior ---

    def test_cache_hit_returns_without_calling_model(self):
        """When cache hits, _load_model should never be called."""
        fixer = svc.ASLGrammarFixer()
        with patch.object(fixer.cache, "lookup") as mock_lookup, \
             patch.object(fixer, "_load_model") as mock_load:
            mock_lookup.return_value = "Hello, how are you?"
            result = fixer.fix_grammar("HELLO HOW YOU")
            assert result == "Hello, how are you?"
            mock_load.assert_not_called()

    def test_cache_hit_with_similar_gloss(self):
        """Similar glosses produce a cache hit (real cache, mocked embedder)."""
        fixer = svc.ASLGrammarFixer()
        with patch.object(fixer.cache, "_get_embedder") as mock_emb:
            # Store one entry
            emb_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            mock_emb.return_value.encode.return_value = emb_a
            fixer.cache.store("HELLO HOW YOU", "Hello, how are you?")

            # Query with similar but different gloss
            emb_b = np.array([0.95, 0.1, 0.0], dtype=np.float32)
            mock_emb.return_value.encode.return_value = emb_b
            with patch.object(fixer, "_load_model") as mock_load:
                result = fixer.fix_grammar("HI HOW ARE YOU")
                assert result == "Hello, how are you?"
                mock_load.assert_not_called()

    # --- Cache miss behavior ---

    def test_cache_miss_invokes_model_and_stores_result(self, mock_flan5):
        """Cache miss should call Flan-T5 and store the result."""
        fixer = svc.ASLGrammarFixer()
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(fixer, "_load_model", lambda: mock_flan5)

        with patch.object(fixer.cache, "lookup", return_value=None), \
             patch.object(fixer.cache, "store") as mock_store, \
             patch.object(fixer.cache, "_get_embedder") as mock_emb:
            mock_emb.return_value.encode.return_value = np.array([1.0, 0.0])

            result = fixer.fix_grammar("ME HUNGRY")
            assert result == "I am hungry."
            mock_store.assert_called_once_with("ME HUNGRY", "I am hungry.")

    def test_cache_miss_then_hit_for_duplicate(self):
        """Same input twice: first miss, second hit (no model call on second)."""
        fixer = svc.ASLGrammarFixer()

        # We'll use real cache store/lookup but mock the embedder and the model
        emb = np.array([1.0, 0.0], dtype=np.float32)

        class FakeTokenizer:
            def __call__(self, *_a, **_kw):
                return {"input_ids": [1]}
            def decode(self, *_a, **_kw):
                return "I am hungry."

        class FakeModel:
            device = "cpu"
            def generate(self, **_kw):
                return [[101]]

        with patch.object(fixer.cache, "_get_embedder") as mock_emb, \
             patch.object(fixer, "_load_model") as mock_load:
            # First call: cache miss, model loads
            mock_emb.return_value.encode.return_value = emb
            mock_load.return_value = (FakeTokenizer(), FakeModel())

            result1 = fixer.fix_grammar("ME HUNGRY")
            assert result1 == "I am hungry."
            assert mock_load.call_count == 1

            # Second call: cache hit, model NOT called
            mock_load.reset_mock()
            mock_emb.return_value.encode.return_value = emb
            result2 = fixer.fix_grammar("ME HUNGRY")
            assert result2 == "I am hungry."
            mock_load.assert_not_called()

    # --- Invalid output not cached ---

    def test_invalid_output_not_stored(self):
        """If Flan-T5 returns invalid (all-caps), cache.store should not be called."""
        class FakeTokenizer:
            def __call__(self, *_a, **_kw):
                return {"input_ids": [1]}
            def decode(self, *_a, **_kw):
                return "I DRINK WATER"  # all-caps → invalid

        class FakeModel:
            device = "cpu"
            def generate(self, **_kw):
                return [[101]]

        fixer = svc.ASLGrammarFixer()
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(fixer, "_load_model", lambda: (FakeTokenizer(), FakeModel()))

        with patch.object(fixer.cache, "lookup", return_value=None), \
             patch.object(fixer.cache, "store") as mock_store:
            with pytest.raises(ValueError, match="invalid English output"):
                fixer.fix_grammar("I DRINK WATER")
            mock_store.assert_not_called()

    # --- Empty input ---

    def test_empty_input_skips_cache_and_model(self):
        fixer = svc.ASLGrammarFixer()
        with patch.object(fixer.cache, "lookup") as mock_lookup, \
             patch.object(fixer, "_load_model") as mock_load:
            result = fixer.fix_grammar("")
            assert result == ""
            mock_lookup.assert_not_called()
            mock_load.assert_not_called()

    def test_whitespace_input_skips_cache_and_model(self):
        fixer = svc.ASLGrammarFixer()
        with patch.object(fixer.cache, "lookup") as mock_lookup, \
             patch.object(fixer, "_load_model") as mock_load:
            result = fixer.fix_grammar("   ")
            assert result == ""
            mock_lookup.assert_not_called()
            mock_load.assert_not_called()

    # --- Cache accumulates across calls ---

    def test_cache_accumulates_across_multiple_calls(self):
        """Multiple unique inputs should all be cached."""
        # Each input gets a distinct embedding so they don't collide in cache
        embeddings = {
            "ME HUNGRY": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "ME WANT WATER": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "HELLO HOW YOU": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }
        # Outputs returned in order of first calls
        outputs = iter([
            "I am hungry.",
            "I want water.",
            "Hello, how are you?",
        ])

        class FakeTokenizer:
            def __call__(self, *_a, **_kw):
                return {"input_ids": [1]}
            def decode(self, *_a, **_kw):
                return next(outputs)

        class FakeModel:
            device = "cpu"
            def generate(self, **_kw):
                return [[101]]

        fixer = svc.ASLGrammarFixer()
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(fixer, "_load_model", lambda: (FakeTokenizer(), FakeModel()))

        inputs = [
            ("ME HUNGRY", "I am hungry."),
            ("ME WANT WATER", "I want water."),
            ("HELLO HOW YOU", "Hello, how are you?"),
        ]

        # All unique inputs → each should miss cache, call model, store result
        with patch.object(fixer.cache, "_get_embedder") as mock_emb:
            def encode_side_effect(text):
                return embeddings.get(text, np.array([0.0, 0.0, 1.0], dtype=np.float32))
            mock_emb.return_value.encode.side_effect = encode_side_effect

            for gloss, expected in inputs:
                result = fixer.fix_grammar(gloss)
                assert result == expected, f"Expected '{expected}' for '{gloss}', got '{result}'"

            assert fixer.cache.size == 3

            # Second round: all should hit cache
            with patch.object(fixer, "_load_model") as mock_load:
                for gloss, expected in inputs:
                    result = fixer.fix_grammar(gloss)
                    assert result == expected, f"Cache should return '{expected}' for '{gloss}', got '{result}'"
                mock_load.assert_not_called()


# ===================================================================
# Integration Tests — Cache + HTTP Endpoints
# ===================================================================

class TestCacheHttpIntegration:
    """Tests the cache through the actual HTTP endpoints with mocked Flan-T5."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Clear session state and reset grammar_fixer cache before each test."""
        svc.session_states.clear()
        svc.grammar_fixer.cache = svc.SemanticCache(threshold=0.85)
        # Mock Flan-T5 model loading so tests don't need the real model
        self._patcher = patch.object(svc.grammar_fixer, "_load_model")
        self._mock_load = self._patcher.start()
        self._setup_fake_model()
        yield
        self._patcher.stop()

    def _setup_fake_model(self):
        class FakeInputs(dict):
            def to(self, _device):
                return self

        class FakeTokenizer:
            def __call__(self, *_a, **_kw):
                return FakeInputs({"input_ids": [1]})
            def decode(self, *_a, **_kw):
                return "Hello, how are you?"

        class FakeModel:
            device = "cpu"
            def generate(self, **_kw):
                return [[101]]

        self._mock_load.return_value = (FakeTokenizer(), FakeModel())

    # --- /convert-sentence ---

    def test_convert_sentence_first_call_miss_second_hit(self):
        """First POST /convert-sentence should miss cache, second should hit."""
        emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        with patch.object(svc.grammar_fixer.cache, "_get_embedder") as mock_emb:
            mock_emb.return_value.encode.return_value = emb

            # First call — should call Flan-T5
            r1 = client.post("/convert-sentence", json={"asl_gloss": "HELLO HOW YOU"})
            assert r1.status_code == 200
            data1 = r1.json()
            assert data1["translated"] == "Hello, how are you?"
            assert data1["success"] is True
            assert self._mock_load.call_count == 1

            # Second call — should hit cache, NOT call Flan-T5
            self._mock_load.reset_mock()
            r2 = client.post("/convert-sentence", json={"asl_gloss": "HELLO HOW YOU"})
            assert r2.status_code == 200
            data2 = r2.json()
            assert data2["translated"] == "Hello, how are you?"
            self._mock_load.assert_not_called()

    def test_convert_sentence_similar_input_hits_cache(self):
        """Semantically similar input should hit cache via /convert-sentence."""
        with patch.object(svc.grammar_fixer.cache, "_get_embedder") as mock_emb:
            emb_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            emb_b = np.array([0.95, 0.1, 0.0], dtype=np.float32)  # sim ~0.994

            # Store in cache manually (simulating first call)
            mock_emb.return_value.encode.return_value = emb_a
            svc.grammar_fixer.cache.store("HELLO HOW YOU", "Hello, how are you?")

            # Similar input should hit cache
            mock_emb.return_value.encode.return_value = emb_b
            self._mock_load.reset_mock()
            r = client.post("/convert-sentence", json={"asl_gloss": "HI HOW ARE YOU"})
            assert r.status_code == 200
            assert r.json()["translated"] == "Hello, how are you?"
            self._mock_load.assert_not_called()

    def test_convert_sentence_different_input_misses(self):
        """Completely different input should miss cache, call Flan-T5."""
        with patch.object(svc.grammar_fixer.cache, "_get_embedder") as mock_emb:
            emb_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            emb_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # orthogonal

            # Store one entry
            mock_emb.return_value.encode.return_value = emb_a
            svc.grammar_fixer.cache.store("HELLO HOW YOU", "Hello, how are you?")

            # Different input → should miss
            mock_emb.return_value.encode.return_value = emb_b
            self._mock_load.reset_mock()
            r = client.post("/convert-sentence", json={"asl_gloss": "I WANT WATER"})
            assert r.status_code == 200
            assert self._mock_load.call_count == 1

    def test_convert_sentence_invalid_gloss_handled(self):
        """Invalid gloss (empty) should skip cache and return empty."""
        self._mock_load.reset_mock()
        r = client.post("/convert-sentence", json={"asl_gloss": ""})
        assert r.status_code == 200
        data = r.json()
        assert data["translated"] == ""
        assert data["success"] is True
        self._mock_load.assert_not_called()

    # --- /stop ---

    def test_stop_endpoint_uses_cache(self):
        """The /stop endpoint should benefit from the cache as well."""
        # Create a session and add words
        svc.session_states["test-uuid"] = {
            "predicted_words": ["HELLO", "HOW", "YOU"],
            "sign_count": 3,
        }

        emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        with patch.object(svc.grammar_fixer.cache, "_get_embedder") as mock_emb:
            mock_emb.return_value.encode.return_value = emb
            self._mock_load.reset_mock()

            r = client.post("/stop", json={"uuid": "test-uuid"})
            assert r.status_code == 200
            data = r.json()
            assert data["asl_gloss"] == "HELLO HOW YOU"
            assert data["english"] == "Hello, how are you?"
            assert data["success"] is True
            # Flan-T5 was called
            assert self._mock_load.call_count == 1

            # Second call with same gloss → should hit cache
            svc.session_states["test-uuid-2"] = {
                "predicted_words": ["HELLO", "HOW", "YOU"],
                "sign_count": 3,
            }
            self._mock_load.reset_mock()
            r2 = client.post("/stop", json={"uuid": "test-uuid-2"})
            assert r2.status_code == 200
            assert r2.json()["english"] == "Hello, how are you?"
            self._mock_load.assert_not_called()

    # --- Edge cases ---

    def test_cache_embedder_failure_falls_back(self):
        """If embedding model fails, cache becomes unavailable but Flan-T5 still works."""
        svc.grammar_fixer.cache.available = False
        self._mock_load.reset_mock()

        r = client.post("/convert-sentence", json={"asl_gloss": "HELLO HOW YOU"})
        assert r.status_code == 200
        assert r.json()["translated"] == "Hello, how are you?"
        assert self._mock_load.call_count == 1  # Flan-T5 still runs

    def test_cache_threshold_configurable_via_env(self, monkeypatch):
        """CACHE_SIMILARITY_THRESHOLD env var should be respected."""
        monkeypatch.setenv("CACHE_SIMILARITY_THRESHOLD", "0.99")
        # Re-initialize to pick up env var
        fixer = svc.ASLGrammarFixer()
        assert fixer.cache.threshold == 0.99
