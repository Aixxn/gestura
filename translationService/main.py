import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

# ---- CUDA / XLA setup: find libdevice.10.bc so XLA can compile GPU kernels ----
# This must happen before any keras/tensorflow import.
_CUDA_CANDIDATES = [
    os.environ.get("CUDA_HOME"),
    os.environ.get("CUDA_ROOT"),
    os.environ.get("CUDA_TOOLKIT_ROOT_DIR"),
    "/usr/local/cuda",
    "/opt/cuda",
    "/usr/lib/cuda",
    "/usr/local/cuda-12",
    "/usr/local/cuda-11",
]
_found_cuda = None
for cand in _CUDA_CANDIDATES:
    if cand and os.path.isfile(os.path.join(cand, "nvvm", "libdevice", "libdevice.10.bc")):
        _found_cuda = cand
        break
if not _found_cuda:
    import subprocess
    try:
        _nsmi = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
        if _nsmi.returncode == 0:
            for _root in ["/usr/lib/cuda", "/usr/local/cuda"]:
                if os.path.isfile(os.path.join(_root, "nvvm", "libdevice", "libdevice.10.bc")):
                    _found_cuda = _root
                    break
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    if not _found_cuda:
        try:
            _nvvm = __import__("nvidia.cuda_nvvm", fromlist=[""])
            _pkg_path = os.path.dirname(_nvvm.__file__)
            _candidate = os.path.join(_pkg_path, "nvvm", "libdevice")
            if os.path.isfile(os.path.join(_candidate, "libdevice.10.bc")):
                _found_cuda = os.path.dirname(_candidate)
        except ImportError:
            pass

if _found_cuda:
    os.environ.setdefault("XLA_FLAGS", f"--xla_gpu_cuda_data_dir={_found_cuda}")
    print(f"[CUDA] Found at {_found_cuda}, set XLA_FLAGS")
else:
    print("[CUDA] libdevice.10.bc not found in common locations.")

import base64
import numpy as np
import keras
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv
from converter import Converter, FEATURE_DIM, WINDOW_SIZE
from motion_detector import MotionDetector
from normalize import normalize_frames

load_dotenv()

app = FastAPI(
    debug=True,
    title='Gesture Translation Service',
    description='Combined sign segmentation + ASL translation + local FLAN-T5 grammar correction',
)

# ---------------------------------------------------------------------------
# ASL Grammar Fixer
# ---------------------------------------------------------------------------

class ASLGrammarFixer:
    def __init__(self, model_name: str | None = None, max_new_tokens: int = 100):
        local_model_path = Path("models/flan-t5-asl-mini")
        default_model = str(local_model_path) if local_model_path.exists() else "google/flan-t5-small"
        self.model_name = model_name or os.getenv("FLAN_T5_MODEL", default_model)
        self.max_new_tokens = max_new_tokens
        self._tokenizer = None
        self._model = None
        self.prompt_template = (
            "translate ASL gloss to English: {asl_gloss}"
        )

    def _load_model(self):
        if self._tokenizer is None or self._model is None:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self._model.eval()
        return self._tokenizer, self._model

    def fix_grammar(self, asl_gloss: str) -> str:
        cleaned_gloss = self._clean_gloss(asl_gloss)
        if not cleaned_gloss:
            return ""

        try:
            tokenizer, model = self._load_model()
            prompt = self.prompt_template.format(asl_gloss=cleaned_gloss)
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            )

            if hasattr(inputs, "to"):
                inputs = inputs.to(model.device)

            output_ids = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=4,
                do_sample=False,
            )
            translated = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            translated = self._clean_output(translated)
            if self._is_acceptable_polish(cleaned_gloss, translated):
                return translated
            raise ValueError(f"FLAN-T5 returned invalid English output: {translated}")
        except Exception as e:
            print(f"FLAN-T5 Error: {e}")
            raise e

    def _clean_output(self, text: str) -> str:
        return text.strip().strip('"').strip("'").strip()

    def _clean_gloss(self, asl_gloss: str) -> str:
        tokens = self._collapse_repeated_tokens(self._tokenize_gloss(asl_gloss))
        return " ".join(tokens)

    def _normalize_for_compare(self, text: str) -> list[str]:
        cleaned = text.replace("?", " ").replace(".", " ").replace(",", " ")
        return [token.upper() for token in cleaned.split()]

    def _is_echo(self, asl_gloss: str, translated: str) -> bool:
        return self._normalize_for_compare(asl_gloss) == self._normalize_for_compare(translated)

    def _is_acceptable_polish(self, asl_gloss: str, translated: str) -> bool:
        if not translated:
            return False
        lowered = translated.lower()
        if "asl gloss:" in lowered or "structured english:" in lowered:
            return False
        has_letters = any(char.isalpha() for char in translated)
        if has_letters and translated.upper() == translated:
            return False
        return True

    def _tokenize_gloss(self, asl_gloss: str) -> list[str]:
        return [
            token.strip('.,?!;:"\'').upper()
            for token in asl_gloss.split()
            if token.strip('.,?!;:"\'')
        ]

    def _collapse_repeated_tokens(self, tokens: list[str]) -> list[str]:
        collapsed: list[str] = []
        for token in tokens:
            if collapsed and collapsed[-1] == token:
                continue
            collapsed.append(token)
        return collapsed

    def _rule_based_translate(self, asl_gloss: str) -> str:
        cleaned = " ".join(asl_gloss.split())
        if not cleaned:
            return ""

        is_question = cleaned.rstrip().endswith("?")
        tokens = self._collapse_repeated_tokens(self._tokenize_gloss(cleaned))
        if not tokens:
            return ""

        if tokens == ["ME", "HUNGRY", "EAT", "WANT"]:
            return "I am hungry and want to eat."
        if tokens == ["YESTERDAY", "ME", "GO", "STORE", "BUY", "MILK"]:
            return "Yesterday, I went to the store to buy milk."

        pronouns = {
            "ME": "I",
            "I": "I",
            "YOU": "you",
            "HE": "he",
            "SHE": "she",
            "WE": "we",
            "THEY": "they",
        }
        adjectives = {
            "HUNGRY": "hungry",
            "THIRSTY": "thirsty",
            "HAPPY": "happy",
            "SAD": "sad",
        }
        verbs = {
            "DRINK": "drink",
            "EAT": "eat",
            "LIKE": "like",
            "WANT": "want",
            "GO": "go",
            "BUY": "buy",
        }
        nouns = {
            "WATER": "water",
            "MILK": "milk",
            "COFFEE": "coffee",
            "APPLE": "an apple",
            "BANANA": "a banana",
            "STORE": "the store",
            "FOOD": "food",
        }
        locations = {
            "HOME": "at home",
            "SCHOOL": "at school",
            "STORE": "at the store",
        }

        if tokens[0] == "PLEASE":
            remaining = tokens[1:]
            verb_index = next((i for i, token in enumerate(remaining) if token in verbs), -1)
            if verb_index >= 0:
                verb = verbs[remaining[verb_index]]
                phrase_tokens = remaining[verb_index + 1:]
                object_parts = [
                    nouns.get(token, token.lower())
                    for token in phrase_tokens
                    if token not in locations
                ]
                location_parts = [
                    locations[token]
                    for token in phrase_tokens
                    if token in locations
                ]
                body_parts = [verb, *object_parts, *location_parts]
                return f"Please {' '.join(body_parts)}."

        subject_token = tokens[0]
        verb_token = tokens[1] if len(tokens) > 1 else ""
        object_tokens = tokens[2:]

        if subject_token in pronouns and verb_token in adjectives:
            subject = pronouns[subject_token]
            sentence = f"{subject} am {adjectives[verb_token]}."
            if subject != "I":
                sentence = f"{subject} is {adjectives[verb_token]}."
            return sentence[0].upper() + sentence[1:]

        if subject_token in pronouns and verb_token in verbs:
            subject = pronouns[subject_token]
            verb = verbs[verb_token]
            obj = " ".join(nouns.get(token, token.lower()) for token in object_tokens)

            if is_question and subject == "you":
                return f"Do you {verb}{(' ' + obj) if obj else ''}?"

            if subject in {"he", "she"} and verb not in {"go"}:
                verb = f"{verb}s"
            elif subject in {"he", "she"} and verb == "go":
                verb = "goes"

            sentence = f"{subject} {verb}{(' ' + obj) if obj else ''}."
            return sentence[0].upper() + sentence[1:]

        fallback = " ".join(token.lower() for token in tokens)
        return fallback.capitalize() + ("?" if is_question else ".")

grammar_fixer = ASLGrammarFixer()

# ---------------------------------------------------------------------------
# ML Model (optional — graceful fallback if not present)
# ---------------------------------------------------------------------------

MODEL_WINDOW_SIZE = WINDOW_SIZE  # 35 — same as segmentation's sliding window

_SERVICE_DIR = Path(__file__).resolve().parent
_MODEL_PATH = _SERVICE_DIR / "best_model.keras"
_CLASSES_PATH = _SERVICE_DIR / "sign_classes.npy"

try:
    model = keras.models.load_model(_MODEL_PATH)
    print(f"[TranslationService] Model loaded from '{_MODEL_PATH}' "
          f"(output classes: {model.output_shape[-1]})")
except Exception as e:
    print(f"[TranslationService] No model loaded ({e}) — using fallback predictions")
    model = None

NUM_CLASSES = model.output_shape[-1] if model else 30

WORD_MAPPING = []
try:
    WORD_MAPPING = list(np.load(_CLASSES_PATH, allow_pickle=True))
    if len(WORD_MAPPING) != NUM_CLASSES:
        print(f"[TranslationService] Warning: sign_classes.npy ({len(WORD_MAPPING)} classes)"
              f" doesn't match model ({NUM_CLASSES}) — using fallback labels")
        WORD_MAPPING = []
except Exception as e:
    print(f"[TranslationService] Could not load sign_classes.npy ({e}) — using fallback labels")

if not WORD_MAPPING:
    WORD_MAPPING = [f"word_{i}" for i in range(NUM_CLASSES)]

# ---------------------------------------------------------------------------
# Segmentation pipeline (singletons)
# ---------------------------------------------------------------------------

converter = Converter()
_MD_STILLNESS_FLOOR = float(os.getenv("MD_STILLNESS_FLOOR", "0.5"))
motion_detector = MotionDetector(
    low_factor=0.5,
    high_factor=4.0,
    still_frames_required=8,
    min_sign_duration=5,
    history_size=30,
    feature_dim=FEATURE_DIM,
    motion_smoothing=0.6,
    stillness_floor=_MD_STILLNESS_FLOOR,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

session_states: dict = {}

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class FrameRequest(BaseModel):
    uuid: str
    image_bytes: str
    timestamp_ms: Optional[int] = None

class FrameResponse(BaseModel):
    status: str
    word: Optional[str] = None
    sign_index: Optional[int] = None

class StopRequest(BaseModel):
    uuid: str

class StopResponse(BaseModel):
    asl_gloss: str
    english: str
    words: list[str]
    success: bool

class ConvertSentenceRequest(BaseModel):
    asl_gloss: str

class ConvertSentenceResponse(BaseModel):
    original: str
    translated: str
    success: bool
    error: str | None = None

class WindowInput(BaseModel):
    window_data: list[list[float]] = Field(
        ...,
        example=[[0.1] * FEATURE_DIM] * MODEL_WINDOW_SIZE,
    )

# ---------------------------------------------------------------------------
# Helper: run ML inference on a completed sign
# ---------------------------------------------------------------------------

def _predict_word(keypoints_sequence: list[list[float]]) -> str:
    normalized = normalize_frames(keypoints_sequence, MODEL_WINDOW_SIZE)
    if model is None:
        return "sign_detected"
    window_np = np.array(normalized, dtype=np.float32)
    inp = window_np[np.newaxis, ...]
    probs = model.predict(inp, verbose=0)
    idx = int(np.argmax(probs[0]))
    if idx >= len(WORD_MAPPING):
        return "unknown"
    return WORD_MAPPING[idx]

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/process-frame")
async def process_frame(request: FrameRequest):
    try:
        raw_bytes = base64.b64decode(request.image_bytes)
        keypoints = converter.point_detection(raw_bytes)

        session = session_states.setdefault(request.uuid, {
            "motion_detector": MotionDetector(
                low_factor=0.5,
                high_factor=4.0,
                still_frames_required=8,
                min_sign_duration=5,
                history_size=30,
                feature_dim=FEATURE_DIM,
                motion_smoothing=0.6,
                stillness_floor=_MD_STILLNESS_FLOOR,
            ),
            "predicted_words": [],
            "sign_count": 0,
        })

        md = session["motion_detector"]

        if converter.is_idle:
            md.reset()
            return FrameResponse(status="idle")


        sign_ended, completed_sign = md.update(keypoints)

        if sign_ended and completed_sign is not None:
            kp_list = [kp.tolist() for kp in completed_sign]
            word = _predict_word(kp_list)
            # Skip BACKGROUND predictions — they are not real signs
            if word == "BACKGROUND":
                return FrameResponse(status="background")

            sign_idx = session["sign_count"]
            session["sign_count"] += 1
            session["predicted_words"].append(word)
            return FrameResponse(
                status="word_detected",
                word=word,
                sign_index=sign_idx,
            )

        return FrameResponse(status="processing")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stop")
async def stop_session(request: StopRequest):
    session = session_states.pop(request.uuid, None)
    if not session:
        raise HTTPException(status_code=404, detail="No active session")

    words = session["predicted_words"]
    if not words:
        return StopResponse(
            asl_gloss="",
            english="",
            words=[],
            success=True,
        )

    asl_gloss = " ".join(words)
    try:
        english = grammar_fixer.fix_grammar(asl_gloss)
        return StopResponse(
            asl_gloss=asl_gloss,
            english=english,
            words=words,
            success=True,
        )
    except Exception as e:
        return StopResponse(
            asl_gloss=asl_gloss,
            english=asl_gloss,
            words=words,
            success=False,
            error=str(e),
        )


@app.post("/translate")
async def translate(data: WindowInput):
    if model is None:
        return {"error": "Model not loaded"}, 500

    window_list = data.window_data
    try:
        window_np = np.array(window_list)
    except ValueError:
        return {"error": "Input data structure is not uniform."}, 422

    if window_np.shape != (MODEL_WINDOW_SIZE, FEATURE_DIM):
        return {"error": f"Incorrect shape. Expected ({MODEL_WINDOW_SIZE}, {FEATURE_DIM}), got {window_np.shape}."}, 422

    inp = window_np.astype(np.float32)[np.newaxis, ...]
    probs = model.predict(inp, verbose=0)
    predicted_index = int(np.argmax(probs[0]))

    if predicted_index >= len(WORD_MAPPING):
        return {"error": "Prediction index out of bounds for word mapping."}, 500

    return {"pred": WORD_MAPPING[predicted_index]}


@app.post("/convert-sentence", response_model=ConvertSentenceResponse)
async def convert_sentence(request: ConvertSentenceRequest):
    try:
        translated = grammar_fixer.fix_grammar(request.asl_gloss)
        return ConvertSentenceResponse(
            original=request.asl_gloss,
            translated=translated,
            success=True,
        )
    except Exception as e:
        return ConvertSentenceResponse(
            original=request.asl_gloss,
            translated=request.asl_gloss,
            success=False,
            error=str(e),
        )


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "gesture-translation"}
