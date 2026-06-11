import os
import base64
import numpy as np
import keras
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from groq import Groq
from dotenv import load_dotenv
from converter import Converter, FEATURE_DIM, WINDOW_SIZE
from motion_detector import MotionDetector
from normalize import normalize_frames

load_dotenv()

app = FastAPI(
    debug=True,
    title='Gesture Translation Service',
    description='Combined sign segmentation + ASL translation + Groq grammar correction',
)

# ---------------------------------------------------------------------------
# ASL Grammar Fixer (unchanged)
# ---------------------------------------------------------------------------

class ASLGrammarFixer:
    def __init__(self, api_key: str = None):
        self.client = Groq(api_key=api_key or os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"
        self.system_prompt = """You are an expert in American Sign Language (ASL) grammar conversion.
Convert ASL gloss text (space-separated signs) into natural, grammatically correct English.

Rules:
- ASL uses topic-comment structure
- No verb conjugations in ASL
- Directional verbs indicate subject/object
- Facial expressions add meaning
- Time is established at start

Examples:
ASL: "YESTERDAY ME GO STORE BUY MILK" → "Yesterday, I went to the store to buy milk."
ASL: "ME HUNGRY EAT WANT" → "I am hungry and want to eat."
ASL: "YOU LIKE COFFEE?" → "Do you like coffee?"
"""

    def fix_grammar(self, asl_gloss: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Convert to English: {asl_gloss}"}
                ],
                temperature=0.3,
                max_tokens=100
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM Error: {e}")
            raise e

grammar_fixer = ASLGrammarFixer()

# ---------------------------------------------------------------------------
# ML Model (optional — graceful fallback if not present)
# ---------------------------------------------------------------------------

MODEL_WINDOW_SIZE = WINDOW_SIZE  # 35 — same as segmentation's sliding window

_MODEL_PATH = "best_model.keras"
_CLASSES_PATH = "sign_classes.npy"

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
motion_detector = MotionDetector(
    low_factor=0.5,
    high_factor=4.0,
    still_frames_required=8,
    min_sign_duration=5,
    history_size=30,
    feature_dim=FEATURE_DIM,
    motion_smoothing=0.6,
    stillness_floor=0.3,
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
        converter.process_new_frame(keypoints)

        session = session_states.setdefault(request.uuid, {
            "motion_detector": MotionDetector(
                low_factor=0.5,
                high_factor=4.0,
                still_frames_required=8,
                min_sign_duration=5,
                history_size=30,
                feature_dim=FEATURE_DIM,
                motion_smoothing=0.6,
                stillness_floor=0.3,
            ),
            "predicted_words": [],
            "sign_count": 0,
        })

        md = session["motion_detector"]
        sign_ended, completed_sign = md.update(keypoints)

        if sign_ended and completed_sign is not None:
            kp_list = [kp.tolist() for kp in completed_sign]
            word = _predict_word(kp_list)
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
