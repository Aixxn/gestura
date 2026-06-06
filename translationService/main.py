from fastapi import FastAPI
from pydantic import BaseModel, Field
import numpy as np
import keras
from groq import Groq
import os
from dotenv import load_dotenv
from typing import List, Tuple

# Load environment variables from .env file
load_dotenv()

app = FastAPI(
        debug=True, 
        title='Translator Service', 
        description='This service is responsible for servicing\
                the translator AI model.'
        )
# Request/Response Models
class ConvertSentenceRequest(BaseModel):
    asl_gloss: str

class ConvertSentenceResponse(BaseModel):
    original: str
    translated: str
    success: bool
    error: str | None = None


# ASL Grammar Fixer Class
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
        """Convert ASL gloss to natural English"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Convert to English: {asl_gloss}"}
                ],
                temperature=0.3,  # Lower = more consistent
                max_tokens=100
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM Error: {e}")
            raise e


# Initialize the grammar fixer (singleton)
grammar_fixer = ASLGrammarFixer()

# Load your .keras model (replace 'your_model.keras' with your actual model file)
# It's best practice to load the model outside the path operation function
# to avoid reloading it on every request.
try:
    model = keras.models.load_model('model.keras')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None # Handle this gracefully in a real app

# Define the expected input shape (configurable via .env to avoid mismatches)
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "80"))
FEATURE_DIM = int(os.getenv("FEATURE_DIM", "1663"))
NUM_CLASSES = model.output_shape[-1] if model else 30 # Adjust as needed

# --- Pydantic Data Model ---

# The request body will contain a list of lists, representing the (80, 1663) array
# We use List[List[float]] to enforce the structure for Pydantic's validation
class WindowInput(BaseModel):
    # Using Field for a clearer description in the auto-generated docs (Swagger/ReDoc)
    window_data: List[List[float]] = Field(
        ...,
        example=[[0.1] * FEATURE_DIM] * WINDOW_SIZE, # A sample (80, 1663)
        description=f"A list of {WINDOW_SIZE} lists, where each inner list has {FEATURE_DIM} float elements."
    )
    
    # Custom validator to check the shape after Pydantic validates types (optional but highly recommended)
    # The actual shape check will be done after conversion to numpy inside the endpoint
    
# --- Model Mapping (Replace with your actual word list) ---
# This list maps the class index (output from the model) back to a word.
# Ensure this list's length matches the NUM_CLASSES of your model.
WORD_MAPPING = [f"word_{i}" for i in range(NUM_CLASSES)]


# API Endpoints
@app.post("/translate")
async def translate(data: WindowInput):
    """
    Takes a (WINDOW_SIZE, FEATURE_DIM) array of floats and returns a predicted word.
    """
    if model is None:
        return {"error": "Model not loaded"}, 500

    # 1. Extract the list of lists from the Pydantic model
    window_list = data.window_data

    # 2. Convert the list of lists to a numpy array
    try:
        window_np = np.array(window_list)
    except ValueError:
        # This would catch issues if the inner lists aren't all the same length
        return {"error": "Input data structure is not uniform."}, 422
    
    # 3. Check for the correct shape
    if window_np.shape != (WINDOW_SIZE, FEATURE_DIM):
        return {"error": f"Incorrect array shape. Expected ({WINDOW_SIZE}, {FEATURE_DIM}), but got {window_np.shape}."}, 422

    # 4. Apply the necessary data type conversion and add batch dimension
    # window_np shape -> (WINDOW_SIZE, FEATURE_DIM)
    inp = window_np.astype(np.float32)[np.newaxis, ...]  # (1, W, D)

    # 5. Model Prediction
    probs = model.predict(inp, verbose=0)  # (1, num_classes)
    
    # 6. Get the predicted class index (word index)
    predicted_index = np.argmax(probs[0])

    # 7. Map the index back to a word
    if predicted_index >= len(WORD_MAPPING):
        # This handles a potential mismatch between model output size and your word list
        return {"error": "Prediction index out of bounds for word mapping."}, 500
        
    predicted_word = WORD_MAPPING[predicted_index]

    # 8. Return the word as the API response
    return {'pred': predicted_word}

@app.post('/convert-sentence', response_model=ConvertSentenceResponse)
async def convert_sentence(request: ConvertSentenceRequest):
    """Convert ASL gloss to natural English grammar"""
    try:
        translated = grammar_fixer.fix_grammar(request.asl_gloss)
        return ConvertSentenceResponse(
            original=request.asl_gloss,
            translated=translated,
            success=True
        )
    except Exception as e:
        return ConvertSentenceResponse(
            original=request.asl_gloss,
            translated=request.asl_gloss,  # Return original on error
            success=False,
            error=str(e)
        )
