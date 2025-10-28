from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
import os
from dotenv import load_dotenv

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
@app.post('/translate')
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


# API Endpoints
@app.post('/translate')
async def translate():
    """Legacy endpoint - placeholder"""
    return {"message": "Use /convert-sentence endpoint"}


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


async def translate():
    """Legacy endpoint - placeholder"""
    return {"message": "Use /convert-sentence endpoint"}


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
