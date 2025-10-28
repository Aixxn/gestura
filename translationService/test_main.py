"""
Unit tests for the Translation Service
Run with: pytest test_main.py -v
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import os

# Import the app
from main import app, ASLGrammarFixer

# Create test client
client = TestClient(app)


class TestASLGrammarFixer:
    """Test the ASLGrammarFixer class"""
    
    @patch('main.Groq')
    def test_init_with_api_key(self, mock_groq):
        """Test initialization with provided API key"""
        fixer = ASLGrammarFixer(api_key="test_key")
        mock_groq.assert_called_once_with(api_key="test_key")
    
    @patch.dict(os.environ, {'GROQ_API_KEY': 'env_key'})
    @patch('main.Groq')
    def test_init_with_env_var(self, mock_groq):
        """Test initialization with environment variable"""
        fixer = ASLGrammarFixer()
        mock_groq.assert_called_once_with(api_key="env_key")
    
    @patch('main.Groq')
    def test_fix_grammar_success(self, mock_groq):
        """Test successful grammar conversion"""
        # Mock the Groq client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "I am hungry and want to eat."
        mock_client.chat.completions.create.return_value = mock_response
        mock_groq.return_value = mock_client
        
        fixer = ASLGrammarFixer(api_key="test_key")
        result = fixer.fix_grammar("ME HUNGRY EAT WANT")
        
        assert result == "I am hungry and want to eat."
        mock_client.chat.completions.create.assert_called_once()
    
    @patch('main.Groq')
    def test_fix_grammar_error(self, mock_groq):
        """Test error handling in grammar conversion"""
        # Mock the Groq client to raise an error
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_groq.return_value = mock_client
        
        fixer = ASLGrammarFixer(api_key="test_key")
        
        with pytest.raises(Exception):
            fixer.fix_grammar("ME HUNGRY EAT WANT")


class TestAPIEndpoints:
    """Test FastAPI endpoints"""
    
    def test_translate_endpoint(self):
        """Test the /translate endpoint"""
        response = client.get("/translate")
        assert response.status_code == 200
        assert response.json() == {"message": "Use /convert-sentence endpoint"}
    
    @patch('main.grammar_fixer.fix_grammar')
    def test_convert_sentence_success(self, mock_fix_grammar):
        """Test successful sentence conversion"""
        mock_fix_grammar.return_value = "I am hungry and want to eat."
        
        response = client.post(
            "/convert-sentence",
            json={"asl_gloss": "ME HUNGRY EAT WANT"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert data["original"] == "ME HUNGRY EAT WANT"
        assert data["translated"] == "I am hungry and want to eat."
        assert data["error"] is None
    
    @patch('main.grammar_fixer.fix_grammar')
    def test_convert_sentence_error(self, mock_fix_grammar):
        """Test error handling in conversion"""
        mock_fix_grammar.side_effect = Exception("LLM Error")
        
        response = client.post(
            "/convert-sentence",
            json={"asl_gloss": "ME HUNGRY EAT WANT"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == False
        assert data["original"] == "ME HUNGRY EAT WANT"
        assert data["translated"] == "ME HUNGRY EAT WANT"  # Returns original on error
        assert "LLM Error" in data["error"]
    
    def test_convert_sentence_invalid_request(self):
        """Test validation with invalid request"""
        response = client.post(
            "/convert-sentence",
            json={"wrong_field": "test"}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_convert_sentence_empty_gloss(self):
        """Test with empty ASL gloss"""
        response = client.post(
            "/convert-sentence",
            json={"asl_gloss": ""}
        )
        
        assert response.status_code == 200
        # Should still process, even if empty


class TestSystemPrompt:
    """Test the system prompt configuration"""
    
    @patch('main.Groq')
    def test_system_prompt_includes_rules(self, mock_groq):
        """Test that system prompt includes ASL rules"""
        mock_client = Mock()
        mock_groq.return_value = mock_client
        
        fixer = ASLGrammarFixer(api_key="test_key")
        
        assert "ASL" in fixer.system_prompt
        assert "topic-comment" in fixer.system_prompt
        assert "Examples:" in fixer.system_prompt
    
    @patch('main.Groq')
    def test_model_configuration(self, mock_groq):
        """Test that correct model is configured"""
        mock_client = Mock()
        mock_groq.return_value = mock_client
        
        fixer = ASLGrammarFixer(api_key="test_key")
        
        assert fixer.model == "llama-3.3-70b-versatile"


class TestIntegrationExamples:
    """Integration test examples with real ASL patterns"""
    
    @patch('main.grammar_fixer.fix_grammar')
    def test_common_asl_patterns(self, mock_fix_grammar):
        """Test common ASL sentence patterns"""
        test_cases = [
            ("ME HUNGRY", "I am hungry."),
            ("YOU WANT COFFEE?", "Do you want coffee?"),
            ("YESTERDAY ME GO STORE", "Yesterday, I went to the store."),
            ("ME LIKE PIZZA", "I like pizza."),
        ]
        
        for asl_input, expected_output in test_cases:
            mock_fix_grammar.return_value = expected_output
            
            response = client.post(
                "/convert-sentence",
                json={"asl_gloss": asl_input}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            assert data["translated"] == expected_output


# Pytest fixtures
@pytest.fixture
def mock_groq_client():
    """Fixture to provide a mocked Groq client"""
    with patch('main.Groq') as mock:
        yield mock


@pytest.fixture
def sample_asl_data():
    """Fixture providing sample ASL test data"""
    return {
        "simple": "ME HUNGRY",
        "question": "YOU WANT COFFEE?",
        "past_tense": "YESTERDAY ME GO STORE",
        "complex": "ME HUNGRY EAT WANT NOW",
    }


# Run tests with: pytest test_main.py -v
# Run with coverage: pytest test_main.py --cov=main --cov-report=html
