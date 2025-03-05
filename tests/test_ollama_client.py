"""
Tests for Ollama client.
"""

import pytest
import responses
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.ollama_client import OllamaClient

# Constants
MODEL_NAME = "deepseek-r1:8b"
BASE_URL = "http://localhost:11434"

@pytest.fixture
def ollama_client():
    return OllamaClient(model=MODEL_NAME)

@responses.activate
def test_generate_success(ollama_client):
    """Test successful text generation."""
    # Mock successful response
    responses.add(
        responses.POST,
        f"{BASE_URL}/api/generate",
        json={"response": "Generated text"},
        status=200
    )
    
    result = ollama_client.generate(
        system_prompt="You are a helpful research assistant.",
        user_prompt="Summarize this paper."
    )
    
    assert result == "Generated text"

@responses.activate
def test_generate_connection_error(ollama_client):
    """Test connection error handling."""
    # Mock connection error
    responses.add(
        responses.POST,
        f"{BASE_URL}/api/generate",
        status=500
    )
    
    with pytest.raises(ConnectionError):
        ollama_client.generate(
            system_prompt="You are a helpful research assistant.",
            user_prompt="Summarize this paper."
        )

@responses.activate
def test_is_available_true(ollama_client):
    """Test model availability check when model is available."""
    # Mock successful response with model available
    responses.add(
        responses.GET,
        f"{BASE_URL}/api/tags",
        json={"models": [{"name": MODEL_NAME}]},
        status=200
    )
    
    assert ollama_client.is_available() is True

@responses.activate
def test_is_available_false(ollama_client):
    """Test model availability check when model is not available."""
    # Mock successful response with model not available
    responses.add(
        responses.GET,
        f"{BASE_URL}/api/tags",
        json={"models": [{"name": "llama2"}]},
        status=200
    )
    
    assert ollama_client.is_available() is False

@responses.activate
def test_load_model_success(ollama_client):
    """Test successful model loading."""
    # Mock model not available initially
    responses.add(
        responses.GET,
        f"{BASE_URL}/api/tags",
        json={"models": []},
        status=200
    )
    
    # Mock successful model pull
    responses.add(
        responses.POST,
        f"{BASE_URL}/api/pull",
        json={"status": "success"},
        status=200
    )
    
    assert ollama_client.load_model() is True

@responses.activate
def test_load_model_failure(ollama_client):
    """Test model loading failure."""
    # Mock model not available initially
    responses.add(
        responses.GET,
        f"{BASE_URL}/api/tags",
        json={"models": []},
        status=200
    )
    
    # Mock failed model pull
    responses.add(
        responses.POST,
        f"{BASE_URL}/api/pull",
        status=500
    )
    
    assert ollama_client.load_model() is False

def test_custom_temperature(ollama_client):
    """Test custom temperature setting."""
    # Default temperature for research tasks should be lower
    assert ollama_client.temperature == 0.7
    
    # Test custom temperature for more deterministic output
    client_with_custom_temp = OllamaClient(model=MODEL_NAME, temperature=0.3)
    assert client_with_custom_temp.temperature == 0.3

@responses.activate
def test_generate_with_context(ollama_client):
    """Test text generation with research context."""
    # Mock successful response
    responses.add(
        responses.POST,
        f"{BASE_URL}/api/generate",
        json={"response": "Research summary with citations"},
        status=200
    )
    
    result = ollama_client.generate(
        system_prompt="You are a research assistant. Provide summaries with citations.",
        user_prompt="Summarize this research paper about single-cell RNA sequencing.",
        context="Paper discusses advances in scRNA-seq technology."
    )
    
    assert result == "Research summary with citations" 