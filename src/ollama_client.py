"""
Ollama API client for text generation.

This module provides a client for interacting with the Ollama API,
which allows running local LLM models for text generation. Ollama
is an open-source platform for running local language models.
"""

import requests  # For making HTTP requests to the Ollama API
from typing import Optional  # For type hints
import time  # For implementing retry delays
import sys  # For platform detection and path operations

## TODO: 
# Add a class for OpenAI/Anthropic/LMStudio API endpoints
##

class OllamaClient:
    """
    Client for interacting with Ollama API.
    
    This class handles all communication with a local Ollama server,
    including checking server status, listing available models,
    pulling models when needed, and generating text responses.
    """
    
    def __init__(
        self,
        model: str = "deepseek-r1:8b",  # Default model optimized for research tasks
        temperature: float = 0.3,  # Low temperature for more deterministic responses
        base_url: str = "http://localhost:11434"  # Default Ollama server URL
    ):
        """
        Initialize Ollama client with configuration parameters.
        
        Parameters:
        -----------
        model: Name of the LLM to use (default: deepseek-r1:8b)
            Specifies which model Ollama should use for text generation
        temperature: Controls determinism in generated text (0.0 to 1.0)
            Lower values produce more deterministic outputs
            Higher values produce more creative/diverse outputs
        base_url: URL where the Ollama server is running
            Default is the standard localhost port for Ollama
        """
        self.model = model
        self.temperature = temperature
        self.base_url = base_url.rstrip('/')  # Remove trailing slash if present for URL consistency
        # Create persistent session for better performance across multiple API calls
        self.session = requests.Session()
    
    def check_server(self) -> bool:
        """
        Check if Ollama server is running and accessible.
        
        Attempts to connect to the Ollama API endpoint to verify
        that the server is operational before making other requests.
        
        Returns:
        --------
        bool: True if server is running and responding, False otherwise
        """
        try:
            # Try to list models as a simple API health check
            response = self.session.get(f"{self.base_url}/api/tags")
            response.raise_for_status()  # Raise an exception for HTTP errors
            return True
        except requests.exceptions.RequestException:
            # Any request exception means server is not properly accessible
            return False
    
    def list_models(self) -> list:
        """
        Get list of available models downloaded to the local Ollama server.
        
        This method queries the Ollama API to get all models that are
        currently available for use without needing to download.
        
        Returns:
        --------
        list: List of model names that can be used immediately
              Returns empty list if server unavailable or error occurs
        """
        try:
            # Get models from the Ollama API
            response = self.session.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            # Extract just the model names from the response
            return [model['name'] for model in data['models']]
        except requests.exceptions.RequestException as e:
            # Log error and return empty list on failure
            print(f"Error listing models: {e}")
            return []
    
    def is_model_available(self) -> bool:
        """
        Check if the specified model is available on the local Ollama server.
        
        Determines if the model requested during initialization is already
        downloaded and available for use without needing to pull it.
        
        Returns:
        --------
        bool: True if model is available locally, False otherwise
        """
        try:
            # Get list of available models and check if our model is in it
            models = self.list_models()
            return self.model in models
        except:
            # If any error occurs during check, assume model is not available
            return False
    
    def pull_model(self) -> bool:
        """
        Pull (download) the requested model from Ollama's model repository.
        
        This method is called when a model is not available locally but is
        needed for text generation. Downloads can take time depending on
        model size and internet connection speed.
        
        Returns:
        --------
        bool: True if model was successfully downloaded, False otherwise
        """
        try:
            print(f"Pulling model {self.model}...")
            # Request model download from Ollama
            response = self.session.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model}
            )
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            # Log error and return failure status
            print(f"Error pulling model: {e}")
            return False
    
    def setup_model(self) -> bool:
        """
        Ensure the requested model is ready for use.
        
        This method performs a complete setup check:
        1. Verifies the Ollama server is running
        2. Checks if the requested model is available
        3. Attempts to download the model if not available
        
        Returns:
        --------
        bool: True if model is ready to use, False if setup failed
        """
        # First check if the Ollama server is running
        if not self.check_server():
            print("\nError: Ollama server is not running!")
            print("Please start the Ollama server:")
            # Provide platform-specific instructions
            if sys.platform == "win32":
                print("1. Open Command Prompt")
                print("2. Run: ollama serve")
            else:
                print("1. Open Terminal")
                print("2. Run: ollama serve")
            return False
        
        # If server is running, check if model is available
        if not self.is_model_available():
            print(f"\nModel {self.model} not found locally.")
            # If model isn't available, try to download it
            if not self.pull_model():
                print("Failed to pull model.")
                return False
        
        # If we get here, the model is ready to use
        return True
    
    def generate(self, prompt: str, max_retries: int = 3) -> str:
        """
        Generate text using the Ollama LLM.
        
        This is the core method for text generation. It:
        1. Ensures the model is set up and ready
        2. Sends the prompt to the Ollama API
        3. Handles errors with automatic retries
        
        Parameters:
        -----------
        prompt: The input text prompt to send to the LLM
        max_retries: Number of retry attempts if request fails
            
        Returns:
        --------
        str: The text generated by the LLM
        
        Raises:
        -------
        RuntimeError: If text generation fails after all retries
        """
        # Make sure model is ready before attempting generation
        if not self.setup_model():
            raise RuntimeError("Failed to set up Ollama model")
        
        # Try up to max_retries times
        for attempt in range(max_retries):
            try:
                # Send generation request to Ollama API
                response = self.session.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "temperature": self.temperature,
                        "stream": False  # Get complete response at once, not streaming
                    }
                )
                response.raise_for_status()
                # Extract the generated text from the response
                return response.json()["response"]
                
            except requests.exceptions.RequestException as e:
                # If we've used all retries, raise an error
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Failed to generate text after {max_retries} attempts: {e}")
                # Otherwise, wait and try again
                print(f"Attempt {attempt + 1} failed, retrying...")
                time.sleep(1)  # Wait 1 second before retrying 