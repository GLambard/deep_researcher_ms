"""
Ollama API client for text generation.
"""

import requests
from typing import Optional
import time
import sys

class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(
        self,
        model: str = "deepseek-r1:8b",
        temperature: float = 0.3,
        base_url: str = "http://localhost:11434"
    ):
        """
        Initialize Ollama client.
        
        Parameters
        ----------
        model : str
            Name of the model to use (default: deepseek-r1:8b)
        temperature : float
            Temperature for text generation (0.0 to 1.0)
        base_url : str
            Base URL for Ollama API
        """
        self.model = model
        self.temperature = temperature
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def check_server(self) -> bool:
        """
        Check if Ollama server is running and accessible.
        
        Returns
        -------
        bool
            True if server is running, False otherwise
        """
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException:
            return False
    
    def list_models(self) -> list:
        """
        Get list of available models.
        
        Returns
        -------
        list
            List of model names
        """
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            return [model['name'] for model in data['models']]
        except requests.exceptions.RequestException as e:
            print(f"Error listing models: {e}")
            return []
    
    def is_model_available(self) -> bool:
        """
        Check if the specified model is available.
        
        Returns
        -------
        bool
            True if model is available, False otherwise
        """
        try:
            models = self.list_models()
            return self.model in models
        except:
            return False
    
    def pull_model(self) -> bool:
        """
        Pull the model from Ollama.
        
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            print(f"Pulling model {self.model}...")
            response = self.session.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model}
            )
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            print(f"Error pulling model: {e}")
            return False
    
    def setup_model(self) -> bool:
        """
        Ensure model is available, pulling if necessary.
        
        Returns
        -------
        bool
            True if model is ready, False otherwise
        """
        if not self.check_server():
            print("\nError: Ollama server is not running!")
            print("Please start the Ollama server:")
            if sys.platform == "win32":
                print("1. Open Command Prompt")
                print("2. Run: ollama serve")
            else:
                print("1. Open Terminal")
                print("2. Run: ollama serve")
            return False
        
        if not self.is_model_available():
            print(f"\nModel {self.model} not found locally.")
            if not self.pull_model():
                print("Failed to pull model.")
                return False
        
        return True
    
    def generate(self, prompt: str, max_retries: int = 3) -> str:
        """
        Generate text using Ollama.
        
        Parameters
        ----------
        prompt : str
            Input prompt for text generation
        max_retries : int
            Maximum number of retries on failure
            
        Returns
        -------
        str
            Generated text
        
        Raises
        ------
        RuntimeError
            If text generation fails after retries
        """
        if not self.setup_model():
            raise RuntimeError("Failed to set up Ollama model")
        
        for attempt in range(max_retries):
            try:
                response = self.session.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "temperature": self.temperature,
                        "stream": False
                    }
                )
                response.raise_for_status()
                return response.json()["response"]
                
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Failed to generate text after {max_retries} attempts: {e}")
                print(f"Attempt {attempt + 1} failed, retrying...")
                time.sleep(1) 