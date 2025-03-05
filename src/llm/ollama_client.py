"""
Ollama client for local LLM processing.
"""

import requests
from typing import Dict, List, Any, Optional
import json

class OllamaClient:
    """
    Client for interacting with local Ollama instance.
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "mistral",  # Default to Mistral as it's a good balance of performance and speed
        temperature: float = 0.7
    ):
        """
        Initialize Ollama client.
        
        Parameters
        ----------
        base_url : str
            URL of the Ollama instance
        model : str
            Model to use (e.g., 'llama2', 'mistral', 'codellama')
        temperature : float
            Temperature for generation
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.temperature = temperature
    
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate text using Ollama.
        
        Parameters
        ----------
        system_prompt : str
            System prompt for context
        user_prompt : str
            User prompt for generation
        temperature : float, optional
            Override default temperature
            
        Returns
        -------
        str
            Generated text
        """
        url = f"{self.base_url}/api/generate"
        
        # Combine prompts
        prompt = f"{system_prompt}\n\nUser: {user_prompt}"
        
        data = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature or self.temperature,
            "stream": False
        }
        
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            return response.json()["response"]
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to Ollama: {e}")
        except (KeyError, json.JSONDecodeError) as e:
            raise ValueError(f"Invalid response from Ollama: {e}")
    
    def is_available(self) -> bool:
        """
        Check if Ollama is available and the model is loaded.
        
        Returns
        -------
        bool
            True if Ollama is available and model is loaded
        """
        try:
            url = f"{self.base_url}/api/tags"
            response = requests.get(url)
            response.raise_for_status()
            models = response.json().get("models", [])
            return self.model in [m["name"] for m in models]
        except:
            return False
    
    def load_model(self) -> bool:
        """
        Ensure the model is loaded.
        
        Returns
        -------
        bool
            True if model is successfully loaded
        """
        if self.is_available():
            return True
            
        try:
            url = f"{self.base_url}/api/pull"
            data = {"name": self.model}
            response = requests.post(url, json=data)
            response.raise_for_status()
            return True
        except:
            return False 