"""
Base agent class for the multi-agent system.
"""

from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from ..llm.ollama_client import OllamaClient

class BaseAgent(ABC):
    """
    Base class for all agents in the system.
    """
    
    def __init__(
        self,
        llm_client: OllamaClient,
        model: str = "deepseek-r1:8b",
        temperature: float = 0.7
    ):
        """
        Initialize the base agent.
        
        Parameters
        ----------
        llm_client : OllamaClient
            Ollama client instance
        model : str, optional
            Model to use for LLM calls (default: deepseek-r1:8b)
        temperature : float, optional
            Temperature for LLM generation
        """
        self.llm_client = llm_client
        self.model = model
        self.temperature = temperature
        
    def _llm_call(self, system_prompt: str, user_prompt: str) -> str:
        """
        Make an LLM call with the given prompts.
        
        Parameters
        ----------
        system_prompt : str
            System prompt for the LLM
        user_prompt : str
            User prompt for the LLM
            
        Returns
        -------
        str
            LLM response
        """
        return self.llm_client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )
    
    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data and return results.
        
        Parameters
        ----------
        input_data : dict
            Input data for the agent
            
        Returns
        -------
        dict
            Processing results
        """
        pass 