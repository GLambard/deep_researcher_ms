"""
Base agent class for the multi-agent system.

This module defines the abstract base class that all specialized agents
in the research system must inherit from. It establishes a common interface
and shared functionality for interacting with the LLM.
"""

from typing import Dict, Any, Optional  # For type hints
from abc import ABC, abstractmethod  # For abstract class definition
from ..llm.ollama_client import OllamaClient  # For LLM interaction

class BaseAgent(ABC):
    """
    Base class for all agents in the multi-agent research system.
    
    This abstract class defines:
    1. A common initialization interface for all agents
    2. Shared LLM interaction functionality
    3. An abstract processing method that each agent must implement
    
    The agent design pattern allows for separation of concerns, with
    different agents specializing in specific aspects of the research process.
    """
    
    def __init__(
        self,
        llm_client: OllamaClient,  # Client for interacting with Ollama LLM
        model: str = "deepseek-r1:8b",  # Default to research-optimized model
        temperature: float = 0.7  # Balanced creativity and determinism
    ):
        """
        Initialize the base agent with LLM configuration.
        
        All specialized agents inherit this constructor, ensuring
        consistent LLM interaction across the system.
        
        Parameters:
        -----------
        llm_client: The Ollama client instance for LLM API calls
            Shared across agents to avoid redundant connections
        model: LLM model to use for specialized agent tasks
            Default is deepseek-r1:8b, optimized for research tasks
        temperature: Controls randomness in LLM responses
            Higher values increase creativity but reduce consistency
        """
        self.llm_client = llm_client
        self.model = model
        self.temperature = temperature
        
    def _llm_call(self, system_prompt: str, user_prompt: str) -> str:
        """
        Make an LLM call with the given prompts.
        
        This helper method standardizes how agents interact with the LLM.
        It encapsulates the prompt formatting and API call details,
        allowing derived agents to focus on their specialized logic.
        
        Parameters:
        -----------
        system_prompt: Defines the LLM's role and behavior for this task
            Sets the context for how the model should respond
        user_prompt: The specific input requiring a response
            Contains the task-specific information or query
            
        Returns:
        --------
        str: Raw response text from the LLM
        """
        # Delegate to the Ollama client for actual API interaction
        return self.llm_client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )
    
    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data and return results.
        
        This abstract method defines the primary interface that all
        specialized agents must implement. Each agent will interpret
        input_data based on its specific role and produce appropriate results.
        
        Parameters:
        -----------
        input_data: A dictionary containing the data needed for processing
            Contents will vary depending on the specific agent's purpose
            
        Returns:
        --------
        dict: Processing results specific to the agent's function
            Format will depend on what the agent is designed to produce
        """
        # Must be implemented by each specialized agent
        pass 