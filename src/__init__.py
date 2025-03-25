"""
Deep Researcher - A toolkit for systematic literature review using LLMs.
This module serves as the main entry point for the Deep Researcher application,
providing a comprehensive interface for conducting systematic literature reviews
with the help of Large Language Models and specialized search APIs.
"""

from typing import List, Dict, Tuple, Optional, Any  # Type hints for better code documentation
from datetime import datetime  # For timestamping operations
import os  # For environment variables and file operations
import json  # For parsing JSON responses from LLMs
from dotenv import load_dotenv  # For loading environment variables from .env file

# Import core components from submodules
from .search.literature_manager import LiteratureManager  # Literature search management
from .processing.text_processor import TextProcessor  # Text processing utilities
from .llm.prompts import RESEARCH_PROMPTS  # Predefined prompts for research tasks
from .agents.query_planner import QueryPlannerAgent  # Agent for breaking down research queries
from .agents.relevance_evaluator import RelevanceEvaluatorAgent  # Agent for evaluating paper relevance
from .llm.ollama_client import OllamaClient  # Client for interacting with Ollama LLM API

class DeepResearcher:
    """
    Main class for performing systematic literature reviews using a multi-agent system.
    
    This class orchestrates the entire research workflow:
    1. Breaking down complex queries into manageable sub-queries
    2. Searching for relevant academic papers
    3. Evaluating the relevance of found papers
    4. Generating summaries and synthesis of findings
    5. Suggesting follow-up research directions
    """
    
    def __init__(
        self,
        model: str = "deepseek-r1:8b",  # Using DeepSeek as default model - optimized for research tasks
        temperature: float = 0.7  # Balances creativity and determinism in responses
    ):
        """
        Initialize the DeepResearcher with necessary components and configuration.
        
        Parameters:
        -----------
        model: The Ollama model to use (default: deepseek-r1:8b which excels at academic content)
        temperature: Controls randomness in LLM responses (higher = more creative, lower = more deterministic)
        """
        # Load environment variables from .env file
        load_dotenv()
        
        # Initialize API clients for external services
        self.ollama_client = OllamaClient(model=model, temperature=temperature)
        self.literature_manager = LiteratureManager()
        
        # Initialize text processing component
        self.text_processor = TextProcessor()
        self.model = model
        
        # Initialize specialized agents that handle different aspects of the research process
        self.query_planner = QueryPlannerAgent(self.ollama_client, model=model)
        self.relevance_evaluator = RelevanceEvaluatorAgent(self.ollama_client, model=model)
        
        # Initialize research state to track progress and store results
        self.conversation_history = []  # Tracks interaction history
        self.current_research = {
            "original_query": None,  # The user's initial research question
            "sub_queries": [],  # Broken-down components of the main query
            "papers": [],  # All papers found during research
            "relevant_papers": [],  # Papers determined to be relevant
            "summaries": []  # Summaries generated for each sub-query
        }
    
    async def start_research(self, query: str, sources: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Start a new research conversation with the given query.
        
        This method initiates the research process by:
        1. Planning how to break down the research query
        2. Executing the first sub-query
        3. Providing initial results and suggesting next steps
        
        Parameters:
        -----------
        query: The initial research query from the user
        sources: Optional list of sources to use (e.g., ["tavily", "semantic_scholar", "arxiv", "open_alex", "chemrxiv"])
            
        Returns:
        --------
        dict: Contains the research plan, initial results, and suggested next steps
        """
        # Reset research state to start fresh
        self.current_research = {
            "original_query": query,
            "sub_queries": [],
            "papers": [],
            "relevant_papers": [],
            "summaries": []
        }
        
        # Use query planner agent to break down the complex query into manageable parts
        plan = self.query_planner.process({"query": query})
        self.current_research["plan"] = plan
        
        try:
            # Parse the research plan returned by the LLM
            plan_data = json.loads(plan["plan"])
            sub_queries = plan_data["sub_queries"]
            # Get priority order or use sequential order if not specified
            priority_order = plan_data.get("priority_order", range(len(sub_queries)))
            
            # Execute the highest priority sub-query first
            first_query = sub_queries[priority_order[0]]
            results = await self._execute_sub_query(first_query, sources)
            
            # Return plan, initial results, and suggested next steps
            return {
                "plan": plan_data,
                "initial_results": results,
                "next_steps": self._get_next_steps(plan_data, 0)
            }
        except json.JSONDecodeError:
            # Handle case where the LLM doesn't return valid JSON
            return {
                "plan": plan,
                "error": "Could not parse research plan",
                "next_steps": ["Please rephrase your query"]
            }
    
    async def continue_research(self, feedback: Optional[str] = None, sources: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Continue the research based on current state and optional user feedback.
        
        This method advances the research by:
        1. Checking current progress
        2. Executing the next sub-query based on priority
        3. Generating a final synthesis if all sub-queries are completed
        
        Parameters:
        -----------
        feedback: Optional user feedback to guide further research
        sources: Optional list of sources to use for literature search
            
        Returns:
        --------
        dict: Contains the next set of results and suggested next steps
        """
        # Get the current research plan
        plan_data = json.loads(self.current_research["plan"]["plan"])
        # Track how many sub-queries have been completed so far
        current_progress = len(self.current_research["sub_queries"])
        
        # Check if all sub-queries have been completed
        if current_progress >= len(plan_data["sub_queries"]):
            # Research complete - generate final synthesis and follow-up suggestions
            return await self._generate_final_synthesis()
        
        # Execute the next sub-query based on priority order
        next_query = plan_data["sub_queries"][plan_data["priority_order"][current_progress]]
        results = await self._execute_sub_query(next_query, sources)
        
        # Return results and suggest next steps
        return {
            "results": results,
            "next_steps": self._get_next_steps(plan_data, current_progress + 1)
        }
    
    async def _execute_sub_query(self, query_info: Dict[str, Any], sources: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Execute a single sub-query and process the results.
        
        This private method handles the core research workflow:
        1. Searching for papers related to the sub-query
        2. Processing the papers to extract key information
        3. Evaluating the relevance of each paper
        4. Generating a summary of the findings
        5. Updating the research state
        
        Parameters:
        -----------
        query_info: Dictionary containing the sub-query and related metadata
        sources: Optional list of sources to use for literature search
        
        Returns:
        --------
        dict: Contains the query info, relevant papers, and summary
        """
        # Extract year range if specified in the query info
        year_range = None
        if "year_range" in query_info:
            try:
                # Parse year range string in format "YYYY-YYYY"
                years = query_info["year_range"].split("-")
                if len(years) == 2:
                    year_range = (int(years[0]), int(years[1]))
            except (ValueError, AttributeError):
                # Malformed year range, ignore it
                pass
        
        # Search for papers using the literature manager
        papers = await self.literature_manager.search(
            query=query_info["query"],
            max_papers=10,  # Fetch up to 10 papers for this sub-query
            year_range=year_range,  # Optional date filtering
            sources=sources  # Optional sources to use
        )
        
        # Process papers to extract and structure relevant information
        processed_papers = self.text_processor.process_papers(papers)
        
        # Evaluate relevance of each paper to the sub-query
        evaluation_results = self.relevance_evaluator.process({
            "query": query_info["query"],
            "papers": processed_papers,
            "relevance_threshold": 0.7  # Only keep papers with relevance score ≥ 0.7
        })
        
        # Update the current research state with new findings
        self.current_research["sub_queries"].append(query_info)
        self.current_research["papers"].extend(papers)
        self.current_research["relevant_papers"].extend(evaluation_results["relevant_papers"])
        
        # Generate a summary of the relevant papers for this sub-query
        summary = self._generate_summary(evaluation_results["relevant_papers"])
        self.current_research["summaries"].append({
            "query": query_info["query"],
            "summary": summary
        })
        
        # Return a dictionary with the results
        return {
            "query": query_info,
            "relevant_papers": evaluation_results["relevant_papers"],
            "summary": summary
        }
    
    def _generate_summary(self, papers: List[Dict]) -> str:
        """
        Generate a summary of the papers using the LLM.
        
        Parameters:
        -----------
        papers: List of paper dictionaries to summarize
        
        Returns:
        --------
        str: Generated summary text
        """
        # Use a predefined prompt template for comprehensive summaries
        prompt = RESEARCH_PROMPTS["comprehensive_summary"].format(
            papers=papers
        )
        
        # Generate a summary using the Ollama LLM
        response = self.ollama_client.generate(
            system_prompt="You are a research assistant helping to synthesize scientific literature.",
            user_prompt=prompt
        )
        
        return response
    
    def _get_next_steps(self, plan: Dict[str, Any], current_step: int) -> List[str]:
        """
        Generate suggestions for next steps in the research process.
        
        Parameters:
        -----------
        plan: The research plan
        current_step: The current step number
        
        Returns:
        --------
        list: Suggested next steps
        """
        # Check if we're done with all sub-queries
        if current_step >= len(plan["sub_queries"]):
            return ["Generate final synthesis"]
        
        # Get the next sub-query based on priority order
        next_query_index = plan["priority_order"][current_step]
        next_query = plan["sub_queries"][next_query_index]
        
        # Suggest next steps based on the query plan
        return [
            f"Execute sub-query: {next_query['query']}",
            "Provide feedback on current results",
            "Change literature search sources"
        ]
    
    async def _generate_final_synthesis(self) -> Dict[str, Any]:
        """
        Generate a final synthesis of all research findings.
        
        Returns:
        --------
        dict: Contains the final synthesis and follow-up suggestions
        """
        # Get all summaries from completed sub-queries
        summaries = [item["summary"] for item in self.current_research["summaries"]]
        original_query = self.current_research["original_query"]
        
        # Create a prompt for the synthesis
        synthesis_prompt = RESEARCH_PROMPTS["final_synthesis"].format(
            query=original_query,
            summaries="\n\n".join(summaries)
        )
        
        # Generate the synthesis
        synthesis = self.ollama_client.generate(
            system_prompt="You are a research assistant creating a comprehensive synthesis of findings.",
            user_prompt=synthesis_prompt
        )
        
        # Get statistics about the search
        search_stats = self.literature_manager.get_search_statistics()
        
        # Return the final results
        return {
            "synthesis": synthesis,
            "search_statistics": search_stats,
            "sub_query_results": self.current_research["summaries"],
            "is_complete": True
        }

# Package metadata
"""
Deep Researcher package - version information.
"""

__version__ = "0.1.0"  # Current version of the package 