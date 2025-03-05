"""
Deep Researcher - A toolkit for systematic literature review using LLMs.
"""

from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
import os
import json
from dotenv import load_dotenv

from .search.semantic_scholar import SemanticScholarAPI
from .processing.text_processor import TextProcessor
from .llm.prompts import RESEARCH_PROMPTS
from .agents.query_planner import QueryPlannerAgent
from .agents.relevance_evaluator import RelevanceEvaluatorAgent
from .llm.ollama_client import OllamaClient

class DeepResearcher:
    """
    Main class for performing systematic literature reviews using a multi-agent system.
    """
    
    def __init__(
        self,
        semantic_scholar_api_key: Optional[str] = None,
        model: str = "deepseek-r1:8b",  # Using DeepSeek as default model
        temperature: float = 0.7
    ):
        """
        Initialize the DeepResearcher.
        """
        # Load environment variables
        load_dotenv()
        
        # Initialize API clients
        self.ollama_client = OllamaClient(model=model, temperature=temperature)
        self.semantic_scholar = SemanticScholarAPI(
            api_key=semantic_scholar_api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        )
        
        # Initialize components
        self.text_processor = TextProcessor()
        self.model = model
        
        # Initialize agents
        self.query_planner = QueryPlannerAgent(self.ollama_client, model=model)
        self.relevance_evaluator = RelevanceEvaluatorAgent(self.ollama_client, model=model)
        
        # Initialize research state
        self.conversation_history = []
        self.current_research = {
            "original_query": None,
            "sub_queries": [],
            "papers": [],
            "relevant_papers": [],
            "summaries": []
        }
    
    def start_research(self, query: str) -> Dict[str, Any]:
        """
        Start a new research conversation.
        
        Parameters
        ----------
        query : str
            The initial research query
            
        Returns
        -------
        dict
            Initial research plan and first set of findings
        """
        # Reset research state
        self.current_research = {
            "original_query": query,
            "sub_queries": [],
            "papers": [],
            "relevant_papers": [],
            "summaries": []
        }
        
        # Plan the research
        plan = self.query_planner.process({"query": query})
        self.current_research["plan"] = plan
        
        try:
            # Parse the plan
            plan_data = json.loads(plan["plan"])
            sub_queries = plan_data["sub_queries"]
            priority_order = plan_data.get("priority_order", range(len(sub_queries)))
            
            # Execute the first sub-query
            first_query = sub_queries[priority_order[0]]
            results = self._execute_sub_query(first_query)
            
            return {
                "plan": plan_data,
                "initial_results": results,
                "next_steps": self._get_next_steps(plan_data, 0)
            }
        except json.JSONDecodeError:
            # Handle case where plan is not valid JSON
            return {
                "plan": plan,
                "error": "Could not parse research plan",
                "next_steps": ["Please rephrase your query"]
            }
    
    def continue_research(self, feedback: Optional[str] = None) -> Dict[str, Any]:
        """
        Continue the research based on current state and optional feedback.
        
        Parameters
        ----------
        feedback : str, optional
            User feedback or preferences for next steps
            
        Returns
        -------
        dict
            Next set of findings and suggestions
        """
        plan_data = json.loads(self.current_research["plan"]["plan"])
        current_progress = len(self.current_research["sub_queries"])
        
        if current_progress >= len(plan_data["sub_queries"]):
            # All sub-queries completed, generate synthesis and follow-ups
            return self._generate_final_synthesis()
        
        # Execute next sub-query
        next_query = plan_data["sub_queries"][plan_data["priority_order"][current_progress]]
        results = self._execute_sub_query(next_query)
        
        return {
            "results": results,
            "next_steps": self._get_next_steps(plan_data, current_progress + 1)
        }
    
    def _execute_sub_query(self, query_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single sub-query and process results.
        """
        # Search for papers
        papers = self.semantic_scholar.search(
            query=query_info["query"],
            limit=10,
            year_range=query_info.get("year_range")
        )
        
        # Process papers
        processed_papers = self.text_processor.process_papers(papers)
        
        # Evaluate relevance
        evaluation_results = self.relevance_evaluator.process({
            "query": query_info["query"],
            "papers": processed_papers,
            "relevance_threshold": 0.7
        })
        
        # Update research state
        self.current_research["sub_queries"].append(query_info)
        self.current_research["papers"].extend(papers)
        self.current_research["relevant_papers"].extend(evaluation_results["relevant_papers"])
        
        # Generate summary for this sub-query
        summary = self._generate_summary(evaluation_results["relevant_papers"])
        self.current_research["summaries"].append({
            "query": query_info["query"],
            "summary": summary
        })
        
        return {
            "query": query_info,
            "relevant_papers": evaluation_results["relevant_papers"],
            "summary": summary
        }
    
    def _generate_summary(self, papers: List[Dict]) -> str:
        """
        Generate a summary of the papers.
        """
        prompt = RESEARCH_PROMPTS["comprehensive_summary"].format(
            papers=papers
        )
        
        response = self.ollama_client.generate(
            system_prompt="You are a research assistant helping to synthesize scientific literature.",
            user_prompt=prompt
        )
        
        return response
    
    def _get_next_steps(self, plan: Dict[str, Any], current_step: int) -> List[str]:
        """
        Generate suggestions for next steps.
        """
        if current_step >= len(plan["sub_queries"]):
            return ["Research complete. Would you like to:",
                   "1. Get a final synthesis",
                   "2. Explore follow-up questions",
                   "3. Start a new research query"]
        
        next_query = plan["sub_queries"][plan["priority_order"][current_step]]
        return [
            f"Continue with sub-query: {next_query['query']}",
            "Modify the research plan",
            "Get current synthesis",
            "Start a new research query"
        ]
    
    def _generate_final_synthesis(self) -> Dict[str, Any]:
        """
        Generate final synthesis and follow-up suggestions.
        """
        # Generate comprehensive synthesis
        synthesis = self._generate_summary(self.current_research["relevant_papers"])
        
        # Generate follow-up queries
        follow_ups = self.query_planner.generate_follow_up_queries({
            "original_query": self.current_research["original_query"],
            "summary": synthesis
        })
        
        return {
            "synthesis": synthesis,
            "follow_up_suggestions": follow_ups,
            "paper_count": len(self.current_research["relevant_papers"]),
            "sub_queries_completed": len(self.current_research["sub_queries"])
        }

"""
Deep Researcher package.
"""

__version__ = "0.1.0" 