"""
Relevance evaluation agent for assessing paper relevance.

This module implements a specialized agent that evaluates how relevant
each found academic paper is to the original research query. It provides
structured assessments and filtering of papers based on relevance scores.
"""

from typing import Dict, Any, List  # For type hints
from .base_agent import BaseAgent  # Base class for all agents

class RelevanceEvaluatorAgent(BaseAgent):
    """
    Agent responsible for evaluating paper relevance to research queries.
    
    This agent serves as a critical filter in the research process by:
    1. Assessing how well each paper addresses the research question
    2. Providing structured relevance scores and explanations
    3. Filtering out less relevant papers to focus on valuable information
    4. Identifying key contributions from each relevant paper
    
    The relevance evaluation is crucial because:
    - It prevents information overload from too many tangential papers
    - It ensures research quality by focusing on the most pertinent sources
    - It provides context for why each paper matters to the research question
    """
    
    # System prompt that defines the agent's role and behavior
    # This instructs the LLM to act as a research evaluator specialized in assessing relevance
    SYSTEM_PROMPT = """You are an expert research evaluator who assesses the relevance of academic papers to specific research queries.
Your goal is to determine how well each paper addresses the research question and identify key contributions."""

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate how relevant each paper is to the specific research query.
        
        This method:
        1. Takes a set of papers and the original research query
        2. Evaluates each paper individually against the query
        3. Assigns structured relevance scores and explanations
        4. Filters papers based on a relevance threshold
        
        Parameters:
        -----------
        input_data: Dictionary containing:
            - 'query': The research query to evaluate against
            - 'papers': List of paper dictionaries with metadata
            - 'relevance_threshold': Minimum score (0-1) to consider a paper relevant
            
        Returns:
        --------
        dict: Contains evaluation results:
            - all_papers: All papers with their evaluations
            - relevant_papers: Only papers meeting the threshold
            - total_papers: Count of all papers evaluated
            - relevant_count: Count of papers meeting the threshold
        """
        # Extract required data from input
        query = input_data["query"]
        papers = input_data["papers"]
        # Default threshold is 0.7 if not specified (70% relevance)
        threshold = input_data.get("relevance_threshold", 0.7)
        
        # List to store papers with their evaluations
        evaluated_papers = []
        
        # Evaluate each paper individually
        for paper in papers:
            # Create a detailed prompt for evaluating this specific paper
            # The prompt includes the research query and key paper details
            # It also specifies the exact format we want for the response
            user_prompt = f"""Please evaluate the relevance of this paper to the research query:

Research Query: {query}

Paper Information:
Title: {paper.get('title')}
Abstract: {paper.get('abstract')}
Key Findings: {paper.get('key_findings')}
Methodology: {paper.get('methodology')}

Please evaluate:
1. How directly does this paper address the research query?
2. What specific aspects of the query does it address?
3. Are the methods and findings reliable and significant?

Format your response as JSON with:
{{
    "relevance_score": float (0-1),
    "aspects_addressed": ["aspect1", "aspect2"],
    "reliability_score": float (0-1),
    "key_contributions": ["contribution1", "contribution2"],
    "inclusion_recommendation": bool,
    "rationale": "explanation for the scores"
}}"""

            # Get evaluation from the LLM
            # This will return a JSON-formatted string with the evaluation
            evaluation = self._llm_call(self.SYSTEM_PROMPT, user_prompt)
            
            # Combine the original paper data with its evaluation
            # This preserves all original information while adding the assessment
            paper_with_evaluation = {
                **paper,  # Spread original paper data
                "evaluation": evaluation  # Add evaluation data
            }
            evaluated_papers.append(paper_with_evaluation)
        
        # Filter papers based on the relevance threshold
        # Only keep papers with relevance scores at or above the threshold
        relevant_papers = [
            paper for paper in evaluated_papers
            if self._get_relevance_score(paper["evaluation"]) >= threshold
        ]
        
        # Return comprehensive results dictionary
        return {
            "all_papers": evaluated_papers,  # All papers with evaluations
            "relevant_papers": relevant_papers,  # Only papers meeting threshold
            "total_papers": len(papers),  # Total count of papers evaluated
            "relevant_count": len(relevant_papers)  # Count of papers meeting threshold
        }
    
    def _get_relevance_score(self, evaluation_str: str) -> float:
        """
        Extract the relevance score from the evaluation string.
        
        This helper method handles potential JSON parsing issues gracefully:
        1. First attempts to parse the string as proper JSON
        2. If that fails, tries to extract the score using regex
        3. Returns 0 as a last resort if no score can be extracted
        
        This robust approach helps handle cases where the LLM doesn't
        perfectly follow the instructed JSON format.
        
        Parameters:
        -----------
        evaluation_str: The evaluation string from the LLM
            Expected to be JSON, but may have formatting issues
        
        Returns:
        --------
        float: The extracted relevance score (0-1)
            Returns 0 if score cannot be extracted
        """
        try:
            # First try parsing as proper JSON
            import json
            evaluation = json.loads(evaluation_str)
            return float(evaluation.get("relevance_score", 0))
        except:
            # If JSON parsing fails, fall back to regex extraction
            # This handles cases where the LLM response has invalid JSON
            # but still includes the relevance score in a recognizable format
            import re
            match = re.search(r'"relevance_score":\s*(0\.\d+|1\.0)', evaluation_str)
            if match:
                return float(match.group(1))
            # Last resort: if we can't extract a score, return 0
            return 0 