"""
Relevance evaluation agent for assessing paper relevance.
"""

from typing import Dict, Any, List
from .base_agent import BaseAgent

class RelevanceEvaluatorAgent(BaseAgent):
    """
    Agent responsible for evaluating paper relevance to research queries.
    """
    
    SYSTEM_PROMPT = """You are an expert research evaluator who assesses the relevance of academic papers to specific research queries.
Your goal is to determine how well each paper addresses the research question and identify key contributions."""

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate paper relevance to the research query.
        
        Parameters
        ----------
        input_data : dict
            Must contain:
            - 'query': the research query
            - 'papers': list of paper dictionaries
            - 'relevance_threshold': minimum relevance score (0-1)
            
        Returns
        -------
        dict
            Contains evaluated papers with relevance scores and explanations
        """
        query = input_data["query"]
        papers = input_data["papers"]
        threshold = input_data.get("relevance_threshold", 0.7)
        
        evaluated_papers = []
        
        for paper in papers:
            # Create evaluation prompt for each paper
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

            # Get evaluation
            evaluation = self._llm_call(self.SYSTEM_PROMPT, user_prompt)
            
            # Add evaluation to paper data
            paper_with_evaluation = {
                **paper,
                "evaluation": evaluation
            }
            evaluated_papers.append(paper_with_evaluation)
        
        # Filter papers based on threshold
        relevant_papers = [
            paper for paper in evaluated_papers
            if self._get_relevance_score(paper["evaluation"]) >= threshold
        ]
        
        return {
            "all_papers": evaluated_papers,
            "relevant_papers": relevant_papers,
            "total_papers": len(papers),
            "relevant_count": len(relevant_papers)
        }
    
    def _get_relevance_score(self, evaluation_str: str) -> float:
        """
        Extract relevance score from evaluation string.
        Handles potential JSON parsing issues gracefully.
        """
        try:
            import json
            evaluation = json.loads(evaluation_str)
            return float(evaluation.get("relevance_score", 0))
        except:
            # If parsing fails, try to extract score using regex
            import re
            match = re.search(r'"relevance_score":\s*(0\.\d+|1\.0)', evaluation_str)
            if match:
                return float(match.group(1))
            return 0 