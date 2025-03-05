"""
Query planning agent for breaking down complex research queries.
"""

from typing import Dict, Any, List
from .base_agent import BaseAgent

class QueryPlannerAgent(BaseAgent):
    """
    Agent responsible for breaking down complex queries into sub-queries.
    """
    
    SYSTEM_PROMPT = """You are an expert research assistant who helps break down complex research queries into specific, focused sub-queries.
Your goal is to create a systematic plan for exploring the research question thoroughly.
Break down the query into logical steps, considering different aspects and potential directions."""

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Break down a complex query into sub-queries.
        
        Parameters
        ----------
        input_data : dict
            Must contain 'query' key with the main research query
            
        Returns
        -------
        dict
            Contains 'sub_queries' list and 'search_plan' with the strategy
        """
        main_query = input_data["query"]
        
        # Create prompt for breaking down the query
        user_prompt = f"""Please help me break down this research query into specific sub-queries:

Main Query: {main_query}

Please provide:
1. A list of focused sub-queries that together will help answer the main query
2. A logical order for investigating these sub-queries
3. Key concepts or terms to focus on for each sub-query
4. Any specific time periods or date ranges that might be relevant

Format your response as JSON with the following structure:
{{
    "sub_queries": [
        {{
            "query": "specific sub-query",
            "key_terms": ["term1", "term2"],
            "year_range": [start_year, end_year],
            "rationale": "why this sub-query is important"
        }}
    ],
    "search_strategy": "overall strategy explanation",
    "priority_order": [0, 1, 2]  // indices of sub_queries in recommended order
}}"""

        # Get LLM response and parse it
        response = self._llm_call(self.SYSTEM_PROMPT, user_prompt)
        
        # The response will be in JSON format, but we'll let the parent handle parsing
        return {
            "original_query": main_query,
            "plan": response
        }
    
    def generate_follow_up_queries(self, context: Dict[str, Any]) -> List[str]:
        """
        Generate follow-up queries based on current context.
        
        Parameters
        ----------
        context : dict
            Current research context including previous queries and results
            
        Returns
        -------
        list
            List of suggested follow-up queries
        """
        user_prompt = f"""Based on the current research context, suggest follow-up queries:

Original Query: {context.get('original_query')}
Current Findings: {context.get('summary', '')}

Please suggest 3-5 follow-up queries that would help:
1. Fill gaps in current findings
2. Explore promising directions mentioned in the papers
3. Address potential contradictions or uncertainties

Format each suggestion as a dictionary with:
- query: the follow-up question
- rationale: why this would be valuable to explore"""

        response = self._llm_call(self.SYSTEM_PROMPT, user_prompt)
        return response 