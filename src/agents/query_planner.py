"""
Query planning agent for breaking down complex research queries.

This module implements a specialized agent that takes complex, broad
research questions and breaks them down into targeted sub-queries
for more effective literature search and analysis.
"""

from typing import Dict, Any, List  # For type hints
from .base_agent import BaseAgent  # Base class for all agents

## TODO: 
# Ask a first question to the user to refine the query as the o1 PRO deepresearch model from OpenAI
# Refine the system and user prompts
# Improve how the output formatting is handled
#
class QueryPlannerAgent(BaseAgent):
    """
    Agent responsible for breaking down complex queries into sub-queries.
    
    This agent serves as the entry point for the research process,
    taking a user's broad research question and creating a systematic
    plan for exploring it through more specific, focused inquiries.
    
    Breaking down the query is crucial because:
    1. It makes literature search more precise and comprehensive
    2. It ensures all aspects of a complex topic are explored
    3. It creates a logical research flow from foundations to specifics
    """
    
    # System prompt that defines the agent's role and behavior
    # This instructs the LLM to act as a research assistant specialized in query decomposition
    SYSTEM_PROMPT = """You are an expert research assistant who helps break down complex research queries into specific, focused sub-queries.
Your goal is to create a systematic plan for exploring the research question thoroughly.
Break down the query into logical steps, considering different aspects and potential directions."""

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Break down a complex research query into structured sub-queries.
        
        This method:
        1. Takes a complex research question
        2. Asks the LLM to decompose it into logical components
        3. Returns a structured plan with specific sub-queries
        4. Includes metadata for each sub-query (key terms, year ranges, etc.)
        
        Parameters:
        -----------
        input_data: Dictionary containing the input to process
            Must include a 'query' key with the main research query text
            
        Returns:
        --------
        dict: Contains the original query and the decomposition plan
            The plan is structured as JSON with sub-queries and strategy
        """
        # Extract the main query from the input data
        main_query = input_data["query"]
        
        # Create a detailed prompt for breaking down the query
        # This prompt instructs the LLM to produce a structured decomposition
        # with specific elements that will guide the research process
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

        # Send the prompt to the LLM and get the structured response
        # The LLM will return a JSON string with the decomposed query plan
        response = self._llm_call(self.SYSTEM_PROMPT, user_prompt)
        
        # Return the original query and the plan
        # We don't parse the JSON here, as it will be handled by the parent component
        # This allows for more flexible error handling at the orchestration level
        return {
            "original_query": main_query,
            "plan": response  # Raw response containing JSON-formatted plan
        }
    
    ## TODO: 
    # The context may not be handled correctly because of reduced context window
    # Check how to improve the context handling by successive refinements a-la map-reduce
    # Need to incorporate a tokens counter such as tiktoken
    # 
    def generate_follow_up_queries(self, context: Dict[str, Any]) -> List[str]:
        """
        Generate follow-up queries based on current research findings.
        
        This method is called after initial research has been conducted to:
        1. Identify gaps in the current findings
        2. Suggest new directions based on what's been discovered
        3. Help deepen the research in promising areas
        
        Follow-up queries are crucial for iterative research that
        builds on initial findings rather than starting from scratch.
        
        Parameters:
        -----------
        context: Research context dictionary containing:
            - original_query: The initial research question
            - summary: Current findings and research progress
            
        Returns:
        --------
        str: LLM response containing suggested follow-up queries
             Each with a rationale for why it would be valuable
        """
        # Create a prompt that includes the original query and current findings
        # This gives the LLM context to generate relevant follow-up questions
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

        # Get suggestions from the LLM using the same system prompt
        # This maintains consistent agent behavior
        response = self._llm_call(self.SYSTEM_PROMPT, user_prompt)
        return response 