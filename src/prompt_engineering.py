"""
Prompt engineering module for Deep Researcher.
Handles query processing, response generation, and literature integration.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from .search.paper import Paper
from .ollama_client import OllamaClient
import re

@dataclass
class QueryComponent:
    """Represents a component of the broken-down query."""
    topic: str
    subtopics: List[str]
    year_range: Optional[Tuple[int, int]] = None

@dataclass
class ResearchResponse:
    """Represents the complete research response."""
    initial_response: str
    papers: List[Paper]
    final_summary: str
    citations: List[str]

class PromptEngineer:
    """
    Handles query processing and response generation.
    """
    
    def __init__(self, ollama_client: OllamaClient):
        """
        Initialize the prompt engineer.
        
        Parameters
        ----------
        ollama_client : OllamaClient
            Client for generating text responses
        """
        self.ollama = ollama_client
    
    def process_query(self, query: str) -> List[QueryComponent]:
        """
        Break down a complex query into components.
        
        Parameters
        ----------
        query : str
            The user's research query
            
        Returns
        -------
        list
            List of QueryComponent objects
        """
        # Generate a structured breakdown of the query using LLM
        prompt = f"""
        Break down this research query into main topics and subtopics:
        Query: {query}
        
        Format:
        - Main topic 1
          * Subtopic 1.1
          * Subtopic 1.2
        - Main topic 2
          * Subtopic 2.1
        
        Only output the structured list, nothing else.
        """
        
        response = self.ollama.generate(prompt)
        
        # Parse the response into QueryComponents
        components = []
        current_topic = None
        current_subtopics = []
        
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('- '):
                # If we have a previous topic, save it
                if current_topic:
                    components.append(QueryComponent(
                        topic=current_topic,
                        subtopics=current_subtopics
                    ))
                # Start new topic
                current_topic = line[2:].strip()
                current_subtopics = []
            elif line.startswith('* '):
                current_subtopics.append(line[2:].strip())
        
        # Add the last topic
        if current_topic:
            components.append(QueryComponent(
                topic=current_topic,
                subtopics=current_subtopics
            ))
        
        return components
    
    def generate_initial_response(self, query: str, components: List[QueryComponent]) -> str:
        """
        Generate an initial response to the query before literature search.
        
        Parameters
        ----------
        query : str
            The original query
        components : list
            Broken down query components
            
        Returns
        -------
        str
            Initial response
        """
        # Create a prompt that includes the original query and its components
        topics_str = "\n".join(
            f"- {comp.topic}\n" + "\n".join(f"  * {sub}" for sub in comp.subtopics)
            for comp in components
        )
        
        prompt = f"""
        Generate a comprehensive initial response to this research query:
        
        Query: {query}
        
        The query has been broken down into these components:
        {topics_str}
        
        Provide a detailed response that:
        1. Addresses each component of the query
        2. Highlights key concepts and methodologies
        3. Identifies potential areas where literature support would be valuable
        4. Maintains scientific accuracy and academic tone
        
        Response:
        """
        
        return self.ollama.generate(prompt)
    
    def generate_search_queries(self, components: List[QueryComponent]) -> List[str]:
        """
        Generate specific search queries for each component.
        
        Parameters
        ----------
        components : list
            Query components to generate searches for
            
        Returns
        -------
        list
            List of search queries
        """
        search_queries = []
        
        for component in components:
            # Add main topic query
            search_queries.append(component.topic)
            
            # Add queries combining main topic with each subtopic
            for subtopic in component.subtopics:
                combined_query = f"{component.topic} {subtopic}"
                search_queries.append(combined_query)
        
        return search_queries
    
    def integrate_literature(
        self,
        initial_response: str,
        papers: List[Paper]
    ) -> ResearchResponse:
        """
        Integrate literature findings with the initial response.
        
        Parameters
        ----------
        initial_response : str
            The initial AI-generated response
        papers : list
            List of relevant papers found
            
        Returns
        -------
        ResearchResponse
            Complete response with summary and citations
        """
        # Create a summary of the papers
        papers_summary = "\n\n".join(
            f"Title: {paper.title}\n"
            f"Authors: {', '.join(paper.authors)}\n"
            f"Year: {paper.year}\n"
            f"Abstract: {paper.abstract}\n"
            for paper in papers
        )
        
        # Generate integrated summary
        prompt = f"""
        Integrate this initial response with the findings from academic literature:
        
        Initial Response:
        {initial_response}
        
        Relevant Literature:
        {papers_summary}
        
        Create a comprehensive summary that:
        1. Combines the initial insights with supporting evidence from the papers
        2. Highlights where the literature confirms or extends the initial response
        3. Adds specific citations to support key points
        4. Maintains a clear and academic tone
        
        Provide the response in two parts:
        1. Final Summary
        2. Citations (in IEEE format)
        
        Response:
        """
        
        response = self.ollama.generate(prompt)
        
        # Split response into summary and citations
        parts = response.split("Citations:", 1)
        final_summary = parts[0].strip()
        citations = []
        
        if len(parts) > 1:
            # Extract citations
            citations = [
                cite.strip()
                for cite in parts[1].split("\n")
                if cite.strip()
            ]
        
        return ResearchResponse(
            initial_response=initial_response,
            papers=papers,
            final_summary=final_summary,
            citations=citations
        ) 