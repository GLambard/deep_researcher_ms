"""
Prompt engineering module for Deep Researcher.
Handles query processing, response generation, and literature integration.

This module contains the core prompt engineering logic for breaking down
complex research queries, generating structured responses, and integrating
academic literature findings into coherent research summaries.
"""

from typing import List, Dict, Optional, Tuple  # For type hints
from dataclasses import dataclass  # For creating structured data classes
from .search.paper import Paper  # For paper data model
from .ollama_client import OllamaClient  # For LLM text generation
import re  # For regular expression operations

@dataclass
class QueryComponent:
    """
    Represents a component of the broken-down research query.
    
    Each query component consists of a main topic and related subtopics,
    optionally with a year range for temporal filtering of results.
    """
    topic: str  # The main topic (e.g., "CRISPR gene editing")
    subtopics: List[str]  # Related subtopics (e.g., ["ethical considerations", "clinical applications"])
    year_range: Optional[Tuple[int, int]] = None  # Optional year range for filtering (e.g., (2020, 2023))

@dataclass
class ResearchResponse:
    """
    Represents the complete research response with all components.
    
    This structured class holds all elements of a complete research response,
    including the initial AI-generated content, relevant papers found,
    the final synthesized summary, and formal citations.
    """
    initial_response: str  # Initial LLM response before literature integration
    papers: List[Paper]  # Relevant papers found during search
    final_summary: str  # Final synthesized summary with literature integration
    citations: List[str]  # Formatted academic citations

class PromptEngineer:
    """
    Handles query processing and response generation using prompt engineering techniques.
    
    This class is responsible for:
    1. Breaking down complex research queries into structured components
    2. Generating initial responses based on model knowledge
    3. Creating targeted search queries for literature search
    4. Integrating literature findings into comprehensive research summaries
    """
    
    def __init__(self, ollama_client: OllamaClient):
        """
        Initialize the prompt engineer with an Ollama client for LLM access.
        
        Parameters:
        -----------
        ollama_client: Client for generating text responses using Ollama LLM API
            This client handles all communication with the language model
        """
        self.ollama = ollama_client
    
    ## TODO: 
    # - Check a better structuration of the outputs
    # 
    def process_query(self, query: str) -> List[QueryComponent]:
        """
        Break down a complex research query into structured components.
        
        This method:
        1. Sends the query to the LLM with instructions for structured breakdown
        2. Parses the LLM response into QueryComponent objects
        3. Returns a list of components for systematic research
        
        Parameters:
        -----------
        query: The user's original research query
            
        Returns:
        --------
        list: List of QueryComponent objects representing the structured breakdown
              Each component contains a main topic and related subtopics
        """
        # Generate a structured breakdown of the query using the LLM
        # The prompt instructs the model to format the response in a specific way
        # that can be easily parsed into QueryComponent objects
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
        
        # Get the structured breakdown from the LLM
        response = self.ollama.generate(prompt)
        
        # Parse the response into QueryComponents
        # This section analyzes the text line by line to extract topics and subtopics
        components = []
        current_topic = None
        current_subtopics = []
        
        # Process each line of the response
        for line in response.split('\n'):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            if line.startswith('- '):  # Main topic line
                # If we have a previous topic, save it before starting a new one
                if current_topic:
                    components.append(QueryComponent(
                        topic=current_topic,
                        subtopics=current_subtopics
                    ))
                # Start a new topic
                current_topic = line[2:].strip()
                current_subtopics = []
            elif line.startswith('* '):  # Subtopic line
                current_subtopics.append(line[2:].strip())
        
        # Don't forget to add the last topic after loop ends
        if current_topic:
            components.append(QueryComponent(
                topic=current_topic,
                subtopics=current_subtopics
            ))
        
        return components
    
    def generate_initial_response(self, query: str, components: List[QueryComponent]) -> str:
        """
        Generate an initial response to the query before literature search.
        
        This method creates a well-structured response that:
        1. Addresses each component of the research query
        2. Provides context and background information
        3. Identifies areas where literature support would be valuable
        
        Parameters:
        -----------
        query: The original research query from the user
        components: Broken down query components from process_query()
            
        Returns:
        --------
        str: A comprehensive initial response based on the LLM's knowledge
        """
        # Format the components into a string for the prompt
        # This recreates the hierarchical structure for the LLM
        topics_str = "\n".join(
            f"- {comp.topic}\n" + "\n".join(f"  * {sub}" for sub in comp.subtopics)
            for comp in components
        )
        
        # Create a detailed prompt that guides the LLM to generate
        # a comprehensive initial response addressing all query components
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
        
        # Generate and return the initial response
        return self.ollama.generate(prompt)
    
    def generate_search_queries(self, components: List[QueryComponent]) -> List[str]:
        """
        Generate specific search queries for each component of the research question.
        
        This method creates targeted search queries by:
        1. Using each main topic as a standalone query
        2. Combining each main topic with each of its subtopics
        
        This approach ensures comprehensive coverage of the research area
        while maintaining specificity for relevant results.
        
        Parameters:
        -----------
        components: Query components from process_query()
            
        Returns:
        --------
        list: List of search queries to use for literature search
        """
        search_queries = []
        
        # Process each component to generate search queries
        for component in components:
            # Add the main topic as a standalone query
            # This provides broader context papers
            search_queries.append(component.topic)
            
            # Add queries combining main topic with each subtopic
            # These provide more specific, targeted papers
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
        Integrate literature findings with the initial AI response.
        
        This critical method:
        1. Combines AI knowledge with evidence from academic papers
        2. Creates a cohesive summary that incorporates literature findings
        3. Generates proper academic citations for the papers
        
        Parameters:
        -----------
        initial_response: The AI-generated response before literature search
        papers: List of relevant papers found during search
            
        Returns:
        --------
        ResearchResponse: Complete research response with summary and citations
        """
        # Create a formatted summary of the papers for the LLM
        # This extracts key metadata in a consistent format
        papers_summary = "\n\n".join(
            f"Title: {paper.title}\n"
            f"Authors: {', '.join(paper.authors)}\n"
            f"Year: {paper.year}\n"
            f"Abstract: {paper.abstract}\n"
            for paper in papers
        )
        
        # Generate integrated summary using a detailed prompt
        # This prompt guides the LLM to synthesize information and add citations
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
        
        # Generate the integrated response
        response = self.ollama.generate(prompt)
        
        # Parse the response to separate summary from citations
        # The response should have a "Citations:" section
        parts = response.split("Citations:", 1)
        final_summary = parts[0].strip()
        citations = []
        
        # If citations section exists, extract individual citations
        if len(parts) > 1:
            # Get each citation as a separate item
            citations = [
                cite.strip()
                for cite in parts[1].split("\n")
                if cite.strip()  # Skip empty lines
            ]
        
        # Return a structured ResearchResponse object with all components
        return ResearchResponse(
            initial_response=initial_response,
            papers=papers,
            final_summary=final_summary,
            citations=citations
        ) 