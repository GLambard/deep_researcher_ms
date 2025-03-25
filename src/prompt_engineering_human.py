"""
Prompt engineering module for Deep Researcher - Human Planning Approach.

This module implements a human-like, step-by-step literature review process that
mimics how researchers systematically approach scientific questions. It follows a
defined workflow from defining the research question to synthesizing findings.
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from .search.paper import Paper
from .ollama_client import OllamaClient
import re

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
    Handles research following a human-like literature review process.
    
    This class implements a step-by-step approach that mimics how a human researcher
    would conduct a literature review, progressing from defining the research question
    to a synthesis of findings from the literature.
    """
    
    def __init__(self, ollama_client: OllamaClient):
        """
        Initialize the prompt engineer with an Ollama client for LLM access.
        
        Parameters:
        -----------
        ollama_client: Client for generating text responses using Ollama LLM API
        """
        self.ollama = ollama_client
    
    def define_research_question(self, query: str) -> Dict[str, Any]:
        """
        STEP 1: Define the research question and scope.
        
        This method articulates the research question clearly and defines
        inclusion/exclusion criteria for the literature review.
        
        Parameters:
        -----------
        query: The user's original research query
            
        Returns:
        --------
        dict: Research question definition and scope parameters
        """
        prompt = f"""
        You're conducting a literature review on the following query:
        
        "{query}"
        
        1. Clearly articulate the research question.
        2. Define inclusion/exclusion criteria for studies (e.g., study type, publication date, methodology).
        3. Identify key concepts and terms for searching.
        
        Format your response as a JSON object with these fields:
        - research_question: The clear research question
        - inclusion_criteria: List of criteria for including studies
        - exclusion_criteria: List of criteria for excluding studies
        - key_terms: List of key search terms
        - time_frame: Suggested time range for the search (e.g., "2018-2023")
        
        Only output the JSON object, nothing else.
        """
        
        response = self.ollama.generate(prompt)
        
        # Parse the JSON response
        # For simplicity, we'll use a regex approach here
        import json
        try:
            # Extract JSON from the response
            json_str = re.search(r'\{.*\}', response, re.DOTALL)
            if json_str:
                return json.loads(json_str.group(0))
            else:
                # Fallback if JSON parsing fails
                return {
                    "research_question": query,
                    "inclusion_criteria": ["Recent studies", "Peer-reviewed publications"],
                    "exclusion_criteria": ["Opinion pieces", "Non-English publications"],
                    "key_terms": query.split(),
                    "time_frame": "2018-2023"
                }
        except:
            # Fallback if JSON parsing fails
            return {
                "research_question": query,
                "inclusion_criteria": ["Recent studies", "Peer-reviewed publications"],
                "exclusion_criteria": ["Opinion pieces", "Non-English publications"],
                "key_terms": query.split(),
                "time_frame": "2018-2023"
            }
    
    def identify_sources(self, research_definition: Dict[str, Any]) -> List[str]:
        """
        STEP 2: Identify appropriate sources and databases.
        
        This method selects appropriate sources that cover the field of study.
        
        Parameters:
        -----------
        research_definition: Research question definition from define_research_question()
            
        Returns:
        --------
        list: List of recommended sources to search
        """
        # For our implementation, we'll return a fixed list of sources
        # In a real implementation, this could be more dynamic based on the research area
        return ["arxiv", "semantic_scholar", "open_alex", "chemrxiv"]
    
    def process_query(self, query: str) -> List[QueryComponent]:
        """
        Process the query into structured components based on the defined research question.
        
        This is a wrapper method that:
        1. Defines the research question and scope
        2. Converts the research definition into query components
        
        Parameters:
        -----------
        query: The user's original research query
            
        Returns:
        --------
        list: List of QueryComponent objects representing the structured breakdown
        """
        # Define the research question
        research_def = self.define_research_question(query)
        
        # Create a prompt to break down the research question into components
        prompt = f"""
        Break down this research question into main topics and subtopics:
        
        Research Question: {research_def['research_question']}
        Key Terms: {', '.join(research_def['key_terms'])}
        
        Format:
        - Main topic 1
          * Subtopic 1.1
          * Subtopic 1.2
        - Main topic 2
          * Subtopic 2.1
        
        Only output the structured list, nothing else.
        """
        
        # Get the response from the LLM
        response = self.ollama.generate(prompt)
        
        # Parse the response into QueryComponents
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
            
        # Try to add year range if available in research definition
        if "time_frame" in research_def:
            try:
                time_frame = research_def["time_frame"]
                if "-" in time_frame:
                    start_year, end_year = map(int, time_frame.split("-"))
                    for component in components:
                        component.year_range = (start_year, end_year)
            except:
                pass
        
        return components
    
    def generate_search_queries(self, components: List[QueryComponent]) -> List[str]:
        """
        STEP 3: Generate search queries based on the research components.
        
        This method creates targeted search queries to retrieve relevant articles.
        
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
            search_queries.append(component.topic)
            
            # Add queries combining main topic with each subtopic
            for subtopic in component.subtopics:
                combined_query = f"{component.topic} {subtopic}"
                search_queries.append(combined_query)
        
        return search_queries
    
    def generate_initial_response(self, query: str, components: List[QueryComponent]) -> str:
        """
        Generate an initial assessment of the research question.
        
        This method creates a preliminary response that outlines the current
        understanding of the topic before delving into the literature.
        
        Parameters:
        -----------
        query: The original research query from the user
        components: Broken down query components from process_query()
            
        Returns:
        --------
        str: A preliminary assessment of the research question
        """
        # Format the components into a string for the prompt
        topics_str = "\n".join(
            f"- {comp.topic}\n" + "\n".join(f"  * {sub}" for sub in comp.subtopics)
            for comp in components
        )
        
        prompt = f"""
        Before conducting a literature search, provide a preliminary assessment of this research question:
        
        Research Question: {query}
        
        The question has been broken down into these components:
        {topics_str}
        
        Your assessment should:
        1. Provide background context for the research question
        2. Describe current understanding of the topic based on general knowledge
        3. Identify what specific information needs to be obtained from the literature
        4. Outline expected challenges or controversies in the research area
        
        Your preliminary assessment:
        """
        
        return self.ollama.generate(prompt)
    
    def screen_papers_by_relevance(self, papers: List[Paper], query: str) -> List[Paper]:
        """
        STEP 4 & 5: Screen papers by relevance (title and abstract screening).
        
        This method mimics the human process of:
        1. Scanning titles for direct relevance to the research question
        2. Reading abstracts to evaluate whether the study aligns with the query
        
        Parameters:
        -----------
        papers: List of papers to screen
        query: The research query
        
        Returns:
        --------
        list: List of papers that passed the screening
        """
        if not papers:
            return []
            
        # Create a prompt to evaluate relevance
        papers_info = "\n\n".join([
            f"Paper {i+1}:\nTitle: {paper.title}\nAbstract: {paper.abstract}"
            for i, paper in enumerate(papers)
        ])
        
        prompt = f"""
        Research Question: {query}
        
        Evaluate the relevance of the following papers to the research question:
        
        {papers_info}
        
        For each paper, determine if it is relevant based on:
        1. Whether the title indicates direct relevance
        2. Whether the abstract aligns with the research question
        
        Return ONLY a comma-separated list of paper numbers that are relevant.
        Example: "1, 3, 5" (meaning papers 1, 3, and 5 are relevant)
        
        Relevant papers:
        """
        
        response = self.ollama.generate(prompt)
        
        # Parse the response to get indices of relevant papers
        try:
            # Extract numbers from the response
            relevant_indices = [int(idx.strip()) - 1 for idx in response.replace(" ", "").split(",") if idx.strip().isdigit()]
            
            # Return relevant papers
            return [papers[i] for i in relevant_indices if 0 <= i < len(papers)]
        except:
            # If parsing fails, return all papers
            return papers
    
    def extract_key_findings(self, papers: List[Paper]) -> Dict[str, List[str]]:
        """
        STEP 6: Extract key findings from full-text review.
        
        This method extracts key findings and evidence from the papers,
        which would normally involve a full-text review.
        
        Parameters:
        -----------
        papers: List of relevant papers
        
        Returns:
        --------
        dict: Extracted key findings grouped by paper ID
        """
        findings = {}
        
        for i, paper in enumerate(papers):
            prompt = f"""
            Extract key findings from this paper:
            
            Title: {paper.title}
            Authors: {', '.join(paper.authors)}
            Abstract: {paper.abstract}
            
            Extract:
            1. The main methodology used
            2. Key results or findings
            3. Limitations mentioned
            4. Implications or applications
            
            List your findings as bullet points.
            """
            
            response = self.ollama.generate(prompt)
            
            # Store findings for the paper
            findings[f"paper_{i}"] = [
                line.strip() for line in response.split("\n")
                if line.strip() and line.strip().startswith("-")
            ]
        
        return findings
    
    def synthesize_findings(self, findings: Dict[str, List[str]], query: str) -> str:
        """
        STEP 7: Synthesize extracted data into themes or categories.
        
        This method organizes extracted data into themes, identifies trends,
        and develops a structured summary.
        
        Parameters:
        -----------
        findings: Extracted findings from papers
        query: The research query
        
        Returns:
        --------
        str: Synthesized findings
        """
        # Flatten findings into a single list
        all_findings = []
        for paper_id, paper_findings in findings.items():
            all_findings.extend(paper_findings)
        
        findings_text = "\n".join(all_findings)
        
        prompt = f"""
        Research Question: {query}
        
        Key findings from literature:
        {findings_text}
        
        Synthesize these findings into a cohesive summary:
        1. Organize findings into themes or categories
        2. Identify trends, conflicting results, or research gaps
        3. Provide a structured synthesis that addresses the research question
        
        Your synthesis:
        """
        
        return self.ollama.generate(prompt)
    
    def integrate_literature(self, initial_response: str, papers: List[Paper]) -> ResearchResponse:
        """
        STEPS 7-8: Integrate literature findings into a final response.
        
        This method implements the complete workflow from literature screening
        to final synthesis, creating a comprehensive research response.
        
        Parameters:
        -----------
        initial_response: Preliminary assessment of the research question
        papers: List of papers found during search
        
        Returns:
        --------
        ResearchResponse: Complete research response with summary and citations
        """
        # If no papers found, return the initial response
        if not papers:
            return ResearchResponse(
                initial_response=initial_response,
                papers=[],
                final_summary=initial_response,
                citations=[]
            )
        
        # Extract a research question from the initial response
        research_question = self._extract_research_question(initial_response)
        
        # STEP 4-5: Screen papers by relevance
        relevant_papers = self.screen_papers_by_relevance(papers, research_question)
        
        # STEP 6: Extract key findings from papers
        findings = self.extract_key_findings(relevant_papers)
        
        # STEP 7: Synthesize findings
        synthesis = self.synthesize_findings(findings, research_question)
        
        # STEP 8: Construct final answer with citations
        papers_summary = "\n\n".join(
            f"Title: {paper.title}\n"
            f"Authors: {', '.join(paper.authors)}\n"
            f"Year: {paper.year}\n"
            f"Abstract: {paper.abstract}\n"
            for paper in relevant_papers
        )
        
        prompt = f"""
        Research Question: {research_question}
        
        Initial Assessment:
        {initial_response}
        
        Literature Synthesis:
        {synthesis}
        
        Papers:
        {papers_summary}
        
        Create a final comprehensive research response that:
        1. Integrates the initial assessment with literature findings
        2. Addresses the research question with evidence from the papers
        3. Adds formal citations for each paper mentioned
        4. Maintains an academic, evidence-based tone
        
        Provide your response in two parts:
        1. Final Summary
        2. Citations (in IEEE format)
        
        Your response:
        """
        
        response = self.ollama.generate(prompt)
        
        # Parse the response to separate summary from citations
        parts = response.split("Citations:", 1)
        final_summary = parts[0].strip()
        citations = []
        
        if len(parts) > 1:
            citations = [
                cite.strip()
                for cite in parts[1].split("\n")
                if cite.strip()
            ]
        
        return ResearchResponse(
            initial_response=initial_response,
            papers=relevant_papers,
            final_summary=final_summary,
            citations=citations
        )
    
    def _extract_research_question(self, text: str) -> str:
        """
        Helper method to extract a research question from text.
        
        Parameters:
        -----------
        text: Text to extract research question from
        
        Returns:
        --------
        str: Extracted research question
        """
        # Use a simple prompt to extract the main research question
        prompt = f"""
        Extract the main research question from this text:
        
        {text}
        
        Return only the research question as a single sentence.
        """
        
        response = self.ollama.generate(prompt)
        
        # Return the first non-empty line
        for line in response.split("\n"):
            if line.strip():
                return line.strip()
                
        # Fallback
        return text 