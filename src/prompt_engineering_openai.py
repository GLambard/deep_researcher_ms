"""
Prompt engineering module for Deep Researcher - OpenAI Deep Research Planning Approach.

This module implements the structured research planning approach outlined in the 
OpenAI deep research planning pseudo-algorithm. It follows a multi-phase approach 
with clear state transitions, managing context window constraints and iterative 
improvements to the final output.
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from .search.paper import Paper
from .ollama_client import OllamaClient
import re
import json

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
    Handles query processing and response generation using the OpenAI research planning approach.
    
    This class implements the multi-phase research process with state transitions:
    1. PLAN_PHASE: Interpret user query and plan the approach
    2. RESEARCH_PHASE: Identify knowledge gaps and gather information
    3. COMPOSING_PHASE: Create draft sections based on gathered information
    4. REVIEW_PHASE: Analyze the draft for issues
    5. REVISION_PHASE: Improve identified issues
    6. FINALIZE_PHASE: Polish the final output
    """
    
    def __init__(self, ollama_client: OllamaClient):
        """
        Initialize the prompt engineer with an Ollama client for LLM access.
        
        Parameters:
        -----------
        ollama_client: Client for generating text responses using Ollama LLM API
        """
        self.ollama = ollama_client
        self.max_external_calls = 10  # Default limit on external API calls
        self.internal_memory = {}  # Structured store for retrieved data, references, etc.
    
    def process_query(self, query: str) -> List[QueryComponent]:
        """
        PLAN_PHASE: Interpret user query and plan the approach.
        
        This method:
        1. Analyzes the query to identify topic, level of detail needed, etc.
        2. Creates an initial outline for how to answer the query
        3. Returns structured query components for the research phase
        
        Parameters:
        -----------
        query: The user's original research query
            
        Returns:
        --------
        list: List of QueryComponent objects representing the structured breakdown
        """
        # Create a planning prompt that asks the LLM to analyze the query
        prompt = f"""
        You are in the PLAN_PHASE of a research process.
        
        ANALYZE this research query: "{query}"
        
        1. Identify the main topic/domain
        2. Determine level of detail needed
        3. Extract potential keywords and subtopics
        
        Then CREATE an initial outline with main topics and subtopics in this format:
        - Main topic 1
          * Subtopic 1.1
          * Subtopic 1.2
        - Main topic 2
          * Subtopic 2.1
        
        Only output the structured outline, nothing else.
        """
        
        # Get the structured breakdown from the LLM
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
        
        # Store components in internal memory for reference
        self.internal_memory['query_components'] = components
        self.internal_memory['original_query'] = query
        self.internal_memory['state'] = 'RESEARCH_PHASE'
        
        return components
    
    def identify_knowledge_gaps(self, components: List[QueryComponent]) -> List[Dict[str, Any]]:
        """
        RESEARCH_PHASE: Identify knowledge gaps that need external data.
        
        This is a critical part of the research phase where the system determines
        what specific information needs to be retrieved.
        
        Parameters:
        -----------
        components: Query components from process_query()
            
        Returns:
        --------
        list: Knowledge gaps that need to be filled
        """
        # Format the components for the prompt
        topics_str = "\n".join(
            f"- {comp.topic}\n" + "\n".join(f"  * {sub}" for sub in comp.subtopics)
            for comp in components
        )
        
        query = self.internal_memory.get('original_query', 'Research query')
        
        prompt = f"""
        You are in the RESEARCH_PHASE of answering this query:
        "{query}"
        
        Based on the query outline:
        {topics_str}
        
        IDENTIFY specific knowledge gaps that need external data or citations.
        For each knowledge gap, specify:
        1. The specific question or information needed
        2. Potential search terms to find this information
        3. Why this information is crucial for answering the query
        
        Format your response as a JSON array of knowledge gaps, each with:
        - "topic": The topic this gap relates to
        - "question": The specific question this gap addresses
        - "search_terms": Array of 2-4 potential search terms
        - "importance": Why this information is important (brief)
        
        Return only the JSON array, nothing else.
        """
        
        response = self.ollama.generate(prompt)
        
        # Extract the JSON array of knowledge gaps
        try:
            # Try to find a JSON array in the response using regex
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                knowledge_gaps = json.loads(json_match.group(0))
            else:
                # Fallback if no JSON array found
                knowledge_gaps = [
                    {
                        "topic": comp.topic,
                        "question": f"What is the latest research on {comp.topic}?",
                        "search_terms": [comp.topic] + comp.subtopics[:2],
                        "importance": "Core component of the research query"
                    }
                    for comp in components
                ]
                
            # Store in internal memory
            self.internal_memory['knowledge_gaps'] = knowledge_gaps
            return knowledge_gaps
            
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            knowledge_gaps = [
                {
                    "topic": comp.topic,
                    "question": f"What is the latest research on {comp.topic}?",
                    "search_terms": [comp.topic] + comp.subtopics[:2],
                    "importance": "Core component of the research query"
                }
                for comp in components
            ]
            
            # Store in internal memory
            self.internal_memory['knowledge_gaps'] = knowledge_gaps
            return knowledge_gaps
    
    def generate_search_queries(self, components: List[QueryComponent]) -> List[str]:
        """
        RESEARCH_PHASE: Generate specific search queries to fill knowledge gaps.
        
        This method:
        1. Identifies knowledge gaps
        2. Formulates targeted search queries to fill these gaps
        
        Parameters:
        -----------
        components: Query components from process_query()
            
        Returns:
        --------
        list: List of search queries to use for literature search
        """
        # First identify knowledge gaps
        knowledge_gaps = self.identify_knowledge_gaps(components)
        
        # Generate search queries based on knowledge gaps
        search_queries = []
        
        for gap in knowledge_gaps:
            # Add the main search terms as individual queries
            for term in gap["search_terms"]:
                if term not in search_queries:
                    search_queries.append(term)
            
            # Add the topic itself
            if gap["topic"] not in search_queries:
                search_queries.append(gap["topic"])
        
        # Ensure we don't exceed our search budget
        # We'll need some budget for later refinement, so limit initial search
        max_initial_queries = min(self.max_external_calls - 2, len(search_queries))
        search_queries = search_queries[:max_initial_queries]
        
        # Track external calls used
        self.internal_memory['external_calls_used'] = len(search_queries)
        
        return search_queries
    
    def generate_initial_response(self, query: str, components: List[QueryComponent]) -> str:
        """
        Generate an initial response that will be refined through research.
        
        This creates a foundation for the final answer based on what the model
        already knows, which will be verified and enhanced with citations later.
        
        Parameters:
        -----------
        query: The original research query from the user
        components: Broken down query components from process_query()
            
        Returns:
        --------
        str: Initial response based on the model's knowledge
        """
        # Format the components into a string for the prompt
        topics_str = "\n".join(
            f"- {comp.topic}\n" + "\n".join(f"  * {sub}" for sub in comp.subtopics)
            for comp in components
        )
        
        # Create a detailed prompt that guides the LLM to generate
        # a comprehensive initial response addressing all query components
        prompt = f"""
        You are in the COMPOSING_PHASE of a research process.

        Create an initial draft response to this query:
        "{query}"
        
        Using the following outline:
        {topics_str}
        
        Your draft should:
        1. Address each component of the query
        2. Be written in clear, organized paragraphs following the outline structure
        3. Indicate where external evidence will be needed (mark with [CITATION NEEDED])
        4. Maintain an academic but accessible tone
        
        This is just an initial draft that will be refined with research.
        """
        
        # Generate and return the initial response
        initial_response = self.ollama.generate(prompt)
        
        # Store in internal memory
        self.internal_memory['initial_response'] = initial_response
        self.internal_memory['state'] = 'RESEARCH_PHASE'
        
        return initial_response
    
    def chunk_and_summarize(self, papers: List[Paper], max_papers_per_chunk: int = 3) -> List[Dict[str, Any]]:
        """
        Chunk papers and summarize them to fit context window.
        
        This method:
        1. Groups papers into manageable chunks
        2. Summarizes each chunk to preserve key information
        3. Associates each chunk with relevant topics
        
        Parameters:
        -----------
        papers: List of papers to process
        max_papers_per_chunk: Maximum papers to include in each chunk
            
        Returns:
        --------
        list: Summarized chunks of papers
        """
        chunks = []
        
        # Split papers into chunks
        for i in range(0, len(papers), max_papers_per_chunk):
            chunk_papers = papers[i:i + max_papers_per_chunk]
            
            # Create a summary of the chunk
            papers_text = "\n\n".join(
                f"Title: {paper.title}\n"
                f"Authors: {', '.join(paper.authors)}\n"
                f"Year: {paper.year}\n"
                f"Abstract: {paper.abstract}"
                for paper in chunk_papers
            )
            
            prompt = f"""
            Summarize the key findings from these papers:
            
            {papers_text}
            
            For each paper, extract:
            1. The main methodology
            2. Key results
            3. Limitations
            4. Implications
            
            Create a concise summary that preserves the essential information.
            """
            
            chunk_summary = self.ollama.generate(prompt)
            
            # Determine which query components this chunk relates to
            components = self.internal_memory.get('query_components', [])
            relevant_topics = []
            
            for comp in components:
                # Check if component topic or subtopics appear in the papers
                if any(comp.topic.lower() in paper.title.lower() or 
                       comp.topic.lower() in paper.abstract.lower() 
                       for paper in chunk_papers):
                    relevant_topics.append(comp.topic)
                    
                for subtopic in comp.subtopics:
                    if any(subtopic.lower() in paper.title.lower() or 
                           subtopic.lower() in paper.abstract.lower() 
                           for paper in chunk_papers):
                        if comp.topic not in relevant_topics:
                            relevant_topics.append(comp.topic)
            
            # Create chunk entry
            chunks.append({
                "papers": chunk_papers,
                "summary": chunk_summary,
                "relevant_topics": relevant_topics
            })
        
        # Store in internal memory
        self.internal_memory['paper_chunks'] = chunks
        return chunks
    
    def compose_section(self, topic: str, paper_chunks: List[Dict[str, Any]]) -> str:
        """
        COMPOSING_PHASE: Compose a section for a specific topic.
        
        This method:
        1. Gathers relevant paper chunks for the topic
        2. Creates a coherent section that incorporates the research
        
        Parameters:
        -----------
        topic: The topic for this section
        paper_chunks: Summarized paper chunks from chunk_and_summarize()
            
        Returns:
        --------
        str: Composed section with citations
        """
        # Gather relevant chunks for this topic
        relevant_chunks = [
            chunk for chunk in paper_chunks
            if topic in chunk.get("relevant_topics", [])
        ]
        
        # If no directly relevant chunks found, use all chunks
        if not relevant_chunks:
            relevant_chunks = paper_chunks
        
        # Combine chunk summaries into input for the LLM
        summaries = "\n\n".join(
            f"Paper Group {i+1}:\n{chunk['summary']}\n"
            f"Papers: {', '.join(p.title for p in chunk['papers'])}"
            for i, chunk in enumerate(relevant_chunks)
        )
        
        # Get the initial response section for this topic
        initial_response = self.internal_memory.get('initial_response', '')
        
        # Try to extract the relevant section from the initial response
        try:
            # Simple approach: find paragraph containing the topic
            paragraphs = initial_response.split('\n\n')
            relevant_paragraphs = [p for p in paragraphs if topic.lower() in p.lower()]
            initial_section = '\n\n'.join(relevant_paragraphs) if relevant_paragraphs else ""
        except:
            initial_section = ""
        
        prompt = f"""
        You are in the COMPOSING_PHASE of a research process.
        
        Compose a coherent section about: "{topic}"
        
        Research summaries from academic papers:
        {summaries}
        
        Initial draft section:
        {initial_section}
        
        Your task:
        1. Write a well-structured section that integrates the research findings
        2. Include specific citations to the papers (use numbering like [1], [2], etc.)
        3. Make sure every major claim has a citation
        4. Use an accessible academic style
        5. Be comprehensive but concise
        
        Section on {topic}:
        """
        
        section = self.ollama.generate(prompt)
        
        # Store in internal memory
        if 'composed_sections' not in self.internal_memory:
            self.internal_memory['composed_sections'] = {}
        
        self.internal_memory['composed_sections'][topic] = {
            "content": section,
            "sources": [
                {"title": paper.title, "authors": paper.authors, "year": paper.year}
                for chunk in relevant_chunks
                for paper in chunk["papers"]
            ]
        }
        
        return section
    
    def integrate_literature(
        self,
        initial_response: str,
        papers: List[Paper]
    ) -> ResearchResponse:
        """
        Main method that orchestrates the entire research process.
        
        This method implements the full workflow:
        1. PLAN_PHASE (already done in process_query())
        2. RESEARCH_PHASE (gathering papers)
        3. COMPOSING_PHASE (creating sections and draft)
        4. REVIEW_PHASE (checking for issues)
        5. REVISION_PHASE (making corrections)
        6. FINALIZE_PHASE (polishing the final answer)
        
        Parameters:
        -----------
        initial_response: The AI-generated response before literature search
        papers: List of relevant papers found during search
            
        Returns:
        --------
        ResearchResponse: Complete research response with summary and citations
        """
        # Store initial response in memory if not already there
        if 'initial_response' not in self.internal_memory:
            self.internal_memory['initial_response'] = initial_response
        
        # RESEARCH_PHASE: Process and chunk papers
        paper_chunks = self.chunk_and_summarize(papers)
        
        # COMPOSING_PHASE: Create sections for each topic
        components = self.internal_memory.get('query_components', [])
        if not components:
            # Extract topics from initial response if components not available
            topics = self._extract_topics_from_response(initial_response)
        else:
            topics = [comp.topic for comp in components]
        
        # Compose each section
        composed_sections = {}
        for topic in topics:
            section = self.compose_section(topic, paper_chunks)
            composed_sections[topic] = section
        
        # Update internal memory state
        self.internal_memory['state'] = 'REVIEW_PHASE'
        
        # REVIEW_PHASE: Check for issues with the draft
        draft_answer = self._combine_sections(composed_sections)
        issues = self._identify_issues(draft_answer, papers)
        
        # REVISION_PHASE: Address identified issues
        if issues:
            draft_answer = self._revise_draft(draft_answer, issues, papers)
        
        # FINALIZE_PHASE: Create final answer with citations
        final_response = self._finalize_answer(draft_answer, papers)
        
        return final_response
    
    def _extract_topics_from_response(self, response: str) -> List[str]:
        """
        Extract topics from the initial response.
        
        Parameters:
        -----------
        response: The initial response text
        
        Returns:
        --------
        list: Extracted topics
        """
        prompt = f"""
        Extract the main topics from this research response:
        
        {response}
        
        List only the main section titles/topics, one per line.
        """
        
        topics_text = self.ollama.generate(prompt)
        
        # Process the response to get topics
        topics = [
            line.strip() for line in topics_text.split('\n')
            if line.strip() and not line.strip().startswith('-')
        ]
        
        return topics
    
    def _combine_sections(self, sections: Dict[str, str]) -> str:
        """
        Combine individual sections into a cohesive draft.
        
        Parameters:
        -----------
        sections: Dictionary of topic to section content
        
        Returns:
        --------
        str: Combined draft
        """
        # Get original query
        query = self.internal_memory.get('original_query', 'Research query')
        
        # Format sections into a string
        sections_text = "\n\n".join(
            f"SECTION: {topic}\n\n{content}"
            for topic, content in sections.items()
        )
        
        prompt = f"""
        You are in the COMPOSING_PHASE of a research process.
        
        Research query: {query}
        
        Individual sections:
        {sections_text}
        
        Combine these sections into a cohesive draft that:
        1. Has a clear introduction that presents the topic
        2. Flows logically between sections
        3. Maintains all citations from the original sections
        4. Concludes with a summary of key points
        
        Create a complete, well-organized draft that answers the research query.
        """
        
        draft = self.ollama.generate(prompt)
        
        # Store in internal memory
        self.internal_memory['draft_answer'] = draft
        
        return draft
    
    def _identify_issues(self, draft: str, papers: List[Paper]) -> List[Dict[str, Any]]:
        """
        REVIEW_PHASE: Identify issues in the draft.
        
        Parameters:
        -----------
        draft: The draft answer
        papers: List of papers for fact-checking
        
        Returns:
        --------
        list: Identified issues
        """
        # Format papers for reference
        papers_text = "\n\n".join(
            f"Paper {i+1}:\nTitle: {paper.title}\n"
            f"Authors: {', '.join(paper.authors)}\n"
            f"Year: {paper.year}\n"
            f"Abstract: {paper.abstract}"
            for i, paper in enumerate(papers[:5])  # Limit to prevent context overflow
        )
        
        prompt = f"""
        You are in the REVIEW_PHASE of a research process.
        
        Review this draft answer:
        {draft}
        
        Key papers:
        {papers_text}
        
        Identify issues such as:
        1. Unclear transitions between sections
        2. Missing citations for major claims
        3. Potential factual inconsistencies with the papers
        4. Unclear explanations that need elaboration
        
        Format issues as a JSON array with:
        - "type": The type of issue (transition, citation, factual, clarity)
        - "location": Where in the draft the issue occurs
        - "description": Description of the problem
        - "suggestion": Suggested fix
        
        Return only the JSON array of issues, nothing else.
        """
        
        response = self.ollama.generate(prompt)
        
        # Parse the JSON response
        try:
            # Try to find a JSON array in the response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                issues = json.loads(json_match.group(0))
                # Store in internal memory
                self.internal_memory['issues'] = issues
                return issues
            else:
                return []
        except:
            return []
    
    def _revise_draft(self, draft: str, issues: List[Dict[str, Any]], papers: List[Paper]) -> str:
        """
        REVISION_PHASE: Revise the draft to address identified issues.
        
        Parameters:
        -----------
        draft: The original draft
        issues: Identified issues from _identify_issues()
        papers: List of papers for reference
        
        Returns:
        --------
        str: Revised draft
        """
        if not issues:
            return draft
            
        # Format issues for the prompt
        issues_text = "\n\n".join(
            f"Issue {i+1}: {issue['type']}\n"
            f"Location: {issue['location']}\n"
            f"Description: {issue['description']}\n"
            f"Suggestion: {issue['suggestion']}"
            for i, issue in enumerate(issues)
        )
        
        prompt = f"""
        You are in the REVISION_PHASE of a research process.
        
        Original draft:
        {draft}
        
        Issues to address:
        {issues_text}
        
        Revise the draft to address all the identified issues.
        Make your corrections directly in the full text, improving clarity,
        fixing transitions, adding citations, and ensuring factual accuracy.
        
        Return the complete revised draft.
        """
        
        revised_draft = self.ollama.generate(prompt)
        
        # Store in internal memory
        self.internal_memory['revised_draft'] = revised_draft
        self.internal_memory['state'] = 'FINALIZE_PHASE'
        
        return revised_draft
    
    def _finalize_answer(self, draft: str, papers: List[Paper]) -> ResearchResponse:
        """
        FINALIZE_PHASE: Polish and finalize the answer.
        
        Parameters:
        -----------
        draft: The revised draft
        papers: List of papers
        
        Returns:
        --------
        ResearchResponse: Final response with summary and citations
        """
        # Format papers for reference
        papers_text = "\n\n".join(
            f"Paper {i+1}:\nTitle: {paper.title}\n"
            f"Authors: {', '.join(paper.authors)}\n"
            f"Year: {paper.year}\n"
            f"Journal/Source: {paper.source_api}"
            for i, paper in enumerate(papers)
        )
        
        prompt = f"""
        You are in the FINALIZE_PHASE of a research process.
        
        Draft answer:
        {draft}
        
        Papers referenced:
        {papers_text}
        
        Create a final polished answer with:
        1. Improved formatting and organization
        2. IEEE-style citations for all referenced papers
        3. A separate "Citations" section at the end listing all references
        
        Your response must include two clearly separated sections:
        1. Final Summary (the main research answer)
        2. Citations (the formal reference list)
        
        Use "Citations:" as the heading for the citations section.
        """
        
        response = self.ollama.generate(prompt)
        
        # Split the response into summary and citations
        parts = response.split("Citations:", 1)
        final_summary = parts[0].strip()
        citations = []
        
        if len(parts) > 1:
            citations = [
                cite.strip()
                for cite in parts[1].split("\n")
                if cite.strip()
            ]
        
        # Get the initial response from memory
        initial_response = self.internal_memory.get('initial_response', '')
        
        # Create and return the final research response
        return ResearchResponse(
            initial_response=initial_response,
            papers=papers,
            final_summary=final_summary,
            citations=citations
        ) 