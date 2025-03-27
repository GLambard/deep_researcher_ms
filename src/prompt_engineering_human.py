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
                    "time_frame": "Undefined"
                }
        except:
            # Fallback if JSON parsing fails
            return {
                "research_question": query,
                "inclusion_criteria": ["Recent studies", "Peer-reviewed publications"],
                "exclusion_criteria": ["Opinion pieces", "Non-English publications"],
                "key_terms": query.split(),
                "time_frame": "Undefined"
            }
    
    def identify_sources(self, research_def: Dict[str, Any]) -> List[str]:
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
        # Default to using all available sources
        # This ensures we get broad coverage across all domains
        available_sources = ["arxiv", "open_alex", "chemrxiv"]
        
        # For most queries, open_alex provides the best coverage across domains
        # We prioritize it first, then use the others as backup
        prioritized_sources = ["open_alex"]
        for source in available_sources:
            if source not in prioritized_sources:
                prioritized_sources.append(source)
                
        # Limit to a maximum of 3 sources to prevent overwhelming
        return prioritized_sources[:3]
    
    def process_query(self, research_def: Dict[str, Any]) -> List[QueryComponent]:
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
    
    def generate_search_queries(self, query: str, initial_response: str, max_queries: int = 3) -> List[str]:
        """
        Generate focused search queries based on the initial response.
        
        Parameters:
        -----------
        query: The original research query
        initial_response: Initial assessment of the research question
        max_queries: Maximum number of search queries to generate (default: 3)
            
        Returns:
        --------
        list: List of search queries to use for literature search
        """
        # Extract key domain terms from the original query
        domain_terms_prompt = f"""
        Extract 5-7 key domain-specific technical terms from this query:
        
        "{query}"
        
        List ONLY the terms, one per line with no numbering or bullets.
        Focus on specialized terminology that defines the research domain and subject area.
        """
        
        domain_terms_response = self.ollama.generate(domain_terms_prompt)
        domain_terms = [term.strip() for term in domain_terms_response.split('\n') if term.strip()]
        
        # Create a domain context statement to anchor the search queries
        domain_context = " AND ".join(domain_terms[:3]) if domain_terms else query
        
        prompt = f"""
        Based on this initial research assessment:
        
        {initial_response}
        
        Generate exactly {max_queries} focused academic search queries to retrieve the most relevant papers about:
        
        {query}
        
        Each query MUST:
        1. Start with core domain terms: "{domain_context}"
        2. Target a specific subtopic or aspect of the research question
        3. Include precise technical terminology from the domain
        4. Use Boolean operators (AND, OR) with parentheses for complex combinations
        5. Be formatted for academic search engines
        6. Maintain focus on the primary domain without drifting to adjacent fields
        7. Be concise (3-6 terms plus operators)
        8. Include a year range filter using proper syntax: "year:[2018 TO 2024]"
        
        Return ONLY the search queries with no numbering, one per line.
        
        Example format:
        ("core terms") AND "specific aspect" AND year:[2018 TO 2024]
        ("core terms") AND ("different aspect" OR "related concept") AND year:[2018 TO 2024]
        title:("core terms") AND key_concept AND year:[2018 TO 2024]
        """
        
        response = self.ollama.generate(prompt)
        
        # Extract queries from the response
        raw_queries = [line.strip() for line in response.split('\n') if line.strip()]
        
        # Filter and process queries
        search_queries = []
        for line in raw_queries:
            # Remove any quotes at beginning/end but preserve internal quotes
            query_text = line.strip('"\'')
            if query_text:
                # Ensure the query contains at least some key domain terms
                if any(term.lower() in query_text.lower() for term in domain_terms[:3]) or domain_context.lower() in query_text.lower():
                    # Add year range if not present
                    if not any(year_pattern in query_text for year_pattern in ["year:[", "[2018 TO", "year>="]):
                        query_text += ' AND year:[2018 TO 2024]'
                    search_queries.append(query_text)
                    
            # Break if we've reached max queries
            if len(search_queries) >= max_queries:
                break
        
        # Ensure we have at least one query by using the domain context as fallback
        if not search_queries and query:
            main_terms = ' '.join(domain_terms[:3]) if domain_terms else query
            search_queries.append(f'("{main_terms}") AND year:[2018 TO 2024]')
        
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
        5. Be concise and to the point

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
    
    def integrate_literature(self, initial_response: str, papers: List[Paper], query: str = None) -> ResearchResponse:
        """
        STEPS 7-8: Integrate literature findings into a final response.
        
        This method implements the complete workflow from literature screening
        to final synthesis, creating a comprehensive research response.
        
        Parameters:
        -----------
        initial_response: Preliminary assessment of the research question
        papers: List of papers found during search
        query: The original query (if provided, ensures better domain anchoring)
        
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
 
        # Set current year to 2025 as per the simulation context
        import datetime
        current_year = datetime.datetime.now().year
        
        # Extract research question from the initial response
        research_question = self._extract_research_question(initial_response)
        
        # Domain anchoring - extract key domain terms explicitly
        domain_terms = []
        if query:
            domain_terms_prompt = f"""
            Extract 5-7 key domain-specific technical terms from this query:
            
            "{query}"
            
            List ONLY the terms, one per line with no numbering or bullets.
            Focus on specialized terminology that defines the domain.
            """
            domain_terms_response = self.ollama.generate(domain_terms_prompt)
            domain_terms = [term.strip() for term in domain_terms_response.split('\n') if term.strip()]
        
        domain_context = " AND ".join(domain_terms[:3]) if domain_terms else (query or research_question)
        
        # STEP 4-5: Screen papers by relevance with domain context
        relevance_prompt = f"""
        Evaluate if each paper is relevant to this research question:
        
        Research Question: {research_question}
        Original Query: {query or research_question}
        
        A paper is considered relevant if it addresses any aspect of the research question,
        even if it's from a different field but provides useful information or methodologies.
        Judge relevance based on content, not just field classification.
        """
        
        # Screen papers with stronger domain focus
        relevant_papers = self.screen_papers_by_relevance_with_domain(
            papers, 
            research_question, 
            domain_context,
            relevance_prompt
        )
        
        # If no relevant papers after screening, try again with a broader interpretation
        if not relevant_papers and papers:
            print("No papers were found relevant with strict criteria. Trying with broader criteria...")
            relevant_papers = self.screen_papers_by_relevance(papers, research_question)
        
        if not relevant_papers:
            return ResearchResponse(
                initial_response=initial_response,
                papers=[],
                final_summary=initial_response + "\n\nNo relevant academic literature was found to address this specific question.",
                citations=[]
            )
        
        # STEP 6: Extract key findings from papers with paper identifiers
        findings_with_metadata = {}
        for i, paper in enumerate(relevant_papers):
            # Create a unique ID for each paper
            paper_id = f"paper_{i}"
            
            # Extract findings with focused prompt
            prompt = f"""
            Extract key findings and information from this paper:
            
            Title: {paper.title}
            Authors: {', '.join(paper.authors)}
            Year: {paper.year if paper.year else 'Unknown'}
            Abstract: {paper.abstract}
            
            Extract:
            1. The main methodology used
            2. Key results or findings
            3. Limitations mentioned
            4. Implications or applications
            
            Format each finding as: "- [Finding description]"
            Be specific and concise. Include numbers, statistics, or concrete findings when available.
            Avoid vague generalizations.
            """
            
            response = self.ollama.generate(prompt)
            
            # Store findings along with paper metadata
            findings = [line.strip()[2:].strip() for line in response.split("\n") 
                       if line.strip() and line.strip().startswith("-")]
            
            findings_with_metadata[paper_id] = {
                "paper": paper,
                "findings": findings
            }
        
        # Format findings for synthesis
        findings_for_synthesis = {}
        for paper_id, data in findings_with_metadata.items():
            findings_for_synthesis[paper_id] = data["findings"]
        
        # STEP 7: Synthesize findings
        synthesis = self.synthesize_findings_with_domain(
            findings_for_synthesis, 
            research_question, 
            domain_context, 
            query
        )
        
        # Extract citation keys from synthesis
        cited_paper_ids = []
        for paper_id in findings_with_metadata.keys():
            if f"[{paper_id}]" in synthesis:
                cited_paper_ids.append(paper_id)
        
        # If no papers were specifically cited, assume all relevant papers were used
        if not cited_paper_ids:
            cited_paper_ids = list(findings_with_metadata.keys())
        
        # Prepare detailed paper information for citation formatting
        papers_for_citation = []
        for paper_id in cited_paper_ids:
            paper_data = findings_with_metadata[paper_id]
            paper = paper_data["paper"]
            
            # Extract journal/venue information with a more comprehensive approach
            venue_prompt = f"""
            As a scientific reference librarian, determine the most likely publication venue (journal or conference) for this paper:
            
            Title: {paper.title}
            Authors: {', '.join(paper.authors)}
            Year: {paper.year}
            Abstract: {paper.abstract}
            
            Based on the title, look for words that might indicate the journal or conference.
            Consider these clues:
            1. Words like "Journal", "Proceedings", "Conference", "Transactions" in the title
            2. Subject matter that matches known journals in the field
            3. Specific formatting or naming patterns common to certain venues
            
            If you can identify a likely venue name, provide it. Otherwise, look at keywords in the title and suggest 
            a relevant journal in that field. For example:
            - For AI: "IEEE Transactions on Neural Networks and Learning Systems"
            - For materials science: "Advanced Materials" or "Journal of Materials Science"
            - For architecture: "Building and Environment" or "Architectural Science Review"
            - For chemistry: "Journal of Chemical Physics" or "Chemical Communications"
            
            Return ONLY the venue name without any other text. If completely uncertain, return "Unknown".
            """
            
            venue_response = self.ollama.generate(venue_prompt).strip()
            venue = venue_response if venue_response and "unknown" not in venue_response.lower() else "Unknown"
            
            # Extract volume and issue information if available
            volume_issue_prompt = f"""
            Extract any potential volume and issue information from this paper:
            
            Title: {paper.title}
            Abstract: {paper.abstract}
            
            Look for patterns like "Vol. X", "Volume X", "Issue Y", or "Number Y" in the text.
            Return ONLY in the format: "volume:X issue:Y" if found, otherwise return "unknown".
            """
            
            volume_issue_info = self.ollama.generate(volume_issue_prompt).strip().lower()
            
            # Parse volume and issue
            volume = "Unknown"
            issue = "Unknown"
            
            if "volume:" in volume_issue_info and volume_issue_info != "unknown":
                try:
                    volume_match = re.search(r"volume:(\d+)", volume_issue_info)
                    if volume_match:
                        volume = volume_match.group(1)
                except:
                    pass
                    
            if "issue:" in volume_issue_info and volume_issue_info != "unknown":
                try:
                    issue_match = re.search(r"issue:(\d+)", volume_issue_info)
                    if issue_match:
                        issue = issue_match.group(1)
                except:
                    pass
            
            papers_for_citation.append({
                "id": paper_id,
                "title": paper.title,
                "authors": paper.authors,
                "year": paper.year,
                "venue": venue,
                "volume": volume,
                "issue": issue
            })
        
        # Format citation information for the prompt
        citation_info = "\n\n".join([
            f"Paper ID: {p['id']}\n"
            f"Title: {p['title']}\n"
            f"Authors: {', '.join(p['authors'])}\n"
            f"Year: {p['year']}\n"
            f"Journal/Conference: {p['venue']}\n"
            f"Volume: {p['volume']}\n"
            f"Issue: {p['issue']}"
            for p in papers_for_citation
        ])
        
        # STEP 8: Construct final answer with proper citations
        prompt = f"""
        Research Question: {research_question}
        Original Query: {query or research_question}
        
        Initial Assessment:
        {initial_response}
        
        Literature Synthesis:
        {synthesis}
        
        Papers to Cite (use ONLY these papers, NO fabrication):
        {citation_info}
        
        Create a comprehensive research response that:
        1. Integrates the initial assessment with the literature findings
        2. Addresses the research question directly using evidence from the papers
        3. Cites ONLY the provided papers using their Paper ID (e.g., "[paper_0]") within the text
        4. Uses multiple papers where possible to provide balanced coverage
        5. Maintains an academic, evidence-based tone
        
        For citations:
        1. Format citations in IEEE format
        2. Include ALL the papers listed above in the citations section
        3. Use the exact author names, titles, and years provided
        4. For journal/venue/volume/issue information, use ONLY what is provided in the fields above
        5. When venue is "Unknown", cite as: Author(s), "Title," Year.
        6. When venue is known but volume or issue is "Unknown", cite as: Author(s), "Title," Journal/Conference, Year.
        7. When venue, volume and issue are known, cite as: Author(s), "Title," Journal/Conference, vol. [Volume], no. [Issue], Year.
        
        Provide your response in two parts:
        1. Final Summary (with in-text citations using paper IDs)
        2. Citations (in IEEE format)
        
        Keep in mind that the final summary should connect to the original question and provide a clear synthesis of findings.
        """
        
        response = self.ollama.generate(prompt)
        
        # Replace paper IDs with citation numbers in the final text
        final_text = response
        for i, paper_id in enumerate(cited_paper_ids):
            citation_number = i + 1
            final_text = final_text.replace(f"[{paper_id}]", f"[{citation_number}]")
        
        # Parse the response to separate summary from citations
        parts = final_text.split("Citations:", 1)
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
    
    def screen_papers_by_relevance_with_domain(self, papers: List[Paper], research_question: str, domain_context: str, relevance_prompt: str = "") -> List[Paper]:
        """
        Enhanced screening of papers with stronger domain anchoring.
        
        Parameters:
        -----------
        papers: List of papers to screen
        research_question: The research question
        domain_context: Key domain terms to focus on
        relevance_prompt: Optional additional prompt text
        
        Returns:
        --------
        list: List of papers that passed the screening
        """
        if not papers:
            return []

        # Set current year as per the simulation context
        import datetime
        current_year = datetime.datetime.now().year
        
        # Extract core research concepts
        core_concepts_prompt = f"""
        Extract 5-8 core CONCEPTS (not just keywords) that are central to this research question:
        
        {research_question}
        
        For each concept, explain WHY it's central to the research question in 1-2 sentences.
        Format as:
        Concept: [Name]
        Importance: [Brief explanation]
        
        Focus on the core ideas and research goals, not just field-specific terminology.
        """
        
        core_concepts_response = self.ollama.generate(core_concepts_prompt)
        
        # Extract interdisciplinary connections
        interdisciplinary_prompt = f"""
        What related fields or disciplines might have relevant knowledge for this research question?
        
        {research_question}
        
        For each related field, explain HOW it connects to the main research question.
        Format as:
        Field: [Name]
        Connection: [How it relates to the research question]
        
        Focus on unexpected but genuinely useful connections across disciplines.
        """
        
        interdisciplinary_response = self.ollama.generate(interdisciplinary_prompt)
        
        # Create a prompt to evaluate relevance with a nuanced approach
        papers_info = "\n\n".join([
            f"Paper {i+1}:\nTitle: {paper.title}\nAuthors: {', '.join(paper.authors)}\n" +
            f"Year: {paper.year if paper.year else 'Unknown'}\nAbstract: {paper.abstract}"
            for i, paper in enumerate(papers)
        ])
        
        prompt = f"""
        {relevance_prompt}
        
        Research Question: {research_question}
        Domain Context: {domain_context}
        
        Core Research Concepts:
        {core_concepts_response}
        
        Potential Interdisciplinary Connections:
        {interdisciplinary_response}
        
        Evaluate each paper's relevance to the research question using a 3-tier system:
        
        TIER 1 (DIRECTLY RELEVANT): Papers that directly address the research question's core concepts
        TIER 2 (INDIRECTLY RELEVANT): Papers that provide useful methodology, data, or frameworks that could be applied
        TIER 3 (TANGENTIALLY RELEVANT): Papers from other domains with concepts that could potentially transfer
        
        For each paper, consider:
        1. How central are the paper's findings to the research question?
        2. Does the paper provide unique data, methods, or perspectives not found in more directly relevant papers?
        3. How much effort would be required to apply the paper's findings to the research question?
        4. Is this paper's approach novel compared to the more directly relevant papers?
        
        Return a comma-separated list in this exact format: 
        "TIER 1: [paper numbers], TIER 2: [paper numbers], TIER 3: [paper numbers]"
        
        Example: "TIER 1: 1, 3, TIER 2: 2, 5, TIER 3: 4"
        If no papers fit a tier, use "TIER X: none"
        
        Papers to evaluate:
        {papers_info}
        
        Relevance assessment:
        """
        
        response = self.ollama.generate(prompt)
        
        # Parse the response to get tiered relevance
        tier1_papers = []
        tier2_papers = []
        tier3_papers = []
        
        try:
            # Extract tier 1 papers
            tier1_match = re.search(r"TIER 1: (.*?)(?:,\s*TIER|$)", response, re.IGNORECASE)
            if tier1_match and "none" not in tier1_match.group(1).lower():
                tier1_indices = [int(idx.strip()) - 1 for idx in tier1_match.group(1).split(",") if idx.strip().isdigit()]
                tier1_papers = [papers[i] for i in tier1_indices if 0 <= i < len(papers)]
            
            # Extract tier 2 papers
            tier2_match = re.search(r"TIER 2: (.*?)(?:,\s*TIER|$)", response, re.IGNORECASE)
            if tier2_match and "none" not in tier2_match.group(1).lower():
                tier2_indices = [int(idx.strip()) - 1 for idx in tier2_match.group(1).split(",") if idx.strip().isdigit()]
                tier2_papers = [papers[i] for i in tier2_indices if 0 <= i < len(papers)]
            
            # Extract tier 3 papers
            tier3_match = re.search(r"TIER 3: (.*?)(?:,\s*TIER|$)", response, re.IGNORECASE)
            if tier3_match and "none" not in tier3_match.group(1).lower():
                tier3_indices = [int(idx.strip()) - 1 for idx in tier3_match.group(1).split(",") if idx.strip().isdigit()]
                tier3_papers = [papers[i] for i in tier3_indices if 0 <= i < len(papers)]
        except:
            # If parsing fails, fall back to a simpler approach
            try:
                # Look for any numbers in the response
                all_numbers = re.findall(r'\d+', response)
                indices = [int(num) - 1 for num in all_numbers if int(num) > 0 and int(num) <= len(papers)]
                # Add unique papers from indices
                relevant_papers = []
                for i in indices:
                    if 0 <= i < len(papers) and papers[i] not in relevant_papers:
                        relevant_papers.append(papers[i])
                return relevant_papers[:min(5, len(relevant_papers))]
            except:
                # Last resort fallback - return the first few papers
                return papers[:min(3, len(papers))]
        
        # Combine papers based on tiers, prioritizing the most relevant
        # Goal: Return up to 5 papers total, with preference given to higher tiers
        
        # First, include all tier 1 papers
        relevant_papers = tier1_papers.copy()
        
        # Add tier 2 papers if we haven't reached our limit
        remaining_slots = 5 - len(relevant_papers)
        if remaining_slots > 0 and tier2_papers:
            relevant_papers.extend(tier2_papers[:remaining_slots])
        
        # Add tier 3 papers if we still haven't reached our limit
        remaining_slots = 5 - len(relevant_papers)
        if remaining_slots > 0 and tier3_papers:
            relevant_papers.extend(tier3_papers[:remaining_slots])
        
        # If we still have no papers, return a small set of the input papers
        if not relevant_papers:
            return papers[:min(3, len(papers))]
            
        return relevant_papers
    
    def synthesize_findings_with_domain(self, findings: Dict[str, List[str]], research_question: str, domain_context: str, query: str = None) -> str:
        """
        Enhanced synthesis with stronger research focus and proper paper citation.
        
        Parameters:
        -----------
        findings: Extracted findings from papers
        research_question: The research question
        domain_context: Key domain terms to focus on
        query: The original query
        
        Returns:
        --------
        str: Synthesized findings with paper citations
        """
        # If there are no findings or only from one paper, use a simplified approach
        if not findings or len(findings) <= 1:
            # Flatten findings into a single list
            all_findings = []
            paper_id = next(iter(findings)) if findings else "no_papers"
            
            for finding in findings.get(paper_id, []):
                all_findings.append(f"[{paper_id}] {finding}")
            
            findings_text = "\n".join(all_findings)
            
            prompt = f"""
            Research Question: {research_question}
            Original Query: {query or research_question}
            
            Synthesize these findings to address the research question:
            
            Key findings:
            {findings_text}
            
            Your synthesis:
            """
            
            return self.ollama.generate(prompt)
        
        # With multiple papers, use a more comprehensive approach
        # Step A: Analyze papers to determine relevance tiers
        paper_analysis_prompt = f"""
        Analyze these papers in terms of their relevance to the research question:
        
        Research Question: {research_question}
        
        For each paper ID below, classify it as:
        - PRIMARY: Directly addresses the research question
        - SECONDARY: Provides useful context, methods, or related insights
        - TERTIARY: Tangentially related or conceptually transferable
        
        PAPER IDS: {list(findings.keys())}
        
        Format your response as:
        PRIMARY: [list of paper IDs]
        SECONDARY: [list of paper IDs]
        TERTIARY: [list of paper IDs]
        
        Base your classification on the paper findings provided after this prompt.
        """
        
        # Step B: Organize findings by paper with paper IDs
        findings_by_paper = {}
        for paper_id, paper_findings in findings.items():
            findings_by_paper[paper_id] = [f"[{paper_id}] {finding}" for finding in paper_findings]
        
        # Organize findings by paper for the analysis
        organized_findings = []
        for paper_id, paper_findings in findings_by_paper.items():
            paper_text = f"PAPER {paper_id}:\n" + "\n".join(paper_findings)
            organized_findings.append(paper_text)
        
        all_organized_findings = "\n\n".join(organized_findings)
        
        # Get the complete paper analysis prompt
        complete_analysis_prompt = paper_analysis_prompt + "\n\n" + all_organized_findings
        
        # Get paper relevance tiers
        tier_response = self.ollama.generate(complete_analysis_prompt)
        
        # Parse tiers
        primary_papers = []
        secondary_papers = []
        tertiary_papers = []
        
        try:
            # Extract primary papers
            primary_match = re.search(r"PRIMARY: (.*?)(?:\n|$)", tier_response, re.IGNORECASE)
            if primary_match:
                primary_papers = re.findall(r'paper_\d+', primary_match.group(1).lower())
            
            # Extract secondary papers
            secondary_match = re.search(r"SECONDARY: (.*?)(?:\n|$)", tier_response, re.IGNORECASE)
            if secondary_match:
                secondary_papers = re.findall(r'paper_\d+', secondary_match.group(1).lower())
            
            # Extract tertiary papers
            tertiary_match = re.search(r"TERTIARY: (.*?)(?:\n|$)", tier_response, re.IGNORECASE)
            if tertiary_match:
                tertiary_papers = re.findall(r'paper_\d+', tertiary_match.group(1).lower())
                
        except:
            # If parsing fails, consider all papers primary
            primary_papers = list(findings.keys())
        
        # If no primary papers identified, use original approach
        if not primary_papers:
            primary_papers = list(findings.keys())
        
        # Step C: Flatten findings with paper IDs included
        all_findings_with_ids = []
        for paper_id, paper_findings in findings_by_paper.items():
            all_findings_with_ids.extend(paper_findings)
        
        findings_text = "\n".join(all_findings_with_ids)
        
        # Step D: Identify key themes across papers for organization
        themes_prompt = f"""
        Based on these findings from multiple papers about:
        
        {research_question}
        
        Findings:
        {findings_text}
        
        Primary Papers: {', '.join(primary_papers)}
        Secondary Papers: {', '.join(secondary_papers)}
        
        Identify 3-5 key themes or topics that appear across the papers, especially focusing on themes in the primary papers.
        List ONLY the theme names, one per line with no numbering or bullets.
        Each theme should capture a significant aspect of the research question.
        """
        
        themes_response = self.ollama.generate(themes_prompt)
        themes = [theme.strip() for theme in themes_response.split('\n') if theme.strip()]
        
        # Use default themes if extraction failed
        if not themes:
            themes = ["Current State of Research", "Key Methodologies", "Main Findings", "Applications and Implications"]
        
        # Step E: Generate synthesis with cross-paper integration based on relevance tiers
        prompt = f"""
        Research Question: {research_question}
        Original Query: {query or research_question}
        
        Key Themes: {', '.join(themes)}
        
        Organization of papers by relevance:
        PRIMARY (directly address research question): {', '.join(primary_papers)}
        SECONDARY (provide useful context/methods): {', '.join(secondary_papers)}
        TERTIARY (tangentially related): {', '.join(tertiary_papers)}
        
        Findings from different papers (with paper IDs in brackets):
        {findings_text}
        
        Create a comprehensive synthesis that:
        1. Addresses the research question directly
        2. Organizes content around the identified themes
        3. Heavily emphasizes findings from PRIMARY papers
        4. Uses SECONDARY papers to provide context and additional support
        5. Only includes TERTIARY papers when they provide unique insights not found in other papers
        6. Includes paper IDs as citations for each finding (e.g., [paper_0])
        7. Makes meaningful connections between primary papers within each theme
        8. Identifies areas of consensus and disagreement between papers
        9. Maintains an academic, evidence-based tone
        
        Synthesis principles:
        - Start each theme by synthesizing findings from PRIMARY papers first
        - Only bring in SECONDARY or TERTIARY papers after establishing the core information
        - When citing papers from different tiers, be explicit about their relationship to the core research
        - If findings from tertiary papers seem disconnected, either find a meaningful connection or omit them
        - Use comparative language (e.g., "While [paper_0] focused on X, [paper_1] approached from Y perspective")
        
        IMPORTANT: Every statement based on the papers MUST include the corresponding paper ID as a citation.
        
        Your synthesis:
        """
        
        synthesis_response = self.ollama.generate(prompt)
        
        # Step F: Check if all primary papers were cited, and fix if needed
        uncited_primary = []
        for paper_id in primary_papers:
            if f"[{paper_id}]" not in synthesis_response:
                uncited_primary.append(paper_id)
        
        if uncited_primary:
            citation_fix_prompt = f"""
            The following PRIMARY papers are not cited in your synthesis:
            {uncited_primary}
            
            Original synthesis:
            {synthesis_response}
            
            Revise the synthesis to include citations to ALL the primary papers, while maintaining coherence and focus.
            Focus on integrating findings from these papers naturally into the existing themes.
            """
            
            synthesis_response = self.ollama.generate(citation_fix_prompt)
        
        return synthesis_response
    
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