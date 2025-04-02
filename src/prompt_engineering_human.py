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
        # First, check and fix any truncated or unbalanced query
        fixed_query = self._validate_and_fix_query(query)
        
        # Parse the query to extract core components
        core_components_prompt = f"""
        Analyze this search query and extract the PRIMARY concepts/technologies/materials:
        
        "{fixed_query}"
        
        For each component identified, provide:
        1. The exact term as it appears in the query
        2. The type of component (e.g., technique, material, application)
        3. A brief description of what it refers to
        
        Format as a numbered list:
        1. Term: [term] | Type: [type] | Description: [1-sentence description]
        2. Term: [term] | Type: [type] | Description: [1-sentence description]
        
        ONLY include terms that are EXPLICITLY mentioned in the query, do NOT add components that aren't present.
        """
        
        components_response = self.ollama.generate(core_components_prompt)
        
        # Extract key terms for context
        key_terms_prompt = f"""
        Extract 5-7 key technical terms from this search query:
        
        "{fixed_query}"
        
        List ONLY the specific technical terms or concepts, one per line (no numbering).
        Focus on domain-specific terminology rather than general words.
        """
        
        key_terms_response = self.ollama.generate(key_terms_prompt)
        key_terms = [term.strip() for term in key_terms_response.split('\n') if term.strip()]
        
        prompt = f"""
        You're conducting a literature review on the following query:
        
        "{fixed_query}"
        
        The query contains these key components:
        {components_response}
        
        And these key technical terms:
        {', '.join(key_terms)}
        
        1. Create a research question that DIRECTLY corresponds to these terms and concepts, without adding unrelated topics.
        2. Define inclusion/exclusion criteria for studies that align with the specific topic in the query.
        3. Identify key concepts and terms for searching that MUST include all major terms from the original query.
        
        IMPORTANT INSTRUCTIONS:
        - Your research question MUST focus ONLY on the relationship between the terms present in the original query.
        - DO NOT introduce topics, applications, or materials that are not explicitly mentioned in the query.
        - If the query mentions specific materials (e.g., "nitrides"), ONLY include those specific materials in your criteria.
        - If the query mentions specific applications (e.g., "electrocatalysis"), ONLY focus on those applications.
        - DO NOT expand the scope beyond what is explicitly stated in the original query.
        
        Format your response as a JSON object with these fields:
        - research_question: The clear research question directly related to the query terms
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
                result = json.loads(json_str.group(0))
                
                # Verify the research question contains the key terms
                research_question = result.get("research_question", "")
                missing_terms = []
                
                # Check if key terms from the query are present in the research question
                for term in key_terms[:3]:  # Check at least the top 3 terms
                    if term.lower() not in research_question.lower():
                        missing_terms.append(term)
                
                # If key terms are missing, add a note to the research question
                if missing_terms:
                    result["research_question"] = research_question + f" (Note: This research specifically focuses on {', '.join(missing_terms)}.)"
                
                return result
            else:
                # Fallback if JSON parsing fails
                return {
                    "research_question": f"How can {' and '.join(key_terms[:3])} be effectively utilized as described in: {fixed_query}",
                    "inclusion_criteria": ["Studies focusing on " + term for term in key_terms[:3]],
                    "exclusion_criteria": ["Studies not related to " + fixed_query, "Opinion pieces", "Non-English publications"],
                    "key_terms": key_terms or fixed_query.split(),
                    "time_frame": "2018-2023"
                }
        except Exception as e:
            print(f"Error parsing research definition: {e}")
            # Fallback if JSON parsing fails
            return {
                "research_question": f"How can {' and '.join(key_terms[:3])} be effectively utilized as described in: {fixed_query}",
                "inclusion_criteria": ["Studies focusing on " + term for term in key_terms[:3]],
                "exclusion_criteria": ["Studies not related to " + fixed_query, "Opinion pieces", "Non-English publications"],
                "key_terms": key_terms or fixed_query.split(),
                "time_frame": "2018-2023"
            }
    
    def _validate_and_fix_query(self, query: str) -> str:
        """
        Validate and fix the query to ensure it's properly formatted.
        
        Parameters:
        -----------
        query: The original query string
        
        Returns:
        --------
        str: Fixed query with balanced parentheses and proper terms
        """
        # Check if query ends abruptly (likely truncated)
        if query.strip().endswith(("AND", "OR", "NOT", "(", " ")):
            # Query likely truncated at a logical operator or opening parenthesis
            print(f"Warning: Query appears to be truncated: {query}")
            # Try to make it a complete query by closing it
            query = query.strip()
            # If it ends with a logical operator, remove it
            if query.endswith(("AND", "OR", "NOT")):
                query = query[:-3].strip()
            # If it ends with an opening parenthesis, remove it
            elif query.endswith("("):
                query = query[:-1].strip()
        
        # Check for truncated or unbalanced parentheses
        open_count = query.count('(')
        close_count = query.count(')')
        
        # If unbalanced, attempt to fix
        if open_count != close_count:
            print(f"Warning: Unbalanced parentheses detected in query: {query}")
            # If more opening parentheses, add closing ones
            if open_count > close_count:
                query = query + ')' * (open_count - close_count)
            # If more closing parentheses, add opening ones at appropriate locations
            else:
                # This is a simplification - in practice, we'd need more sophisticated parsing
                query = '(' * (close_count - open_count) + query
        
        # Check for AND/OR operators in all caps for proper Boolean syntax
        query = re.sub(r'\b(and)\b', 'AND', query, flags=re.IGNORECASE)
        query = re.sub(r'\b(or)\b', 'OR', query, flags=re.IGNORECASE)
        query = re.sub(r'\b(not)\b', 'NOT', query, flags=re.IGNORECASE)
        
        # Check for incomplete quotes - if quotes don't match, assume truncation
        if query.count('"') % 2 != 0:
            print(f"Warning: Unbalanced quotes detected in query: {query}")
            # Count open quotes vs close quotes
            segments = query.split('"')
            if len(segments) > 1:
                # If odd number of segments, add closing quote to the last term
                query = query + '"'
        
        # Preserve already properly quoted terms
        # First, temporarily replace quoted terms with placeholders
        quoted_terms = []
        
        def replace_quoted(match):
            quoted_terms.append(match.group(0))
            return f"__QUOTED_TERM_{len(quoted_terms)-1}__"
        
        # Replace quoted terms with placeholders
        temp_query = re.sub(r'"[^"]*"', replace_quoted, query)
        
        # Now quote unquoted terms
        def ensure_quoted(match):
            term = match.group(1)
            # Skip placeholders and operators
            if term.startswith("__QUOTED_TERM_") or term.upper() in ["AND", "OR", "NOT"]:
                return term
            return f'"{term}"'
        
        # Quote terms that aren't already quoted and aren't operators
        temp_query = re.sub(r'\b([^\s"()AND|OR|NOT]+)\b', ensure_quoted, temp_query)
        
        # Restore original quoted terms
        for i, term in enumerate(quoted_terms):
            temp_query = temp_query.replace(f"__QUOTED_TERM_{i}__", term)
        
        # Fix double spaces and trim
        query = re.sub(r'\s+', ' ', temp_query).strip()
        
        print(f"Parsed query: {query}")
        return query
    
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
        STEP 3: Generate search queries for API calls.
        
        This method generates effective search queries based on the research question,
        formatted appropriately for academic search APIs.
        
        Parameters:
        -----------
        query: The original user query
        initial_response: The initial research question assessment
        max_queries: Maximum number of queries to generate
    
        Returns:
        --------
        list: List of search queries formatted for academic databases
        """
        # Extract key terms from the original query - these are our highest priority
        query_terms = []
        
        # Extract quoted terms from the query
        quoted_terms = re.findall(r'"([^"]+)"', query)
        if quoted_terms:
            query_terms.extend(quoted_terms)
        
        # Extract logical components from the query
        logical_components = re.split(r'\s+AND\s+|\s+OR\s+', query.replace('"', ''))
        logical_components = [comp.strip() for comp in logical_components if comp.strip()]
        logical_components = [comp.strip('()') for comp in logical_components]
        
        # Add logical components not already captured in quoted terms
        for component in logical_components:
            if component not in query_terms and len(component.split()) <= 3:  # Avoid long phrases
                query_terms.append(component)
        
        # Extract research question from initial response
        research_question = self._extract_research_question(initial_response)
        
        # Create queries based directly on the original query terms
        direct_query = ' AND '.join([f'"{term}"' for term in query_terms[:3]])
        
        # Create prompts for search query generation
        prompt = f"""
        You're preparing search queries for academic databases based on:
        
        Original Query: {query}
        Research Question: {research_question}
        
        The most critical terms from the query are: {', '.join(query_terms)}
        
        Create {max_queries} different academic database search queries that:
        1. ALWAYS include the core query terms listed above
        2. Use proper Boolean operators (AND, OR) with parentheses
        3. Are formatted for academic databases
        4. Have different focuses while remaining true to the original query
        5. Use quoted phrases for exact matches
        
        The queries should follow this format:
        1. ("term1" AND "term2") AND ("term3" OR "term4")
        2. ("term1" AND "term5") AND ("term6")
        3. ("term2" AND "term3") AND ("term7" OR "term8")
        
        Focus on generating queries that will yield DIRECTLY relevant papers.
        ALWAYS include the original query terms, then add synonyms or related concepts.
        """
        
        response = self.ollama.generate(prompt)
        
        # Parse the response
        search_queries = []
        
        # Extract the search queries from the response
        lines = response.strip().split('\n')
        for line in lines:
            if re.match(r'^\d+\.', line):
                # This line starts with a number and period, likely a query
                query_text = re.sub(r'^\d+\.\s*', '', line).strip()
                if query_text and len(query_text) > 10:  # Basic validation
                    search_queries.append(query_text)
        
        # Add the direct query if it's not already included
        if direct_query not in search_queries:
            search_queries.insert(0, direct_query)
        
        # Ensure we're using all key terms from the original query
        if query_terms and not any(all(term.lower() in q.lower() for term in query_terms[:2]) for q in search_queries):
            search_queries.insert(0, ' AND '.join([f'"{term}"' for term in query_terms[:3]]))
        
        # Limit to max_queries
        return search_queries[:max_queries]
    
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
    
    def screen_papers_by_title(self, papers: List[Paper]) -> List[Paper]:
        """
        STEP 4: Screen papers by title first, mimicking a human's quick scan.
        
        This method implements a human-like quick scan of titles to eliminate
        obviously irrelevant papers before deeper analysis of abstracts.
        
        Parameters:
        -----------
        papers: List of papers to screen by title
        
        Returns:
        --------
        list: List of papers that passed title screening
        """
        if not papers:
            return []
            
        # Format the titles for evaluation
        papers_info = "\n".join([
            f"Paper {i+1}: {paper.title}" 
            for i, paper in enumerate(papers)
        ])
        
        # Create a prompt that focuses on rapid title evaluation with stricter criteria
        prompt = f"""
        You're a researcher conducting a literature review. 
        Scan these paper titles quickly and identify which ones might be relevant for further review.
        
        When scanning titles, be VERY STRICT and consider:
        1. Direct relevance to the primary research topic
        2. Presence of specific technical terms related to the field
        3. Clear indication of appropriate methodology
        4. Evidence of relevant findings or innovations
        
        Be SELECTIVE - only choose papers that are CLEARLY related to the topic.
        REJECT papers that are only tangentially related or from different domains.
        
        Assess the following papers by title alone:
        {papers_info}
        
        For each paper, respond only with the paper number if it is DIRECTLY relevant.
        Use this format exactly: "Relevant papers: 1, 3, 5" (just the numbers in a comma-separated list)
        If no papers are relevant, respond with: "Relevant papers: none"
        """
        
        response = self.ollama.generate(prompt)
        
        # Extract paper numbers using regex
        import re
        
        # Check if "none" is in the response
        if re.search(r'(?:relevant|relevance).*none', response.lower()):
            return []
        
        numbers = re.findall(r'\d+', response)
        
        try:
            # Convert to indices (0-based)
            indices = [int(num) - 1 for num in numbers if 0 < int(num) <= len(papers)]
            
            # Return papers that passed title screening
            title_passed_papers = [papers[i] for i in indices if 0 <= i < len(papers)]
            
            # If too few papers pass, consider adding a few more to ensure sufficient material
            # But be more selective than before - add at most 2 additional papers
            if len(title_passed_papers) < min(2, len(papers)):
                additional_count = min(2, len(papers)) - len(title_passed_papers)
                # Sort papers by potential relevance (based on keyword matching)
                potentially_relevant = []
                for i, paper in enumerate(papers):
                    if paper not in title_passed_papers:
                        # Count occurrences of keywords in title
                        title_lower = paper.title.lower()
                        keyword_count = sum([1 for kw in ["model", "learning", "prediction", "material", "design", "ai", 
                                                          "machine", "algorithm", "discovery", "generative"]
                                            if kw.lower() in title_lower])
                        if keyword_count > 0:
                            potentially_relevant.append((i, keyword_count, paper))
                
                # Sort by keyword count (descending)
                potentially_relevant.sort(key=lambda x: x[1], reverse=True)
                
                # Add the most relevant papers
                for _, _, paper in potentially_relevant[:additional_count]:
                    title_passed_papers.append(paper)
                        
            return title_passed_papers
            
        except Exception as e:
            print(f"Error during title screening: {e}")
            # Fallback: return a limited number of papers if parsing fails
            return papers[:min(5, len(papers))]
    
    def screen_papers_by_abstract(self, papers: List[Paper], research_question: str, domain_context: str = "") -> List[Paper]:
        """
        STEP 5: Detailed abstract screening of papers that passed title screening.
        
        This method performs a more thorough evaluation of papers by reading
        their abstracts, as a human would do after the initial title scan.
        
        Parameters:
        -----------
        papers: List of papers that passed title screening
        research_question: The research question to compare against
        domain_context: Key domain terms to focus on
        
        Returns:
        --------
        list: List of papers that passed abstract screening
        """
        if not papers:
            return []
        
        # First, extract the primary query keywords from the domain context or research question
        query_keywords_prompt = f"""
        Extract 3-5 PRIMARY technical keywords that MUST be present in relevant papers from:
        
        Domain Context: {domain_context}
        Research Question: {research_question}
        
        Return ONLY the most critical technical terms, separated by commas, that are essential to the topic.
        These will be used as mandatory filter terms for papers.
        """
        
        keywords_response = self.ollama.generate(query_keywords_prompt)
        primary_keywords = [kw.strip().lower() for kw in keywords_response.split(',') if kw.strip()]
        
        # Count keyword occurrence in papers for direct relevance checking
        def count_primary_keywords(paper: Paper) -> int:
            text = (paper.title + " " + paper.abstract).lower()
            return sum(1 for kw in primary_keywords if kw in text)
        
        # Pre-filter to retain papers with at least one primary keyword
        keyword_filtered_papers = []
        for paper in papers:
            keyword_count = count_primary_keywords(paper)
            if keyword_count > 0:
                keyword_filtered_papers.append((paper, keyword_count))
        
        # If we filtered out all papers based on keywords, keep at least a few original papers
        if not keyword_filtered_papers and papers:
            # Try with more lenient filtering if needed
            for paper in papers:
                # Use partial matching for keywords as a fallback
                partial_matches = sum(1 for kw in primary_keywords if any(part in (paper.title + " " + paper.abstract).lower() 
                                    for part in kw.split() if len(part) > 3))
                if partial_matches > 0:
                    keyword_filtered_papers.append((paper, partial_matches))
        
        # Extract core research concepts
        core_concepts_prompt = f"""
        Extract 5-7 core CONCEPTS (not just keywords) that are central to this research question:
        
        {research_question}
        
        For each concept, explain WHY it's central to the research question in 1-2 sentences.
        Format as:
        Concept: [Name]
        Importance: [Brief explanation]
        
        Focus on the core ideas and research goals, not just field-specific terminology.
        """
        
        core_concepts_response = self.ollama.generate(core_concepts_prompt)
        
        # Extract key terms to check for in abstracts
        extract_terms_prompt = f"""
        Based on this research question:
        
        {research_question}
        
        Extract 10-15 SPECIFIC technical terms, methodologies, and concepts that MUST be present in a relevant paper.
        
        These terms will be used to filter papers, so be precise and focus on domain-specific terminology.
        Include terms related to:
        1. The primary techniques/methods (e.g., "transformer architecture", "conformal prediction")
        2. The specific application domain (e.g., "materials discovery", "drug design")
        3. The evaluation metrics or outcomes (e.g., "prediction accuracy", "uncertainty quantification")
        
        Return ONLY a comma-separated list of terms, nothing else.
        """
        
        terms_response = self.ollama.generate(extract_terms_prompt)
        key_terms = [term.strip() for term in terms_response.split(',')]
        
        # Sort keyword-filtered papers by relevance
        sorted_papers = sorted(keyword_filtered_papers, key=lambda x: x[1], reverse=True)
        papers_to_evaluate = [p[0] for p in sorted_papers]
        
        # If no papers made it through the keyword filter, don't waste time on evaluation
        if not papers_to_evaluate:
            return []
        
        # Create a prompt to evaluate abstracts with stricter criteria
        papers_info = "\n\n".join([
            f"Paper {i+1}:\nTitle: {paper.title}\nAbstract: {paper.abstract}"
            for i, paper in enumerate(papers_to_evaluate)
        ])
        
        prompt = f"""
        You are conducting a RIGOROUS literature review for this research question:
        
        Research Question: {research_question}
        Domain Context: {domain_context}
        Primary Keywords (MUST be addressed): {', '.join(primary_keywords)}
        
        Core Research Concepts:
        {core_concepts_response}
        
        Key Terms (a relevant paper should contain several of these concepts):
        {', '.join(key_terms)}
        
        Evaluate each paper using these strict criteria:
        
        RELEVANT (Accept):
        - Paper MUST specifically address AT LEAST ONE of the primary keywords: {', '.join(primary_keywords)}
        - Abstract EXPLICITLY addresses multiple core concepts from the research question
        - Contains at least 3-4 of the key terms listed above
        - Describes methods, results, or applications directly related to the research question
        - Clearly contributes to answering the research question
        
        NOT RELEVANT (Reject):
        - Missing ALL primary keywords
        - Only tangentially related to the research question
        - Mentions key terms but in a different context or application
        - Focuses on a different domain or problem space
        - Does not clearly contribute to answering the research question
        
        Papers to evaluate:
        {papers_info}
        
        For each paper, determine if it meets the RELEVANT criteria.
        Return ONLY the paper numbers that are RELEVANT in this format:
        "Relevant papers: 1, 3, 5"
        
        If no papers are relevant, return "Relevant papers: none"
        """
        
        response = self.ollama.generate(prompt)
        
        # Parse the response to get relevant papers
        relevant_papers = []
        
        try:
            # Check if no papers are relevant
            if re.search(r'(?:relevant|relevance).*none', response.lower()):
                return []
                
            # Extract paper numbers
            matches = re.search(r'Relevant papers:?(.*)', response, re.IGNORECASE)
            if matches:
                numbers_str = matches.group(1)
                numbers = re.findall(r'\d+', numbers_str)
                
                # Convert to 0-based indices within the papers_to_evaluate list
                indices = [int(num) - 1 for num in numbers if 0 < int(num) <= len(papers_to_evaluate)]
                
                # Get the relevant papers
                relevant_papers = [papers_to_evaluate[i] for i in indices if 0 <= i < len(papers_to_evaluate)]
        except Exception as e:
            print(f"Error during abstract screening: {e}")
        
        # Add a safety check to ensure papers contain at least one primary keyword
        final_papers = []
        for paper in relevant_papers:
            keyword_count = count_primary_keywords(paper)
            if keyword_count > 0:
                final_papers.append(paper)
        
        # If we've filtered out everything but have papers from keyword filtering, use those
        if not final_papers and keyword_filtered_papers:
            # Take the top 2 papers by keyword count
            return [p[0] for p in sorted_papers[:min(2, len(sorted_papers))]]
        
        return final_papers
    
    def screen_papers_by_relevance_with_domain(self, papers: List[Paper], research_question: str, domain_context: str, relevance_prompt: str = "") -> List[Paper]:
        """
        Enhanced screening of papers with stronger domain anchoring.
        This method now applies the two-step screening process: first titles, then abstracts.
        
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
            
        # STEP 4: Title Screening - quick first pass to eliminate obviously irrelevant papers
        title_passed_papers = self.screen_papers_by_title(papers)
        print(f"Title screening: {len(title_passed_papers)}/{len(papers)} papers passed")
        
        # STEP 5: Abstract Screening - detailed evaluation of papers that passed title screening
        abstract_passed_papers = self.screen_papers_by_abstract(
            title_passed_papers, 
            research_question, 
            domain_context
        )
        print(f"Abstract screening: {len(abstract_passed_papers)}/{len(title_passed_papers)} papers passed")
        
        return abstract_passed_papers
    
    def screen_papers_by_relevance(self, papers: List[Paper], query: str) -> List[Paper]:
        """
        Basic screening of papers by relevance, used as a fallback.
        
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
            return papers[:min(5, len(papers))]
    
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
            Extract 10-15 key domain-specific technical terms from this query:
            
            "{query}"
            
            Research Question: {research_question}
            
            List specialized terminology related to this field of study, including:
            1. Main concepts and their variants
            2. Methodological approaches
            3. Materials or techniques mentioned
            4. Application domains
            
            Format your response as a simple list, one term per line (no numbering or bullets).
            Prioritize specific scientific/technical terms over general words.
            """
            domain_terms_response = self.ollama.generate(domain_terms_prompt)
            domain_terms = [term.strip() for term in domain_terms_response.split('\n') if term.strip()]
        
        # Create a richer domain context by combining multiple terms with proper Boolean operators
        if domain_terms:
            # Group terms into categories for a structured search
            main_concepts = domain_terms[:3]  # First 3 terms are likely main concepts
            secondary_concepts = domain_terms[3:6]  # Next few are likely secondary concepts
            
            # Create a structured query combining these groups
            if len(main_concepts) >= 2 and len(secondary_concepts) >= 1:
                domain_context = f"({' AND '.join(main_concepts)}) AND ({' OR '.join(secondary_concepts)})"
            elif main_concepts:
                domain_context = " AND ".join(main_concepts)
            else:
                domain_context = query or research_question
        else:
            domain_context = query or research_question
        
        # Important: Use all papers that have already gone through title/abstract screening
        # Skip additional relevance filtering since papers have already been screened
        relevant_papers = papers
        
        print(f"Using all {len(papers)} papers that passed title and abstract screening")
        
        # Track paper usage to ensure all papers are considered
        paper_usage = {i: {"used": False, "relevance": 0} for i in range(len(relevant_papers))}
        
        # Extract key findings from papers with paper identifiers
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
        
        # Assess methodology quality for better synthesis
        methodology_assessments = self.assess_methodology_quality(relevant_papers)
        
        # STEP 7: Synthesize findings with domain context
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
                paper_usage[int(paper_id.split('_')[1])]["used"] = True
        
        # If papers weren't cited, create a supplementary synthesis
        if len(cited_paper_ids) < len(findings_with_metadata) // 2:
            # Create a prompt to integrate uncited papers
            uncited_papers = [paper_id for paper_id in findings_with_metadata.keys() if paper_id not in cited_paper_ids]
            
            if uncited_papers:
                uncited_findings = {}
                for paper_id in uncited_papers:
                    uncited_findings[paper_id] = findings_with_metadata[paper_id]["findings"]
                
                supplementary_synthesis = self.synthesize_findings_with_domain(
                    uncited_findings,
                    research_question,
                    domain_context,
                    query
                )
                
                # Add the supplementary synthesis
                synthesis += "\n\n" + supplementary_synthesis
                
                # Update cited papers
                for paper_id in uncited_papers:
                    if f"[{paper_id}]" in supplementary_synthesis:
                        cited_paper_ids.append(paper_id)
                        paper_usage[int(paper_id.split('_')[1])]["used"] = True
        
        # Prepare detailed paper information for citation formatting
        papers_for_citation = []
        for paper_id in findings_with_metadata.keys():  # Include ALL papers in citations
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
            f"Year: {p['year'] if p['year'] else 'Unknown'}\n"
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
        3. Cites ALL the papers using their Paper ID (e.g., "[paper_0]") within the text
        4. Uses multiple papers where possible to provide balanced coverage
        5. Maintains an academic, evidence-based tone
        
        For citations:
        1. Format citations in IEEE format correctly as follows:
           - Single author: A. Author, "Title," Journal, vol. X, no. Y, pp. Z, Year.
           - Two authors: A. Author and B. Author, "Title," Journal, vol. X, no. Y, pp. Z, Year.
           - Three+ authors: A. Author et al., "Title," Journal, vol. X, no. Y, pp. Z, Year.
        
        2. When creating citations, follow these rules:
           - For each author, use only their last name and first initial (e.g., "J. Smith")
           - Do NOT include "vol. Unknown" or "no. Unknown" - omit these elements entirely
           - For preprint servers (arXiv, chemrxiv), use format: A. Author et al., "Title," [Server name] preprint, Year.
           - For papers without a specified journal, use the format: A. Author et al., "Title," Year.
           - For papers with a specified journal but no volume/issue, use: A. Author et al., "Title," Journal, Year.
           - NEVER insert the placeholder text "Unknown" in citations
           - NEVER include non-existent page numbers or use "pp. Z" as a placeholder
        
        Provide your response in two parts:
        1. Final Summary (with in-text citations using paper IDs)
        2. Citations (in IEEE format)
        
        The final summary MUST focus specifically on addressing the original research question about {query}.
        Ensure all papers are cited at least once in the summary.
        """
        
        response = self.ollama.generate(prompt)
        
        # Replace paper IDs with citation numbers in the final text
        final_text = response
        citation_mapping = {}
        
        # Map all papers (even uncited ones) to ensure complete citations
        for i, paper_id in enumerate(findings_with_metadata.keys()):
            citation_number = i + 1
            citation_mapping[paper_id] = citation_number
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
    
    def synthesize_findings_with_domain(self, findings: Dict[str, List[str]], research_question: str, domain_context: str = "", query: str = None) -> str:
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
            
            Synthesize these findings to directly address the research question:
            
            Key findings:
            {findings_text}
            
            Your synthesis should:
            1. Directly answer the research question
            2. Maintain focus on the original query
            3. Cite the paper ID in square brackets [paper_id]
            
            Your synthesis:
            """
            
            return self.ollama.generate(prompt)
        
        # With multiple papers, use a more comprehensive approach
        # First reminder of the original query to prevent topic drift
        original_query_reminder = f"""
        IMPORTANT: The original research question is: "{research_question}"
        Original query: "{query or research_question}"
        
        Your synthesis must DIRECTLY address this question, avoiding topic drift or excessive focus on tangential aspects.
        """
        
        # Record count of findings per paper for balance tracking
        paper_finding_counts = {paper_id: len(findings_list) for paper_id, findings_list in findings.items()}
        paper_count = len(findings)
        
        # Step A: Organize findings by paper with paper IDs for traceability
        findings_by_paper = {}
        for paper_id, paper_findings in findings.items():
            findings_by_paper[paper_id] = [f"[{paper_id}] {finding}" for finding in paper_findings]
        
        # Organize findings by paper with clear separation
        organized_findings = []
        for paper_id, paper_findings in findings_by_paper.items():
            paper_text = f"PAPER {paper_id} (findings: {len(paper_findings)}):\n" + "\n".join(paper_findings)
            organized_findings.append(paper_text)
        
        all_organized_findings = "\n\n".join(organized_findings)
        
        # Step B: Identify themes across papers
        theme_identification_prompt = f"""
        {original_query_reminder}
        
        Analyze these findings from multiple papers and identify 3-5 key themes or patterns that emerge:
        
        {all_organized_findings}
        
        For each theme:
        1. Give it a descriptive title that directly relates to the research question
        2. List which paper IDs contain findings related to this theme
        3. Explain how the theme addresses the research question
        
        Format:
        THEME 1: [Title]
        Papers: [list of paper IDs]
        Relation to research question: [explanation]
        
        THEME 2: [Title]
        Papers: [list of paper IDs] 
        Relation to research question: [explanation]
        
        And so on.
        
        IMPORTANT: Ensure themes directly address the research question and don't drift to tangential topics.
        Focus on SYNTHESIZING findings rather than just summarizing papers.
        """
        
        # Get themes
        themes_response = self.ollama.generate(theme_identification_prompt)
        
        # Step C: Create synthesis with balanced citation distribution
        synthesis_prompt = f"""
        {original_query_reminder}
        
        Paper findings:
        {all_organized_findings}
        
        Identified themes:
        {themes_response}
        
        Now create a comprehensive synthesis that:
        
        1. DIRECTLY addresses the research question: "{research_question}"
        2. Integrates findings across ALL papers (use EVERY paper ID at least once)
        3. Organizes content by themes while maintaining coherence
        4. Highlights agreements and disagreements between papers
        5. Cites specific findings by including the paper ID in square brackets [paper_id]
        6. Maintains balance - papers with more findings ({paper_finding_counts}) should be cited proportionally
        7. Avoids topic drift - stay FOCUSED on the ORIGINAL research question
        
        CRITICAL REQUIREMENTS:
        - EVERY paper MUST be cited at least once
        - Cite papers with MORE findings more frequently
        - Use the EXACT paper IDs in square brackets: [paper_0], [paper_1], etc.
        - Keep focus STRICTLY on the ORIGINAL research question
        - Include ALL major themes identified
        - Be specific and concrete about findings, not vague
        - Synthesize across papers rather than simply summarizing each paper
        
        Your synthesis:
        """
        
        # Generate synthesis
        synthesis = self.ollama.generate(synthesis_prompt)
        
        # Verify all papers are cited
        missing_citations = []
        for paper_id in findings.keys():
            if f"[{paper_id}]" not in synthesis:
                missing_citations.append(paper_id)
        
        # If any papers are missing, generate a supplementary paragraph to include them
        if missing_citations:
            missing_findings = []
            for paper_id in missing_citations:
                for finding in findings.get(paper_id, []):
                    missing_findings.append(f"[{paper_id}] {finding}")
        
            missing_findings_text = "\n".join(missing_findings)
            
            supplement_prompt = f"""
            {original_query_reminder}
            
            Your synthesis is missing citations to these papers: {missing_citations}
            
            Add a paragraph that integrates these missing findings into your synthesis:
            
            Missing findings:
            {missing_findings_text}
            
            Create a brief paragraph that:
            1. Connects these findings DIRECTLY to the research question: "{research_question}"
            2. Cites each missing paper at least once using [paper_id]
            3. Fits coherently with the rest of the synthesis
            4. Maintains focus on the original research query
            
            Additional paragraph:
            """
            
            supplement = self.ollama.generate(supplement_prompt)
            
            # Add the supplement to the synthesis
            synthesis += "\n\n" + supplement
        
        return synthesis
    
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
    
    def generate_refined_search_queries(self, initial_papers: List[Paper], research_question: str) -> List[str]:
        """
        Generate refined search queries based on the initial papers found.
        
        This implements a human-like approach where researchers adjust their search
        strategy based on what they found in initial papers.
        
        Parameters:
        -----------
        initial_papers: List of papers found in initial search
        research_question: The research question
        
        Returns:
        --------
        list: List of refined search queries
        """
        if not initial_papers:
            return []
            
        # Extract specialized terminology from the papers first
        specialized_terms = self.extract_specialized_terminology(initial_papers)
        
        # Identify the key themes from the abstracts
        papers_info = "\n\n".join([
            f"Paper {i+1}:\nTitle: {paper.title}\nAbstract: {paper.abstract}"
            for i, paper in enumerate(initial_papers[:5])  # Limit to first 5 papers for manageability
        ])
        
        # Extract key methods, materials, and applications from the papers
        extraction_prompt = f"""
        Analyze these papers related to the research question:
        
        Research Question: {research_question}
        
        Based on the titles and abstracts, extract:
        1. SPECIFIC methods/techniques mentioned (e.g., "attention mechanism", "conformal prediction")
        2. SPECIFIC materials or domains discussed (e.g., "perovskite solar cells", "metal-organic frameworks")
        3. SPECIFIC applications targeted (e.g., "drug discovery", "battery design")
        4. SPECIFIC performance metrics used (e.g., "prediction accuracy", "uncertainty calibration")
        
        For each category, list the 3-5 most important terms found across the papers.
        
        Papers:
        {papers_info}
        
        Format your response as:
        
        METHODS: [comma-separated list]
        MATERIALS: [comma-separated list]
        APPLICATIONS: [comma-separated list]
        METRICS: [comma-separated list]
        """
        
        extraction_response = self.ollama.generate(extraction_prompt)
        
        # Parse the extracted terms
        methods = []
        materials = []
        applications = []
        metrics = []
        
        # Extract each category
        methods_match = re.search(r'METHODS:(.+?)(?=MATERIALS:|APPLICATIONS:|METRICS:|$)', extraction_response, re.DOTALL)
        if methods_match:
            methods = [m.strip() for m in methods_match.group(1).split(',') if m.strip()]
        
        materials_match = re.search(r'MATERIALS:(.+?)(?=METHODS:|APPLICATIONS:|METRICS:|$)', extraction_response, re.DOTALL)
        if materials_match:
            materials = [m.strip() for m in materials_match.group(1).split(',') if m.strip()]
        
        applications_match = re.search(r'APPLICATIONS:(.+?)(?=METHODS:|MATERIALS:|METRICS:|$)', extraction_response, re.DOTALL)
        if applications_match:
            applications = [a.strip() for a in applications_match.group(1).split(',') if a.strip()]
        
        metrics_match = re.search(r'METRICS:(.+?)(?=METHODS:|MATERIALS:|APPLICATIONS:|$)', extraction_response, re.DOTALL)
        if metrics_match:
            metrics = [m.strip() for m in metrics_match.group(1).split(',') if m.strip()]
        
        # Create a prompt to generate database search queries
        query_prompt = f"""
        You're a database search expert helping a researcher refine their search queries.
        
        Research Question: {research_question}
        
        Based on initial papers, we've identified these key elements:
        
        Methods/Techniques: {', '.join(methods[:5])}
        Materials/Domains: {', '.join(materials[:5])}
        Applications: {', '.join(applications[:5])}
        Metrics: {', '.join(metrics[:5])}
        Specialized Terminology: {', '.join(specialized_terms[:7])}
        
        Generate 3 highly specific academic database search queries that will find NEW relevant papers.
        Create queries that search for different aspects or combinations of the research question.
        
        Each query MUST:
        1. Use proper Boolean operators (AND, OR) with parentheses
        2. Include specific technical terms from the lists above
        3. Include a year range filter [2014-2024]
        4. Focus on retrieving highly relevant papers
        5. Use standard database search syntax
        6. Be DIFFERENT from each other (explore different aspects)
        
        Format each query as:
        1. (term1 AND term2) AND (term3 OR term4) AND year:[2014 TO 2024]
        2. (term5 AND term6) AND (term7 OR term8) AND year:[2014 TO 2024]
        3. (term9 AND term10) AND (term11 OR term12) AND year:[2014 TO 2024]
        
        ONLY provide the 3 queries numbered as shown above, with no other text.
        """
        
        response = self.ollama.generate(query_prompt)
        
        # Parse the response to extract refined queries
        refined_queries = []
        
        for line in response.split('\n'):
            line = line.strip()
            # Look for numbered lines with search queries
            if re.match(r'^\d+\.', line):
                # Extract the query part
                query_match = re.search(r'^\d+\.\s*(.+?)(?:\s*-|$)', line)
                if query_match:
                    query = query_match.group(1).strip()
                    # Validate that it's a proper query (contains Boolean operators)
                    if query and ("AND" in query or "OR" in query) and len(query) > 10:
                        refined_queries.append(query)
        
        # If we don't have enough valid queries, try a different approach
        if len(refined_queries) < 3:
            # Combine different aspects from our extracted terms
            all_terms = []
            all_terms.extend(methods)
            all_terms.extend(specialized_terms)
            all_terms.extend(materials)
            all_terms.extend(applications)
            
            # Remove duplicates
            all_terms = list(set([term for term in all_terms if term]))
            
            if len(all_terms) >= 6:
                # Create balanced queries
                if len(refined_queries) < 1 and len(all_terms) >= 2:
                    refined_queries.append(f"({all_terms[0]} AND {all_terms[1]}) AND year:[2014 TO 2024]")
                
                if len(refined_queries) < 2 and len(all_terms) >= 4:
                    refined_queries.append(f"({all_terms[2]} AND {all_terms[3]}) AND year:[2014 TO 2024]")
                
                if len(refined_queries) < 3 and len(all_terms) >= 6:
                    refined_queries.append(f"({all_terms[4]} AND {all_terms[5]}) AND year:[2014 TO 2024]")
        
        # If we still have fewer than 3 queries, add some from the research question itself
        if len(refined_queries) < 3:
            # Extract key terms from the research question
            question_terms = re.findall(r'\b[A-Za-z][A-Za-z-]+\b', research_question)
            question_terms = [term for term in question_terms if len(term) > 4 and term.lower() not in ['what', 'where', 'when', 'which', 'there', 'their', 'about', 'would', 'could', 'should']]
            
            if len(question_terms) >= 4:
                if len(refined_queries) < 1:
                    refined_queries.append(f"({question_terms[0]} AND {question_terms[1]}) AND year:[2014 TO 2024]")
                
                if len(refined_queries) < 2:
                    refined_queries.append(f"({question_terms[2]} AND {question_terms[3]}) AND year:[2014 TO 2024]")
                
                if len(refined_queries) < 3 and len(question_terms) >= 6:
                    refined_queries.append(f"({question_terms[4]} AND {question_terms[5]}) AND year:[2014 TO 2024]")
        
        # Ensure we have at least one query
        if not refined_queries and research_question:
            # Create a basic query from the research question
            words = [w for w in research_question.split() if len(w) > 4]
            if len(words) >= 2:
                refined_queries.append(f"({words[0]} AND {words[1]}) AND year:[2014 TO 2024]")
        
        # Remove duplicates and limit to 3 queries
        unique_queries = []
        for query in refined_queries:
            if query not in unique_queries:
                unique_queries.append(query)
                if len(unique_queries) >= 3:
                    break
        
        return unique_queries
        
    def extract_specialized_terminology(self, papers: List[Paper]) -> List[str]:
        """
        Extract specialized terminology from papers to improve search refinement.
        
        Parameters:
        -----------
        papers: List of papers to analyze
        
        Returns:
        --------
        list: List of specialized terms found in the papers
        """
        if not papers:
            return []
            
        # Combine abstracts for analysis
        combined_text = "\n\n".join([
            f"Title: {paper.title}\nAbstract: {paper.abstract}"
            for paper in papers[:5]  # Limit to first 5 papers for manageability
        ])
        
        prompt = f"""
        Extract specialized technical terminology from these scientific papers that could help refine literature searches.
        Look for:
        1. Domain-specific technical terms
        2. Methodological approaches
        3. Theoretical frameworks
        4. Research-specific abbreviations or acronyms (with their meanings)
        5. Names of specific techniques, algorithms, or materials
        
        For each term, briefly explain why it's significant for searches.
        
        Papers:
        {combined_text}
        
        Format each term:
        Term: [specialized term]
        Type: [type of term from the list above]
        Significance: [brief explanation of why this term would help refine searches]
        
        Provide at least 5-7 specialized terms.
        """
        
        response = self.ollama.generate(prompt)
        
        # Extract the terms
        terms = []
        current_term = None
        
        for line in response.split('\n'):
            line = line.strip()
            
            # Look for term lines
            term_match = re.search(r'^Term:\s*(.+)$', line)
            if term_match:
                current_term = term_match.group(1).strip()
                if current_term:
                    terms.append(current_term)
        
        # If regex extraction failed, fall back to simpler approach
        if not terms:
            # Look for any capitalized technical terms
            terms = re.findall(r'[A-Z][a-zA-Z\-]+(?:\s+[a-zA-Z][a-zA-Z\-]+){0,3}', response)
            
        # Remove duplicates and limit length
        unique_terms = []
        for term in terms:
            if term.lower() not in [t.lower() for t in unique_terms]:
                unique_terms.append(term)
                
        return unique_terms[:10]  # Return up to 10 unique terms

    def identify_citation_leads(self, papers: List[Paper]) -> List[str]:
        """
        Identify promising citation leads from papers to expand the literature search.
        
        This mimics how human researchers follow citation trails by checking
        the most cited works or most relevant-sounding references.
        
        Parameters:
        -----------
        papers: List of papers to analyze
        
        Returns:
        --------
        list: List of search queries based on citation leads
        """
        if not papers:
            return []
            
        # Combine paper information including authors
        papers_info = "\n\n".join([
            f"Paper {i+1}:\nTitle: {paper.title}\nAbstract: {paper.abstract}\nAuthors: {', '.join(paper.authors)}\nYear: {paper.year if paper.year else 'Unknown'}"
            for i, paper in enumerate(papers[:3])  # Limit to first 3 for focus
        ])
        
        # Extract any author names to use in search queries
        author_names = []
        for paper in papers[:3]:
            if paper.authors:
                # Take first and last author's last names
                if len(paper.authors) > 0:
                    first_author = paper.authors[0].split()[-1] if paper.authors[0].split() else ""
                    if first_author and len(first_author) > 2:
                        author_names.append(first_author)
                        
                if len(paper.authors) > 1:
                    last_author = paper.authors[-1].split()[-1] if paper.authors[-1].split() else ""
                    if last_author and len(last_author) > 2 and last_author != first_author:
                        author_names.append(last_author)
        
        # Extract key methods and technical terms from papers
        methods_prompt = f"""
        From these papers, extract:
        1. Key methods or algorithms mentioned
        2. Technical terms that might be useful for finding related papers
        3. Names of theories, frameworks, or models referenced
        
        Return ONLY a comma-separated list of the extracted terms (no explanations).
        
        Papers:
        {papers_info}
        """
        
        methods_response = self.ollama.generate(methods_prompt)
        key_methods = [term.strip() for term in methods_response.split(',') if term.strip()]
        
        # Create prompt to identify potential citation leads
        prompt = f"""
        You're creating academic database search queries to find papers that are:
        
        1. Frequently cited by these papers (seminal works)
        2. Use similar methods but in different applications
        3. Provide methodological foundations for these papers
        
        Based on these papers, create specific database search queries:
        
        {papers_info}
        
        Generate 3 different search queries in proper database syntax that will find papers in the reference lists of the above papers.
        
        Each query must:
        1. Use Boolean operators (AND, OR) with parentheses
        2. Include specific technical terms, methods, or author names
        3. Include a year range filter
        4. Focus on finding CITED papers (papers referenced by the above)
        
        Use these author names when relevant: {', '.join(author_names)}
        Use these key methods/terms when relevant: {', '.join(key_methods[:7] if key_methods else [])}
        
        Format EXACTLY as follows (use proper database syntax):
        1. (term1 AND term2) AND (term3 OR author:"Name") AND year:[1990 TO 2022]
        2. (term4 AND term5) AND (term6 OR author:"Name") AND year:[1990 TO 2022]
        3. (term7 AND term8) AND (term9 OR author:"Name") AND year:[1990 TO 2022]
        
        ONLY provide the numbered queries, nothing else.
        """
        
        response = self.ollama.generate(prompt)
        
        # Parse the response to extract search queries
        citation_queries = []
        
        for line in response.split('\n'):
            line = line.strip()
            
            # Look for numbered lines with proper search queries
            if re.match(r'^\d+\.', line):
                # Extract the actual query
                query_match = re.search(r'^\d+\.\s*(.+?)(?:\s*-|$)', line)
                if query_match:
                    query = query_match.group(1).strip()
                    if query and ("AND" in query or "OR" in query) and len(query) > 10:  # Basic validation
                        citation_queries.append(query)
        
        # If we don't have enough queries, create author-based searches
        if len(citation_queries) < 3 and author_names:
            for author in author_names:
                if len(citation_queries) >= 3:
                    break
                    
                # Create query with author name and keywords
                keywords = key_methods[:3] if key_methods else []
                if keywords:
                    query = f'(author:"{author}") AND ({" OR ".join(keywords)}) AND year:[1990 TO 2020]'
                else:
                    query = f'author:"{author}" AND year:[1990 TO 2020]'
                    
                if query not in citation_queries:
                    citation_queries.append(query)
        
        # If still insufficient, create keyword-based queries
        if len(citation_queries) < 3 and key_methods:
            # Use pairs of keywords
            for i in range(0, len(key_methods)-1, 2):
                if len(citation_queries) >= 3:
                    break
                    
                if i+1 < len(key_methods):
                    query = f'({key_methods[i]} AND {key_methods[i+1]}) AND year:[1990 TO 2020]'
                    if query not in citation_queries:
                        citation_queries.append(query)
        
        # If still insufficient, fall back to paper titles
        if len(citation_queries) < 3:
            for paper in papers[:3]:
                if len(citation_queries) >= 3:
                    break
                    
                # Extract 2-3 important terms from the title
                title_words = [word for word in paper.title.split() 
                            if len(word) > 4 and word.lower() not in ["and", "or", "the", "for", "from", "with"]]
                
                if len(title_words) >= 2:
                    query = f"({title_words[0]}) AND ({title_words[1]}) AND year:[1990 TO 2020]"
                    if query not in citation_queries and len(query) > 15:
                        citation_queries.append(query)
                    
        # Remove duplicates and limit to 3 queries
        unique_queries = []
        for query in citation_queries:
            if query not in unique_queries:
                unique_queries.append(query)
                if len(unique_queries) >= 3:
                    break
                
        return unique_queries
        
    def follow_citation_trail(self, initial_papers: List[Paper], research_question: str, literature_manager: Any) -> List[Paper]:
        """
        Follow citation trails to expand literature search.
        
        Parameters:
        -----------
        initial_papers: List of papers from the initial search
        research_question: The research question
        literature_manager: LiteratureManager instance for searching
        
        Returns:
        --------
        list: List of papers found through citation trails
        """
        # Identify potential citation leads from initial papers
        citation_queries = self.identify_citation_leads(initial_papers)
        
        if not citation_queries:
            return []
            
        # Search for papers based on citation leads
        citation_papers = []
        
        print(f"\nFollowing citation trails with {len(citation_queries)} queries...")
        for i, query in enumerate(citation_queries):
            print(f"Citation trail search {i+1}/{len(citation_queries)}: '{query}'")
            
            try:
                # Use LiteratureManager to search
                papers = literature_manager.search(
                    query=query,
                    max_papers=3,  # Limit per query to avoid overwhelming
                    sources=["open_alex", "arxiv"]  # Focus on academic sources
                )
                
                print(f"Found {len(papers)} papers from citation trail")
                
                # Add unique papers not already in initial papers
                for paper in papers:
                    if paper not in initial_papers and paper not in citation_papers:
                        citation_papers.append(paper)
                        
                # Small delay between searches
                if i < len(citation_queries) - 1:
                    import time
                    time.sleep(2)  # 2-second delay between citation searches
                    
            except Exception as e:
                print(f"Error in citation trail search: {e}")
        
        # Screen citation papers for relevance to the research question
        citation_papers = self.screen_papers_by_title(citation_papers)
        print(f"Title screening for citation papers: {len(citation_papers)} papers passed")
        
        return citation_papers

    def assess_methodology_quality(self, papers: List[Paper]) -> Dict[str, Dict[str, Any]]:
        """
        Assess the methodology quality of papers to prioritize higher quality evidence.
        
        Researchers evaluate study methodology to determine the strength of evidence.
        This method mimics that critical assessment process.
        
        Parameters:
        -----------
        papers: List of papers to assess
        
        Returns:
        --------
        dict: Dictionary mapping paper IDs to methodology assessment results
        """
        if not papers:
            return {}
            
        # Prepare paper information
        papers_info = []
        for i, paper in enumerate(papers):
            paper_id = f"paper_{i}"
            papers_info.append(f"Paper ID: {paper_id}\nTitle: {paper.title}\nAbstract: {paper.abstract}")
            
        papers_text = "\n\n".join(papers_info)
        
        # Create prompt for methodology assessment
        prompt = f"""
        As a methodologist, critically assess the research methodologies described in these papers.
        For each paper, evaluate:
        
        1. Study design strength (e.g., RCT, cohort study, case-control, cross-sectional, case series)
        2. Sample size and representativeness
        3. Control of confounding variables
        4. Appropriateness of statistical methods (if mentioned)
        5. Potential for bias
        6. Generalizability of findings
        
        Base your assessment only on information provided in the abstracts.
        
        Papers to evaluate:
        {papers_text}
        
        For each paper, provide:
        1. Study design (if identifiable)
        2. Methodological strengths (list key strengths)
        3. Methodological limitations (list key limitations)
        4. Evidence quality rating (High/Medium/Low/Insufficient Information)
        
        Format each assessment with the exact Paper ID as shown above, followed by the assessment.
        """
        
        response = self.ollama.generate(prompt)
        
        # Parse the response to extract methodology assessments
        assessments = {}
        current_paper_id = None
        current_assessment = {"design": "", "strengths": [], "limitations": [], "quality": ""}
        
        # Process each line of the response
        for line in response.split('\n'):
            line = line.strip()
            
            # Look for paper ID line
            paper_id_match = re.search(r'paper_\d+', line, re.IGNORECASE)
            if paper_id_match:
                # If we were processing a paper, save it before starting the next one
                if current_paper_id:
                    assessments[current_paper_id] = current_assessment.copy()
                    
                # Start a new paper assessment
                current_paper_id = paper_id_match.group(0)
                current_assessment = {"design": "", "strengths": [], "limitations": [], "quality": ""}
                continue
                
            # Look for study design
            if re.search(r'study design|design', line, re.IGNORECASE):
                design_match = re.search(r'(?:study design|design)[:\s]+(.*)', line, re.IGNORECASE)
                if design_match:
                    current_assessment["design"] = design_match.group(1).strip()
                    
            # Look for strengths
            elif re.search(r'strength', line, re.IGNORECASE):
                # If the line contains both "strengths" and a colon, extract what follows
                strengths_match = re.search(r'strength[s]?[:\s]+(.*)', line, re.IGNORECASE)
                if strengths_match:
                    strength = strengths_match.group(1).strip()
                    if strength:
                        current_assessment["strengths"].append(strength)
                        
            # Look for limitations
            elif re.search(r'limitation', line, re.IGNORECASE):
                limitations_match = re.search(r'limitation[s]?[:\s]+(.*)', line, re.IGNORECASE)
                if limitations_match:
                    limitation = limitations_match.group(1).strip()
                    if limitation:
                        current_assessment["limitations"].append(limitation)
                        
            # Look for quality rating
            elif re.search(r'quality|evidence', line, re.IGNORECASE):
                quality_match = re.search(r'(?:quality|evidence)[:\s]+(.*)', line, re.IGNORECASE)
                if quality_match:
                    quality = quality_match.group(1).strip()
                    if re.search(r'high|medium|low|insufficient', quality, re.IGNORECASE):
                        current_assessment["quality"] = quality
                        
            # Look for bullet points that might be strengths or limitations
            elif line.startswith('-') or line.startswith('*'):
                # If we've seen a strengths line recently, this might be a continuation
                if current_assessment["strengths"] and not current_assessment["limitations"]:
                    strength = line[1:].strip()
                    if strength:
                        current_assessment["strengths"].append(strength)
                # If we've seen a limitations line recently, this might be a continuation
                elif current_assessment["limitations"]:
                    limitation = line[1:].strip()
                    if limitation:
                        current_assessment["limitations"].append(limitation)
        
        # Don't forget to add the last paper
        if current_paper_id and current_paper_id not in assessments:
            assessments[current_paper_id] = current_assessment
            
        return assessments
        
    def identify_conflicts_and_agreements(self, papers: List[Paper], key_findings: Dict[str, List[str]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Identify conflicts and agreements between papers to support critical analysis.
        
        Human researchers critically compare findings across papers to identify
        areas of consensus and conflict.
        
        Parameters:
        -----------
        papers: List of papers
        key_findings: Dictionary of paper ID to list of key findings
        
        Returns:
        --------
        dict: Dictionary containing conflicts and agreements
        """
        if not papers or len(papers) < 2:
            return {"conflicts": [], "agreements": []}
            
        # Prepare paper information with findings
        papers_with_findings = []
        
        for i, paper in enumerate(papers):
            paper_id = f"paper_{i}"
            if paper_id in key_findings and key_findings[paper_id]:
                findings_text = "\n".join([f"- {finding}" for finding in key_findings[paper_id]])
                
                papers_with_findings.append(
                    f"Paper {i+1} (ID: {paper_id}):\n"
                    f"Title: {paper.title}\n"
                    f"Authors: {', '.join(paper.authors)}\n"
                    f"Year: {paper.year if paper.year else 'Unknown'}\n"
                    f"Key Findings:\n{findings_text}"
                )
        
        if len(papers_with_findings) < 2:
            return {"conflicts": [], "agreements": []}
            
        papers_text = "\n\n".join(papers_with_findings)
        
        # Create prompt to identify conflicts and agreements
        prompt = f"""
        As a critical research synthesizer, compare the findings across these papers to identify:
        1. Points of agreement/consensus
        2. Points of conflict/contradiction
        3. Areas where findings build upon each other
        
        Paper information and findings:
        {papers_text}
        
        Format your response as follows:
        
        AGREEMENTS:
        1. [Specific point of agreement] - Found in papers: [paper IDs]
        [Explanation of the agreement and its significance]
        
        2. [Next point of agreement] - Found in papers: [paper IDs]
        [Explanation]
        
        CONFLICTS:
        1. [Specific point of conflict] - Between papers: [paper IDs with conflicting views]
        [Explanation of the conflicting positions and possible reasons]
        
        2. [Next point of conflict] - Between papers: [paper IDs]
        [Explanation]
        
        BUILDING RELATIONSHIPS:
        1. [How one paper builds on another] - Papers: [paper IDs in relationship]
        [Explanation of how the findings connect or extend each other]
        
        Focus on substantive conflicts and agreements related to research findings, not superficial similarities or differences.
        """
        
        response = self.ollama.generate(prompt)
        
        # Parse the response to extract conflicts and agreements
        conflicts = []
        agreements = []
        
        # Extract agreements section
        agreements_section = re.search(r'AGREEMENTS:(.*?)(?:CONFLICTS:|BUILDING RELATIONSHIPS:|$)', response, re.DOTALL)
        if agreements_section:
            agreements_text = agreements_section.group(1).strip()
            # Extract individual agreement points
            agreement_points = re.findall(r'\d+\.\s*(.*?)(?=\d+\.|$)', agreements_text, re.DOTALL)
            
            for point in agreement_points:
                point = point.strip()
                if point:
                    # Extract the point and the paper IDs
                    point_match = re.search(r'(.*?)\s*-\s*Found in papers:\s*(.*?)(?:\n|$)', point, re.DOTALL)
                    if point_match:
                        point_text = point_match.group(1).strip()
                        paper_ids_text = point_match.group(2).strip()
                        paper_ids = re.findall(r'paper_\d+', paper_ids_text)
                        
                        explanation = re.sub(r'.*?(?:\n|$)', '', point, count=1).strip()
                        
                        agreements.append({
                            "point": point_text,
                            "paper_ids": paper_ids,
                            "explanation": explanation
                        })
        
        # Extract conflicts section
        conflicts_section = re.search(r'CONFLICTS:(.*?)(?:BUILDING RELATIONSHIPS:|$)', response, re.DOTALL)
        if conflicts_section:
            conflicts_text = conflicts_section.group(1).strip()
            # Extract individual conflict points
            conflict_points = re.findall(r'\d+\.\s*(.*?)(?=\d+\.|$)', conflicts_text, re.DOTALL)
            
            for point in conflict_points:
                point = point.strip()
                if point:
                    # Extract the point and the paper IDs
                    point_match = re.search(r'(.*?)\s*-\s*Between papers:\s*(.*?)(?:\n|$)', point, re.DOTALL)
                    if point_match:
                        point_text = point_match.group(1).strip()
                        paper_ids_text = point_match.group(2).strip()
                        paper_ids = re.findall(r'paper_\d+', paper_ids_text)
                        
                        explanation = re.sub(r'.*?(?:\n|$)', '', point, count=1).strip()
                        
                        conflicts.append({
                            "point": point_text,
                            "paper_ids": paper_ids,
                            "explanation": explanation
                        })
        
        return {
            "conflicts": conflicts,
            "agreements": agreements
        }
        
    def synthesize_findings_with_critical_analysis(self, papers: List[Paper], research_question: str) -> str:
        """
        Synthesize findings with critical analysis of methodology and conflicting evidence.
        
        This enhanced synthesis incorporates methodology quality assessment and 
        explicit analysis of conflicting findings for a more nuanced summary.
        
        Parameters:
        -----------
        papers: List of papers to synthesize
        research_question: The research question
        
        Returns:
        --------
        str: Synthesized findings with critical analysis
        """
        if not papers:
            return "No papers available for synthesis."
            
        # First extract key findings from each paper
        key_findings = {}
        for i, paper in enumerate(papers):
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
            
            # Parse findings
            findings = [line.strip()[2:].strip() for line in response.split("\n") 
                       if line.strip() and line.strip().startswith("-")]
            
            key_findings[paper_id] = findings
            
        # Assess methodology quality
        methodology_assessments = self.assess_methodology_quality(papers)
        
        # Identify conflicts and agreements
        analysis = self.identify_conflicts_and_agreements(papers, key_findings)
        
        # Generate paper info text with quality assessments
        papers_with_assessments = []
        for i, paper in enumerate(papers):
            paper_id = f"paper_{i}"
            
            quality_info = ""
            if paper_id in methodology_assessments:
                assessment = methodology_assessments[paper_id]
                quality = assessment.get("quality", "").strip()
                design = assessment.get("design", "").strip()
                
                if quality or design:
                    quality_info = f"\nMethodology: {design}\nEvidence Quality: {quality}"
                    
                    if assessment.get("strengths"):
                        strengths = "; ".join(assessment["strengths"])
                        quality_info += f"\nStrengths: {strengths}"
                        
                    if assessment.get("limitations"):
                        limitations = "; ".join(assessment["limitations"])
                        quality_info += f"\nLimitations: {limitations}"
            
            findings_text = ""
            if paper_id in key_findings and key_findings[paper_id]:
                findings_text = "\nFindings:\n" + "\n".join([f"- {finding}" for finding in key_findings[paper_id]])
                
            papers_with_assessments.append(
                f"Paper {i+1} (ID: {paper_id}):\n"
                f"Title: {paper.title}\n"
                f"Authors: {', '.join(paper.authors)}\n"
                f"Year: {paper.year if paper.year else 'Unknown'}"
                f"{quality_info}"
                f"{findings_text}"
            )
            
        papers_text = "\n\n".join(papers_with_assessments)
        
        # Format conflicts and agreements for synthesis
        conflicts_text = ""
        if analysis["conflicts"]:
            conflicts_text = "Conflicts in the Literature:\n"
            for i, conflict in enumerate(analysis["conflicts"]):
                conflicts_text += f"{i+1}. {conflict['point']} - Between papers: {', '.join(conflict['paper_ids'])}\n"
                conflicts_text += f"   {conflict['explanation']}\n\n"
                
        agreements_text = ""
        if analysis["agreements"]:
            agreements_text = "Agreements in the Literature:\n"
            for i, agreement in enumerate(analysis["agreements"]):
                agreements_text += f"{i+1}. {agreement['point']} - Found in papers: {', '.join(agreement['paper_ids'])}\n"
                agreements_text += f"   {agreement['explanation']}\n\n"
                
        # Create enhanced synthesis prompt
        prompt = f"""
        Research Question: {research_question}
        
        Paper Information with Quality Assessment:
        {papers_text}
        
        {conflicts_text}
        {agreements_text}
        
        As a critical research synthesizer, create a comprehensive synthesis of the literature that:
        
        1. Addresses the research question directly
        2. Weighs evidence based on methodology quality (prioritizing higher quality studies)
        3. Explicitly addresses conflicting findings and potential reasons for disagreements
        4. Acknowledges limitations in the literature
        5. Maintains a balanced view of the evidence
        6. Cites specific papers using their IDs (e.g., [paper_0])
        
        Structure your synthesis with these sections:
        - Overview of the Literature
        - Synthesis of Key Findings (with reference to methodology quality)
        - Analysis of Conflicts and Agreements
        - Limitations of Current Evidence
        - Conclusions and Implications
        
        The synthesis should be scholarly, nuanced, and reflect the current state of evidence, including uncertainties.
        """
        
        synthesis = self.ollama.generate(prompt)
        
        return synthesis 