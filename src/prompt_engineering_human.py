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
        list: Search queries formatted for academic databases
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
        
        # Create a direct query based on the original query terms
        # If no specific terms found, extract significant words from the research question
        if not query_terms:
            # Extract significant words from the research question, avoiding common words
            common_words = {"and", "or", "the", "in", "on", "at", "by", "for", "with", 
                           "a", "an", "of", "to", "is", "are", "were", "was", "be", 
                           "have", "has", "had", "can", "could", "would", "should",
                           "what", "how", "why", "when", "where", "which", "who", "latest"}
            
            significant_words = [word.lower() for word in re.findall(r'\b\w+\b', research_question) 
                               if word.lower() not in common_words and len(word) > 3]
            
            # Use the most significant words from the research question
            query_terms = significant_words[:5]
        
        # Ensure we have at least some query terms
        if not query_terms:
            # Fallback to extracting nouns from the query if everything else fails
            nouns = []
            for word in query.split():
                if len(word) > 3 and word.lower() not in {"and", "or", "the", "for", "with", "what", "how"}:
                    nouns.append(word)
            
            if nouns:
                query_terms = nouns[:5]
            else:
                # Absolute last resort - use the first few words of the query
                query_terms = [word for word in query.split()[:5] if len(word) > 3]
        
        # Create queries based on the query terms
        search_queries = []
        
        # Ensure we have at least one valid query term
        if query_terms:
            # Basic query - combine the main terms with AND
            direct_query = ' AND '.join([term for term in query_terms[:3]])
            search_queries.append(direct_query)
        else:
            # If we still have no terms, use the entire research question as a fallback
            search_queries.append(research_question)
        
        # Create prompts for additional search query generation
        if len(query_terms) > 1:
            prompt = f"""
            You're preparing search queries for academic databases based on:
            
            Original Query: {query}
            Research Question: {research_question}
            
            The most critical terms from the query are: {', '.join(query_terms)}
            
            Create {max_queries - 1} different academic database search queries that:
            1. ALWAYS include the core query terms listed above
            2. Use proper Boolean operators (AND, OR) with parentheses
            3. Are formatted for academic databases
            4. Have different focuses while remaining true to the original query
            5. Use quoted phrases for exact matches
            
            The queries should follow this format:
            1. ("term1" AND "term2") AND ("term3" OR "term4")
            2. ("term1" AND "term5") AND ("term6")
            
            Focus on generating queries that will yield DIRECTLY relevant papers.
            ALWAYS include the original query terms, then add synonyms or related concepts.
            """
            
            response = self.ollama.generate(prompt)
            
            # Parse the response
            lines = response.strip().split('\n')
            for line in lines:
                if re.match(r'^\d+\.', line):
                    # This line starts with a number and period, likely a query
                    query_text = re.sub(r'^\d+\.\s*', '', line).strip()
                    if query_text and len(query_text) > 10:  # Basic validation
                        search_queries.append(query_text)
        
        # Ensure we have at least one query and not more than max_queries
        if not search_queries:
            # Last resort fallback
            search_queries = [query.replace("?", "")]
        
        # Ensure all queries have valid content
        search_queries = [q for q in search_queries if q and len(q.strip()) > 3]
        
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
        import re
        current_year = datetime.datetime.now().year
        
        # Extract research question from the initial response
        research_question = self._extract_research_question(initial_response)
        
        # Filter papers to ensure they match the time frame if specified in the research question
        time_frame_match = re.search(r'(\d{4})[-–](\d{4})', research_question)
        filtered_papers = papers
        
        if time_frame_match:
            start_year, end_year = map(int, time_frame_match.groups())
            # Filter papers by publication year
            filtered_papers = [
                paper for paper in papers 
                if paper.year and int(paper.year) >= start_year and int(paper.year) <= end_year
            ]
            
            # If filtering removed all papers, use original set but note the discrepancy
            if not filtered_papers:
                filtered_papers = papers
        
        # Domain anchoring - extract key domain terms explicitly
        domain_terms = []
        
        if query:
            # Extract domain terms from both query and research question
            all_text = query + " " + research_question
            # Get significant words, filter out common words
            words = [word.lower() for word in re.findall(r'\b\w+\b', all_text)]
            common_words = {"and", "or", "the", "in", "on", "at", "by", "for", "with", 
                           "a", "an", "of", "to", "is", "are", "were", "was", "be", 
                           "have", "has", "had", "can", "could", "would", "should", 
                           "what", "how", "why", "when", "where", "which", "who"}
            domain_terms = [word for word in words if word not in common_words]
            # Keep only terms that appear in at least 30% of paper titles or abstracts
            if filtered_papers:
                domain_terms = [
                    term for term in domain_terms
                    if sum(1 for p in filtered_papers if term.lower() in p.title.lower() or term.lower() in p.abstract.lower())
                    >= 0.3 * len(filtered_papers)
                ]
        
        # Key paper information for reasoning
        paper_info = []
        for i, paper in enumerate(filtered_papers):
            paper_info.append(f"Paper {i}: {paper.title} ({paper.year})")
            
        # Hierarchical analysis for deeper synthesis
        
        # First, basic findings extraction from each paper
        findings = {}
        for i, paper in enumerate(filtered_papers):
            paper_id = f"paper_{i}"
            
            # Generate analysis prompt for this paper
            prompt = f"""
            Analyze this scientific paper, focusing on its most significant findings and contributions:
            
            Title: {paper.title}
            Authors: {', '.join(paper.authors)}
            Year: {paper.year}
            Abstract: {paper.abstract}
            
            Extract 3-5 key findings from this paper. Focus on:
            1. Main results and conclusions
            2. Methodological approaches used
            3. Limitations acknowledged
            4. How this relates to the research question: "{research_question}"
            
            For each finding, write a 1-2 sentence summary that captures the essence of what was discovered.
            Ensure each finding is specific and directly tied to evidence provided in the paper.
            
            Format your response as a simple list of findings, one per line.
            """
            
            # Extract findings
            response = self.ollama.generate(prompt)
            
            # Process response into list of findings
            paper_findings = [
                line.strip().strip('•-*').strip()
                for line in response.split('\n')
                if line.strip() and not line.startswith('#') and not line.startswith('Finding')
            ]
            
            # Filter out any non-finding lines and limit to 5 max
            paper_findings = [f for f in paper_findings if len(f) > 20 and '.' in f][:5]
            
            if paper_findings:
                findings[paper_id] = paper_findings
                
        # Next, find interrelationships between papers
        synthesis = self.synthesize_findings_with_domain(
            findings,
            research_question,
            domain_context=" ".join(domain_terms),
            query=query
        )
        
        # Identify papers suitable for citation (those with findings)
        findings_with_metadata = {}
        
        # Add metadata for papers with findings
        for i, paper in enumerate(filtered_papers):
            paper_id = f"paper_{i}"
            if paper_id in findings:
                # Clean journal/venue name
                venue = paper.venue if paper.venue and str(paper.venue).lower() != "none" else ""
                
                # Process volume and issue - Paper class doesn't have these attributes
                # So set them as empty strings by default
                volume = ""
                issue = ""
                
                # Add to citation-ready papers
                findings_with_metadata[paper_id] = {
                    "id": paper_id,
                    "title": paper.title,
                    "authors": paper.authors,
                    "year": paper.year if paper.year else "2025",
                    "venue": venue,
                    "volume": volume,
                    "issue": issue
                }
        
        # Gather papers with findings to be used for citation
        papers_for_citation = []
        relevant_papers = []
        
        for i, paper in enumerate(filtered_papers):
            paper_id = f"paper_{i}"
            if paper_id in findings_with_metadata:
                papers_for_citation.append(findings_with_metadata[paper_id])
                relevant_papers.append(paper)
        
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
           - Each citation must be UNIQUE - never repeat the same citation multiple times
           - Number citations sequentially as [1], [2], etc. with each paper having exactly ONE citation entry
           - Use the ACTUAL information from the papers, not placeholder text like "A. Author"
           - IMPORTANT: Format the authors' names correctly (Last name, First initial.)
           - MOST IMPORTANT: Keep the numbering consistent between in-text citations and the reference list
        
        Provide your response in two parts:
        1. Final Summary (with in-text citations using paper IDs)
        2. Citations (in IEEE format, with each citation appearing exactly once)
        
        MOST IMPORTANT INSTRUCTION: The final summary MUST focus specifically on addressing the original research question about {query or research_question}.
        Ensure all papers are cited at least once in the summary.
        
        ADDITIONAL IMPORTANT INSTRUCTIONS:
        - In your final summary, when you refer to papers, use numerical references that match the citation list (e.g., [1], [2], etc.)
        - Make sure that if a paper is numbered [1] in your "Part 2: Citations" list, it's also numbered [1] in your "Part 1: Final Summary"
        - Keep your citation numbering consistent throughout both sections
        - AVOID long series of references like [1, 2, 3, 4, 5] - group only when critically necessary
        - For each citation in Part 2, include the COMPLETE paper information based on what's provided
        - If the papers don't directly address the research question, you MUST relate their findings to the research question by drawing appropriate connections
        - The final summary must be a direct answer to the original research question, even if that requires interpreting how the paper findings relate to the question
        - Make sure to begin your summary with a clear statement addressing the original research question
        - If the papers focus on a different domain (e.g., biomedical) than the research question (e.g., architecture), explicitly discuss how principles from one domain can be transferred to the other
        - Acknowledge any limitations in addressing the research question based on the available papers
        
        MOST IMPORTANT: When creating the citations section, use the ACTUAL paper information:
        - For paper_0, use the title "{papers_for_citation[0]['title'] if len(papers_for_citation) > 0 else 'No title'}" by {', '.join(papers_for_citation[0]['authors']) if len(papers_for_citation) > 0 else 'No authors'} ({papers_for_citation[0]['year'] if len(papers_for_citation) > 0 else '2023'})
        - For paper_1, use the title "{papers_for_citation[1]['title'] if len(papers_for_citation) > 1 else 'No title'}" by {', '.join(papers_for_citation[1]['authors']) if len(papers_for_citation) > 1 else 'No authors'} ({papers_for_citation[1]['year'] if len(papers_for_citation) > 1 else '2023'})
        - For paper_2, use the title "{papers_for_citation[2]['title'] if len(papers_for_citation) > 2 else 'No title'}" by {', '.join(papers_for_citation[2]['authors']) if len(papers_for_citation) > 2 else 'No authors'} ({papers_for_citation[2]['year'] if len(papers_for_citation) > 2 else '2023'})
        - For paper_3, use the title "{papers_for_citation[3]['title'] if len(papers_for_citation) > 3 else 'No title'}" by {', '.join(papers_for_citation[3]['authors']) if len(papers_for_citation) > 3 else 'No authors'} ({papers_for_citation[3]['year'] if len(papers_for_citation) > 3 else '2023'})
        """
        
        response = self.ollama.generate(prompt)
        
        # Map all paper IDs to citation numbers
        paper_ids = list(findings_with_metadata.keys())
        citation_mapping = {paper_id: i+1 for i, paper_id in enumerate(paper_ids)}
        
        # Replace paper IDs with citation numbers consistently
        final_text = response
        for paper_id, citation_number in citation_mapping.items():
            final_text = re.sub(f"\\[{paper_id}\\]", f"[{citation_number}]", final_text)
        
        # Parse the response to separate summary from citations
        parts = final_text.split("Citations:", 1)
        if len(parts) < 2:
            parts = final_text.split("Part 2:", 1)
        
        final_summary = parts[0].strip()
        citations = []
        
        if len(parts) > 1:
            # Process citation part to ensure uniqueness and correct numbering
            raw_citations = [
                cite.strip()
                for cite in parts[1].split("\n")
                if cite.strip() and not cite.strip().startswith("Part")
            ]
            
            # Ensure citations are properly numbered and unique
            seen_citations = set()
            numbered_citations = {}
            
            for citation in raw_citations:
                # Extract citation number if present
                num_match = re.match(r'^\[?(\d+)\]?\.?\s+', citation)
                if num_match:
                    num = int(num_match.group(1))
                    citation_text = re.sub(r'^\[?(\d+)\]?\.?\s+', '', citation).strip()
                    
                    # Skip if we've already seen this citation
                    if citation_text in seen_citations:
                        continue
                    
                    seen_citations.add(citation_text)
                    numbered_citations[num] = citation_text
                else:
                    # For unnumbered citations, add them with a high number to process later
                    citation_text = citation.strip()
                    if citation_text not in seen_citations:
                        seen_citations.add(citation_text)
                        numbered_citations[100 + len(numbered_citations)] = citation_text
            
            # Create final numbered citations in order
            for i, paper_id in enumerate(paper_ids):
                num = i + 1
                if num in numbered_citations:
                    citations.append(f"{num}. {numbered_citations[num]}")
                elif i < len(papers_for_citation):
                    # Generate citation from paper data if missing
                    paper_data = papers_for_citation[i]
                    citations.append(self._generate_citation_from_paper(num, paper_data))
        
        # If citations still contain placeholder text or are empty, generate proper citations from papers data
        if len(citations) == 0 or any("A. Author" in cite for cite in citations):
            # Generate proper IEEE citations from paper data
            citations = []
            for i, paper_id in enumerate(paper_ids):
                if i < len(papers_for_citation):
                    paper_data = papers_for_citation[i]
                    citations.append(self._generate_citation_from_paper(i+1, paper_data))
        
        return ResearchResponse(
            initial_response=initial_response,
            papers=relevant_papers,
            final_summary=final_summary,
            citations=citations
        )
    
    def _generate_citation_from_paper(self, number: int, paper_data: Dict[str, Any]) -> str:
        """
        Generate a properly formatted citation from paper data.
        
        Parameters:
        -----------
        number: Citation number to use
        paper_data: Dictionary containing paper metadata
            
        Returns:
        --------
        str: Formatted citation
        """
        authors = paper_data.get("authors", [])
        title = paper_data.get("title", "")
        venue = paper_data.get("venue", "")
        year = paper_data.get("year", "2023")
        
        # Format author names according to IEEE style
        if len(authors) == 1:
            # Handle single author
            parts = authors[0].split()
            if len(parts) > 1:
                author_text = f"{parts[-1]} {parts[0][0]}."
            else:
                author_text = authors[0]
        elif len(authors) == 2:
            # Handle two authors
            parts1 = authors[0].split()
            parts2 = authors[1].split()
            
            if len(parts1) > 1:
                author1 = f"{parts1[-1]} {parts1[0][0]}."
            else:
                author1 = authors[0]
                
            if len(parts2) > 1:
                author2 = f"{parts2[-1]} {parts2[0][0]}."
            else:
                author2 = authors[1]
                
            author_text = f"{author1} and {author2}"
        else:
            # Handle three or more authors
            parts = authors[0].split()
            if len(parts) > 1:
                author_text = f"{parts[-1]} {parts[0][0]}. et al."
            else:
                author_text = f"{authors[0]} et al."
        
        # Create citation string
        if venue:
            citation = f"{number}. {author_text}, \"{title},\" {venue}, {year}."
        else:
            citation = f"{number}. {author_text}, \"{title},\" {year}."
        
        return citation
    
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
        strategy based on what they found in initial papers, while ensuring
        alignment with the original research question.
        
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
        
        # First, extract key terms from the research question to maintain focus
        research_terms_prompt = f"""
        Extract 5-7 key technical terms that are CENTRAL to this research question:
        
        {research_question}
        
        Return ONLY a comma-separated list of the most important technical terms and concepts.
        These terms will be used to ensure search queries stay on topic.
        """
        
        research_terms_response = self.ollama.generate(research_terms_prompt)
        research_terms = [term.strip() for term in research_terms_response.split(',') if term.strip()]
        
        # Ensure we have at least some research terms
        if not research_terms:
            research_terms = [term for term in research_question.split() 
                             if len(term) > 4 and term.lower() not in ['what', 'where', 'when', 'which', 'there', 'their', 'about', 'would', 'could', 'should', 'utilize', 'utilized']]
        
        # Extract specialized terminology from the papers
        specialized_terms = self.extract_specialized_terminology(initial_papers)
        
        # Identify potentially relevant methods, materials and concepts from the papers
        papers_info = "\n\n".join([
            f"Paper {i+1}:\nTitle: {paper.title}\nAbstract: {paper.abstract}"
            for i, paper in enumerate(initial_papers[:5])  # Limit to first 5 papers for manageability
        ])
        
        # Validate potential terms against the original research question
        validation_prompt = f"""
        The original research question is: "{research_question}"
        
        The key terms from this research question are: {', '.join(research_terms)}
        
        Based on initial papers, I've extracted these specialized terms: {', '.join(specialized_terms)}
        
        Please identify which of these specialized terms are DIRECTLY RELEVANT to the original research question.
        For each term, respond with YES or NO followed by brief justification.
        
        Format: 
        Term: [term] | Relevant: [YES/NO] | Reason: [brief explanation]
        
        Be strict - only mark terms as relevant if they directly relate to the original research question.
        """
        
        validation_response = self.ollama.generate(validation_prompt)
        
        # Extract validated terms
        validated_terms = []
        for line in validation_response.split('\n'):
            if 'Relevant: YES' in line or 'Relevant: yes' in line:
                term_match = re.search(r'Term:\s*([^|]+)', line)
                if term_match:
                    validated_term = term_match.group(1).strip()
                    if validated_term:
                        validated_terms.append(validated_term)
        
        # Combine validated terms with research terms, ensuring original focus is maintained
        combined_terms = research_terms.copy()
        for term in validated_terms:
            if term not in combined_terms:
                combined_terms.append(term)
        
        # If we didn't get enough validated terms, use the central research terms
        if len(combined_terms) < 5:
            # Extract core concepts from research question
            core_concepts_prompt = f"""
            Extract 3-5 CORE CONCEPTS from this research question:
            
            {research_question}
            
            Format as a comma-separated list.
            """
            core_concepts_response = self.ollama.generate(core_concepts_prompt)
            core_concepts = [concept.strip() for concept in core_concepts_response.split(',') if concept.strip()]
            
            # Add unique core concepts
            for concept in core_concepts:
                if concept not in combined_terms:
                    combined_terms.append(concept)
        
        # Now generate search queries using these combined terms, always ensuring original research focus
        query_prompt = f"""
        Create 3 search queries for academic databases based on this research question:
        
        {research_question}
        
        Use these terms to create focused queries:
        {', '.join(combined_terms)}
        
        Each query MUST:
        1. STAY FOCUSED on the original research question
        2. Use proper Boolean operators (AND, OR) with parentheses
        3. Include 2-4 terms combined with appropriate operators
        4. Include a year range filter [2014 TO 2024]
        5. Be specific enough to return relevant papers
        
        Format each query on a new line, like this:
        1. (term1 AND term2) AND (term3 OR term4) AND year:[2014 TO 2024]
        2. (term5 AND term6) AND (term7 OR term8) AND year:[2014 TO 2024]
        3. (term9 AND term10) AND (term11 OR term12) AND year:[2014 TO 2024]
        
        IMPORTANT: EACH query MUST include at least 2 terms from the original research question to ensure relevance.
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
        
        # Sanity check: Ensure queries relate to the original research question
        # Add initial "safety" query directly derived from research question
        safety_query = self._create_safety_query(research_question, research_terms)
        if safety_query and safety_query not in refined_queries:
            refined_queries.insert(0, safety_query)
        
        # Ensure we have at least one query
        if not refined_queries and research_question:
            # Create a basic query using key terms from the research question
            main_terms = research_terms[:3] if research_terms else [term for term in research_question.split() if len(term) > 4]
            if len(main_terms) >= 2:
                refined_queries.append(f"({main_terms[0]} AND {main_terms[1]}) AND year:[2014 TO 2024]")
        
        # Limit to 3 queries
        return refined_queries[:3]

    def _create_safety_query(self, research_question: str, research_terms: List[str]) -> str:
        """
        Create a safety query directly from the research question to prevent topic drift.
        
        Parameters:
        -----------
        research_question: The original research question
        research_terms: Key terms extracted from the research question
        
        Returns:
        --------
        str: A search query directly based on the original research question
        """
        # Extract main topics from the research question
        main_topics_prompt = f"""
        Extract the 2-3 MAIN TOPICS addressed in this research question:
        
        {research_question}
        
        Return ONLY a comma-separated list of the main topics (e.g., "artificial intelligence, ethics, healthcare").
        """
        
        main_topics_response = self.ollama.generate(main_topics_prompt)
        main_topics = [topic.strip() for topic in main_topics_response.split(',') if topic.strip()]
        
        # If we have main topics, create a query from them
        if main_topics and len(main_topics) >= 2:
            return f"({main_topics[0]} AND {main_topics[1]}) AND year:[2014 TO 2024]"
        
        # Otherwise, use research terms
        if research_terms and len(research_terms) >= 2:
            return f"({research_terms[0]} AND {research_terms[1]}) AND year:[2014 TO 2024]"
        
        # Last resort: extract most important words from the question
        words = [w.lower() for w in research_question.split() if len(w) > 4 and w.lower() not in ['what', 'where', 'when', 'which', 'there', 'their', 'about', 'would', 'could', 'should', 'utilize', 'utilized']]
        if len(words) >= 2:
            return f"({words[0]} AND {words[1]}) AND year:[2014 TO 2024]"
        
        return ""
    
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
        Follow citation trails to expand literature search while ensuring relevance to the original research.
        
        Parameters:
        -----------
        initial_papers: List of papers from the initial search
        research_question: The research question
        literature_manager: LiteratureManager instance for searching
        
        Returns:
        --------
        list: List of papers found through citation trails
        """
        # Extract key terms from research question for validation
        research_terms_prompt = f"""
        Extract 3-5 key technical terms from this research question:
        
        {research_question}
        
        Return ONLY a comma-separated list of the most important technical terms.
        """
        
        research_terms_response = self.ollama.generate(research_terms_prompt)
        research_terms = [term.strip() for term in research_terms_response.split(',') if term.strip()]
        
        # Identify potential citation leads from initial papers
        citation_queries = self.identify_citation_leads(initial_papers)
        
        # Validate citation queries against research question
        validated_queries = []
        for query in citation_queries:
            # Create a validation prompt
            validation_prompt = f"""
            Determine if this search query is DIRECTLY RELEVANT to the original research question.
            
            Research Question: {research_question}
            Key Terms: {', '.join(research_terms)}
            
            Search Query: {query}
            
            Answer with ONLY "YES" if relevant or "NO" if not relevant.
            A query is relevant if it contains terms that directly relate to the research question.
            """
            
            validation_response = self.ollama.generate(validation_prompt).strip().upper()
            if "YES" in validation_response:
                validated_queries.append(query)
        
        # If no validated queries, create basic ones from research terms
        if not validated_queries and research_terms:
            if len(research_terms) >= 2:
                validated_queries.append(f"({research_terms[0]} AND {research_terms[1]}) AND year:[1990 TO 2022]")
            if len(research_terms) >= 3:
                validated_queries.append(f"({research_terms[0]} AND {research_terms[2]}) AND year:[1990 TO 2022]")
        
        # Use original queries as fallback if validation removed all of them
        if not validated_queries:
            validated_queries = citation_queries[:2]  # Limit to 2 to reduce off-topic results
        
        if not validated_queries:
            return []
        
        # Search for papers based on validated citation leads
        citation_papers = []
        
        print(f"\nFollowing citation trails with {len(validated_queries)} queries...")
        for i, query in enumerate(validated_queries):
            print(f"Citation trail search {i+1}/{len(validated_queries)}: '{query}'")
            
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
                if i < len(validated_queries) - 1:
                    import time
                    time.sleep(2)  # 2-second delay between citation searches
                    
            except Exception as e:
                print(f"Error in citation trail search: {e}")
        
        # Screen citation papers using both title and abstract
        title_screened = self.screen_papers_by_title(citation_papers)
        print(f"Title screening for citation papers: {len(title_screened)}/{len(citation_papers)} papers passed")
        
        abstract_screened = self.screen_papers_by_abstract(title_screened, research_question)
        print(f"Abstract screening for citation papers: {len(abstract_screened)}/{len(title_screened)} papers passed")
        
        return abstract_screened
    
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
        Synthesize research findings with critical analysis.
        
        Parameters:
        -----------
        papers: List of relevant papers
        research_question: The research question
        
        Returns:
        --------
        str: Critical synthesis
        """
        # If no papers found, return a message about lack of evidence
        if not papers:
            return "No relevant papers were found to address the research question."
        
        # Filter papers to ensure they match the time frame if specified in the research question
        time_frame_match = re.search(r'(\d{4})[-–](\d{4})', research_question)
        filtered_papers = papers
        
        if time_frame_match:
            start_year, end_year = map(int, time_frame_match.groups())
            # Filter papers by publication year
            filtered_papers = [
                paper for paper in papers 
                if paper.year and int(paper.year) >= start_year and int(paper.year) <= end_year
            ]
            
            # If filtering removed all papers, use original set but note the discrepancy
            if not filtered_papers:
                filtered_papers = papers
        
        # Extract paper info for the prompt
        paper_info = []
        for i, paper in enumerate(filtered_papers):
            authors_str = ", ".join(paper.authors[:3])
            if len(paper.authors) > 3:
                authors_str += " et al."
                
            paper_info.append(
                f"Paper_{i}:\n"
                f"Title: {paper.title}\n"
                f"Authors: {authors_str}\n"
                f"Year: {paper.year or 'Unknown'}\n"
                f"Abstract: {paper.abstract}\n"
            )
            
        papers_text = "\n\n".join(paper_info)
        
        # Ensure synthesis focuses on the specific research question
        prompt = f"""
        Research Question: {research_question}
        
        Conduct a comprehensive critical synthesis of the following papers, addressing the research question directly.
        
        {papers_text}
        
        Structure your synthesis as follows:
        
        **1. Overview of the Literature:**
        Summarize the current state of research on this topic based on the available papers. Assess how well the collection of papers addresses the research question. Identify any gaps or limitations in the literature.
        
        **2. Synthesis of Key Findings:**
        For each paper, extract and synthesize the most relevant findings that address the research question. Use the paper ID (e.g., Paper_0) when referencing specific papers.
        
        **3. Analysis of Conflicts and Agreements:**
        Identify where the papers agree or disagree on key points. Analyze the potential reasons for any conflicts or inconsistencies in the literature.
        
        **4. Limitations of Current Evidence:**
        Critically assess the quality, validity, and generalizability of the evidence presented across these papers. What methodological limitations might affect the reliability of findings?
        
        **5. Conclusions and Implications:**
        Based on this analysis, provide conclusions that directly address the research question. Discuss implications for theory, practice, and future research.
        
        **Additional Relevant Research**
        In one paragraph, mention any highly relevant research areas or potential papers that would be valuable to consult but aren't included in the analyzed set.
        
        **Research Question Alignment**
        In one final paragraph, critically evaluate how well this synthesis addresses the original research question. If there are gaps or misalignments, explicitly note what additional information would be needed to fully address the question.
        
        IMPORTANT:
        1. Maintain strict focus on the research question: "{research_question}"
        2. Be critical and analytical, not just descriptive
        3. When using Paper_X references, ensure they accurately reflect the content of each paper
        4. Do NOT fabricate or exaggerate findings
        5. Do NOT add citations using [X] format - use Paper_X format instead
        6. Acknowledge when evidence is insufficient to draw strong conclusions
        7. Ensure your analysis is DIRECTLY relevant to the exact topic asked about
        8. If the papers don't fully address the research question, explicitly acknowledge this limitation
        """
        
        # Generate the synthesis
        synthesis = self.ollama.generate(prompt)
        
        return synthesis