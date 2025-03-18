"""
Text processing module for handling paper content.

This module contains utilities for processing academic paper content,
extracting structured information, and preparing text for LLM analysis.
It helps identify key components of papers such as findings, methodology,
and conclusions even when not explicitly labeled in the source.
"""

from typing import List, Dict  # For type hints
import re  # For regular expression pattern matching

class TextProcessor:
    """
    Process and prepare paper content for LLM analysis.
    
    This class handles tasks such as:
    1. Cleaning and normalizing text from various sources
    2. Extracting structured information from unstructured text
    3. Formatting content consistently for LLM consumption
    4. Identifying key sections like findings and methodology
    """
    
    ## TODO: Extend to include main text when available
    def process_papers(self, papers: List[Dict]) -> List[Dict]:
        """
        Process a list of papers to prepare them for LLM analysis.
        
        This method transforms raw paper data into a structured format
        with enhanced information extraction. It performs cleaning,
        extraction of key components, and consistent formatting.
        
        Parameters:
        -----------
        papers: List of paper dictionaries from the search API
            Each paper may have inconsistent fields and formatting
        
        Returns:
        --------
        list: Processed papers with standardized fields and enhanced content
            Contains extracted key findings, methodology, and conclusions
        """
        processed_papers = []
        
        # Process each paper individually
        for paper in papers:
            # Create a standardized structure with enhanced information
            processed_paper = {
                "title": self._clean_text(paper.get("title", "")),  # Clean the title
                "abstract": self._clean_text(paper.get("abstract", "")),  # Clean the abstract
                "year": paper.get("year"),  # Publication year
                "authors": self._format_authors(paper.get("authors", [])),  # Format author list
                "venue": paper.get("venue", ""),  # Publication venue (journal/conference)
                "citation_count": paper.get("citationCount", 0),  # Number of citations
                "url": paper.get("url", ""),  # Link to the paper
                # Extract additional structured information from the abstract
                "key_findings": self._extract_key_findings(paper.get("abstract", "")),
                "methodology": self._extract_methodology(paper.get("abstract", "")),
                "conclusions": self._extract_conclusions(paper.get("abstract", ""))
            }
            processed_papers.append(processed_paper)
        
        return processed_papers
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text content for consistent processing.
        
        This method:
        1. Removes excessive whitespace and normalizes spacing
        2. Strips unwanted special characters while keeping essential punctuation
        3. Ensures text is consistently formatted for further processing
        
        Parameters:
        -----------
        text: Input text to clean, which may contain formatting issues
        
        Returns:
        --------
        str: Cleaned and normalized text ready for analysis
        """
        if not text:
            return ""  # Return empty string for None or empty input
        
        # Remove extra whitespace (multiple spaces, tabs, newlines)
        # This creates consistent spacing throughout the text
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        # This preserves the meaning while eliminating problematic characters
        text = re.sub(r'[^\w\s.,;:?!-]', '', text)
        
        # Normalize whitespace at beginning and end
        text = text.strip()
        
        return text
    
    ## TODO: Check if authors formatting is always possible and correct
    def _format_authors(self, authors: List[Dict]) -> str:
        """
        Format author list into a readable string.
        
        This method handles different author list formats and
        creates a consistent author string for citation purposes.
        
        Parameters:
        -----------
        authors: List of author dictionaries, which may have
                different structures based on the source API
        
        Returns:
        --------
        str: Comma-separated list of author names or "Unknown" if none found
        """
        if not authors:
            return "Unknown"  # Default when author information is missing
        
        # Extract author names from potentially different formats
        author_names = []
        for author in authors:
            # Get name field, which might be structured differently based on API
            name = author.get("name", "")
            if name:  # Only add non-empty names
                author_names.append(name)
        
        # Join all authors with comma separator
        return ", ".join(author_names) if author_names else "Unknown"
    
    ## TODO: Check if key findings extraction is always possible and correct
    ## TODO: Check if key findings extraction is better done with LLM
    def _extract_key_findings(self, abstract: str) -> str:
        """
        Extract key findings from the abstract using pattern matching.
        
        This method uses regex patterns to identify sentences that likely
        describe research findings, even when not explicitly labeled.
        The patterns look for common phrases and structures used when
        presenting findings in academic abstracts.
        
        Parameters:
        -----------
        abstract: Paper abstract text to analyze
        
        Returns:
        --------
        str: Extracted text describing key findings or a default message
             if no findings could be identified
        """
        # Look for common phrases indicating key findings
        # These patterns capture sentences containing words that typically
        # introduce or describe research findings
        findings_patterns = [
            # Verbs that typically introduce findings
            r"(?:found|discovered|identified|demonstrated|showed|revealed|established).*?[.!?]",
            # Nouns that typically label findings
            r"(?:results|findings|conclusion).*?[.!?]",
            # Adjectives that typically qualify findings
            r"(?:key|main|primary|major).*?[.!?]"
        ]
        
        # Collect all matching sentences
        findings = []
        for pattern in findings_patterns:
            matches = re.finditer(pattern, abstract, re.IGNORECASE)
            findings.extend(match.group(0) for match in matches)
        
        # Return joined findings or default message
        return " ".join(findings) if findings else "No key findings explicitly stated."
    
    ## TODO: Check if methodology extraction is always possible and correct
    ## TODO: Check if methodology extraction is better done with LLM
    def _extract_methodology(self, abstract: str) -> str:
        """
        Extract methodology information from the abstract.
        
        This method identifies sentences that describe research methods,
        techniques, or approaches used in the study. It captures text
        that explains how the research was conducted.
        
        Parameters:
        -----------
        abstract: Paper abstract text to analyze
        
        Returns:
        --------
        str: Extracted text describing methodology or a default message
             if no methodology information could be identified
        """
        # Look for common phrases indicating methodology
        # These patterns capture sentences that typically describe methods
        method_patterns = [
            # Nouns that typically label methods
            r"(?:method|approach|technique|procedure|protocol).*?[.!?]",
            # Verbs that typically describe method application
            r"(?:using|employed|utilized|applied).*?[.!?]",
            # Verbs that typically describe method creation
            r"(?:developed|designed|implemented).*?[.!?]"
        ]
        
        # Collect all matching sentences
        methods = []
        for pattern in method_patterns:
            matches = re.finditer(pattern, abstract, re.IGNORECASE)
            methods.extend(match.group(0) for match in matches)
        
        # Return joined methods or default message
        return " ".join(methods) if methods else "No methodology explicitly stated."
    
    def _extract_conclusions(self, abstract: str) -> str:
        """
        Extract conclusions from the abstract.
        
        This method identifies sentences that represent the conclusions,
        implications, or final takeaways from the research. It captures
        text that summarizes what the findings mean or suggest.
        
        Parameters:
        -----------
        abstract: Paper abstract text to analyze
        
        Returns:
        --------
        str: Extracted text describing conclusions or a default message
             if no conclusions could be identified
        """
        # Look for common phrases indicating conclusions
        # These patterns capture sentences that typically present conclusions
        conclusion_patterns = [
            # Words that explicitly label conclusions
            r"(?:concluded|conclusion|summary|overall|in summary).*?[.!?]",
            # Words that introduce logical conclusions
            r"(?:therefore|thus|hence|consequently).*?[.!?]",
            # Verbs that typically present implications
            r"(?:suggest|imply|indicate|demonstrate).*?[.!?]"
        ]
        
        # Collect all matching sentences
        conclusions = []
        for pattern in conclusion_patterns:
            matches = re.finditer(pattern, abstract, re.IGNORECASE)
            conclusions.extend(match.group(0) for match in matches)
        
        # Return joined conclusions or default message
        return " ".join(conclusions) if conclusions else "No conclusions explicitly stated." 