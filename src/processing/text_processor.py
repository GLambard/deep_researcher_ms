"""
Text processing module for handling paper content.
"""

from typing import List, Dict
import re

class TextProcessor:
    """
    Process and prepare paper content for LLM analysis.
    """
    
    def process_papers(self, papers: List[Dict]) -> List[Dict]:
        """
        Process a list of papers to prepare them for LLM analysis.
        
        Parameters
        ----------
        papers : list
            List of paper dictionaries from the API
        
        Returns
        -------
        list
            Processed papers with cleaned and formatted content
        """
        processed_papers = []
        
        for paper in papers:
            processed_paper = {
                "title": self._clean_text(paper.get("title", "")),
                "abstract": self._clean_text(paper.get("abstract", "")),
                "year": paper.get("year"),
                "authors": self._format_authors(paper.get("authors", [])),
                "venue": paper.get("venue", ""),
                "citation_count": paper.get("citationCount", 0),
                "url": paper.get("url", ""),
                "key_findings": self._extract_key_findings(paper.get("abstract", "")),
                "methodology": self._extract_methodology(paper.get("abstract", "")),
                "conclusions": self._extract_conclusions(paper.get("abstract", ""))
            }
            processed_papers.append(processed_paper)
        
        return processed_papers
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Parameters
        ----------
        text : str
            Input text to clean
        
        Returns
        -------
        str
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,;:?!-]', '', text)
        
        # Normalize whitespace
        text = text.strip()
        
        return text
    
    def _format_authors(self, authors: List[Dict]) -> str:
        """
        Format author list into a string.
        
        Parameters
        ----------
        authors : list
            List of author dictionaries
        
        Returns
        -------
        str
            Formatted author string
        """
        if not authors:
            return "Unknown"
        
        author_names = []
        for author in authors:
            name = author.get("name", "")
            if name:
                author_names.append(name)
        
        return ", ".join(author_names)
    
    def _extract_key_findings(self, abstract: str) -> str:
        """
        Extract key findings from the abstract.
        
        Parameters
        ----------
        abstract : str
            Paper abstract
        
        Returns
        -------
        str
            Extracted key findings
        """
        # Look for common phrases indicating key findings
        findings_patterns = [
            r"(?:found|discovered|identified|demonstrated|showed|revealed|established).*?[.!?]",
            r"(?:results|findings|conclusion).*?[.!?]",
            r"(?:key|main|primary|major).*?[.!?]"
        ]
        
        findings = []
        for pattern in findings_patterns:
            matches = re.finditer(pattern, abstract, re.IGNORECASE)
            findings.extend(match.group(0) for match in matches)
        
        return " ".join(findings) if findings else "No key findings explicitly stated."
    
    def _extract_methodology(self, abstract: str) -> str:
        """
        Extract methodology information from the abstract.
        
        Parameters
        ----------
        abstract : str
            Paper abstract
        
        Returns
        -------
        str
            Extracted methodology
        """
        # Look for common phrases indicating methodology
        method_patterns = [
            r"(?:method|approach|technique|procedure|protocol).*?[.!?]",
            r"(?:using|employed|utilized|applied).*?[.!?]",
            r"(?:developed|designed|implemented).*?[.!?]"
        ]
        
        methods = []
        for pattern in method_patterns:
            matches = re.finditer(pattern, abstract, re.IGNORECASE)
            methods.extend(match.group(0) for match in matches)
        
        return " ".join(methods) if methods else "No methodology explicitly stated."
    
    def _extract_conclusions(self, abstract: str) -> str:
        """
        Extract conclusions from the abstract.
        
        Parameters
        ----------
        abstract : str
            Paper abstract
        
        Returns
        -------
        str
            Extracted conclusions
        """
        # Look for common phrases indicating conclusions
        conclusion_patterns = [
            r"(?:concluded|conclusion|summary|overall|in summary).*?[.!?]",
            r"(?:therefore|thus|hence|consequently).*?[.!?]",
            r"(?:suggest|imply|indicate|demonstrate).*?[.!?]"
        ]
        
        conclusions = []
        for pattern in conclusion_patterns:
            matches = re.finditer(pattern, abstract, re.IGNORECASE)
            conclusions.extend(match.group(0) for match in matches)
        
        return " ".join(conclusions) if conclusions else "No conclusions explicitly stated." 