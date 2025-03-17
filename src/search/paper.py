"""
Paper data model for standardized representation across different APIs.

This module provides a consistent data structure for academic papers
retrieved from different sources (Semantic Scholar, Tavily, etc.).
It ensures that paper data follows a uniform format regardless of origin.
"""

from typing import List, Optional  # For type hints
from dataclasses import dataclass  # For creating immutable data classes
import hashlib  # For generating unique paper hashes

@dataclass
class Paper:
    """
    Standardized paper representation across different APIs.
    
    This class provides a unified data structure for academic papers
    regardless of their source API. It captures essential metadata
    needed for academic research and ensures consistent handling
    throughout the application.
    
    Using a dataclass provides automatic initialization, representation,
    and comparison methods, making the code more maintainable.
    """
    title: str  # Paper title
    abstract: str  # Paper abstract/summary
    authors: List[str]  # List of author names
    year: Optional[int]  # Publication year (if available)
    doi: Optional[str]  # Digital Object Identifier (if available)
    url: Optional[str]  # URL to access the paper
    source_api: str  # Which API provided this paper (e.g., "semantic_scholar", "tavily")
    venue: Optional[str] = None  # Journal or conference name where published
    
    def get_hash(self) -> str:
        """
        Generate a unique hash for the paper based on title and authors.
        
        This method creates a consistent identifier for papers to detect
        and prevent duplicates. The hash combines the lowercase title and
        sorted author names to ensure that minor formatting differences
        don't result in duplicate papers being treated as distinct.
        
        Returns:
        --------
        str: MD5 hexadecimal hash string that uniquely identifies the paper
        """
        # Combine title (converted to lowercase) and sorted author names
        # This approach handles cases where the same paper might be returned
        # from different APIs with slight variations in capitalization or author order
        content = f"{self.title.lower()}{''.join(sorted(self.authors))}".encode('utf-8')
        
        # Generate MD5 hash of the combined content
        # MD5 is sufficient for deduplication purposes in this context
        return hashlib.md5(content).hexdigest() 