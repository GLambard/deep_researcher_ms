"""
Common data models for the scholarly literature aggregator.

This module defines the core data structures used throughout the application.
These models provide a unified representation of scholarly literature data
regardless of the source API (Semantic Scholar, arXiv, etc.).
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Union, Any


@dataclass
class Author:
    """
    Represents an author of a scholarly paper.
    
    This class stores information about paper authors that is common across
    different academic APIs, allowing for normalized representation.
    
    Attributes:
        name: The author's full name.
        id: Unique identifier for the author (if available from the source).
        affiliations: List of institutions the author is affiliated with.
        url: URL to the author's profile page (if available).
    """
    name: str
    id: Optional[str] = None
    affiliations: List[str] = field(default_factory=list)
    url: Optional[str] = None


@dataclass
class PaperMetadata:
    """
    Common data model for paper metadata across different sources.
    
    This class serves as the unified representation for scholarly papers,
    normalizing data from various sources (Semantic Scholar, arXiv, etc.)
    into a consistent format.
    
    Attributes:
        title: Paper title.
        source: Origin of the data (e.g., "Semantic Scholar", "arXiv").
        authors: List of paper authors.
        abstract: Paper abstract or summary text.
        doi: Digital Object Identifier (DOI).
        url: URL to the paper's landing page.
        publication_date: Date when the paper was published.
        pdf_url: Direct URL to download the PDF (if available).
        year: Publication year (separate from date for easier querying).
        venue: Journal or conference where the paper was published.
        citation_count: Number of citations the paper has received.
        fields_of_study: Research areas associated with the paper.
        publication_types: Type of publication (journal, conference, etc.).
        external_ids: Dictionary mapping ID types to values (e.g., "arxiv" -> "1234.5678").
        raw_data: Original API response data for debugging or custom processing.
    """
    title: str
    source: str  # Origin (arXiv, BioRxiv, Semantic Scholar, etc.)
    authors: List[Author] = field(default_factory=list)
    abstract: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    publication_date: Optional[datetime] = None
    pdf_url: Optional[str] = None
    # Additional optional fields
    year: Optional[int] = None
    venue: Optional[str] = None
    citation_count: Optional[int] = None
    fields_of_study: List[str] = field(default_factory=list)
    publication_types: List[str] = field(default_factory=list)
    # Source-specific IDs (arxiv_id, s2_id, etc.)
    external_ids: Dict[str, str] = field(default_factory=dict)
    # Raw response data for debugging or custom processing
    raw_data: Optional[Dict] = None


@dataclass
class QueryParams:
    """
    Parameters for querying literature sources.
    
    This class encapsulates all the parameters that can be used to search
    for papers across different APIs. It provides a unified interface for
    filtering and pagination that can be adapted to each specific API's
    requirements.
    
    Attributes:
        keywords: Search terms to find in title, abstract, or full text.
        year: Publication year or year range (e.g., "2020-2023").
        fields_of_study: Filter by research areas (e.g., "Computer Science") separated by commas.
        limit: Maximum number of results to return per page.
        offset: Pagination offset for retrieving subsequent pages.
        open_access_only: Whether to return only papers with open access PDFs.
        sort_by: Criteria for ordering results.
    """
    keywords: Optional[str] = None
    year: Optional[Union[int, str]] = None  # Can be specific year or range like "2020-2023"
    fields_of_study: Optional[str] = None
    limit: int = 100
    offset: int = 0
    open_access_only: bool = False
    sort_by: Optional[str] = None  # e.g., "relevance", "date", "citations"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the object to a dictionary for JSON serialization.
        
        This method is important for caching, as it ensures the QueryParams
        object can be properly serialized to JSON when generating cache keys.
        
        Returns:
            Dictionary representation of the query parameters with datetime
            objects converted to ISO-format strings.
        """
        result = {}
        for key, value in self.__dict__.items():
            # Handle datetime objects
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            else:
                result[key] = value
        return result
    
    def __str__(self) -> str:
        """
        Return a string representation of the object.
        
        Returns a human-readable string showing only the non-None parameters,
        which is useful for debugging and logging.
        """
        return f"QueryParams({', '.join(f'{k}={v}' for k, v in self.to_dict().items() if v is not None)})"


@dataclass
class QueryResult:
    """
    Results of a query to a literature source.
    
    This class encapsulates the papers returned from a search query,
    along with pagination metadata for retrieving subsequent pages.
    
    Attributes:
        papers: List of paper metadata objects.
        total_results: Total number of results matching the query (may exceed the length of papers).
        next_offset: Offset value to use when retrieving the next page.
        next_token: Token-based pagination value for APIs that use it.
        has_next: Boolean indicating if there are more results available.
    """
    papers: List[PaperMetadata] = field(default_factory=list)
    total_results: Optional[int] = None
    next_offset: Optional[int] = None
    next_token: Optional[str] = None  # For APIs that use token-based pagination
    has_next: bool = False 