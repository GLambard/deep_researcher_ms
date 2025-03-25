"""
Base adapter class for scholarly literature sources.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from .models import PaperMetadata, QueryParams, QueryResult


class LiteratureAdapter(ABC):
    """
    Abstract base class for all literature source adapters.
    
    This class defines the interface that all adapters must implement
    to work with the aggregator.
    """
    
    @abstractmethod
    async def search_papers(self, params: QueryParams) -> QueryResult:
        """
        Search for papers based on the provided query parameters.
        
        Args:
            params: Query parameters for the search
            
        Returns:
            QueryResult containing matching papers and pagination info
        """
        pass
    
    @abstractmethod
    async def get_paper_by_id(self, paper_id: str) -> Optional[PaperMetadata]:
        """
        Retrieve a paper by its ID in the source system.
        
        Args:
            paper_id: ID of the paper in the source system
            
        Returns:
            Paper metadata if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def get_paper_by_doi(self, doi: str) -> Optional[PaperMetadata]:
        """
        Retrieve a paper by its DOI.
        
        Args:
            doi: DOI of the paper
            
        Returns:
            Paper metadata if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def get_citations(self, paper_id: str, limit: int = 100) -> List[PaperMetadata]:
        """
        Get papers that cite the specified paper.
        
        Args:
            paper_id: ID of the paper in the source system
            limit: Maximum number of citations to retrieve
            
        Returns:
            List of papers citing the specified paper
        """
        pass
    
    @abstractmethod
    async def get_references(self, paper_id: str, limit: int = 100) -> List[PaperMetadata]:
        """
        Get papers referenced by the specified paper.
        
        Args:
            paper_id: ID of the paper in the source system
            limit: Maximum number of references to retrieve
            
        Returns:
            List of papers referenced by the specified paper
        """
        pass
    
    @abstractmethod
    def source_name(self) -> str:
        """
        Get the name of this literature source.
        
        Returns:
            Name of the source (e.g., "Semantic Scholar", "arXiv")
        """
        pass 