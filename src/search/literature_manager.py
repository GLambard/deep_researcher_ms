"""
Literature search manager using Tavily API.
"""

import os
from typing import List, Dict, Any, Optional, Set
from .tavily import TavilyAPI
from .paper import Paper

class LiteratureManager:
    """
    Manages literature search using Tavily API.
    """
    
    def __init__(self):
        """Initialize Tavily API and paper tracking."""
        api_key = os.getenv('TAVILY_API_KEY')
        self.tavily = TavilyAPI(api_key=api_key)
        self.seen_papers: Set[str] = set()  # Track paper hashes
        self.search_history: List[Dict] = []
    
    def search(
        self,
        query: str,
        max_papers: int = 10,
        year_range: Optional[tuple] = None
    ) -> List[Paper]:
        """
        Search for papers using Tavily API.
        
        Parameters
        ----------
        query : str
            Search query
        max_papers : int
            Maximum number of papers to return
        year_range : tuple, optional
            (start_year, end_year) for filtering
            
        Returns
        -------
        list
            List of unique papers
        """
        try:
            papers = self.tavily.search(
                query=query,
                limit=max_papers,
                year_range=year_range
            )
            
            # Remove duplicates
            unique_papers = self._remove_duplicates(papers)
            
            # Update search history
            self.search_history.append({
                "query": query,
                "total_results": len(unique_papers),
                "api_used": "tavily"
            })
            
            return unique_papers
            
        except Exception as e:
            print(f"Search failed: {e}")
            return []
    
    def _remove_duplicates(self, papers: List[Paper]) -> List[Paper]:
        """Remove duplicate papers based on title."""
        unique_papers = []
        for paper in papers:
            paper_hash = paper.get_hash()
            if paper_hash not in self.seen_papers:
                self.seen_papers.add(paper_hash)
                unique_papers.append(paper)
        return unique_papers
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get statistics about the searches performed."""
        return {
            "total_searches": len(self.search_history),
            "total_unique_papers": len(self.seen_papers),
            "searches_by_api": {
                "tavily": len(self.search_history)
            }
        } 