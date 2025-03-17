"""
Literature search manager using Tavily API.

This module implements a manager for academic literature searches
using the Tavily API. It provides a centralized interface for
searching, tracking, and deduplicating academic papers.
"""

import os  # For accessing environment variables
from typing import List, Dict, Any, Optional, Set  # For type hints
from .tavily import TavilyAPI  # API client for Tavily search engine
from .paper import Paper  # Paper data model

class LiteratureManager:
    """
    Manages literature search using Tavily API.
    
    This class acts as a facade for performing academic literature searches,
    handling tasks such as:
    1. Interfacing with the Tavily API for paper retrieval
    2. Deduplicating papers across multiple searches
    3. Tracking search history
    4. Providing usage statistics
    """
    
    def __init__(self):
        """
        Initialize the literature manager.
        
        Sets up:
        1. The Tavily API client with the API key from environment variables
        2. A set to track previously seen papers (for deduplication)
        3. A list to track search history
        """
        # Get API key from environment variables (should be set in .env file)
        api_key = os.getenv('TAVILY_API_KEY')
        # Initialize the Tavily API client
        self.tavily = TavilyAPI(api_key=api_key)
        # Set to track papers we've seen (using paper hashes)
        # This prevents duplicates across multiple searches
        self.seen_papers: Set[str] = set()  # Track paper hashes
        # List to store history of all searches performed
        self.search_history: List[Dict] = []
    
    def search(
        self,
        query: str,
        max_papers: int = 10,
        year_range: Optional[tuple] = None
    ) -> List[Paper]:
        """
        Search for academic papers using Tavily API.
        
        This method:
        1. Calls the Tavily API with the specified query parameters
        2. Deduplicates results against previously seen papers
        3. Updates search history for tracking
        4. Handles errors gracefully
        
        Parameters:
        -----------
        query: The search query string for finding relevant papers
        max_papers: Maximum number of papers to return (default: 10)
        year_range: Optional tuple of (start_year, end_year) for filtering papers by publication date
            
        Returns:
        --------
        list: List of unique Paper objects matching the query
              Empty list if search fails or no results found
        """
        try:
            # Perform the search using Tavily API
            papers = self.tavily.search(
                query=query,
                limit=max_papers,
                year_range=year_range
            )
            
            # Remove duplicate papers that we've seen before
            # This is important when performing multiple related searches
            unique_papers = self._remove_duplicates(papers)
            
            # Track this search in our history
            # This helps with generating statistics and debugging
            self.search_history.append({
                "query": query,
                "total_results": len(unique_papers),
                "api_used": "tavily"
            })
            
            return unique_papers
            
        except Exception as e:
            # Handle any errors that occur during the search
            # This ensures a failed search doesn't crash the application
            print(f"Search failed: {e}")
            return []  # Return empty list on failure
    
    def _remove_duplicates(self, papers: List[Paper]) -> List[Paper]:
        """
        Remove duplicate papers based on paper hash.
        
        This method:
        1. Checks each paper against previously seen papers
        2. Adds new papers to the tracking set
        3. Returns only papers that haven't been seen before
        
        Deduplication is important because:
        - Different queries may return the same papers
        - It prevents information overload for the user
        - It ensures more diverse results overall
        
        Parameters:
        -----------
        papers: List of papers to deduplicate
        
        Returns:
        --------
        list: List containing only unique papers not seen in previous searches
        """
        unique_papers = []
        for paper in papers:
            # Get paper hash based on title and authors
            paper_hash = paper.get_hash()
            # Check if we've seen this paper before
            if paper_hash not in self.seen_papers:
                # If not, add it to our tracking set
                self.seen_papers.add(paper_hash)
                # And include it in the results
                unique_papers.append(paper)
        return unique_papers
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the searches performed.
        
        This method provides insight into the search activity,
        including total searches and unique papers found.
        
        Returns:
        --------
        dict: Dictionary containing search statistics:
            - total_searches: Number of searches performed
            - total_unique_papers: Number of unique papers found
            - searches_by_api: Breakdown of searches by API used
        """
        return {
            "total_searches": len(self.search_history),  # Total number of searches performed
            "total_unique_papers": len(self.seen_papers),  # Total unique papers found
            "searches_by_api": {
                "tavily": len(self.search_history)  # All searches currently use Tavily
            }
        } 