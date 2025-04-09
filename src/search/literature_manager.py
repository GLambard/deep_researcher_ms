"""
Literature search manager for various academic APIs.

This module implements a manager for academic literature searches
using multiple sources including Tavily API, OpenAlex, arXiv, and ChemRxiv.
It provides a centralized interface for searching, tracking, and 
deduplicating academic papers across different academic databases.
"""

import os  # For accessing environment variables
from typing import List, Dict, Any, Optional, Set, Union, Tuple  # For type hints
import asyncio  # For asynchronous operations
from .tavily import TavilyAPI  # API client for Tavily search engine
from .paper import Paper  # Paper data model

# Import the adapter classes from the adapters package
from adapters.semantic_scholar import SemanticScholarAdapter
from adapters.open_alex import OpenAlexAdapter
from adapters.arxiv import ArXivAdapter
from adapters.chemrxiv import ChemRxivAdapter

# Import the common models
from common.models import QueryParams

class LiteratureManager:
    """
    Manages literature search using multiple API sources.
    
    This class acts as a facade for performing academic literature searches,
    handling tasks such as:
    1. Interfacing with multiple academic APIs for paper retrieval
    2. Deduplicating papers across multiple searches
    3. Tracking search history
    4. Providing usage statistics
    """
    
    def __init__(self):
        """
        Initialize the literature manager.
        
        Sets up:
        1. All API clients with their API keys from environment variables
        2. A set to track previously seen papers (for deduplication)
        3. A list to track search history
        """
        # Get API keys from environment variables
        tavily_api_key = os.getenv('TAVILY_API_KEY')
        # semantic_scholar_api_key = os.getenv('SEMANTIC_SCHOLAR_API_KEY')
        
        # Initialize the API clients
        self.tavily = TavilyAPI(api_key=tavily_api_key)
        
        # Initialize the adapter classes
        # self.semantic_scholar = SemanticScholarAdapter(api_key=semantic_scholar_api_key)
        self.open_alex = OpenAlexAdapter(email=os.getenv('OPEN_ALEX_EMAIL', ''))
        self.arxiv = ArXivAdapter()
        self.chemrxiv = ChemRxivAdapter()
        
        # Set to track papers we've seen (using paper hashes)
        # This prevents duplicates across multiple searches
        self.seen_papers: Set[str] = set()  # Track paper hashes
        
        # List to store history of all searches performed
        self.search_history: List[Dict] = []
    
    def reset_tracking(self):
        """
        Reset the paper tracking and search history.
        
        This method clears the internal tracking of seen papers and search history,
        which is useful when starting a completely new research query to avoid
        any influence from previous searches.
        """
        # Clear the set of seen papers
        self.seen_papers.clear()
        
        # Optionally, also clear search history if desired
        self.search_history.clear()
    
    def search(
        self,
        query: str,
        max_papers: int = 10,
        year_range: Optional[Tuple[int, int]] = None,
        sources: Optional[List[str]] = None
    ) -> List[Paper]:
        """
        Search for papers across selected API sources.
        
        This is a synchronous wrapper around the async search method,
        making it easier to use in non-async contexts.
        
        Parameters:
        -----------
        query: The search query string
        max_papers: Maximum number of papers to return per source
        year_range: Optional tuple of (start_year, end_year)
        sources: Optional list of source APIs to use
            
        Returns:
        --------
        list: List of Paper objects
        """
        # Create an event loop for async execution
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run the async search method in the loop
            return loop.run_until_complete(
                self.async_search(
                    query=query, 
                    max_papers=max_papers,
                    year_range=year_range,
                    sources=sources
                )
            )
        finally:
            # Clean up the loop
            loop.close()

    async def async_search(
        self,
        query: str,
        max_papers: int = 10,
        year_range: Optional[Tuple[int, int]] = None,
        sources: Optional[List[str]] = None
    ) -> List[Paper]:
        """
        Search for academic papers using all available sources.
        
        This method:
        1. Calls multiple APIs with the specified query parameters
        2. Combines and deduplicates results from all sources
        3. Updates search history for tracking
        4. Handles errors gracefully
        
        Parameters:
        -----------
        query: The search query string for finding relevant papers
        max_papers: Maximum number of papers to return from each source (default: 10)
        year_range: Optional tuple of (start_year, end_year) for filtering papers by publication date
        sources: Optional list of sources to use (e.g., ["tavily", "semantic_scholar", "arxiv"])
                If None, all available sources will be used
            
        Returns:
        --------
        list: List of unique Paper objects matching the query
              Empty list if search fails or no results found
        """
        try:
            # Map of source keys to search functions
            source_map = {
                #"tavily": self._search_tavily, # TODO: Add this back in but with another API (expensive)
                #"semantic_scholar": self._search_semantic_scholar,
                "open_alex": self._search_open_alex,
                "arxiv": self._search_arxiv,
                "chemrxiv": self._search_chemrxiv
            }
            
            # Determine which sources to use
            if sources is None:
                # Default to all sources
                sources = list(source_map.keys())
            
            # Create a list of tasks for concurrent execution
            tasks = []
            for source in sources:
                if source in source_map:
                    # Add search task to the list
                    search_func = source_map[source]
                    tasks.append(search_func(query, max_papers, year_range))
            
            # Execute all search tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results, excluding any that resulted in exceptions
            all_papers = []
            for i, result in enumerate(results):
                # Skip exceptions
                if isinstance(result, Exception):
                    print(f"Search error in {sources[i]}: {result}")
                    continue
                
                # Add papers to our combined list
                all_papers.extend(result)
            
            # Remove duplicate papers that we've seen before
            unique_papers = self._remove_duplicates(all_papers)
            
            # Track this search in our history
            self.search_history.append({
                "query": query,
                "total_results": len(unique_papers),
                "sources_used": sources
            })
            
            return unique_papers
            
        except Exception as e:
            # Handle any errors that occur during the search
            print(f"Search failed: {e}")
            return []  # Return empty list on failure
    
    async def _search_tavily(
        self, 
        query: str, 
        max_papers: int, 
        year_range: Optional[Tuple[int, int]]
    ) -> List[Paper]:
        """
        Search for papers using the Tavily API.
        
        Parameters:
        -----------
        query: The search query string
        max_papers: Maximum number of papers to return
        year_range: Optional tuple of (start_year, end_year)
            
        Returns:
        --------
        list: List of Paper objects
        """
        # Call the synchronous Tavily API
        return self.tavily.search(
            query=query,
            limit=max_papers,
            year_range=year_range
        )
    
    async def _search_semantic_scholar(
        self, 
        query: str, 
        max_papers: int, 
        year_range: Optional[Tuple[int, int]]
    ) -> List[Paper]:
        """
        Search for papers using the Semantic Scholar API.
        
        Parameters:
        -----------
        query: The search query string
        max_papers: Maximum number of papers to return
        year_range: Optional tuple of (start_year, end_year)
            
        Returns:
        --------
        list: List of Paper objects
        """
        # Create query parameters for the adapter
        params = QueryParams(
            keywords=query,
            limit=max_papers,
            year=f"{year_range[0]}-{year_range[1]}" if year_range else None
        )
        
        # Search using the adapter
        results = await self.semantic_scholar.search_papers(params)
        
        # Convert from PaperMetadata to Paper objects
        papers = []
        for metadata in results.papers:
            paper = Paper(
                title=metadata.title,
                abstract=metadata.abstract or "",
                authors=[author.name for author in metadata.authors],
                year=metadata.year,
                doi=metadata.doi,
                url=metadata.url,
                source_api="semantic_scholar"
            )
            papers.append(paper)
            
        return papers
    
    async def _search_open_alex(
        self, 
        query: str, 
        max_papers: int, 
        year_range: Optional[Tuple[int, int]]
    ) -> List[Paper]:
        """
        Search for papers using the OpenAlex API.
        
        Parameters:
        -----------
        query: The search query string
        max_papers: Maximum number of papers to return
        year_range: Optional tuple of (start_year, end_year)
            
        Returns:
        --------
        list: List of Paper objects
        """
        # Create query parameters for the adapter
        params = QueryParams(
            keywords=query,
            limit=max_papers,
            year=f"{year_range[0]}-{year_range[1]}" if year_range else None
        )
        
        # Search using the adapter
        results = await self.open_alex.search_papers(params)
        
        # Convert from PaperMetadata to Paper objects
        papers = []
        for metadata in results.papers:
            paper = Paper(
                title=metadata.title,
                abstract=metadata.abstract or "",
                authors=[author.name for author in metadata.authors],
                year=metadata.year,
                doi=metadata.doi,
                url=metadata.url,
                source_api="open_alex"
            )
            papers.append(paper)
            
        return papers
    
    async def _search_arxiv(
        self, 
        query: str, 
        max_papers: int, 
        year_range: Optional[Tuple[int, int]]
    ) -> List[Paper]:
        """
        Search for papers using the arXiv API.
        
        Parameters:
        -----------
        query: The search query string
        max_papers: Maximum number of papers to return
        year_range: Optional tuple of (start_year, end_year)
            
        Returns:
        --------
        list: List of Paper objects
        """
        # Create query parameters for the adapter
        params = QueryParams(
            keywords=query,
            limit=max_papers,
            year=f"{year_range[0]}-{year_range[1]}" if year_range else None
        )
        
        # Search using the adapter
        results = await self.arxiv.search_papers(params)
        
        # Convert from PaperMetadata to Paper objects
        papers = []
        for metadata in results.papers:
            paper = Paper(
                title=metadata.title,
                abstract=metadata.abstract or "",
                authors=[author.name for author in metadata.authors],
                year=metadata.year,
                doi=metadata.doi,
                url=metadata.url,
                source_api="arxiv"
            )
            papers.append(paper)
            
        return papers
    
    async def _search_chemrxiv(
        self, 
        query: str, 
        max_papers: int, 
        year_range: Optional[Tuple[int, int]]
    ) -> List[Paper]:
        """
        Search for papers using the ChemRxiv API.
        
        Parameters:
        -----------
        query: The search query string
        max_papers: Maximum number of papers to return
        year_range: Optional tuple of (start_year, end_year)
            
        Returns:
        --------
        list: List of Paper objects
        """
        # ChemRxiv adapter requires a string query rather than QueryParams
        # So we'll handle it differently
        year_filter = {}
        if year_range:
            year_filter["year"] = year_range[0]  # ChemRxiv only supports single year filtering
            
        # Search using the adapter with the query string and additional parameters
        results = await self.chemrxiv.search_papers(query, page=1, per_page=max_papers, **year_filter)
        
        # Convert from PaperMetadata to Paper objects
        papers = []
        for metadata in results.papers:
            paper = Paper(
                title=metadata.title,
                abstract=metadata.abstract or "",
                authors=[author.name for author in metadata.authors],
                year=metadata.year,
                doi=metadata.doi,
                url=metadata.url,
                source_api="chemrxiv"
            )
            papers.append(paper)
            
        return papers
    
    def _remove_duplicates(self, papers: List[Paper]) -> List[Paper]:
        """
        Remove duplicate papers based on paper hash.
        
        This method:
        1. Checks each paper against previously seen papers
        2. Adds new papers to the tracking set
        3. Returns only papers that haven't been seen before
        
        Deduplication is important because:
        - Different sources may return the same papers
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
        # Count searches by source
        searches_by_api = {}
        for search in self.search_history:
            for source in search["sources_used"]:
                searches_by_api[source] = searches_by_api.get(source, 0) + 1
        
        return {
            "total_searches": len(self.search_history),
            "total_unique_papers": len(self.seen_papers),
            "searches_by_api": searches_by_api
        } 