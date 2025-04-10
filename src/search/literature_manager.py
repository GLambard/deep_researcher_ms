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
import logging
from datetime import datetime

# Import the adapter classes from the adapters package
from adapters.semantic_scholar import SemanticScholarAdapter
from adapters.open_alex import OpenAlexAdapter
from adapters.arxiv import ArXivAdapter
from adapters.chemrxiv import ChemRxivAdapter

# Import the common models
from common.models import QueryParams, PaperMetadata

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("search.log"),
        logging.StreamHandler()
    ]
)

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
        # Setup logger
        self.logger = logging.getLogger("LiteratureManager")
        
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

    def validate_query(self, query: str) -> Tuple[bool, str]:
        """
        Validate a search query for proper syntax.
        
        This method checks for:
        1. Balanced parentheses
        2. Balanced quotes
        3. Valid Boolean operators
        
        Parameters:
        -----------
        query: The search query string to validate
        
        Returns:
        --------
        tuple: (is_valid, error_message)
            - is_valid: Boolean indicating if the query is valid
            - error_message: Description of the issue if invalid, empty string if valid
        """
        # Check for balanced parentheses
        if query.count('(') != query.count(')'):
            return False, "Unbalanced parentheses in query"
        
        # Check for balanced quotes
        if query.count('"') % 2 != 0:
            return False, "Unbalanced quotes in query"
        
        # Check for valid Boolean operators
        boolean_ops = ['AND', 'OR', 'NOT']
        words = query.split()
        for i, word in enumerate(words):
            if word.upper() in boolean_ops:
                # Check if operator is at the beginning or end of query
                if i == 0 or i == len(words) - 1:
                    return False, f"Boolean operator '{word}' at the beginning or end of query"
                
                # Check if operator is followed by another operator
                if i < len(words) - 1 and words[i+1].upper() in boolean_ops:
                    return False, f"Consecutive Boolean operators: '{word} {words[i+1]}'"
        
        return True, ""
        
    def fix_query(self, query: str) -> Tuple[str, bool, str]:
        """
        Validate and automatically fix common query syntax issues.
        
        This method repairs:
        1. Unbalanced parentheses
        2. Unbalanced quotes
        3. Boolean operators at the beginning/end of query or consecutive operators
        
        Parameters:
        -----------
        query: The query string to validate and fix
        
        Returns:
        --------
        tuple: (fixed_query, was_modified, modification_message)
            - fixed_query: The repaired query string
            - was_modified: Boolean indicating if the query was modified
            - modification_message: Description of the changes made, if any
        """
        original_query = query
        fixed_query = query
        modifications = []
        
        # Fix unbalanced parentheses
        open_count = fixed_query.count('(')
        close_count = fixed_query.count(')')
        
        if open_count != close_count:
            if open_count > close_count:
                # Add missing closing parentheses at the end
                fixed_query += ')' * (open_count - close_count)
                modifications.append(f"Added {open_count - close_count} closing parentheses")
            else:
                # Add missing opening parentheses at the beginning
                fixed_query = '(' * (close_count - open_count) + fixed_query
                modifications.append(f"Added {close_count - open_count} opening parentheses")
        
        # Fix unbalanced quotes using a more intelligent approach
        if fixed_query.count('"') % 2 != 0:
            # Find positions of all quotes
            quote_positions = [i for i, char in enumerate(fixed_query) if char == '"']
            
            # If we have an odd number of quotes, find the best place to add a quote
            if len(quote_positions) % 2 == 1:
                # Analyze the string to locate quote pairs and find the unpaired quote
                paired = set()
                for i in range(0, len(quote_positions) - 1, 2):
                    # If we've run out of quotes to pair, the rest are unpaired
                    if i + 1 >= len(quote_positions):
                        break
                    paired.add(quote_positions[i])
                    paired.add(quote_positions[i + 1])
                
                # Find the position of the unpaired quote
                unpaired = [pos for pos in quote_positions if pos not in paired]
                
                if unpaired:
                    unpaired_pos = unpaired[0]
                    # If the unpaired quote is at the start, add one at the end of that term
                    if unpaired_pos == quote_positions[0]:
                        # Find the next space or logical operator after this quote
                        space_after = fixed_query[unpaired_pos+1:].find(' ')
                        if space_after == -1:  # No space found, add at the end
                            fixed_query += '"'
                        else:
                            # Insert quote before the space
                            fixed_query = fixed_query[:unpaired_pos+1+space_after] + '"' + fixed_query[unpaired_pos+1+space_after:]
                    else:
                        # Otherwise, add a quote at the end of the query
                        fixed_query += '"'
                else:
                    # Fallback: just add a quote at the end
                    fixed_query += '"'
                
                modifications.append("Added balancing quote mark")
        
        # Fix Boolean operators
        boolean_ops = ['AND', 'OR', 'NOT']
        words = fixed_query.split()
        
        # Remove operator from beginning if present
        if words and words[0].upper() in boolean_ops:
            words = words[1:]
            modifications.append(f"Removed leading Boolean operator")
        
        # Remove operator from end if present
        if words and words[-1].upper() in boolean_ops:
            words = words[:-1]
            modifications.append(f"Removed trailing Boolean operator")
        
        # Fix consecutive operators
        i = 0
        consecutives_fixed = False
        while i < len(words) - 1:
            if words[i].upper() in boolean_ops and words[i+1].upper() in boolean_ops:
                # Remove the second operator
                words.pop(i+1)
                consecutives_fixed = True
            else:
                i += 1
                
        if consecutives_fixed:
            modifications.append("Fixed consecutive Boolean operators")
        
        # Reconstruct the query if words were modified
        if words != fixed_query.split():
            fixed_query = ' '.join(words)
        
        # Check if fix introduced or failed to fix all issues
        is_valid, remaining_error = self.validate_query(fixed_query)
        if not is_valid:
            # If we still have errors, try one more round of fixes
            if "Unbalanced parentheses" in remaining_error:
                open_count = fixed_query.count('(')
                close_count = fixed_query.count(')')
                if open_count > close_count:
                    fixed_query += ')' * (open_count - close_count)
                    modifications.append(f"Added {open_count - close_count} more closing parentheses")
                else:
                    fixed_query = '(' * (close_count - open_count) + fixed_query
                    modifications.append(f"Added {close_count - open_count} more opening parentheses")
            
            if "Unbalanced quotes" in remaining_error:
                fixed_query += '"'
                modifications.append("Added another closing quote mark")
        
        # Do one final validation check
        is_valid, remaining_error = self.validate_query(fixed_query)
        if not is_valid:
            # Log that we couldn't fully fix the query
            self.logger.warning(f"Query still has issues after fixing: {remaining_error}")
            modifications.append(f"Query may still have issues: {remaining_error}")
        
        # Determine if any changes were made
        was_modified = fixed_query != original_query
        modification_message = "; ".join(modifications) if modifications else "No modifications needed"
        
        return fixed_query, was_modified, modification_message

    async def async_search(
        self,
        query: str,
        max_papers: int = 10,
        year_range: Optional[Tuple[int, int]] = None,
        sources: Optional[List[str]] = None,
        max_retries: int = 3,
        retry_delay_base: float = 2.0
    ) -> List[Paper]:
        """
        Search for academic papers using all available sources.
        
        This method:
        1. Validates and fixes the query syntax if needed
        2. Calls multiple APIs with the specified query parameters
        3. Combines and deduplicates results from all sources
        4. Updates search history for tracking
        5. Handles errors gracefully with retry logic
        
        Parameters:
        -----------
        query: The search query string for finding relevant papers
        max_papers: Maximum number of papers to return from each source (default: 10)
        year_range: Optional tuple of (start_year, end_year) for filtering papers by publication date
        sources: Optional list of sources to use (e.g., ["tavily", "semantic_scholar", "arxiv"])
                If None, all available sources will be used
        max_retries: Maximum number of retry attempts for failed searches
        retry_delay_base: Base delay for exponential backoff (in seconds)
            
        Returns:
        --------
        list: List of unique Paper objects matching the query
              Empty list if search fails or no results found
        """
        self.logger.info(f"Starting search with query: {query}")
        
        # Validate and fix query syntax if needed
        is_valid, error_message = self.validate_query(query)
        if not is_valid:
            self.logger.warning(f"Invalid query syntax: {error_message}")
            fixed_query, was_modified, modification_message = self.fix_query(query)
            
            if was_modified:
                self.logger.info(f"Query automatically fixed: {modification_message}")
                self.logger.info(f"Using fixed query: {fixed_query}")
                query = fixed_query
            else:
                self.logger.error(f"Query could not be fixed automatically")
                return []  # Return empty list for invalid queries that couldn't be fixed
        
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
            
            self.logger.info(f"Using sources: {', '.join(sources)}")
            
            # Create a list of tasks for concurrent execution
            tasks = []
            for source in sources:
                if source in source_map:
                    # Add search task to the list
                    search_func = source_map[source]
                    tasks.append(self._search_with_retry(search_func, query, max_papers, year_range, source, max_retries, retry_delay_base))
            
            # Execute all search tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results, excluding any that resulted in exceptions
            all_papers = []
            for i, result in enumerate(results):
                # Skip exceptions
                if isinstance(result, Exception):
                    self.logger.error(f"Search error in {sources[i]}: {result}")
                    continue
                
                # Add papers to our combined list
                all_papers.extend(result)
                self.logger.info(f"Found {len(result)} papers from {sources[i]}")
            
            # Remove duplicate papers that we've seen before
            unique_papers = self._remove_duplicates(all_papers)
            
            # Track this search in our history
            self.search_history.append({
                "query": query,
                "total_results": len(unique_papers),
                "sources_used": sources,
                "timestamp": datetime.now().isoformat()
            })
            
            self.logger.info(f"Search completed. Total unique papers: {len(unique_papers)}")
            return unique_papers
            
        except Exception as e:
            # Handle any errors that occur during the search
            self.logger.exception(f"Search failed with exception: {str(e)}")
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

    async def _search_with_retry(self, search_func, query, max_papers, year_range, source_name, max_retries, retry_delay_base):
        """
        Execute a search with automatic retry logic for failed attempts.
        
        Parameters:
        -----------
        search_func: The search function to call
        query: The search query
        max_papers: Maximum number of papers to return
        year_range: Optional year range filter
        source_name: Name of the source for logging
        max_retries: Maximum number of retry attempts
        retry_delay_base: Base delay for exponential backoff
        
        Returns:
        --------
        list: List of Paper objects
        """
        retry_count = 0
        last_error = None
        
        while retry_count <= max_retries:
            try:
                if retry_count > 0:
                    self.logger.info(f"Retry attempt {retry_count}/{max_retries} for {source_name}")
                
                # Execute the search
                return await search_func(query, max_papers, year_range)
                
            except Exception as e:
                last_error = e
                retry_count += 1
                
                if retry_count <= max_retries:
                    # Calculate delay with exponential backoff and jitter
                    delay = retry_delay_base * (2 ** (retry_count - 1))
                    jitter = 0.1 * delay * (2 * asyncio.get_event_loop().time() % 1)  # Add up to 10% jitter
                    delay += jitter
                    
                    self.logger.warning(f"Search attempt failed for {source_name}: {str(e)}. Retrying in {delay:.2f} seconds...")
                    await asyncio.sleep(delay)
                else:
                    self.logger.error(f"All retry attempts failed for {source_name}: {str(e)}")
        
        # If we get here, all retries failed
        return Exception(f"Max retries exceeded for {source_name}: {str(last_error)}") 