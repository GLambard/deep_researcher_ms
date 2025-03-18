"""
Semantic Scholar API client for searching academic papers.

This module provides a client for interacting with the Semantic Scholar API,
which allows searching for academic papers and retrieving metadata.
Semantic Scholar is a free academic search engine that provides access to
millions of research papers across various disciplines.
"""

import requests  # For making HTTP requests to the API
from typing import List, Dict, Optional, Tuple  # For type hints
import time  # For implementing rate limiting
from .paper import Paper  # For paper data model

class SemanticScholarAPI:
    """
    Client for the Semantic Scholar API.
    
    This class handles all communication with the Semantic Scholar API,
    including search requests, rate limiting, and converting API responses
    into standardized Paper objects for consistent handling throughout
    the application.
    """
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1"  # API endpoint base URL
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Semantic Scholar API client with optional API key.
        
        Parameters:
        -----------
        api_key: Optional API key for Semantic Scholar
            Using an API key increases rate limits and provides more reliable service
            Without an API key, requests are subject to stricter rate limits
        """
        self.api_key = api_key
        self.last_request_time = 0  # Track time of last request for rate limiting
        # Set minimum interval between requests to comply with API rate limits
        # Without an API key, limit is 100 requests per 5 minutes (3 seconds between requests)
        self.min_request_interval = 3.0  # 3 seconds between requests (rate limit: 100/5min)
    
    def search(
        self,
        query: str,
        limit: int = 10,
        year_range: Optional[Tuple[int, int]] = None
    ) -> List[Paper]:
        """
        Search for academic papers using the Semantic Scholar API.
        
        This method:
        1. Constructs a search query with appropriate parameters
        2. Handles rate limiting to comply with API restrictions
        3. Processes the API response into standardized Paper objects
        4. Handles errors gracefully
        
        Parameters:
        -----------
        query: The search query string for finding relevant papers
        limit: Maximum number of results to return (default: 10)
        year_range: Optional tuple of (start_year, end_year) for filtering papers by publication date
            
        Returns:
        --------
        list: List of Paper objects matching the query
              Empty list if search fails or no results found
        """

        ## TODO: Check list of fields to retrieve

        # Prepare search parameters for the API request
        # Specify which fields we want to retrieve for each paper
        params = {
            "query": query,
            "limit": limit,
            "fields": "title,abstract,authors,year,doi,url,venue"  # Request specific fields
        }
        
        # Add year range filter if specified
        # Format: "2018-2023" for papers published between 2018 and 2023
        if year_range:
            params["year"] = f"{year_range[0]}-{year_range[1]}"
        
        # Set up request headers
        # User-Agent identifies our application to the API
        headers = {"User-Agent": "DeepResearcher/1.0"}
        # Add API key to headers if available for higher rate limits
        if self.api_key:
            headers["x-api-key"] = self.api_key
        
        # Apply rate limiting to avoid exceeding API limits
        self._rate_limit()
        
        try:
            # Make the HTTP request to the Semantic Scholar API
            response = requests.get(
                f"{self.BASE_URL}/paper/search",
                params=params,
                headers=headers
            )
            # Raise an exception for HTTP errors (4xx, 5xx)
            response.raise_for_status()
            
            # Parse the JSON response
            data = response.json()
            # Extract the list of paper data from the response
            results = data.get("data", [])
            
            # Convert each result to a standardized Paper object
            papers = []
            for result in results:
                if not result:  # Skip empty results
                    continue
                    
                # Create a Paper object with data from the API response
                paper = Paper(
                    title=result.get("title", ""),  # Paper title
                    abstract=result.get("abstract", ""),  # Abstract/summary
                    authors=[a.get("name", "") for a in result.get("authors", [])],  # Author list
                    year=result.get("year"),  # Publication year
                    doi=result.get("doi"),  # Digital Object Identifier
                    url=result.get("url"),  # URL to access the paper
                    source_api="semantic_scholar"  # Track which API provided this paper
                )
                papers.append(paper)
            
            return papers
            
        except requests.exceptions.RequestException as e:
            # Handle any errors that occur during the request
            # This ensures a failed search doesn't crash the application
            print(f"Semantic Scholar API request failed: {e}")
            return []  # Return empty list on failure
    
    def _rate_limit(self):
        """
        Implement rate limiting to comply with API restrictions.
        
        This method ensures we don't exceed the API's rate limits by:
        1. Tracking the time of the last request
        2. Calculating time elapsed since last request
        3. Sleeping if needed to maintain minimum interval between requests
        
        Rate limiting is important to:
        - Avoid getting blocked by the API
        - Ensure fair usage of the service
        - Maintain reliable operation of the application
        """
        # Get current time
        current_time = time.time()
        # Calculate time elapsed since last request
        time_since_last_request = current_time - self.last_request_time
        
        # If not enough time has passed, sleep for the remaining time
        if time_since_last_request < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last_request)
        
        # Update the last request time
        self.last_request_time = time.time() 