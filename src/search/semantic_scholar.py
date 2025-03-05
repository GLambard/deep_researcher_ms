"""
Semantic Scholar API client for searching academic papers.
"""

import requests
from typing import List, Dict, Optional, Tuple
import time
from .paper import Paper

class SemanticScholarAPI:
    """
    Client for the Semantic Scholar API.
    """
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Semantic Scholar API client.
        """
        self.api_key = api_key
        self.last_request_time = 0
        self.min_request_interval = 3.0  # 3 seconds between requests (rate limit: 100/5min)
    
    def search(
        self,
        query: str,
        limit: int = 10,
        year_range: Optional[Tuple[int, int]] = None
    ) -> List[Paper]:
        """
        Search for papers using Semantic Scholar API.
        
        Parameters
        ----------
        query : str
            Search query
        limit : int, optional
            Maximum number of results
        year_range : tuple, optional
            (start_year, end_year) for filtering
            
        Returns
        -------
        list
            List of Paper objects
        """
        # Prepare search parameters
        params = {
            "query": query,
            "limit": limit,
            "fields": "title,abstract,authors,year,doi,url,venue"
        }
        
        # Add year range filter if specified
        if year_range:
            params["year"] = f"{year_range[0]}-{year_range[1]}"
        
        # Add API key if available
        headers = {"User-Agent": "DeepResearcher/1.0"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        
        # Rate limiting
        self._rate_limit()
        
        try:
            # Make request
            response = requests.get(
                f"{self.BASE_URL}/paper/search",
                params=params,
                headers=headers
            )
            response.raise_for_status()
            
            # Extract results
            data = response.json()
            results = data.get("data", [])
            
            # Convert to Paper objects
            papers = []
            for result in results:
                if not result:
                    continue
                    
                paper = Paper(
                    title=result.get("title", ""),
                    abstract=result.get("abstract", ""),
                    authors=[a.get("name", "") for a in result.get("authors", [])],
                    year=result.get("year"),
                    doi=result.get("doi"),
                    url=result.get("url"),
                    source_api="semantic_scholar"
                )
                papers.append(paper)
            
            return papers
            
        except requests.exceptions.RequestException as e:
            print(f"Semantic Scholar API request failed: {e}")
            return []
    
    def _rate_limit(self):
        """
        Implement rate limiting.
        """
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last_request)
        
        self.last_request_time = time.time() 