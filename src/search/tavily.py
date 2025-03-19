"""
Tavily API client for searching academic papers.
Documentation: https://docs.tavily.com/api-reference/introduction
"""

import requests
from typing import List, Dict, Optional, Tuple
import time
import re
from .paper import Paper

class TavilyAPI:
    """
    Client for the Tavily API.
    """
    
    BASE_URL = "https://api.tavily.com"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Tavily API client.
        
        Parameters
        ----------
        api_key : str
            Tavily API key
        """
        self.api_key = api_key
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 second between requests
        self.session = requests.Session()
        retry_adapter = requests.adapters.HTTPAdapter(
            max_retries=3,
            pool_connections=10,
            pool_maxsize=10
        )
        self.session.mount('https://', retry_adapter)
        self.session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}" if api_key else ""
        })
    
    ## TODO: 
    # Check if an LLM is needed to extract the year from the content
    #
    def _extract_year(self, content: str) -> Optional[int]:
        """
        Extract publication year from content using various patterns.
        
        Parameters
        ----------
        content : str
            Paper content to extract year from
            
        Returns
        -------
        int or None
            Publication year if found
        """
        # Look for explicit publication year patterns
        pub_year_patterns = [
            r'published in (\d{4})',
            r'published .*?(\d{4})',
            r'\((\d{4})\)',
            r'©\s*(\d{4})',
            r'copyright\s*(\d{4})',
        ]
        
        for pattern in pub_year_patterns:
            matches = re.findall(pattern, content.lower())
            if matches:
                year = int(matches[0])
                if 1900 <= year <= 2024:  # Sanity check
                    return year
        
        # Extract all years and take the most recent one
        years = re.findall(r'\b(19|20)\d{2}\b', content)
        if years:
            valid_years = [int(y) for y in years if 1900 <= int(y) <= 2024]
            if valid_years:
                return max(valid_years)
        
        return None
    
    ## DONE: 
    # Added the following domains to the search:
    # - chemrxiv.org
    # - biorxiv.org
    # - medrxiv.org
    # - springeropen.com
    # - onlinelibrary.wiley.com
    # - link.springer.com
    # - pubs.acs.org
    # - pubs.rsc.org
    # - ieeexplore.ieee.org
    # - iopscience.iop.org
    # - pubs.aip.org
    # - tandfonline.com
    # - mdpi.com
    # - frontiersin.org
    # - plos.org
    # - hindawi.com
    # - nature.com/ncomms
    # - nature.com/srep
    # - science.org/journal/sciadv
    #
    # Removed the following domains:
    # - researchgate.net
    # - mdpi.com
    # - ncbi.nlm.nih.gov
    #
    ## TODO: 
    # Remove the domains that are not needed, e.g. researchgate.net
    # May use the OpenAlex API to get the DOI
    # Check if an LLM is needed to extract the authors from the content
    #
    def search(
        self,
        query: str,
        limit: int = 10,
        year_range: Optional[Tuple[int, int]] = None
    ) -> List[Paper]:
        """
        Search for academic papers using Tavily API.
        
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
        if not self.api_key:
            print("Tavily API key not provided")
            return []
            
        # Add year range to query if specified
        if year_range:
            query = f"{query} year:{year_range[0]}-{year_range[1]}"
            
        # Prepare search parameters
        params = {
            "query": query,
            "search_depth": "advanced",
            "include_answer": False,  # We don't need the AI answer
            "include_raw_content": True,  # Get full content when available
            "include_domains": [
                "scholar.google.com",
                "arxiv.org",
                "chemrxiv.org",
                "biorxiv.org",
                "medrxiv.org",
                #"researchgate.net",
                "semanticscholar.org",
                #"ncbi.nlm.nih.gov",
                "sciencedirect.com",
                "springer.com",
                "nature.com",
                "science.org",
                "wiley.com", 
                #"mdpi.com",
                "frontiersin.org",
                "plos.org",
                "hindawi.com", 
                "nature.com/ncomms",
                "nature.com/srep",
                "science.org/journal/sciadv",   
                "pubs.acs.org/journal/acsodf",
                "onlinelibrary.wiley.com/journal/21983844",
                "pubs.rsc.org/en/journals/journal/RA",  
                "pubs.aip.org/aip/adv", 
                "springeropen.com",
                "onlinelibrary.wiley.com",
                "link.springer.com", 
                "pubs.acs.org",
                "pubs.rsc.org",
                "ieeexplore.ieee.org",
                "iopscience.iop.org",
                "pubs.aip.org",
                "tandfonline.com", 
            ],
            "max_results": min(limit, 20),  # Tavily's limit is 20
            "search_type": "search"
        }
        
        # Rate limiting
        self._rate_limit()
        
        try:
            # Make request
            response = self.session.post(
                f"{self.BASE_URL}/search",
                json=params
            )
            response.raise_for_status()
            
            # Extract results
            data = response.json()
            results = data.get("results", [])
            
            # Convert to Paper objects
            papers = []
            for result in results:
                if not result:
                    continue
                    
                content = result.get("content", "")
                
                # Extract year using improved method
                year = self._extract_year(content)
                
                # Try to extract authors from content
                authors = []
                try:
                    # Look for common author patterns in content
                    # Look for "by [Author Name]" or "Authors: [Author Names]"
                    author_matches = re.findall(r'(?:by|authors?:)\s+([^\.]+)', content, re.IGNORECASE)
                    if author_matches:
                        # Split on common separators and clean up
                        authors = [
                            author.strip()
                            for author in author_matches[0].split(',')
                            if author.strip() and len(author.strip()) > 2
                        ]
                except:
                    pass
                
                paper = Paper(
                    title=result.get("title", ""),
                    abstract=result.get("snippet", ""),
                    authors=authors,
                    year=year,
                    doi=None,  # Would need additional processing to extract DOI
                    url=result.get("url"),
                    source_api="tavily",
                    venue=result.get("domain", "")  # Using domain as venue
                )
                papers.append(paper)
            
            return papers[:limit]
            
        except requests.exceptions.RequestException as e:
            print(f"Tavily API request failed: {e}")
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