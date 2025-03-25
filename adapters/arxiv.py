"""
arXiv API adapter for retrieving scholarly literature.

This module implements an adapter for the arXiv API, which provides access
to preprints in various fields of science, mathematics, and engineering.
The adapter handles request formatting, XML response parsing, error handling,
rate limiting, and caching, following arXiv's API guidelines.

arXiv API documentation:
- https://info.arxiv.org/help/api/basics.html
- https://info.arxiv.org/help/api/user-manual.html

This adapter follows arXiv's rate limiting requirements:
- No more than 1 request every 3 seconds
- Single connection at a time
"""
import asyncio
import hashlib
import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Set
from urllib.parse import quote, urlencode

import aiohttp
from dateutil import parser as date_parser
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from common.adapter_base import LiteratureAdapter
from common.models import Author, PaperMetadata, QueryParams, QueryResult
from utils.cache import cached
from utils.rate_limiter import get_rate_limiter


class ArXivAPIError(Exception):
    """
    Exception raised for arXiv API errors.
    
    This custom exception encapsulates errors returned by the API,
    such as rate limiting, invalid requests, or server errors.
    """
    pass


class ArXivAdapter(LiteratureAdapter):
    """
    Adapter for the arXiv API.
    
    This adapter implements the LiteratureAdapter interface and provides methods
    to search and retrieve paper metadata from the arXiv API. It handles:
    
    1. Request formatting according to API specifications
    2. Response parsing into our common data model
    3. Error handling with retries for transient failures
    4. Rate limiting to comply with API usage policies
    5. Caching of responses to reduce redundant requests
    
    The adapter uses asynchronous HTTP requests for better performance while
    respecting arXiv's rate limits.
    """
    
    # Base URL for the arXiv API
    _BASE_URL = "http://export.arxiv.org/api"
    
    # XML namespaces used in the arXiv API responses
    _NAMESPACES = {
        "atom": "http://www.w3.org/2005/Atom",
        "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
        "arxiv": "http://arxiv.org/schemas/atom"
    }
    
    def __init__(
        self, 
        timeout: int = 30,
        disable_cache: bool = False,
    ):
        """
        Initialize the arXiv adapter.
        
        Args:
            timeout: Request timeout in seconds. Default is 30s which balances
                    reliability with responsiveness.
            disable_cache: Whether to disable caching. This is primarily useful
                          for testing or when fresh data is always required.
        """
        self._timeout = timeout
        self._logger = logging.getLogger(__name__)
        self._rate_limiter = get_rate_limiter("arxiv")
        self._disable_cache = disable_cache
    
    async def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> str:
        """
        Make a request to the arXiv API.
        
        Args:
            endpoint: API endpoint, starting with "/"
            params: Query parameters
            
        Returns:
            Raw XML response as a string
            
        Raises:
            ArXivAPIError: If the API returns an error
        """
        url = f"{self._BASE_URL}{endpoint}"
        
        # Initialize parameters if None
        if params is None:
            params = {}
            
        # URL encode parameters
        if params:
            query_string = urlencode(params, doseq=True)
            url = f"{url}?{query_string}"
        
        try:
            # Use retrying to handle transient errors
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=1, max=10),
                retry=retry_if_exception_type((aiohttp.ClientError, TimeoutError)),
                reraise=True,
            ):
                with attempt:
                    # Use the rate limiter - arXiv requires at least 3s between requests
                    await self._rate_limiter.acquire()
                    
                    async with aiohttp.ClientSession() as session:
                        async with session.get(
                            url, 
                            timeout=self._timeout
                        ) as response:
                            # Check for errors
                            if response.status != 200:
                                error_text = await response.text()
                                raise ArXivAPIError(
                                    f"API error {response.status}: {error_text}"
                                )
                            
                            # Return raw XML response
                            return await response.text()
                            
        except RetryError as e:
            raise ArXivAPIError(f"Max retries exceeded: {str(e.last_attempt.exception())}")
        except aiohttp.ClientError as e:
            raise ArXivAPIError(f"HTTP error: {str(e)}")
    
    def _parse_author(self, author_element: ET.Element) -> Author:
        """
        Parse author data from XML element.
        
        Args:
            author_element: XML element containing author data
            
        Returns:
            Author object
        """
        # Extract author name
        name = author_element.find("atom:name", self._NAMESPACES)
        author_name = name.text if name is not None else "Unknown"
        
        # arXiv doesn't provide author IDs or affiliations in the API
        return Author(
            name=author_name,
            id=None,
            url=None,
            affiliations=[]
        )
    
    def _parse_paper(self, entry_element: ET.Element) -> PaperMetadata:
        """
        Parse paper data from XML entry element.
        
        Args:
            entry_element: XML element containing paper data
            
        Returns:
            PaperMetadata object
        """
        # Extract basic metadata
        title_elem = entry_element.find("atom:title", self._NAMESPACES)
        title = title_elem.text if title_elem is not None else "Untitled"
        
        # Clean up title - remove newlines and extra whitespace
        title = re.sub(r'\s+', ' ', title).strip()
        
        # Extract abstract
        summary_elem = entry_element.find("atom:summary", self._NAMESPACES)
        abstract = summary_elem.text if summary_elem is not None else None
        
        # Clean up abstract - remove newlines and extra whitespace
        if abstract:
            abstract = re.sub(r'\s+', ' ', abstract).strip()
        
        # Extract arXiv ID and DOI
        id_elem = entry_element.find("atom:id", self._NAMESPACES)
        arxiv_url = id_elem.text if id_elem is not None else None
        
        # ArXiv ID is the last part of the URL
        arxiv_id = None
        if arxiv_url:
            arxiv_id = arxiv_url.split('/')[-1]
        
        # Extract DOI if available (stored in arxiv:doi element)
        doi_elem = entry_element.find("arxiv:doi", self._NAMESPACES)
        doi = doi_elem.text if doi_elem is not None else None
        
        # Extract publication date
        published_elem = entry_element.find("atom:published", self._NAMESPACES)
        publication_date = None
        year = None
        if published_elem is not None and published_elem.text:
            try:
                publication_date = date_parser.parse(published_elem.text)
                publication_date = publication_date.strftime("%Y-%m-%d")
                year = publication_date.split("-")[0]
            except (ValueError, TypeError):
                pass
        
        # Extract authors
        authors = []
        for author_elem in entry_element.findall("atom:author", self._NAMESPACES):
            authors.append(self._parse_author(author_elem))
        
        # Extract links
        url = None
        pdf_url = None
        for link_elem in entry_element.findall("atom:link", self._NAMESPACES):
            rel = link_elem.get("rel")
            href = link_elem.get("href")
            if rel == "alternate" and href:
                url = href
            elif rel == "related" and href and href.endswith(".pdf"):
                pdf_url = href
        
        # Extract categories/fields of study
        fields_of_study = []
        for category_elem in entry_element.findall("atom:category", self._NAMESPACES):
            term = category_elem.get("term")
            if term:
                fields_of_study.append(term)
        
        # Set default URL to arXiv if not found
        if not url and arxiv_id:
            url = f"https://arxiv.org/abs/{arxiv_id}"
        
        # Set default PDF URL if not found
        if not pdf_url and arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        
        # Extract journal reference (if published)
        venue = None
        journal_ref_elem = entry_element.find("arxiv:journal_ref", self._NAMESPACES)
        if journal_ref_elem is not None and journal_ref_elem.text:
            venue = journal_ref_elem.text
        
        # Create external IDs dictionary
        external_ids = {}
        if arxiv_id:
            external_ids["arxiv"] = arxiv_id
        if doi:
            external_ids["doi"] = doi
            
        # Store all raw data for reference
        raw_data = {elem.tag.split('}')[-1]: elem.text for elem in entry_element}
        
        return PaperMetadata(
            title=title,
            source="arXiv",
            authors=authors,
            abstract=abstract,
            doi=doi,
            url=url,
            publication_date=publication_date,
            pdf_url=pdf_url,
            year=year,
            venue=venue,
            citation_count=None,  # arXiv doesn't provide citation counts
            fields_of_study=fields_of_study,
            publication_types=["preprint"],  # All arXiv papers are preprints
            external_ids=external_ids,
            raw_data=raw_data
        )
    
    def _build_search_query(self, params: QueryParams) -> Dict[str, Any]:
        """
        Build a search query parameters dictionary from QueryParams.
        
        Args:
            params: Query parameters
            
        Returns:
            Dictionary of search parameters for the arXiv API
        """
        query_params = {
            "start": params.offset,
            "max_results": params.limit
        }
        
        # Build search query parts
        search_parts = []
        
        # Add keywords (search in all fields if no specific field mentioned)
        if params.keywords:
            # Check if the keywords already contain field specifiers (au:, ti:, etc.)
            # If not, search in all fields
            if ":" in params.keywords:
                search_parts.append(params.keywords)
            else:
                search_parts.append(f"all:{params.keywords}")
            
        # Add year filter using submittedDate if specified
        if params.year:
            if isinstance(params.year, str) and "-" in params.year:
                # Handle year range (e.g., "2020-2023")
                start_year, end_year = params.year.split("-")
                search_parts.append(
                    f"submittedDate:[{start_year}0101 TO {end_year}1231]"
                )
            else:
                # Handle single year
                year = str(params.year)
                search_parts.append(
                    f"submittedDate:[{year}0101 TO {year}1231]"
                )
                
        # Add fields of study filter (arXiv uses 'cat' prefix for categories)
        if params.fields_of_study:
            search_parts.append(f"cat:{params.fields_of_study}")
                
        # Combine search parts with AND operator
        if search_parts:
            query_params["search_query"] = " AND ".join(search_parts)
            
        # arXiv doesn't support sorting in their API directly
        # If sorting by date is requested, we'll need to sort results in memory
        
        return query_params
    
    async def search_papers(self, params: QueryParams) -> QueryResult:
        """
        Search for papers based on the provided query parameters.
        
        Args:
            params: Query parameters for the search
            
        Returns:
            QueryResult containing matching papers and pagination info
        """
        # Skip caching if disabled
        if self._disable_cache:
            return await self._search_papers_impl(params)
        else:
            return await self._search_papers_cached(params)
    
    @cached(ttl=3600, prefix="arxiv_search")
    async def _search_papers_cached(self, params: QueryParams) -> QueryResult:
        """Cached implementation of search_papers."""
        return await self._search_papers_impl(params)
    
    async def _search_papers_impl(self, params: QueryParams) -> QueryResult:
        """Internal implementation of search_papers."""
        api_params = self._build_search_query(params)
        
        try:
            # Make the API request
            response_xml = await self._make_request("/query", api_params)
            
            # Parse XML response
            root = ET.fromstring(response_xml)
            
            # Extract total results count
            total_results = 0
            total_elem = root.find(".//opensearch:totalResults", self._NAMESPACES)
            if total_elem is not None and total_elem.text:
                try:
                    total_results = int(total_elem.text)
                except ValueError:
                    pass
            
            # Extract entries and convert to PaperMetadata objects
            papers = []
            for entry in root.findall(".//atom:entry", self._NAMESPACES):
                papers.append(self._parse_paper(entry))
            
            # Calculate pagination info
            next_offset = params.offset + len(papers)
            has_next = next_offset < total_results
            
            return QueryResult(
                papers=papers,
                total_results=total_results,
                next_offset=next_offset if has_next else None,
                has_next=has_next,
            )
            
        except ArXivAPIError as e:
            self._logger.error(f"Error searching papers: {str(e)}")
            # Return empty result on error
            return QueryResult(papers=[], has_next=False)
    
    async def get_paper_by_id(self, paper_id: str) -> Optional[PaperMetadata]:
        """
        Retrieve a paper by its arXiv ID.
        
        Args:
            paper_id: arXiv paper ID (e.g., "2307.06715" or "cond-mat/0101020")
            
        Returns:
            Paper metadata if found, None otherwise
        """
        # Skip caching if disabled
        if self._disable_cache:
            return await self._get_paper_by_id_impl(paper_id)
        else:
            return await self._get_paper_by_id_cached(paper_id)
    
    @cached(ttl=86400, prefix="arxiv_paper")  # Cache for 1 day
    async def _get_paper_by_id_cached(self, paper_id: str) -> Optional[PaperMetadata]:
        """Cached implementation of get_paper_by_id."""
        return await self._get_paper_by_id_impl(paper_id)
    
    async def _get_paper_by_id_impl(self, paper_id: str) -> Optional[PaperMetadata]:
        """Internal implementation of get_paper_by_id."""
        try:
            # Use id_list parameter to get a specific paper
            response_xml = await self._make_request("/query", {"id_list": paper_id})
            
            # Parse XML response
            root = ET.fromstring(response_xml)
            
            # Find the entry element
            entry = root.find(".//atom:entry", self._NAMESPACES)
            if entry is None:
                # No paper found
                return None
                
            # Parse paper data
            return self._parse_paper(entry)
            
        except ArXivAPIError as e:
            self._logger.error(f"Error getting paper by ID: {str(e)}")
            return None
    
    async def get_paper_by_doi(self, doi: str) -> Optional[PaperMetadata]:
        """
        Retrieve a paper by its DOI.
        
        Note: arXiv doesn't provide a direct way to search by DOI.
        This implementation uses a workaround by searching for the DOI
        in the metadata.
        
        Args:
            doi: DOI of the paper
            
        Returns:
            Paper metadata if found, None otherwise
        """
        # Skip caching if disabled
        if self._disable_cache:
            return await self._get_paper_by_doi_impl(doi)
        else:
            return await self._get_paper_by_doi_cached(doi)
    
    @cached(ttl=86400, prefix="arxiv_paper_doi")  # Cache for 1 day
    async def _get_paper_by_doi_cached(self, doi: str) -> Optional[PaperMetadata]:
        """Cached implementation of get_paper_by_doi."""
        return await self._get_paper_by_doi_impl(doi)
    
    async def _get_paper_by_doi_impl(self, doi: str) -> Optional[PaperMetadata]:
        """Internal implementation of get_paper_by_doi."""
        try:
            # Search for papers with the given DOI
            # We need to handle DOIs that contain special characters
            encoded_doi = quote(doi, safe="")
            search_query = f"doi:{encoded_doi}"
            
            response_xml = await self._make_request("/query", {
                "search_query": search_query,
                "max_results": 1
            })
            
            # Parse XML response
            root = ET.fromstring(response_xml)
            
            # Find the entry element
            entry = root.find(".//atom:entry", self._NAMESPACES)
            if entry is None:
                # No paper found
                return None
                
            # Parse paper data
            return self._parse_paper(entry)
            
        except ArXivAPIError as e:
            self._logger.error(f"Error getting paper by DOI: {str(e)}")
            return None
    
    async def get_citations(self, paper_id: str, limit: int = 100) -> List[PaperMetadata]:
        """
        Get papers that cite the specified paper.
        
        Note: arXiv doesn't provide citation data directly.
        This method returns an empty list as a placeholder.
        
        Args:
            paper_id: arXiv paper ID
            limit: Maximum number of citations to retrieve
            
        Returns:
            Empty list (arXiv doesn't provide citation data)
        """
        # arXiv doesn't provide citation information
        self._logger.info("Citation information is not available through the arXiv API")
        return []
    
    async def get_references(self, paper_id: str, limit: int = 100) -> List[PaperMetadata]:
        """
        Get papers referenced by the specified paper.
        
        Note: arXiv doesn't provide reference data directly.
        This method returns an empty list as a placeholder.
        
        Args:
            paper_id: arXiv paper ID
            limit: Maximum number of references to retrieve
            
        Returns:
            Empty list (arXiv doesn't provide reference data)
        """
        # arXiv doesn't provide reference information
        self._logger.info("Reference information is not available through the arXiv API")
        return []
    
    def source_name(self) -> str:
        """
        Get the name of this literature source.
        
        Returns:
            Name of the source
        """
        return "arXiv" 