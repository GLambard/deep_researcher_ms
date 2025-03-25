"""
ChemRxiv API adapter for retrieving scholarly literature.

This module implements an adapter for the ChemRxiv API, which provides access
to preprints in chemistry and related fields. The adapter handles request
formatting, response parsing, error handling, rate limiting, and caching,
following ChemRxiv's API guidelines.

ChemRxiv API documentation:
- https://chemrxiv.org/engage/chemrxiv/public-api/documentation
"""
import asyncio
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
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


class ChemRxivAPIError(Exception):
    """
    Exception raised for ChemRxiv API errors.
    
    This custom exception encapsulates errors returned by the API,
    such as rate limiting, invalid requests, or server errors.
    """
    pass


class ChemRxivAdapter(LiteratureAdapter):
    """
    Adapter for the ChemRxiv API.
    
    This adapter implements the LiteratureAdapter interface and provides methods
    to search and retrieve paper metadata from the ChemRxiv API. It handles:
    
    1. Request formatting according to API specifications
    2. Response parsing into our common data model
    3. Error handling with retries for transient failures
    4. Rate limiting to comply with API usage policies
    5. Caching of responses to reduce redundant requests
    
    The adapter uses asynchronous HTTP requests for better performance.
    """
    
    # Base URL for the ChemRxiv API
    _BASE_URL = "https://chemrxiv.org/engage/chemrxiv/public-api/v1"
    
    def __init__(
        self, 
        timeout: int = 30,
        disable_cache: bool = False,
    ):
        """
        Initialize the ChemRxiv adapter.
        
        Args:
            timeout: Request timeout in seconds. Default is 30s which balances
                    reliability with responsiveness.
            disable_cache: Whether to disable caching. This is primarily useful
                          for testing or when fresh data is always required.
        """
        self._timeout = timeout
        self.logger = logging.getLogger(__name__)
        self._rate_limiter = get_rate_limiter("chemrxiv")
        self._disable_cache = disable_cache
    
    async def _make_request(
        self, 
        endpoint: str, 
        params: Optional[Dict[str, Any]] = None,
        method: str = "GET",
    ) -> Dict[str, Any]:
        """
        Make a request to the ChemRxiv API.
        
        Args:
            endpoint: API endpoint, starting with "/"
            params: Query parameters
            method: HTTP method to use (GET or POST)
            
        Returns:
            JSON response as a dictionary
            
        Raises:
            ChemRxivAPIError: If the API returns an error
        """
        url = f"{self._BASE_URL}{endpoint}"
        
        # Initialize parameters if None
        if params is None:
            params = {}
            
        try:
            # Use retrying to handle transient errors
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=1, max=10),
                retry=retry_if_exception_type((aiohttp.ClientError, TimeoutError)),
                reraise=True,
            ):
                with attempt:
                    # Use the rate limiter to avoid hitting API limits
                    await self._rate_limiter.acquire()
                    
                    async with aiohttp.ClientSession() as session:
                        if method.upper() == "GET":
                            # For GET requests, add params to the URL
                            async with session.get(
                                url, 
                                params=params,
                                timeout=self._timeout
                            ) as response:
                                return await self._process_response(response)
                        else:
                            # For POST requests, add params to the body
                            async with session.post(
                                url, 
                                json=params,
                                timeout=self._timeout
                            ) as response:
                                return await self._process_response(response)
                            
        except RetryError as e:
            raise ChemRxivAPIError(f"Max retries exceeded: {str(e.last_attempt.exception())}")
        except aiohttp.ClientError as e:
            raise ChemRxivAPIError(f"HTTP error: {str(e)}")
    
    async def _process_response(self, response: aiohttp.ClientResponse) -> Dict[str, Any]:
        """
        Process the API response and handle errors.
        
        Args:
            response: HTTP response from the API
            
        Returns:
            JSON response as a dictionary
            
        Raises:
            ChemRxivAPIError: If the API returns an error
        """
        if response.status != 200:
            error_text = await response.text()
            raise ChemRxivAPIError(f"API error {response.status}: {error_text}")
        
        try:
            return await response.json()
        except ValueError:
            raise ChemRxivAPIError("Invalid JSON in response")
    
    def _parse_author(self, author_data: Dict[str, Any]) -> Author:
        """
        Parse author data from API response.
        
        Args:
            author_data: Dictionary containing author data
            
        Returns:
            Author object
        """
        # Extract author name (fullName or firstName + lastName)
        name = author_data.get("fullName")
        if not name and "firstName" in author_data and "lastName" in author_data:
            name = f"{author_data['firstName']} {author_data['lastName']}".strip()
        
        if not name:
            name = "Unknown Author"
        
        # Extract author ID if available
        author_id = author_data.get("id") or author_data.get("authorId")
        
        # Extract affiliations if available
        affiliations = []
        if "institutions" in author_data and author_data["institutions"]:
            for institution in author_data["institutions"]:
                if isinstance(institution, dict) and "name" in institution:
                    affiliations.append(institution["name"])
                elif isinstance(institution, str):
                    affiliations.append(institution)
        
        # Extract URL if available
        url = None
        if "authorUrl" in author_data:
            url = author_data["authorUrl"]
        
        return Author(
            name=name,
            id=author_id,
            affiliations=affiliations,
            url=url
        )
    
    def _parse_paper(self, paper_data: Dict[str, Any]) -> PaperMetadata:
        """
        Parse paper data from API response.
        
        Args:
            paper_data: Dictionary containing paper data
            
        Returns:
            PaperMetadata object
        """
        # Extract basic metadata
        title = paper_data.get("title", "Untitled")
        
        # Extract abstract
        abstract = paper_data.get("abstract")
        
        # Extract DOI
        doi = paper_data.get("doi")
        
        # Extract publication date
        publication_date = None
        year = None
        
        # Check various date fields
        if "publishedDate" in paper_data and paper_data["publishedDate"]:
            try:
                publication_date = date_parser.parse(paper_data["publishedDate"])
                year = publication_date.year
            except (ValueError, TypeError):
                pass
        elif "submittedDate" in paper_data and paper_data["submittedDate"]:
            try:
                publication_date = date_parser.parse(paper_data["submittedDate"])
                year = publication_date.year
            except (ValueError, TypeError):
                pass
        
        # Convert publication_date to string format
        if publication_date:
            publication_date = publication_date.strftime("%Y-%m-%d")
        
        # Extract authors
        authors = []
        if "authors" in paper_data and paper_data["authors"]:
            for author_data in paper_data["authors"]:
                # Combine firstName and lastName
                name = None
                if "firstName" in author_data and "lastName" in author_data:
                    name = f"{author_data['firstName']} {author_data['lastName']}".strip()
                
                # Create author with ChemRxiv-specific fields
                author = Author(
                    name=name or "Unknown Author",
                    id=author_data.get("orcid"),
                    affiliations=[]
                )
                
                # Add affiliations if available
                if "institutions" in author_data and author_data["institutions"]:
                    for institution in author_data["institutions"]:
                        if isinstance(institution, dict) and "name" in institution:
                            author.affiliations.append(institution["name"])
                
                authors.append(author)
        
        # Extract URL for the paper
        url = None
        if "id" in paper_data:
            url = f"https://chemrxiv.org/engage/chemrxiv/article-details/{paper_data['id']}"
        
        # Extract PDF URL if available
        pdf_url = None
        if "asset" in paper_data and paper_data["asset"] and "original" in paper_data["asset"]:
            pdf_url = paper_data["asset"]["original"].get("url")
        
        # Extract fields of study from categories and keywords
        fields_of_study = []
        
        # Add categories
        if "categories" in paper_data and paper_data["categories"]:
            for category in paper_data["categories"]:
                if isinstance(category, dict) and "name" in category:
                    fields_of_study.append(category["name"])
        
        # Add keywords
        if "keywords" in paper_data and paper_data["keywords"]:
            fields_of_study.extend(paper_data["keywords"])
        
        # Extract venue (always ChemRxiv)
        venue = "ChemRxiv"
        
        # Extract citation count if available in metrics
        citation_count = None
        if "metrics" in paper_data and paper_data["metrics"]:
            for metric in paper_data["metrics"]:
                if metric.get("description") == "Citations":
                    citation_count = metric.get("value")
                    break
        
        # ChemRxiv ID
        chemrxiv_id = paper_data.get("id")
        
        # Create external IDs dictionary
        external_ids = {}
        if chemrxiv_id:
            external_ids["chemrxiv"] = chemrxiv_id
        if doi:
            external_ids["doi"] = doi
        
        # Extract license information
        license_info = paper_data.get("license")
        if license_info:
            if isinstance(license_info, dict):
                license_name = license_info.get("name")
                if license_name:
                    fields_of_study.append(f"License: {license_name}")
            elif isinstance(license_info, str):
                fields_of_study.append(f"License: {license_info}")
        
        return PaperMetadata(
            title=title,
            source="ChemRxiv",
            authors=authors,
            abstract=abstract,
            doi=doi,
            url=url,
            publication_date=publication_date,
            pdf_url=pdf_url,
            year=year,
            venue=venue,
            citation_count=citation_count,
            fields_of_study=fields_of_study,
            publication_types=["preprint"],  # ChemRxiv papers are preprints
            external_ids=external_ids,
            raw_data=paper_data
        )
    
    def _format_date(self, date_str: str) -> str:
        """
        Format a date string as required by the ChemRxiv API.
        
        Args:
            date_str: Date string in format YYYY, YYYY-MM, or YYYY-MM-DD
            
        Returns:
            Formatted date string
        """
        try:
            date_obj = date_parser.parse(date_str)
            return date_obj.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        except (ValueError, TypeError):
            return date_str
    
    def _build_search_query(self, params):
        """
        Build a search query for the ChemRxiv API.
        
        Args:
            params: Either a search string or a QueryParams object with search parameters
            
        Returns:
            Dictionary of query parameters for the ChemRxiv API
        """
        api_params = {}
        
        # Handle the case where params is a string (simple query)
        if isinstance(params, str):
            api_params["term"] = params
            api_params["limit"] = 3  # Default limit
            api_params["skip"] = 0    # Default offset
            return api_params
        
        # Handle the case where params is a QueryParams object
        if params.keywords:
            api_params["term"] = params.keywords
        
        # Year filter - for now, we'll use searchDateFrom and searchDateTo instead of filter
        if params.year:
            if isinstance(params.year, str) and "-" in params.year:
                # Year range format: "2020-2023"
                start_year, end_year = params.year.split("-")
                api_params["searchDateFrom"] = f"{start_year}-01-01T00:00:00.000Z"
                api_params["searchDateTo"] = f"{end_year}-12-31T23:59:59.999Z"
            else:
                # Single year
                year = str(params.year)
                api_params["searchDateFrom"] = f"{year}-01-01T00:00:00.000Z"
                api_params["searchDateTo"] = f"{year}-12-31T23:59:59.999Z"
        
        # Fields of study - we'll use the term parameter to include these
        if params.fields_of_study:
            fields = params.fields_of_study.split(",")
            if "term" in api_params:
                # Add fields to the existing term
                api_params["term"] += " " + " ".join(field.strip() for field in fields)
            else:
                # Create a new term
                api_params["term"] = " ".join(field.strip() for field in fields)
        
        # Open access filter - no direct filter available, we'll handle this in post-processing
        
        # Pagination
        api_params["skip"] = params.offset if params.offset is not None else 0
        api_params["limit"] = params.limit if params.limit is not None else 3
        
        # Sorting
        if params.sort_by:
            # ChemRxiv supports sorting by date, relevance, etc.
            sort_mapping = {
                "date": "PUBLISHED_DATE_DESC",
                "relevance": "RELEVANCE",
                "citations": "VIEWS_COUNT_DESC"  # Using views as proxy for citations
            }
            
            if params.sort_by.lower() in sort_mapping:
                api_params["sort"] = sort_mapping[params.sort_by.lower()]
        
        self.logger.debug(f"Built ChemRxiv API params: {api_params}")
        return api_params
    
    @cached(ttl=3600, prefix="chemrxiv_search")
    async def _search_papers_cached(self, params: QueryParams) -> QueryResult:
        """Cached implementation of search_papers."""
        return await self._search_papers_impl(params)
    
    async def _search_papers_impl(self, params: QueryParams) -> QueryResult:
        """Internal implementation of search_papers."""
        api_params = self._build_search_query(params)
        
        try:
            # Make the API request
            response = await self._make_request("/items", api_params)
            
            # Extract total results count - this is in the correct place
            total_results = response.get("totalCount", 0)
            
            # Extract items and convert to PaperMetadata objects
            papers = []
            
            # ChemRxiv uses 'itemHits' which contains objects with an 'item' property
            item_hits = response.get("itemHits", [])
            for item_hit in item_hits:
                if "item" in item_hit:
                    item = item_hit["item"]
                    papers.append(self._parse_paper(item))
            
            # Calculate pagination info
            next_offset = params.offset + len(papers)
            has_next = next_offset < total_results
            
            return QueryResult(
                papers=papers,
                total_results=total_results,
                next_offset=next_offset if has_next else None,
                has_next=has_next,
            )
            
        except ChemRxivAPIError as e:
            self.logger.error(f"Error searching papers: {str(e)}")
            # Return empty result on error
            return QueryResult(papers=[], has_next=False)
    
    @cached(ttl=3600, prefix="chemrxiv_search")
    async def search_papers(self, 
                           query_or_params, 
                           page=1, 
                           per_page=10,
                           **kwargs) -> QueryResult:
        """
        Search for papers in ChemRxiv.
        
        Args:
            query_or_params: Either a search string or a QueryParams object
            page: Page number for pagination (1-indexed)
            per_page: Number of results per page
            **kwargs: Additional search parameters (keywords, year, author, venue, etc.)
        
        Returns:
            QueryResult containing papers and pagination info
        """
        # Convert string query to QueryParams
        if isinstance(query_or_params, str):
            from common.models import QueryParams
            params = QueryParams(
                keywords=query_or_params,
                offset=(page - 1) * per_page,
                limit=per_page,
                **kwargs
            )
        else:
            params = query_or_params
        
        # Skip caching if disabled
        if self._disable_cache:
            return await self._search_papers_impl(params)
        else:
            return await self._search_papers_cached(params)
    
    async def get_paper_by_id(self, paper_id: str) -> Optional[PaperMetadata]:
        """
        Retrieve a paper by its ChemRxiv ID.
        
        Args:
            paper_id: ChemRxiv paper ID
            
        Returns:
            Paper metadata if found, None otherwise
        """
        # Skip caching if disabled
        if self._disable_cache:
            return await self._get_paper_by_id_impl(paper_id)
        else:
            return await self._get_paper_by_id_cached(paper_id)
    
    @cached(ttl=86400, prefix="chemrxiv_paper")  # Cache for 1 day
    async def _get_paper_by_id_cached(self, paper_id: str) -> Optional[PaperMetadata]:
        """Cached implementation of get_paper_by_id."""
        return await self._get_paper_by_id_impl(paper_id)
    
    async def _get_paper_by_id_impl(self, paper_id: str) -> Optional[PaperMetadata]:
        """Internal implementation of get_paper_by_id."""
        try:
            # Request the specific paper by ID
            response = await self._make_request(f"/items/{paper_id}")
            
            # The response for a single item might be the item directly
            # or it might be in a different format
            if "item" in response:
                # If the response has an 'item' field, use that
                return self._parse_paper(response["item"])
            else:
                # Otherwise try to parse the response directly
                return self._parse_paper(response)
            
        except ChemRxivAPIError as e:
            self.logger.error(f"Error getting paper by ID: {str(e)}")
            return None
    
    async def get_paper_by_doi(self, doi: str) -> Optional[PaperMetadata]:
        """
        Retrieve a paper by its DOI.
        
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
    
    @cached(ttl=86400, prefix="chemrxiv_paper_doi")  # Cache for 1 day
    async def _get_paper_by_doi_cached(self, doi: str) -> Optional[PaperMetadata]:
        """Cached implementation of get_paper_by_doi."""
        return await self._get_paper_by_doi_impl(doi)
    
    async def _get_paper_by_doi_impl(self, doi: str) -> Optional[PaperMetadata]:
        """Internal implementation of get_paper_by_doi."""
        try:
            # Search for papers with the given DOI
            encoded_doi = quote(doi, safe="")
            search_params = {
                "term": f"doi:{encoded_doi}",
                "limit": 1
            }
            
            response = await self._make_request("/items", search_params)
            
            # Check if any papers were found - ChemRxiv uses 'itemHits'
            item_hits = response.get("itemHits", [])
            if not item_hits:
                return None
            
            # Return the first matching paper
            if "item" in item_hits[0]:
                return self._parse_paper(item_hits[0]["item"])
            
            return None
            
        except ChemRxivAPIError as e:
            self.logger.error(f"Error getting paper by DOI: {str(e)}")
            return None
    
    async def get_citations(self, paper_id: str, limit: int = 100) -> List[PaperMetadata]:
        """
        Get papers that cite the specified paper.
        
        Note: ChemRxiv API might not provide direct citation data.
        If not available, this method returns an empty list.
        
        Args:
            paper_id: ChemRxiv paper ID
            limit: Maximum number of citations to retrieve
            
        Returns:
            List of papers citing the specified paper
        """
        try:
            # Attempt to get citations if the API supports it
            response = await self._make_request(f"/items/{paper_id}/citations", {"limit": limit})
            
            # Parse citations - ChemRxiv uses 'itemHits'
            citations = []
            item_hits = response.get("itemHits", [])
            for item_hit in item_hits:
                if "item" in item_hit:
                    citations.append(self._parse_paper(item_hit["item"]))
            
            return citations
            
        except ChemRxivAPIError as e:
            # The API might not support citations, so log and return empty list
            self.logger.info(f"Unable to retrieve citations: {str(e)}")
            return []
    
    async def get_references(self, paper_id: str, limit: int = 100) -> List[PaperMetadata]:
        """
        Get papers referenced by the specified paper.
        
        Note: ChemRxiv API might not provide direct reference data.
        If not available, this method returns an empty list.
        
        Args:
            paper_id: ChemRxiv paper ID
            limit: Maximum number of references to retrieve
            
        Returns:
            List of papers referenced by the specified paper
        """
        try:
            # Attempt to get references if the API supports it
            response = await self._make_request(f"/items/{paper_id}/references", {"limit": limit})
            
            # Parse references - ChemRxiv uses 'itemHits'
            references = []
            item_hits = response.get("itemHits", [])
            for item_hit in item_hits:
                if "item" in item_hit:
                    references.append(self._parse_paper(item_hit["item"]))
            
            return references
            
        except ChemRxivAPIError as e:
            # The API might not support references, so log and return empty list
            self.logger.info(f"Unable to retrieve references: {str(e)}")
            return []
    
    def source_name(self) -> str:
        """
        Get the name of this literature source.
        
        Returns:
            Name of the source
        """
        return "ChemRxiv" 