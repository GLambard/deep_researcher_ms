"""
Semantic Scholar API adapter for retrieving scholarly literature.

This module implements an adapter for the Semantic Scholar Academic Graph API,
which provides access to a vast database of academic papers, authors, and citations.
The adapter handles authentication, request formatting, response parsing, error
handling, rate limiting, and caching.

The Semantic Scholar API (https://api.semanticscholar.org/) offers:
- Paper search with filtering options
- Paper metadata retrieval by ID or DOI
- Citation and reference data
- Author information

This adapter normalizes the API responses to our common data model.
"""
import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
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

from ..common.adapter_base import LiteratureAdapter
from ..common.models import Author, PaperMetadata, QueryParams, QueryResult
from ..utils.cache import cached
from ..utils.rate_limiter import get_rate_limiter, rate_limited


class SemanticScholarAPIError(Exception):
    """
    Exception raised for Semantic Scholar API errors.
    
    This custom exception encapsulates errors returned by the API,
    such as rate limiting, authentication failures, or invalid requests.
    """
    pass


class SemanticScholarAdapter(LiteratureAdapter):
    """
    Adapter for the Semantic Scholar Academic Graph API.
    
    This adapter implements the LiteratureAdapter interface and provides methods
    to search and retrieve paper metadata from the Semantic Scholar API. It
    handles the complexities of API interactions including:
    
    1. Authentication with API keys
    2. Request formatting according to API specifications
    3. Response parsing into our common data model
    4. Error handling with retries for transient failures
    5. Rate limiting to comply with API usage policies
    6. Caching of responses to reduce duplicate requests
    
    The adapter uses asynchronous HTTP requests for better performance and
    implements caching to reduce redundant API calls.
    """
    
    # Base URL for the Semantic Scholar API
    # Using a class constant allows for easy changes if the API endpoint changes
    _BASE_URL = "https://api.semanticscholar.org/graph/v1"
    
    # Default fields to request from the API
    # These are the most commonly needed fields which balance completeness
    # with request size/performance. Additional fields can be specified
    # in the constructor if needed.
    
    # _DEFAULT_FIELDS = (
    #     "paperId,externalIds,url,title,authors,abstract,venue,"
    #     "referenceCount,citationCount," 
    #     "isOpenAccess,openAccessPdf,fieldsOfStudy,s2FieldsOfStudy,"
    #     "publicationTypes,publicationDate,journal,citationStyles,"
    #     "citations,references"
    # )

    _DEFAULT_FIELDS = (
        "paperId,externalIds,url,title,authors,abstract,venue,"
        "referenceCount,citationCount," 
        "isOpenAccess,fieldsOfStudy,"
        "publicationTypes,publicationDate,journal"
    )

    _DEFAULT_FIELDS_OF_STUDY = (
        "Computer Science,Chemistry,Biology,"
        "Materials Science,Physics,Geology,Philosophy, Mathematics,"
        "Engineering, Environmental Science, Agricultural and Food Science"
    )
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        fields: Optional[str] = None,
        fields_of_study: Optional[str] = None,
        timeout: int = 30,
        disable_cache: bool = False,
    ):
        """
        Initialize the Semantic Scholar adapter.
        
        Args:
            api_key: Optional API key for higher rate limits. Without an API key,
                    requests are limited to 100 per 5 minutes. With an API key,
                    limits are higher and can be customized.
            fields: Comma-separated list of fields to include in responses,
                   or None to use the default fields. See API documentation for
                   available fields: https://api.semanticscholar.org/api-docs/
            timeout: Request timeout in seconds. Default is 30s which balances
                    reliability with responsiveness.
            disable_cache: Whether to disable caching. This is primarily useful
                          for testing or when fresh data is always required.
        """
        self._api_key = api_key
        self._fields = fields or self._DEFAULT_FIELDS
        self._fields_of_study = fields_of_study or self._DEFAULT_FIELDS_OF_STUDY
        self._timeout = timeout
        self._logger = logging.getLogger(__name__)
        self._rate_limiter = get_rate_limiter("semantic_scholar")
        self._disable_cache = disable_cache
        
    def _get_headers(self) -> Dict[str, str]:
        """
        Get HTTP headers for API requests.
        
        Returns:
            Dictionary of HTTP headers
        """
        headers = {
            "Accept": "application/json",
        }
        
        if self._api_key:
            headers["x-api-key"] = self._api_key
            
        return headers
    
    async def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make a request to the Semantic Scholar API.
        
        Args:
            endpoint: API endpoint, starting with "/"
            params: Query parameters
            
        Returns:
            Parsed JSON response
            
        Raises:
            SemanticScholarAPIError: If the API returns an error
        """
        url = f"{self._BASE_URL}{endpoint}"
        
        # Add the fields parameter if not already present
        if params is None:
            params = {}
        #if "fields" not in params:
        #    params["fields"] = self._fields
            
        # URL encode parameters
        if params:
            query_string = urlencode(params, doseq=True)
            url = f"{url}?{query_string}"
        
        headers = self._get_headers()
        
        try:
            # Use retrying to handle transient errors
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=1, max=10),
                retry=retry_if_exception_type((aiohttp.ClientError, TimeoutError)),
                reraise=True,
            ):
                with attempt:
                    # Use the rate limiter
                    await self._rate_limiter.acquire()
                    
                    async with aiohttp.ClientSession() as session:
                        async with session.get(
                            url, 
                            headers=headers, 
                            timeout=self._timeout
                        ) as response:
                            # Check for errors
                            if response.status != 200:
                                error_text = await response.text()
                                raise SemanticScholarAPIError(
                                    f"API error {response.status}: {error_text}"
                                )
                            
                            # Parse JSON response
                            return await response.json()
                            
        except RetryError as e:
            raise SemanticScholarAPIError(f"Max retries exceeded: {str(e.last_attempt.exception())}")
        except aiohttp.ClientError as e:
            raise SemanticScholarAPIError(f"HTTP error: {str(e)}")
        except json.JSONDecodeError:
            raise SemanticScholarAPIError("Invalid JSON response")
    
    def _parse_author(self, author_data: Dict[str, Any]) -> Author:
        """
        Parse author data from API response.
        
        Args:
            author_data: Author data from API response
            
        Returns:
            Author object
        """
        return Author(
            name=author_data.get("name", "Unknown"),
            id=author_data.get("authorId"),
            url=f"https://www.semanticscholar.org/author/{author_data.get('authorId')}" if author_data.get("authorId") else None,
            # API doesn't always include affiliations
            affiliations=author_data.get("affiliations", []),
        )
    
    def _parse_paper(self, paper_data: Dict[str, Any]) -> PaperMetadata:
        """
        Parse paper data from API response.
        
        Args:
            paper_data: Paper data from API response
            
        Returns:
            PaperMetadata object
        """
        # Parse authors
        authors = [
            self._parse_author(author)
            for author in paper_data.get("authors", [])
        ]
        
        # Extract DOI from externalIds
        doi = None
        if "externalIds" in paper_data and "DOI" in paper_data["externalIds"]:
            doi = paper_data["externalIds"]["DOI"]
            
        # Extract publication date
        publication_date = None
        if paper_data.get("publicationDate"):
            try:
                publication_date = date_parser.parse(paper_data["publicationDate"])
            except (ValueError, TypeError):
                pass
                
        # Extract PDF URL if available
        pdf_url = None
        if "openAccessPdf" in paper_data and paper_data["openAccessPdf"]:
            pdf_url = paper_data["openAccessPdf"].get("url")
        
        # Extract external IDs
        external_ids = {}
        if "externalIds" in paper_data:
            for id_type, id_value in paper_data["externalIds"].items():
                external_ids[id_type.lower()] = id_value
                
        # Create PaperMetadata object
        return PaperMetadata(
            title=paper_data.get("title", "Untitled"),
            authors=authors,
            abstract=paper_data.get("abstract"),
            doi=doi,
            source="Semantic Scholar",
            url=paper_data.get("url"),
            publication_date=publication_date,
            pdf_url=pdf_url,
            year=paper_data.get("year"),
            venue=paper_data.get("venue"),
            citation_count=paper_data.get("citationCount"),
            fields_of_study=paper_data.get("fieldsOfStudy", []),
            publication_types=paper_data.get("publicationTypes", []),
            external_ids=external_ids,
            raw_data=paper_data,
        )
        
    def _build_search_query(self, params: QueryParams) -> str:
        """
        Build a search query string from QueryParams.
        
        Args:
            params: Query parameters
            
        Returns:
            Search query string
        """
        parts = []
        
        # Add keywords
        if params.keywords:
            parts.append(params.keywords)
            
        # Add year filter
        if params.year:
            parts.append(f"year:{params.year}")
            
        # Add venue filter
        if params.venue:
            parts.append(f"venue:\"{params.venue}\"")
            
        # Add author filters
        if params.authors:
            for author in params.authors:
                parts.append(f"author:\"{author}\"")
                
        return " ".join(parts)
    
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
    
    @cached(ttl=3600, prefix="semantic_scholar_search")
    async def _search_papers_cached(self, params: QueryParams) -> QueryResult:
        """Cached implementation of search_papers."""
        return await self._search_papers_impl(params)
    
    async def _search_papers_impl(self, params: QueryParams) -> QueryResult:
        """Internal implementation of search_papers."""
        query = self._build_search_query(params)
        if not query:
            # If no search terms provided, return empty result
            return QueryResult(papers=[], total_results=0, has_next=False)
        
        api_params = {
            "query": query,
            "limit": min(params.limit, 100),  # API limit is 100
            "offset": params.offset,
            "fields": self._fields,
            "fieldsOfStudy": self._fields_of_study,
        }
        
        # Add open access filter if requested
        if params.open_access_only:
            api_params["openAccessPdf"] = "true"
            
        # Add year filter if not already in query
        if params.year and "year:" not in query:
            api_params["year"] = params.year
            
        try:
            response = await self._make_request("/paper/search", api_params)
            #print(response)
            
            # Parse results
            papers = [self._parse_paper(paper) for paper in response.get("data", [])]
            
            # Extract pagination info
            total = response.get("total", 0)
            next_offset = params.offset + len(papers)
            has_next = next_offset < total
            
            return QueryResult(
                papers=papers,
                total_results=total,
                next_offset=next_offset if has_next else None,
                has_next=has_next,
            )
            
        except SemanticScholarAPIError as e:
            self._logger.error(f"Error searching papers: {str(e)}")
            # Return empty result on error
            return QueryResult(papers=[], has_next=False)
    
    async def get_paper_by_id(self, paper_id: str) -> Optional[PaperMetadata]:
        """
        Retrieve a paper by its Semantic Scholar ID.
        
        Args:
            paper_id: Semantic Scholar paper ID
            
        Returns:
            Paper metadata if found, None otherwise
        """
        # Skip caching if disabled
        if self._disable_cache:
            return await self._get_paper_by_id_impl(paper_id)
        else:
            return await self._get_paper_by_id_cached(paper_id)
    
    @cached(ttl=86400, prefix="semantic_scholar_paper")  # Cache for 1 day
    async def _get_paper_by_id_cached(self, paper_id: str) -> Optional[PaperMetadata]:
        """Cached implementation of get_paper_by_id."""
        return await self._get_paper_by_id_impl(paper_id)
    
    async def _get_paper_by_id_impl(self, paper_id: str) -> Optional[PaperMetadata]:
        """Internal implementation of get_paper_by_id."""
        try:
            response = await self._make_request(f"/paper/{paper_id}")
            return self._parse_paper(response)
        except SemanticScholarAPIError as e:
            self._logger.error(f"Error getting paper by ID: {str(e)}")
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
    
    @cached(ttl=86400, prefix="semantic_scholar_paper_doi")  # Cache for 1 day
    async def _get_paper_by_doi_cached(self, doi: str) -> Optional[PaperMetadata]:
        """Cached implementation of get_paper_by_doi."""
        return await self._get_paper_by_doi_impl(doi)
    
    async def _get_paper_by_doi_impl(self, doi: str) -> Optional[PaperMetadata]:
        """Internal implementation of get_paper_by_doi."""
        # URL encode the DOI
        encoded_doi = quote(doi, safe="")
        
        try:
            response = await self._make_request(f"/paper/DOI:{encoded_doi}")
            return self._parse_paper(response)
        except SemanticScholarAPIError as e:
            self._logger.error(f"Error getting paper by DOI: {str(e)}")
            return None
    
    async def get_citations(self, paper_id: str, limit: int = 100) -> List[PaperMetadata]:
        """
        Get papers that cite the specified paper.
        
        Args:
            paper_id: Semantic Scholar paper ID
            limit: Maximum number of citations to retrieve
            
        Returns:
            List of papers citing the specified paper
        """
        # Skip caching if disabled
        if self._disable_cache:
            return await self._get_citations_impl(paper_id, limit)
        else:
            return await self._get_citations_cached(paper_id, limit)
    
    @cached(ttl=86400, prefix="semantic_scholar_citations")  # Cache for 1 day
    async def _get_citations_cached(self, paper_id: str, limit: int = 100) -> List[PaperMetadata]:
        """Cached implementation of get_citations."""
        return await self._get_citations_impl(paper_id, limit)
    
    async def _get_citations_impl(self, paper_id: str, limit: int = 100) -> List[PaperMetadata]:
        """Internal implementation of get_citations."""
        try:
            response = await self._make_request(
                f"/paper/{paper_id}/citations",
                params={
                    "limit": min(limit, 1000),  # API limit is 1000
                    "fields": self._fields,
                }
            )
            
            # Parse results - the citing papers are in the "citingPaper" field of each item
            papers = [
                self._parse_paper(item.get("citingPaper", {}))
                for item in response.get("data", [])
                if "citingPaper" in item
            ]
            
            return papers
            
        except SemanticScholarAPIError as e:
            self._logger.error(f"Error getting citations: {str(e)}")
            return []
    
    async def get_references(self, paper_id: str, limit: int = 100) -> List[PaperMetadata]:
        """
        Get papers referenced by the specified paper.
        
        Args:
            paper_id: Semantic Scholar paper ID
            limit: Maximum number of references to retrieve
            
        Returns:
            List of papers referenced by the specified paper
        """
        # Skip caching if disabled
        if self._disable_cache:
            return await self._get_references_impl(paper_id, limit)
        else:
            return await self._get_references_cached(paper_id, limit)
    
    @cached(ttl=86400, prefix="semantic_scholar_references")  # Cache for 1 day
    async def _get_references_cached(self, paper_id: str, limit: int = 100) -> List[PaperMetadata]:
        """Cached implementation of get_references."""
        return await self._get_references_impl(paper_id, limit)
    
    async def _get_references_impl(self, paper_id: str, limit: int = 100) -> List[PaperMetadata]:
        """Internal implementation of get_references."""
        try:
            response = await self._make_request(
                f"/paper/{paper_id}/references",
                params={
                    "limit": min(limit, 1000),  # API limit is 1000
                    "fields": self._fields,
                }
            )
            
            # Parse results - the referenced papers are in the "citedPaper" field of each item
            papers = [
                self._parse_paper(item.get("citedPaper", {}))
                for item in response.get("data", [])
                if "citedPaper" in item
            ]
            
            return papers
            
        except SemanticScholarAPIError as e:
            self._logger.error(f"Error getting references: {str(e)}")
            return []
    
    def source_name(self) -> str:
        """
        Get the name of this literature source.
        
        Returns:
            Name of the source
        """
        return "Semantic Scholar"

    async def get_paper_details_with_citations_references(
        self, paper_id: str, citations_limit: int = 3, references_limit: int = 3
    ) -> Dict[str, Any]:
        """
        Retrieve a paper's details, citations, and references in a single optimized call.
        
        This method batches multiple API requests together to reduce the number of
        separate HTTP calls, improving performance and reducing the likelihood
        of hitting API rate limits.
        
        Args:
            paper_id: Semantic Scholar paper ID
            citations_limit: Maximum number of citations to retrieve
            references_limit: Maximum number of references to retrieve
            
        Returns:
            Dictionary containing:
                'paper': PaperMetadata for the requested paper
                'citations': List of PaperMetadata for papers citing the requested paper
                'references': List of PaperMetadata for papers referenced by the requested paper
        """
        # Skip caching if disabled
        if self._disable_cache:
            return await self._get_paper_details_with_citations_references_impl(
                paper_id, citations_limit, references_limit
            )
        else:
            return await self._get_paper_details_with_citations_references_cached(
                paper_id, citations_limit, references_limit
            )
    
    @cached(ttl=86400, prefix="semantic_scholar_paper_with_refs_cits")  # Cache for 1 day
    async def _get_paper_details_with_citations_references_cached(
        self, paper_id: str, citations_limit: int = 3, references_limit: int = 3
    ) -> Dict[str, Any]:
        """Cached implementation of get_paper_details_with_citations_references."""
        return await self._get_paper_details_with_citations_references_impl(
            paper_id, citations_limit, references_limit
        )
    
    async def _get_paper_details_with_citations_references_impl(
        self, paper_id: str, citations_limit: int = 3, references_limit: int = 3
    ) -> Dict[str, Any]:
        """Internal implementation of get_paper_details_with_citations_references."""
        # Use asyncio.gather to run all three requests concurrently
        try:
            paper_task = self._get_paper_by_id_impl(paper_id)
            citations_task = self._get_citations_impl(paper_id, citations_limit)
            references_task = self._get_references_impl(paper_id, references_limit)
            
            # Wait for all tasks to complete
            paper, citations, references = await asyncio.gather(
                paper_task, citations_task, references_task
            )
            
            return {
                "paper": paper,
                "citations": citations,
                "references": references
            }
            
        except SemanticScholarAPIError as e:
            self._logger.error(f"Error in batched paper details request: {str(e)}")
            # Return partial results if available
            return {
                "paper": None,
                "citations": [],
                "references": []
            } 