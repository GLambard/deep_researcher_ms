"""
OpenAlex API adapter for retrieving scholarly literature.

This module implements an adapter for the OpenAlex API, which provides access
to a vast database of scholarly documents, authors, sources, institutions,
and more. The adapter handles request formatting, response parsing, error
handling, rate limiting, and caching.

The OpenAlex API (https://api.openalex.org/) offers:
- Work search with filtering options
- Paper metadata retrieval by ID or DOI
- Citation and reference data
- Author information
- Source and institution data

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
from ..utils.rate_limiter import get_rate_limiter


class OpenAlexAPIError(Exception):
    """
    Exception raised for OpenAlex API errors.
    
    This custom exception encapsulates errors returned by the API,
    such as rate limiting, authentication failures, or invalid requests.
    """
    pass


class OpenAlexAdapter(LiteratureAdapter):
    """
    Adapter for the OpenAlex API.
    
    This adapter implements the LiteratureAdapter interface and provides methods
    to search and retrieve paper metadata from the OpenAlex API. It handles:
    
    1. Request formatting according to API specifications
    2. Response parsing into our common data model
    3. Error handling with retries for transient failures
    4. Rate limiting to comply with API usage policies
    5. Caching of responses to reduce duplicate requests
    
    The adapter uses asynchronous HTTP requests for better performance and
    implements caching to reduce redundant API calls.
    """
    
    # Base URL for the OpenAlex API
    _BASE_URL = "https://api.openalex.org"
    
    def __init__(
        self, 
        email: Optional[str] = None,
        timeout: int = 30,
        disable_cache: bool = False,
    ):
        """
        Initialize the OpenAlex adapter.
        
        Args:
            email: Optional email for polite pool. OpenAlex recommends adding your
                  email to all API requests for better service.
            timeout: Request timeout in seconds. Default is 30s which balances
                    reliability with responsiveness.
            disable_cache: Whether to disable caching. This is primarily useful
                          for testing or when fresh data is always required.
        """
        self._email = email
        self._timeout = timeout
        self._logger = logging.getLogger(__name__)
        self._rate_limiter = get_rate_limiter("openalex")
        self._disable_cache = disable_cache
        
    async def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make a request to the OpenAlex API.
        
        Args:
            endpoint: API endpoint, starting with "/"
            params: Query parameters
            
        Returns:
            Parsed JSON response
            
        Raises:
            OpenAlexAPIError: If the API returns an error
        """
        url = f"{self._BASE_URL}{endpoint}"
        
        # Initialize parameters if None
        if params is None:
            params = {}
            
        # Add email for polite pool if provided
        if self._email:
            params["mailto"] = self._email
            
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
                    # Use the rate limiter
                    await self._rate_limiter.acquire()
                    
                    async with aiohttp.ClientSession() as session:
                        async with session.get(
                            url, 
                            headers={"Accept": "application/json"}, 
                            timeout=self._timeout
                        ) as response:
                            # Check for errors
                            if response.status != 200:
                                error_text = await response.text()
                                raise OpenAlexAPIError(
                                    f"API error {response.status}: {error_text}"
                                )
                            
                            # Parse JSON response
                            return await response.json()
                            
        except RetryError as e:
            raise OpenAlexAPIError(f"Max retries exceeded: {str(e.last_attempt.exception())}")
        except aiohttp.ClientError as e:
            raise OpenAlexAPIError(f"HTTP error: {str(e)}")
        except json.JSONDecodeError:
            raise OpenAlexAPIError("Invalid JSON response")
    
    def _parse_author(self, author_data: Dict[str, Any]) -> Author:
        """
        Parse author data from API response.
        
        Args:
            author_data: Author data from API response
            
        Returns:
            Author object
        """
        # Extract author ID from URL if available
        author_id = None
        if "id" in author_data:
            # OpenAlex IDs are URLs like https://openalex.org/A1234567890
            # Extract the ID part after the last slash
            author_id = author_data["id"].split("/")[-1] if author_data["id"] else None
            
        return Author(
            name=author_data.get("display_name", "Unknown"),
            id=author_id,
            url=author_data.get("id"),  # OpenAlex uses URLs as IDs which can be used directly
            affiliations=[], # OpenAlex stores affiliations at the work level, not in the author object
        )
    
    def _parse_paper(self, paper_data: Dict[str, Any]) -> PaperMetadata:
        """
        Parse paper data from API response.
        
        Args:
            paper_data: Paper data from API response
            
        Returns:
            PaperMetadata object
        """

        # Extract title
        title = paper_data.get("title", "Untitled")

        # Extract authors
        authors = []
        for authorship in paper_data.get("authorships", []):
            if "author" in authorship:
                authors.append(self._parse_author(authorship["author"]))
        
        # Extract abstract
        abstract = paper_data.get("abstract_inverted_index", {})
        if abstract:
            abstract = " ".join(abstract.keys())
        else:
            abstract = None

        # Extract DOI
        doi = paper_data.get("doi")
            
        # Extract publication date - Format: 2024-01-01
        publication_date = None
        if paper_data.get("publication_date"):
            try:
                publication_date = date_parser.parse(paper_data["publication_date"])
                publication_date = publication_date.strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                pass

        # Extract primary URL 
        url = None
        # Extract PDF URL if available
        pdf_url = None
        # Check primary_location first, then best_oa_location
        if "primary_location" in paper_data and paper_data["primary_location"]:
            if "landing_page_url" in paper_data["primary_location"] and paper_data["primary_location"]["landing_page_url"]:
                url = paper_data["primary_location"]["landing_page_url"]
            if "pdf_url" in paper_data["primary_location"] and paper_data["primary_location"]["pdf_url"]:
                pdf_url = paper_data["primary_location"]["pdf_url"]
        
        # Skip if we already have a URL
        if (not url or not pdf_url) and "best_oa_location" in paper_data and paper_data["best_oa_location"]:
            if not url and "landing_page_url" in paper_data["best_oa_location"] and paper_data["best_oa_location"]["landing_page_url"]:
                    url = paper_data["best_oa_location"]["landing_page_url"]
            if not pdf_url and "pdf_url" in paper_data["best_oa_location"] and paper_data["best_oa_location"]["pdf_url"]:
                    pdf_url = paper_data["best_oa_location"]["pdf_url"]

        if not pdf_url and "locations" in paper_data and paper_data["locations"]:
            for location in paper_data["locations"]:
                if "pdf_url" in location and location["pdf_url"]:
                    pdf_url = location["pdf_url"]
                    break
        
        # Extract year
        year = None
        if paper_data.get("publication_year"):
            year = paper_data["publication_year"]

        # Extract venue/source information
        venue = None
        if "primary_location" in paper_data and paper_data["primary_location"]:
            if "source" in paper_data["primary_location"] and paper_data["primary_location"]["source"]:
                venue = paper_data["primary_location"]["source"].get("display_name")
        
        # Extract citation count
        citation_count = None
        if paper_data.get("cited_by_count"):
            citation_count = paper_data["cited_by_count"]

        # Extract fields of study (topics or concepts in OpenAlex)
        fields_of_study = []
        for topic in paper_data.get("topics", []):
            if "display_name" in topic:
                fields_of_study.append(topic["display_name"])

        # Extract publication types
        publication_types = []
        if paper_data.get("type"):
            publication_types.append(paper_data["type"])

        # Extract external IDs
        external_ids = {}
        if "ids" in paper_data:
            # Convert OpenAlex IDs to our format
            ids = paper_data["ids"]
            if "doi" in ids:
                external_ids["doi"] = ids["doi"].replace("https://doi.org/", "")
            if "pmid" in ids:
                external_ids["pmid"] = ids["pmid"].replace("https://pubmed.ncbi.nlm.nih.gov/", "").rstrip("/")
            if "pmcid" in ids:
                external_ids["pmcid"] = ids["pmcid"].replace("https://www.ncbi.nlm.nih.gov/pmc/articles/", "").rstrip("/")
            if "mag" in ids:
                external_ids["mag"] = ids["mag"]
            # Add openalex ID itself
            if "openalex" in ids:
                external_ids["openalex"] = ids["openalex"].replace("https://openalex.org/", "")
                
        # Create PaperMetadata object
        return PaperMetadata(
            title=title,
            authors=authors,
            abstract=abstract, 
            doi=doi,
            source="OpenAlex",
            url=url,
            publication_date=publication_date,
            pdf_url=pdf_url,
            year=year,
            venue=venue,
            citation_count=citation_count,
            fields_of_study=fields_of_study,
            publication_types=publication_types,
            external_ids=external_ids,
            raw_data=paper_data,
        )
        
    def _build_search_query(self, params: QueryParams) -> Dict[str, Any]:
        """
        Build a search query parameters dictionary from QueryParams.
        
        Args:
            params: Query parameters
            
        Returns:
            Dictionary of search parameters for the OpenAlex API
        """
        query_params = {}
        
        # Add keywords as search query
        if params.keywords:
            query_params["search"] = params.keywords
            
        # Add filters
        filters = []
        
        # Add year filter
        if params.year:
            # Handle year range or specific year
            if isinstance(params.year, str):
                filters.append(f"publication_year:{params.year}")
            else:
                raise ValueError(f"Invalid year: {params.year}. It must be a string.")

        # TODO: Add fields of study filter
            
        # Add open access filter
        if params.open_access_only:
            filters.append("is_oa:true")
            
        # Combine filters
        if filters:
            query_params["filter"] = ",".join(filters)
            
        # Add pagination
        query_params["per_page"] = params.limit
        query_params["page"] = (params.offset // params.limit) + 1
            
        # Add sort
        if params.sort_by:
            # Map our sort options to OpenAlex sort options
            sort_mapping = {
                "relevance": "relevance_score:desc",
                "date": "publication_date:desc",
                "citations": "cited_by_count:desc"
            }
            
            if params.sort_by in sort_mapping:
                query_params["sort"] = sort_mapping[params.sort_by]
            else:
                query_params["sort"] = params.sort_by
        else:
            # Default sort by relevance
            query_params["sort"] = "relevance_score:desc"
            
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
    
    @cached(ttl=3600, prefix="openalex_search")
    async def _search_papers_cached(self, params: QueryParams) -> QueryResult:
        """Cached implementation of search_papers."""
        return await self._search_papers_impl(params)
    
    async def _search_papers_impl(self, params: QueryParams) -> QueryResult:
        """Internal implementation of search_papers."""
        api_params = self._build_search_query(params)
        
        try:
            response = await self._make_request("/works", api_params)
            
            # Parse results
            papers = [self._parse_paper(paper) for paper in response.get("results", [])]
            
            # Extract pagination info
            total = response.get("meta", {}).get("count", 0)
            next_offset = params.offset + len(papers)
            has_next = next_offset < total
            
            return QueryResult(
                papers=papers,
                total_results=total,
                next_offset=next_offset if has_next else None,
                has_next=has_next,
            )
            
        except OpenAlexAPIError as e:
            self._logger.error(f"Error searching papers: {str(e)}")
            # Return empty result on error
            return QueryResult(papers=[], has_next=False)

    async def get_paper_by_id(self, paper_id: str) -> Optional[PaperMetadata]:
        """
        Retrieve a paper by its OpenAlex ID.
        
        Args:
            paper_id: OpenAlex paper ID (can be the full URL or just the ID part)
            
        Returns:
            Paper metadata if found, None otherwise
        """
        # Skip caching if disabled
        if self._disable_cache:
            return await self._get_paper_by_id_impl(paper_id)
        else:
            return await self._get_paper_by_id_cached(paper_id)
    
    @cached(ttl=86400, prefix="openalex_paper")  # Cache for 1 day
    async def _get_paper_by_id_cached(self, paper_id: str) -> Optional[PaperMetadata]:
        """Cached implementation of get_paper_by_id."""
        return await self._get_paper_by_id_impl(paper_id)
    
    async def _get_paper_by_id_impl(self, paper_id: str) -> Optional[PaperMetadata]:
        """Internal implementation of get_paper_by_id."""
        # If the paper_id is a full URL, extract just the ID part
        if paper_id.startswith("https://openalex.org/"):
            paper_id = paper_id.replace("https://openalex.org/", "")
            
        try:
            response = await self._make_request(f"/works/{paper_id}")
            return self._parse_paper(response)
        except OpenAlexAPIError as e:
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
    
    @cached(ttl=86400, prefix="openalex_paper_doi")  # Cache for 1 day
    async def _get_paper_by_doi_cached(self, doi: str) -> Optional[PaperMetadata]:
        """Cached implementation of get_paper_by_doi."""
        return await self._get_paper_by_doi_impl(doi)
    
    async def _get_paper_by_doi_impl(self, doi: str) -> Optional[PaperMetadata]:
        """Internal implementation of get_paper_by_doi."""
        # URL encode the DOI
        encoded_doi = quote(doi, safe="")
        
        try:
            # OpenAlex can lookup by DOI directly
            response = await self._make_request(f"/works/doi:{encoded_doi}")
            return self._parse_paper(response)
        except OpenAlexAPIError as e:
            self._logger.error(f"Error getting paper by DOI: {str(e)}")
            return None
    
    async def get_citations(self, paper_id: str, limit: int = 100) -> List[PaperMetadata]:
        """
        Get papers that cite the specified paper.
        
        Args:
            paper_id: OpenAlex paper ID (can be the full URL or just the ID part)
            limit: Maximum number of citations to retrieve
            
        Returns:
            List of papers citing the specified paper
        """
        # Skip caching if disabled
        if self._disable_cache:
            return await self._get_citations_impl(paper_id, limit)
        else:
            return await self._get_citations_cached(paper_id, limit)
    
    @cached(ttl=86400, prefix="openalex_citations")  # Cache for 1 day
    async def _get_citations_cached(self, paper_id: str, limit: int = 100) -> List[PaperMetadata]:
        """Cached implementation of get_citations."""
        return await self._get_citations_impl(paper_id, limit)
    
    async def _get_citations_impl(self, paper_id: str, limit: int = 100) -> List[PaperMetadata]:
        """Internal implementation of get_citations."""
        # If the paper_id is a full URL, extract just the ID part
        if paper_id.startswith("https://openalex.org/"):
            paper_id = paper_id.replace("https://openalex.org/", "")
            
        try:
            # In OpenAlex, you get citing papers by using the filter cites:paper_id
            api_params = {
                "filter": f"cites:{paper_id}",
                "per_page": min(limit, 200),  # OpenAlex limit is 200 per page
                "page": 1
            }
            
            response = await self._make_request("/works", api_params)
            
            # Parse results
            papers = [self._parse_paper(paper) for paper in response.get("results", [])]
            
            return papers
            
        except OpenAlexAPIError as e:
            self._logger.error(f"Error getting citations: {str(e)}")
            return []
    
    async def get_references(self, paper_id: str, limit: int = 100) -> List[PaperMetadata]:
        """
        Get papers referenced by the specified paper.
        
        Args:
            paper_id: OpenAlex paper ID (can be the full URL or just the ID part)
            limit: Maximum number of references to retrieve
            
        Returns:
            List of papers referenced by the specified paper
        """
        # Skip caching if disabled
        if self._disable_cache:
            return await self._get_references_impl(paper_id, limit)
        else:
            return await self._get_references_cached(paper_id, limit)
    
    @cached(ttl=86400, prefix="openalex_references")  # Cache for 1 day
    async def _get_references_cached(self, paper_id: str, limit: int = 100) -> List[PaperMetadata]:
        """Cached implementation of get_references."""
        return await self._get_references_impl(paper_id, limit)
    
    async def _get_references_impl(self, paper_id: str, limit: int = 100) -> List[PaperMetadata]:
        """Internal implementation of get_references."""
        # If the paper_id is a full URL, extract just the ID part
        if paper_id.startswith("https://openalex.org/"):
            paper_id = paper_id.replace("https://openalex.org/", "")
            
        try:
            # First, get the paper to access its referenced_works
            paper_response = await self._make_request(f"/works/{paper_id}")
            
            # Extract referenced work IDs
            referenced_works = paper_response.get("referenced_works", [])
            
            # Limit to the requested number
            referenced_works = referenced_works[:limit]
            
            # If no references, return empty list
            if not referenced_works:
                return []
                
            # Get the details for each referenced work
            # Create tasks for concurrent execution
            tasks = []
            for ref_id in referenced_works:
                # Extract just the ID part if it's a URL
                if ref_id.startswith("https://openalex.org/"):
                    ref_id = ref_id.replace("https://openalex.org/", "")
                tasks.append(self._get_paper_by_id_impl(ref_id))
                
            # Execute all tasks concurrently for better performance
            papers = await asyncio.gather(*tasks)
            
            # Filter out None values (papers that couldn't be retrieved)
            return [paper for paper in papers if paper is not None]
            
        except OpenAlexAPIError as e:
            self._logger.error(f"Error getting references: {str(e)}")
            return []
    
    async def get_paper_details_with_citations_references(
        self, paper_id: str, citations_limit: int = 3, references_limit: int = 3
    ) -> Dict[str, Any]:
        """
        Retrieve a paper's details, citations, and references in a single optimized call.
        
        This method batches multiple API requests together to reduce the number of
        separate HTTP calls, improving performance and reducing the likelihood
        of hitting API rate limits.
        
        Args:
            paper_id: OpenAlex paper ID
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
    
    @cached(ttl=86400, prefix="openalex_paper_with_refs_cits")  # Cache for 1 day
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
            
        except OpenAlexAPIError as e:
            self._logger.error(f"Error in batched paper details request: {str(e)}")
            # Return partial results if available
            return {
                "paper": None,
                "citations": [],
                "references": []
            }
    
    def source_name(self) -> str:
        """
        Get the name of this literature source.
        
        Returns:
            Name of the source
        """
        return "OpenAlex" 