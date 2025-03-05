"""
Tests for Tavily API client.
"""

import pytest
import responses
from src.search.tavily import TavilyAPI
from src.search.paper import Paper

# Mock response data
TAVILY_RESPONSE = {
    "results": [{
        "title": "Single-cell RNA sequencing analysis: A comprehensive review",
        "url": "https://example.com/paper1",
        "content": "Published in 2023 by John Smith, Jane Doe. This comprehensive review discusses...",
        "snippet": "A comprehensive review of single-cell RNA sequencing analysis methods.",
        "domain": "nature.com"
    }, {
        "title": "Advanced methods in scRNA-seq data processing",
        "url": "https://example.com/paper2",
        "content": "Authors: Alice Johnson, Bob Wilson (2022). This paper presents...",
        "snippet": "Novel computational methods for processing scRNA-seq data.",
        "domain": "sciencedirect.com"
    }]
}

@pytest.fixture
def tavily_api():
    """Create a TavilyAPI instance with test API key."""
    return TavilyAPI(api_key="test-api-key")

@responses.activate
def test_search_success(tavily_api):
    """Test successful paper search."""
    # Mock API response
    responses.add(
        responses.POST,
        "https://api.tavily.com/search",
        json=TAVILY_RESPONSE,
        status=200
    )
    
    results = tavily_api.search("single cell RNA sequencing", limit=2)
    
    assert len(results) == 2
    assert isinstance(results[0], Paper)
    assert results[0].title == "Single-cell RNA sequencing analysis: A comprehensive review"
    assert results[0].year == 2023
    assert "John Smith" in results[0].authors
    assert "Jane Doe" in results[0].authors
    assert results[0].venue == "nature.com"

@responses.activate
def test_search_no_results(tavily_api):
    """Test search with no results."""
    responses.add(
        responses.POST,
        "https://api.tavily.com/search",
        json={"results": []},
        status=200
    )
    
    results = tavily_api.search("nonexistent topic")
    assert len(results) == 0

@responses.activate
def test_search_api_error(tavily_api):
    """Test handling of API errors."""
    responses.add(
        responses.POST,
        "https://api.tavily.com/search",
        status=500
    )
    
    results = tavily_api.search("test query")
    assert len(results) == 0

def test_no_api_key():
    """Test behavior when no API key is provided."""
    api = TavilyAPI()
    results = api.search("test query")
    assert len(results) == 0

@responses.activate
def test_search_with_year_range(tavily_api):
    """Test search with year range filter."""
    responses.add(
        responses.POST,
        "https://api.tavily.com/search",
        json=TAVILY_RESPONSE,
        status=200
    )
    
    results = tavily_api.search("test query", year_range=(2020, 2023))
    assert len(results) > 0
    # Verify the year range was added to the query
    request = responses.calls[0].request
    assert "2020" in request.body.decode()
    assert "2023" in request.body.decode()

@responses.activate
def test_rate_limiting(tavily_api):
    """Test rate limiting behavior."""
    responses.add(
        responses.POST,
        "https://api.tavily.com/search",
        json=TAVILY_RESPONSE,
        status=200
    )
    
    import time
    start_time = time.time()
    
    # Make two quick requests
    tavily_api.search("query 1")
    tavily_api.search("query 2")
    
    elapsed = time.time() - start_time
    # Should take at least 1 second due to rate limiting
    assert elapsed >= 1.0

@responses.activate
def test_authentication_header(tavily_api):
    """Test that authentication header is set correctly."""
    responses.add(
        responses.POST,
        "https://api.tavily.com/search",
        json=TAVILY_RESPONSE,
        status=200
    )
    
    tavily_api.search("test query")
    
    # Verify the auth header was set
    request = responses.calls[0].request
    assert request.headers["Authorization"] == "Bearer test-api-key" 