"""
Tests for literature search components.
"""

import pytest
import responses
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.search.literature_manager import LiteratureManager
from src.search.paper import Paper

# Test data
TAVILY_RESPONSE = {
    "results": [{
        "title": "Test Paper 1",
        "url": "https://example.com/paper1",
        "content": "Published in 2023 by John Doe, Jane Smith. This paper discusses...",
        "snippet": "This is a test paper about RNA sequencing.",
        "domain": "nature.com"
    }, {
        "title": "Test Paper 2",
        "url": "https://example.com/paper2",
        "content": "Authors: Alice Johnson, Bob Wilson (2022). Another paper about...",
        "snippet": "Another test paper about sequencing.",
        "domain": "sciencedirect.com"
    }]
}

@pytest.fixture
def literature_manager():
    """Create a LiteratureManager instance."""
    return LiteratureManager()

def test_paper_hash():
    """Test paper hash generation."""
    paper1 = Paper(
        title="Test Paper",
        abstract="This is a test paper.",
        authors=["John Doe", "Jane Smith"],
        year=2023,
        doi=None,
        url="https://example.com/paper",
        source_api="tavily",
        venue="Test Journal"
    )
    paper2 = Paper(
        title="Test Paper",
        abstract="Different abstract",
        authors=["John Doe", "Jane Smith"],
        year=2023,
        doi=None,
        url="https://example.com/paper",
        source_api="tavily",
        venue="Different Journal"
    )
    paper3 = Paper(
        title="Different Title",
        abstract="This is a test paper.",
        authors=["John Doe", "Jane Smith"],
        year=2023,
        doi=None,
        url="https://example.com/paper",
        source_api="tavily",
        venue="Test Journal"
    )
    
    # Same title and authors should have same hash
    assert paper1.get_hash() == paper2.get_hash()
    # Different title should have different hash
    assert paper1.get_hash() != paper3.get_hash()

@responses.activate
def test_literature_manager_search(literature_manager):
    """Test literature manager search with Tavily API."""
    # Mock Tavily API response
    responses.add(
        responses.POST,
        "https://api.tavily.com/search",
        json=TAVILY_RESPONSE,
        status=200
    )
    
    results = literature_manager.search("test query", max_papers=2)
    
    assert len(results) == 2
    assert isinstance(results[0], Paper)
    assert results[0].title == "Test Paper 1"
    assert "John Doe" in results[0].authors
    assert results[0].venue == "nature.com"

@responses.activate
def test_literature_manager_deduplication(literature_manager):
    """Test deduplication of search results."""
    # Create response with duplicate papers
    duplicate_response = {
        "results": [
            TAVILY_RESPONSE["results"][0],  # First paper
            TAVILY_RESPONSE["results"][0],  # Same paper again
            TAVILY_RESPONSE["results"][1]   # Different paper
        ]
    }
    
    responses.add(
        responses.POST,
        "https://api.tavily.com/search",
        json=duplicate_response,
        status=200
    )
    
    results = literature_manager.search("test query", max_papers=3)
    
    # Should only return 2 papers after deduplication
    assert len(results) == 2

def test_literature_manager_statistics(literature_manager):
    """Test literature manager statistics."""
    # Perform some searches
    with responses.RequestsMock() as rsps:
        rsps.add(
            responses.POST,
            "https://api.tavily.com/search",
            json=TAVILY_RESPONSE,
            status=200
        )
        
        literature_manager.search("test query")
    
    stats = literature_manager.get_search_statistics()
    assert stats["total_searches"] == 1
    assert stats["total_unique_papers"] > 0
    assert stats["searches_by_api"]["tavily"] == 1 