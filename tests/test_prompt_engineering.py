"""
Tests for prompt engineering module.
"""

import pytest
from unittest.mock import Mock, patch
from src.prompt_engineering import PromptEngineer, QueryComponent, ResearchResponse
from src.search.paper import Paper

# Test data
SAMPLE_QUERY = "What are the latest developments in single-cell RNA sequencing analysis methods for cancer research?"

SAMPLE_QUERY_BREAKDOWN = """
- Single-cell RNA sequencing analysis
  * Data preprocessing and quality control
  * Normalization techniques
  * Dimensionality reduction
- Cancer research applications
  * Tumor heterogeneity analysis
  * Drug response prediction
  * Biomarker identification
"""

SAMPLE_INITIAL_RESPONSE = """
Single-cell RNA sequencing (scRNA-seq) has revolutionized our understanding of cellular heterogeneity in cancer...
"""

SAMPLE_PAPERS = [
    Paper(
        title="Advanced scRNA-seq Analysis Methods",
        abstract="This paper presents novel approaches to single-cell RNA sequencing analysis...",
        authors=["John Smith", "Jane Doe"],
        year=2023,
        doi="10.1234/example",
        url="https://example.com/paper1",
        source_api="tavily",
        venue="Nature Methods"
    ),
    Paper(
        title="Cancer Research Using scRNA-seq",
        abstract="We demonstrate applications of scRNA-seq in cancer research...",
        authors=["Alice Johnson", "Bob Wilson"],
        year=2023,
        doi="10.1234/example2",
        url="https://example.com/paper2",
        source_api="tavily",
        venue="Cancer Cell"
    )
]

SAMPLE_FINAL_RESPONSE = """
Final Summary:
Recent developments in single-cell RNA sequencing analysis have significantly advanced our understanding of cancer...

Citations:
[1] J. Smith and J. Doe, "Advanced scRNA-seq Analysis Methods," Nature Methods, 2023.
[2] A. Johnson and B. Wilson, "Cancer Research Using scRNA-seq," Cancer Cell, 2023.
"""

@pytest.fixture
def mock_ollama_client():
    """Create a mock Ollama client."""
    mock = Mock()
    mock.generate.side_effect = [
        SAMPLE_QUERY_BREAKDOWN,
        SAMPLE_INITIAL_RESPONSE,
        SAMPLE_FINAL_RESPONSE
    ]
    return mock

@pytest.fixture
def prompt_engineer(mock_ollama_client):
    """Create a PromptEngineer instance with mock Ollama client."""
    return PromptEngineer(ollama_client=mock_ollama_client)

def test_process_query(prompt_engineer):
    """Test query processing."""
    components = prompt_engineer.process_query(SAMPLE_QUERY)
    
    assert len(components) == 2
    assert components[0].topic == "Single-cell RNA sequencing analysis"
    assert len(components[0].subtopics) == 3
    assert "Data preprocessing and quality control" in components[0].subtopics
    
    assert components[1].topic == "Cancer research applications"
    assert len(components[1].subtopics) == 3
    assert "Tumor heterogeneity analysis" in components[1].subtopics

def test_generate_initial_response(prompt_engineer):
    """Test initial response generation."""
    components = [
        QueryComponent(
            topic="Single-cell RNA sequencing analysis",
            subtopics=["Data preprocessing", "Normalization"]
        ),
        QueryComponent(
            topic="Cancer research applications",
            subtopics=["Tumor heterogeneity"]
        )
    ]
    
    response = prompt_engineer.generate_initial_response(SAMPLE_QUERY, components)
    assert response == SAMPLE_INITIAL_RESPONSE
    assert "scRNA-seq" in response

def test_generate_search_queries(prompt_engineer):
    """Test search query generation."""
    components = [
        QueryComponent(
            topic="Single-cell RNA sequencing analysis",
            subtopics=["Data preprocessing", "Normalization"]
        )
    ]
    
    queries = prompt_engineer.generate_search_queries(components)
    
    assert len(queries) == 3
    assert "Single-cell RNA sequencing analysis" in queries
    assert "Single-cell RNA sequencing analysis Data preprocessing" in queries
    assert "Single-cell RNA sequencing analysis Normalization" in queries

def test_integrate_literature(prompt_engineer):
    """Test literature integration."""
    response = prompt_engineer.integrate_literature(
        initial_response=SAMPLE_INITIAL_RESPONSE,
        papers=SAMPLE_PAPERS
    )
    
    assert isinstance(response, ResearchResponse)
    assert response.initial_response == SAMPLE_INITIAL_RESPONSE
    assert response.papers == SAMPLE_PAPERS
    assert "Recent developments" in response.final_summary
    assert len(response.citations) == 2
    assert any("Smith" in cite for cite in response.citations)
    assert any("Johnson" in cite for cite in response.citations)

def test_empty_papers_integration(prompt_engineer):
    """Test literature integration with no papers."""
    response = prompt_engineer.integrate_literature(
        initial_response=SAMPLE_INITIAL_RESPONSE,
        papers=[]
    )
    
    assert isinstance(response, ResearchResponse)
    assert response.initial_response == SAMPLE_INITIAL_RESPONSE
    assert len(response.papers) == 0

def test_query_component_creation():
    """Test QueryComponent creation and validation."""
    component = QueryComponent(
        topic="Test Topic",
        subtopics=["Sub 1", "Sub 2"],
        year_range=(2020, 2023)
    )
    
    assert component.topic == "Test Topic"
    assert len(component.subtopics) == 2
    assert component.year_range == (2020, 2023)

def test_research_response_creation():
    """Test ResearchResponse creation and validation."""
    response = ResearchResponse(
        initial_response="Initial",
        papers=SAMPLE_PAPERS,
        final_summary="Final",
        citations=["Citation 1", "Citation 2"]
    )
    
    assert response.initial_response == "Initial"
    assert len(response.papers) == 2
    assert response.final_summary == "Final"
    assert len(response.citations) == 2 