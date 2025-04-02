#!/usr/bin/env python
"""
Test script for query parsing and paper filtering improvements.
This script tests the fixed implementation to ensure proper handling of scientific queries.
"""

import sys
import os

# Add the project root directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ollama_client import OllamaClient
from src.prompt_engineering_human import PromptEngineer
from src.search.paper import Paper

def test_query_validation():
    """Test the query validation and fixing functionality."""
    # Initialize clients
    ollama_client = OllamaClient()
    prompt_engineer = PromptEngineer(ollama_client)
    
    # Test cases
    test_queries = [
        '("Conformal Prediction" AND "reaction kinetics") AND ("metal',  # Unbalanced parentheses
        '(large language models AND materials science',                  # Missing closing parenthesis
        'conformal prediction or bayesian optimization',                 # Lowercase operators
        'machine learning drug discovery',                               # No operators or quotes
    ]
    
    print("\n=== TESTING QUERY VALIDATION ===")
    for i, query in enumerate(test_queries):
        fixed_query = prompt_engineer._validate_and_fix_query(query)
        print(f"\nOriginal Query {i+1}: {query}")
        print(f"Fixed Query {i+1}: {fixed_query}")

def test_research_definition():
    """Test the research question definition functionality."""
    # Initialize clients
    ollama_client = OllamaClient()
    prompt_engineer = PromptEngineer(ollama_client)
    
    # Test query
    query = '("Conformal Prediction" AND "reaction kinetics") AND ("metal catalysis")'
    
    print("\n=== TESTING RESEARCH DEFINITION ===")
    print(f"Query: {query}")
    
    # Get research definition
    research_def = prompt_engineer.define_research_question(query)
    
    # Print results
    print("\nResearch Question:", research_def.get("research_question"))
    print("\nInclusion Criteria:")
    for criterion in research_def.get("inclusion_criteria", []):
        print(f"- {criterion}")
    
    print("\nKey Terms:")
    for term in research_def.get("key_terms", []):
        print(f"- {term}")
    
    print("\nTime Frame:", research_def.get("time_frame"))

def test_paper_filtering():
    """Test the improved paper filtering functionality."""
    # Initialize clients
    ollama_client = OllamaClient()
    prompt_engineer = PromptEngineer(ollama_client)
    
    # Create test papers - mix of relevant and irrelevant
    papers = [
        Paper(
            title="Conformal Prediction for Uncertainty Quantification in Reaction Kinetics",
            abstract="This study explores the application of conformal prediction to quantify uncertainty in metal catalysis reaction kinetics models. The results demonstrate improved calibration of prediction intervals.",
            authors=["John Smith", "Jane Doe"],
            year=2022,
            doi=None,
            url=None,
            source_api="test",
            venue="Journal of Catalysis"
        ),
        Paper(
            title="Machine Learning Applications in Drug Discovery",
            abstract="Recent advances in machine learning have accelerated drug discovery processes. This review examines various ML techniques used in pharmaceutical research.",
            authors=["Alice Johnson", "Bob Williams"],
            year=2021,
            doi=None,
            url=None,
            source_api="test",
            venue="Journal of Medicinal Chemistry"
        ),
        Paper(
            title="Examining Anxiety and Depression in COPD Patients",
            abstract="A systematic review of how anxiety and depression influence hospital admissions for COPD patients.",
            authors=["Michael Brown"],
            year=2019,
            doi=None,
            url=None,
            source_api="test",
            venue="BMC Pulmonary Medicine"
        ),
        Paper(
            title="Metal Catalysis with Uncertainty Quantification",
            abstract="This work applies conformal prediction methods to properly quantify uncertainties in reaction kinetics studies involving metal catalysts.",
            authors=["Sarah Wilson", "Thomas Davis"],
            year=2023,
            doi=None,
            url=None,
            source_api="test",
            venue="ACS Catalysis"
        ),
    ]
    
    # Define research question focused on conformal prediction and reaction kinetics
    research_question = "How can conformal prediction methods be used to quantify uncertainty in reaction kinetics models for metal catalysis?"
    domain_context = "Conformal Prediction, reaction kinetics, metal catalysis, uncertainty quantification"
    
    print("\n=== TESTING PAPER FILTERING ===")
    print(f"Research Question: {research_question}")
    
    # Screen papers by title
    print("\nScreening papers by title...")
    title_passed = prompt_engineer.screen_papers_by_title(papers)
    print(f"Papers passed title screening: {len(title_passed)}/{len(papers)}")
    for paper in title_passed:
        print(f"- {paper.title}")
    
    # Screen papers by abstract
    print("\nScreening papers by abstract...")
    abstract_passed = prompt_engineer.screen_papers_by_abstract(title_passed, research_question, domain_context)
    print(f"Papers passed abstract screening: {len(abstract_passed)}/{len(title_passed)}")
    for paper in abstract_passed:
        print(f"- {paper.title}")

if __name__ == "__main__":
    print("Testing improved query parsing and paper filtering...")
    
    # Run tests
    test_query_validation()
    test_research_definition()
    test_paper_filtering()
    
    print("\nTests completed.") 