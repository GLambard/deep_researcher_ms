#!/usr/bin/env python
"""
Test script to verify the fixes for truncated queries and search query generation.
"""

import sys
import os
import re

# Add the project root directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ollama_client import OllamaClient
from src.prompt_engineering_human import PromptEngineer

def test_truncated_queries():
    """Test the handling of truncated queries."""
    # Initialize clients
    ollama_client = OllamaClient()
    prompt_engineer = PromptEngineer(ollama_client)
    
    # Test cases of truncated queries
    truncated_queries = [
        '("Conformal Prediction" AND "reaction kinetics") AND ("metal',
        '("active learning" AND "electrocatalysis" AND "nitrides") AND ("data',
        '"machine learning" AND "drug discovery',
        '(reinforcement learning AND',
        '"quantum computing" OR "quantum',
    ]
    
    print("\n=== TESTING TRUNCATED QUERY HANDLING ===")
    for i, query in enumerate(truncated_queries):
        fixed_query = prompt_engineer._validate_and_fix_query(query)
        print(f"\nOriginal Query {i+1}: {query}")
        print(f"Fixed Query {i+1}: {fixed_query}")

def test_search_query_generation():
    """Test the generation of search queries from original query."""
    # Initialize clients
    ollama_client = OllamaClient()
    prompt_engineer = PromptEngineer(ollama_client)
    
    # Test with a complete query
    query = '("Conformal Prediction" AND "reaction kinetics") AND ("metal catalysis")'
    
    # Mock initial response for testing
    initial_response = """
    Research Question: How can conformal prediction techniques be applied to improve 
    the accuracy and reliability of reaction kinetics models in metal catalysis processes?
    """
    
    print("\n=== TESTING SEARCH QUERY GENERATION ===")
    print(f"Original Query: {query}")
    print(f"Research Question: {initial_response.strip()}")
    
    search_queries = prompt_engineer.generate_search_queries(query, initial_response)
    
    print("\nGenerated Search Queries:")
    for i, search_query in enumerate(search_queries):
        print(f"{i+1}. {search_query}")
    
    # Test with a truncated query
    truncated_query = '("active learning" AND "electrocatalysis" AND "nitrides") AND ("data'
    fixed_query = prompt_engineer._validate_and_fix_query(truncated_query)
    
    print("\nTesting with fixed truncated query:")
    print(f"Original Truncated Query: {truncated_query}")
    print(f"Fixed Query: {fixed_query}")
    
    search_queries = prompt_engineer.generate_search_queries(fixed_query, initial_response)
    
    print("\nGenerated Search Queries from Fixed Query:")
    for i, search_query in enumerate(search_queries):
        print(f"{i+1}. {search_query}")

if __name__ == "__main__":
    print("Testing query handling improvements...")
    
    # Run tests
    test_truncated_queries()
    test_search_query_generation()
    
    print("\nTests completed.") 