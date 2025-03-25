#!/usr/bin/env python3
"""
Test script for the Literature Manager with multiple API sources.

This script demonstrates how to use the LiteratureManager to search for papers
across multiple academic API sources including Tavily, Semantic Scholar, OpenAlex,
ArXiv, and ChemArXiv.
"""

import os
import sys
import asyncio
from dotenv import load_dotenv

# Add the project root to the Python path
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

# Add the adapters, common, and utils directories to the Python path
sys.path.insert(0, os.path.join(project_root, 'adapters'))
sys.path.insert(0, os.path.join(project_root, 'common'))
sys.path.insert(0, os.path.join(project_root, 'utils'))

# Import the LiteratureManager
from src.search.literature_manager import LiteratureManager

async def run_search(query, sources=None, max_papers=5):
    """
    Run a search using the LiteratureManager.
    
    Parameters:
    -----------
    query: The search query
    sources: Optional list of sources to use
    max_papers: Maximum number of papers to return per source
    """
    print(f"\n=== Searching for: '{query}' ===")
    if sources:
        print(f"Using sources: {', '.join(sources)}")
    else:
        print("Using all available sources")
    
    # Initialize the literature manager
    literature_manager = LiteratureManager()
    
    try:
        # Search for papers
        papers = await literature_manager.search(
            query=query,
            max_papers=max_papers,
            sources=sources
        )
        
        # Print the results
        print(f"\nFound {len(papers)} unique papers")
        
        if not papers:
            print("No papers found. Possible causes:")
            print("- API key issues (check your .env file)")
            print("- Rate limiting by the APIs")
            print("- No papers match the query")
            return
        
        # Group papers by source
        papers_by_source = {}
        for paper in papers:
            source = paper.source_api
            if source not in papers_by_source:
                papers_by_source[source] = []
            papers_by_source[source].append(paper)
        
        # Print papers by source
        for source, source_papers in papers_by_source.items():
            print(f"\n== Papers from {source} ({len(source_papers)}) ==")
            for i, paper in enumerate(source_papers):
                print(f"\n{i+1}. {paper.title}")
                if paper.authors:
                    print(f"   Authors: {', '.join(paper.authors[:3])}" + 
                         (f" et al." if len(paper.authors) > 3 else ""))
                if paper.year:
                    print(f"   Year: {paper.year}")
                if paper.doi:
                    print(f"   DOI: {paper.doi}")
                print(f"   URL: {paper.url}" if paper.url else "   URL: None")
        
        # Print search statistics
        stats = literature_manager.get_search_statistics()
        print("\n=== Search Statistics ===")
        print(f"Total searches: {stats['total_searches']}")
        print(f"Total unique papers: {stats['total_unique_papers']}")
        print(f"Searches by API: {stats['searches_by_api']}")
    
    except Exception as e:
        print(f"Error during search: {e}")

async def main():
    """Main function to run the test."""
    # Load environment variables
    load_dotenv()
    
    print("Deep Researcher API Test")
    print("=======================")
    print("This script tests the literature search APIs")
    print("If you see API errors, check your .env file and ensure you have the correct API keys:")
    print("- TAVILY_API_KEY: Required for Tavily API")
    print("- SEMANTIC_SCHOLAR_API_KEY: Optional for Semantic Scholar API")
    print("- OPEN_ALEX_EMAIL: Recommended for OpenAlex API")
    print("\nNote: Some APIs may still work without keys but with reduced rate limits")
    
    # Define test queries
    queries = [
        "CRISPR gene editing ethics",
        "single-cell RNA sequencing cancer"
    ]
    
    # List of all available sources
    all_sources = ["tavily", "semantic_scholar", "open_alex", "arxiv", "chemrxiv"]
    
    # Try running with only APIs that don't require authentication first
    print("\nTesting APIs that don't require authentication...")
    await run_search(queries[0], ["arxiv", "chemrxiv", "open_alex"])
    
    # Then try all APIs
    #print("\nTesting all APIs...")
    #await run_search(queries[1])

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main()) 