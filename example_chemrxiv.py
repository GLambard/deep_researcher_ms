"""
Example script demonstrating the usage of the ChemRxivAdapter.

This script shows how to search for papers on ChemRxiv, retrieve paper details,
and access citations and references.
"""

import asyncio
import logging

from src.adapters.chemrxiv import ChemRxivAdapter

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    """
    Demonstrates the usage of the ChemRxivAdapter.
    """
    # Initialize the adapter
    adapter = ChemRxivAdapter()
    
    # Search for papers with basic query
    query = "water"
    print(f"\n=== Searching for papers about '{query}' ===")
    results = await adapter.search_papers(query, page=1, per_page=3)
    print(f"Found {results.total_results} papers")
    print(f"Received {len(results.papers)} papers in response")
    
    if results.papers:
        for i, paper in enumerate(results.papers):
            # Print key information from each paper
            print(f"\n{i+1}. {paper.title}")
            if paper.authors:
                if len(paper.authors) > 3:
                    print(f"Authors: {', '.join(author.name for author in paper.authors[:3])} et al.")
                else:
                    print(f"Authors: {', '.join(author.name for author in paper.authors)}")
            else:
                print("Authors: None")

            print(f"Abstract: {paper.abstract[:150]}..." if paper.abstract else "Abstract: None")
            print(f"DOI: {paper.doi}")
            print(f"Publication date: {paper.publication_date}")
            print(f"URL: {paper.url}")
            print(f"PDF URL: {paper.pdf_url}")
            print(f"Citation count: {paper.citation_count}")
            print(f"Fields of study: {', '.join(paper.fields_of_study[:5])}{'...' if len(paper.fields_of_study) > 5 else ''}")
        
        # # Get paper by ID
        # paper_id = paper.external_ids["chemrxiv"]
        # print(f"\n=== Retrieving paper by ID {paper_id} ===")
        # paper_by_id = await adapter.get_paper_by_id(paper_id)
        # if paper_by_id:
        #     print(f"Successfully retrieved paper: {paper_by_id.title}")
        # else:
        #     print("Failed to retrieve paper by ID")
        
        # # Get paper by DOI
        # if paper.doi:
        #     print(f"\n=== Retrieving paper by DOI {paper.doi} ===")
        #     paper_by_doi = await adapter.get_paper_by_doi(paper.doi)
        #     if paper_by_doi:
        #         print(f"Successfully retrieved paper: {paper_by_doi.title}")
        #     else:
        #         print("Failed to retrieve paper by DOI")
        
        # # Get citations and references
        # print(f"\n=== Retrieving citations for paper {paper_id} ===")
        # citations = await adapter.get_citations(paper_id)
        # print(f"Found {len(citations)} citations")
        
        # print(f"\n=== Retrieving references for paper {paper_id} ===")
        # references = await adapter.get_references(paper_id)
        # print(f"Found {len(references)} references")
    
    # # Advanced search with filters
    # print("\n=== Advanced search ===")
    # keywords = "nanoparticles"
    # advanced_params = {
    #     "year": 2025  # Just use year, no fields_of_study
    # }
    # advanced_results = await adapter.search_papers(keywords, **advanced_params)
    # if advanced_results and advanced_results.total_results:
    #     print(f"Found {advanced_results.total_results} papers matching advanced criteria")
    #     if advanced_results.papers:
    #         print(f"First result: {advanced_results.papers[0].title}")
    # else:
    #     print("No papers found or error occurred during advanced search")
    
    # # Pagination example
    # print("\n=== Pagination example ===")
    # page1 = await adapter.search_papers("catalyst", page=1, per_page=5)
    # print(f"Page 1: {len(page1.papers)} papers of {page1.total_results} total")
    # if page1.papers:
    #     print("Page 1 papers:")
    #     for i, paper in enumerate(page1.papers, 1):
    #         print(f"  {i}. {paper.title}")
    
    # page2 = await adapter.search_papers("catalyst", page=2, per_page=5)
    # print(f"\nPage 2: {len(page2.papers)} papers of {page2.total_results} total")
    # if page2.papers:
    #     print("Page 2 papers:")
    #     for i, paper in enumerate(page2.papers, 1):
    #         print(f"  {i}. {paper.title}")


if __name__ == "__main__":
    asyncio.run(main()) 