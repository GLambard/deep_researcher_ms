"""
Example usage of the arXiv adapter.

This script demonstrates how to use the ArXivAdapter to perform
common operations such as searching for papers, retrieving paper details,
and showing how to use the various search parameters.

The arXiv API does not provide citation or reference data, so these
methods will return empty lists.
"""
import asyncio
import logging
import sys
from datetime import datetime

# Import from the correct package
from src.adapters.arxiv import ArXivAdapter
from src.common.models import QueryParams


async def main():
    """
    Run examples of arXiv adapter usage.
    
    This function demonstrates the core functionality of the ArXivAdapter:
    1. Searching for papers with various filters
    2. Displaying detailed metadata from search results
    3. Retrieving papers by arXiv ID
    4. Retrieving papers by DOI (when available)
    5. Demonstrating advanced search queries
    
    Each step includes printing the results to show the data structure.
    """
    # Configure logging to see informational and error messages
    # This helps with debugging and understanding API interactions
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )
    
    # Create adapter instance
    adapter = ArXivAdapter(disable_cache=True)
    
    # Example 1: Search for papers about a topic
    # This demonstrates the basic search functionality with simple parameters
    print("\n=== Search for papers about quantum computing ===")
    search_params = QueryParams(
        keywords="quantum computing",
        #year="1990-2000",
        limit=3,  # Limit to 3 results for brevity
    )
    
    # The search_papers method returns a QueryResult object
    # containing metadata for matching papers
    results = await adapter.search_papers(search_params)
    
    print(f"Found {results.total_results} papers")
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
        print(f"    DOI: {paper.doi}" if paper.doi else "    DOI: None")
        print(f"    arXiv ID: {paper.external_ids.get('arxiv')}" if paper.external_ids and 'arxiv' in paper.external_ids else "    arXiv ID: None")
        print(f"    Publication date: {paper.publication_date}" if paper.publication_date else "    Publication date: None")
        print(f"    Fields of study: {', '.join(paper.fields_of_study)}" if paper.fields_of_study else "    Fields of study: None")
        print(f"    URL: {paper.url}" if paper.url else "    URL: None")
        print(f"    PDF: {paper.pdf_url}" if paper.pdf_url else "    PDF: None")
    
    # # If we found papers, use the first one for the next examples
    # # This shows how to chain API calls based on initial results
    # if results.papers and len(results.papers) > 0:
    #     # Use the first paper from the search results
    #     first_paper = results.papers[0]
    #     arxiv_id = first_paper.external_ids.get("arxiv")
        
    #     if arxiv_id:
    #         # Example 2: Retrieve paper by ID
    #         print("\n=== Getting paper by arXiv ID ===")
    #         paper = await adapter.get_paper_by_id(arxiv_id)
            
    #         if paper:
    #             print(f"Retrieved paper: {paper.title}")
    #             print(f"Authors: {', '.join(author.name for author in paper.authors)}")
    #             print(f"Abstract: {paper.abstract[:150]}..." if paper.abstract else "Abstract: None")
    #             print(f"DOI: {paper.doi}" if paper.doi else "DOI: None")
    #         else:
    #             print(f"Failed to retrieve paper with ID {arxiv_id}")
        
    #     # Example 3: Try to get citations (will return empty list as arXiv doesn't provide this)
    #     print("\n=== Attempt to get citations (not supported by arXiv) ===")
    #     citations = await adapter.get_citations(arxiv_id, limit=3)
    #     print(f"Number of citations found: {len(citations)}")
        
    #     # Example 4: Try to get references (will return empty list as arXiv doesn't provide this)
    #     print("\n=== Attempt to get references (not supported by arXiv) ===")
    #     references = await adapter.get_references(arxiv_id, limit=3)
    #     print(f"Number of references found: {len(references)}")
        
    #     # Example 5: If the paper has a DOI, try to retrieve it by DOI
    #     if first_paper.doi:
    #         print("\n=== Getting paper by DOI ===")
    #         paper_by_doi = await adapter.get_paper_by_doi(first_paper.doi)
            
    #         if paper_by_doi:
    #             print(f"Retrieved paper by DOI: {paper_by_doi.title}")
    #             print(f"arXiv ID: {paper_by_doi.external_ids.get('arxiv', 'None')}")
    #         else:
    #             print(f"Failed to retrieve paper with DOI {first_paper.doi}")
    
    # # Example 6: Search with more complex parameters
    # # This demonstrates advanced filtering options
    # print("\n=== Advanced search with filters ===")
    # advanced_params = QueryParams(
    #     keywords="ti:neural AND cat:cs.LG",  # Search for "neural" in title, in the Machine Learning category
    #     year="2022",                         # Papers from 2022
    #     limit=3,                             # Limit results
    # )
    
    # results = await adapter.search_papers(advanced_params)
    
    # print(f"Found {results.total_results} papers")
    # for i, paper in enumerate(results.papers):
    #     print(f"\n{i+1}. {paper.title}")
    #     print(f"   Authors: {', '.join(author.name for author in paper.authors)}")
    #     print(f"   Year: {paper.year}" if paper.year else "   Year: None")
    #     print(f"   Fields of study: {', '.join(paper.fields_of_study)}" if paper.fields_of_study else "   Fields of study: None")
    #     print(f"   URL: {paper.url}" if paper.url else "   URL: None")
    
    # # Example 7: Search with year range
    # # This demonstrates how to filter by a date range
    # print("\n=== Search with year range ===")
    # year_range_params = QueryParams(
    #     keywords="supersymmetry",
    #     year="2020-2023",  # Date range from 2020 to 2023
    #     limit=3,
    # )
    
    # results = await adapter.search_papers(year_range_params)
    
    # print(f"Found {results.total_results} papers")
    # for i, paper in enumerate(results.papers):
    #     print(f"\n{i+1}. {paper.title}")
    #     print(f"   Year: {paper.year}" if paper.year else "   Year: None")
    #     print(f"   Publication date: {paper.publication_date}" if paper.publication_date else "   Publication date: None")
    
    # # Example 8: Demonstrate pagination
    # # This shows how to get the next batch of results
    # print("\n=== Pagination example ===")
    # pagination_params = QueryParams(
    #     keywords="machine learning",
    #     limit=2,           # Small limit to demonstrate pagination
    #     offset=0,          # Start at the beginning
    # )
    
    # # First page
    # first_page = await adapter.search_papers(pagination_params)
    # print(f"Page 1: {len(first_page.papers)} papers, total: {first_page.total_results}")
    # for i, paper in enumerate(first_page.papers):
    #     print(f"{i+1}. {paper.title}")
    
    # # Check if there are more results
    # if first_page.has_next:
    #     # Update offset to get next page
    #     pagination_params.offset = first_page.next_offset
    #     second_page = await adapter.search_papers(pagination_params)
        
    #     print(f"\nPage 2: {len(second_page.papers)} papers")
    #     for i, paper in enumerate(second_page.papers):
    #         print(f"{i+1}. {paper.title}")


if __name__ == "__main__":
    # Use asyncio.run to execute the async main function
    asyncio.run(main()) 