"""
Example usage of the OpenAlex adapter.

This script demonstrates how to use the OpenAlexAdapter to perform
common operations such as searching for papers, retrieving paper details,
and obtaining citations and references.

This serves as both a usage example and a simple test that the adapter
is working correctly.
"""
import asyncio
import logging
import sys
from datetime import datetime

# Import from the correct package
from src.adapters.open_alex import OpenAlexAdapter
from src.common.models import QueryParams


async def main():
    """
    Run examples of OpenAlex adapter usage.
    
    This function demonstrates the core functionality of the OpenAlexAdapter:
    1. Searching for papers with various filters
    2. Displaying detailed metadata from search results
    3. Getting citations for a paper
    4. Getting references for a paper
    5. Performing complex searches with multiple filters
    6. Using the optimized batched request method to get paper details, citations, and references
       in a single call
    
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
    # You can optionally provide your email for the polite pool
    adapter = OpenAlexAdapter(email="your.email@example.com", disable_cache=True)
    
    # Example 1: Search for papers about a topic
    # This demonstrates the basic search functionality with simple parameters
    print("\n=== Search for papers ===")
    search_params = QueryParams(
        keywords="dark matter",
        #year="1980-1990",
        limit=3,  # Limit to 3 results for brevity
        open_access_only=True,  # Only return papers with open access
        sort_by="relevance",  # Sort by relevance
    )
    
    # The search_papers method returns a QueryResult object
    # containing metadata for matching papers
    results = await adapter.search_papers(search_params)
    print(results.papers[0].raw_data.keys())
    
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
        print(f"    Publication date: {paper.publication_date}" if paper.publication_date else "    Publication date: None")
        print(f"    Fields of study: {', '.join(paper.fields_of_study)}" if paper.fields_of_study else "    Fields of study: None")
        print(f"    Citations: {paper.citation_count}" if paper.citation_count else "    Citations: None")
        print(f"    URL: {paper.url}" if paper.url else "    URL: None")
        print(f"    PDF: {paper.pdf_url}" if paper.pdf_url else "    PDF: None")
    
    # # If we found papers, use the first one for the next examples
    # # This shows how to chain API calls based on initial results
    # if results.papers and len(results.papers) > 0:
    #     # Use the first paper from the search results
    #     # We already have all the paper details from the search, no need to call get_paper_by_id
    #     first_paper = results.papers[0]
        
    #     # Example 2: Display detailed paper information
    #     # Using data we already have instead of making another API call
    #     print("\n=== Paper details (from search results) ===")
    #     print(f"Title: {first_paper.title}")
    #     print(f"Abstract: {first_paper.abstract[:150]}..." if first_paper.abstract else "Abstract: None")
    #     print(f"DOI: {first_paper.doi}" if first_paper.doi else "DOI: None")
    #     print(f"Publication date: {first_paper.publication_date}" if first_paper.publication_date else "Publication date: None")
    #     print(f"Fields of study: {', '.join(first_paper.fields_of_study)}" if first_paper.fields_of_study else "Fields of study: None")
        
    #     # Extract the paper ID from URL or external_ids
    #     paper_id = None
    #     if first_paper.url:
    #         # OpenAlex uses URLs like https://openalex.org/W1234567890
    #         paper_id = first_paper.url.split("/")[-1] if first_paper.url else None
    #     elif first_paper.external_ids and "openalex" in first_paper.external_ids:
    #         paper_id = first_paper.external_ids["openalex"]
            
    #     if paper_id:
    #         # Example 3: STANDARD APPROACH - Separate API calls for citations and references
    #         # This demonstrates the individual API call approach
    #         print("\n=== STANDARD APPROACH: Separate API calls for details, citations, and references ===")
            
    #         # For citations and references, we make separate API calls
    #         # as this information is not included in the search results
    #         print("\n=== Get citations ===")
    #         citations = await adapter.get_citations(paper_id, limit=3)
            
    #         print(f"Found {len(citations)} citations")
    #         for i, citation in enumerate(citations):
    #             print(f"\n{i+1}. {citation.title}")
    #             print(f"   Year: {citation.year}" if citation.year else "   Year: None")
    #             print(f"   URL: {citation.url}" if citation.url else "   URL: None")
            
    #         print("\n=== Get references ===")
    #         references = await adapter.get_references(paper_id, limit=3)
            
    #         print(f"Found {len(references)} references")
    #         for i, reference in enumerate(references):
    #             print(f"\n{i+1}. {reference.title}")
    #             print(f"   Year: {reference.year}" if reference.year else "   Year: None")
    #             print(f"   URL: {reference.url}" if reference.url else "   URL: None")
            
    #         # Example 4: OPTIMIZED APPROACH - Batched API calls
    #         # This demonstrates the new optimized method that combines all three requests
    #         # into a single method call, reducing API interaction complexity
    #         print("\n=== OPTIMIZED APPROACH: Batched API call for details, citations, and references ===")
    #         batched_results = await adapter.get_paper_details_with_citations_references(
    #             paper_id, citations_limit=3, references_limit=3
    #         )
            
    #         # Access the paper details
    #         paper = batched_results["paper"]
    #         if paper:
    #             print(f"\nPaper: {paper.title}")
    #             print(f"Abstract: {paper.abstract[:150]}..." if paper.abstract else "Abstract: None")
            
    #         # Access the citations
    #         batch_citations = batched_results["citations"]
    #         print(f"\nFound {len(batch_citations)} citations (batched)")
    #         for i, citation in enumerate(batch_citations):
    #             print(f"{i+1}. {citation.title}")
            
    #         # Access the references
    #         batch_references = batched_results["references"]
    #         print(f"\nFound {len(batch_references)} references (batched)")
    #         for i, reference in enumerate(batch_references):
    #             print(f"{i+1}. {reference.title}")
            
    #         # Explain the optimization
    #         print("\n=== Optimization Analysis ===")
    #         print("Standard approach: 3 separate API calls")
    #         print("Optimized approach: 1 method call (3 concurrent API calls internally)")
    #         print("Benefits: Reduced latency, cleaner code, better error handling")
    
    # # Example 5: Search with more complex parameters
    # # This demonstrates advanced filtering options
    # print("\n=== Advanced search with filters ===")
    # advanced_params = QueryParams(
    #     keywords="artificial intelligence ethics",  # Topic keywords
    #     year=2022,                                  # Single year instead of range
    #     # Removed venue filter as it requires specialized formatting
    #     limit=3,                                    # Limit results
    #     sort_by="citations",                        # Sort by citation count
    # )
    
    # results = await adapter.search_papers(advanced_params)
    
    # print(f"Found {results.total_results} papers")
    # for i, paper in enumerate(results.papers):
    #     print(f"\n{i+1}. {paper.title}")
    #     print(f"   Venue: {paper.venue}" if paper.venue else "   Venue: None")
    #     print(f"   Year: {paper.year}" if paper.year else "   Year: None")
    #     print(f"   Citations: {paper.citation_count}" if paper.citation_count else "   Citations: None")
    #     print(f"   URL: {paper.url}" if paper.url else "   URL: None")


if __name__ == "__main__":
    # Use asyncio.run to execute the async main function
    asyncio.run(main()) 