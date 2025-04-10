#!/usr/bin/env python3
"""
Test script for Deep Researcher using the human-like literature review approach.

This script demonstrates the step-by-step process a human researcher would follow
when conducting a literature review, from defining the research question through
to synthesizing findings into a comprehensive answer.
"""

import os
import sys
import time
import random
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
import re

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from the project root's .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

from src.search.literature_manager import LiteratureManager
from src.ollama_client import OllamaClient
from src.prompt_engineering_human import PromptEngineer
from utils.cache import clear_cache  # Import the clear_cache function

def check_environment():
    """Check if all required components are set up."""
    # Check Tavily API key
    if not os.getenv('TAVILY_API_KEY'):
        print("\nWarning: Tavily API key not found. Some search functionality may be limited.")
    
    # Check Semantic Scholar API key
    if not os.getenv('SEMANTIC_SCHOLAR_API_KEY'):
        print("\nWarning: Semantic Scholar API key not found. Some search functionality may be limited.")
    
    # Check OpenAlex email
    if not os.getenv('OPEN_ALEX_EMAIL'):
        print("\nWarning: OpenAlex email not provided. Rate limits may be lower.")
    
    return True

def setup_components():
    """Set up all necessary components."""
    try:
        # Initialize Ollama client
        print("\nInitializing Ollama client...")
        ollama = OllamaClient(
            model="gemma3:4b",
            temperature=0.3
        )
        
        # Check Ollama server
        if not ollama.check_server():
            print("\nError: Ollama server is not running!")
            print("Please start the Ollama server:")
            if sys.platform == "win32":
                print("1. Open Command Prompt")
                print("2. Run: ollama serve")
            else:
                print("1. Open Terminal")
                print("2. Run: ollama serve")
            return (None,) * 3
        
        # Check if model is available
        print("Checking Ollama model...")
        if not ollama.setup_model():
            return (None,) * 3
        
        print("✓ Ollama setup complete")
        
        # Initialize literature manager
        print("\nInitializing literature manager...")
        literature_manager = LiteratureManager()
        print("✓ Literature manager ready")
        
        # Initialize prompt engineer
        print("\nInitializing prompt engineer...")
        prompt_engineer = PromptEngineer(ollama_client=ollama)
        print("✓ Prompt engineer ready")
        
        return ollama, literature_manager, prompt_engineer
        
    except Exception as e:
        print(f"\nError during setup: {e}")
        return (None,) * 3

def process_query(query: str, prompt_engineer: PromptEngineer, literature_manager: LiteratureManager):
    """
    Process a research query following a human-like literature review approach.
    
    This function implements the step-by-step process described in the human planning
    pseudo-algorithm, from defining the research question to synthesizing findings.
    """
    try:
        # Clear the cache before starting a new query to avoid data leakage
        print("\n[STEP 0] Clearing cache to ensure clean results...")
        clear_cache()
        literature_manager.reset_tracking()
        print("✓ Cache cleared and tracking reset successfully")
        
        # Store the original query for reference
        original_query = query
        
        # STEP 0A: Generate clarification questions
        print("\n[STEP 0A] Generating clarification questions...")
        clarification_questions = prompt_engineer.generate_clarification_questions(query)
        
        if clarification_questions:
            print("\nTo better focus the search, please answer these clarification questions:")
            
            clarifications = {}
            for i, question in enumerate(clarification_questions, 1):
                print(f"\n{i}. {question}")
                answer = input("Your answer: ").strip()
                clarifications[question] = answer
            
            # STEP 0B: Construct refined query
            print("\n[STEP 0B] Constructing refined query based on clarifications...")
            refined_query = prompt_engineer.construct_refined_query(original_query, clarifications)
            print(f"\nRefined Query: {refined_query}")
            
            # Use the refined query for subsequent steps
            query = refined_query
        
        # STEP 1: Define the Research Question & Scope
        print("\n[STEP 1] Defining the research question and scope...")
        research_def = prompt_engineer.define_research_question(query)
        
        print(f"\nResearch Question: {research_def['research_question']}")
        print("Inclusion Criteria:")
        for criteria in research_def['inclusion_criteria']:
            print(f"  - {criteria}")
        print("Exclusion Criteria:")
        for criteria in research_def['exclusion_criteria']:
            print(f"  - {criteria}")
        print("Key Terms:")
        for term in research_def.get('primary_keyphrases', []):
            print(f"  - {term}")
        if 'time_frame' in research_def:
            print(f"Time Frame: {research_def['time_frame']}")
        
        # STEP 2: Identify Sources and Databases
        print("\n[STEP 2] Identifying appropriate sources...")
        sources = prompt_engineer.identify_sources(research_def)
        print(f"Selected sources: {', '.join(sources)}")
        
        # Get the search queries formulated in Step 1
        search_queries = research_def.get("search_queries", [])
        if not search_queries:
             print("\nWarning: No search queries were generated by the LLM. Check the define_research_question method and LLM response.")
             # Attempt fallback using primary keyphrases if available
             fallback_keys = research_def.get("primary_keyphrases", [])
             if fallback_keys:
                 search_queries = [" ".join([f'"{key}"' for key in fallback_keys])]
                 print(f"Using fallback search query: {search_queries[0]}")
             else:
                 print("Error: Cannot proceed without search queries.")
                 return

        print("\n[STEP 3] Using generated search queries:")
        for q in search_queries:
            print(f"  - {q}")
        
        # STEP 3: Initial Search and Retrieval
        print("\n[STEP 3] Searching for initial papers...")
        all_papers = []
        for i, search_query in enumerate(search_queries):
            # Don't remove quotes - they're important for precision
            # search_query = search_query.replace("\"", "")
            print(f"\nSearching for: '{search_query}' ({i+1}/{len(search_queries)})")
            
            try:
                # Set limit lower for multiple queries to avoid overwhelming the system
                papers = literature_manager.search(
                    query=search_query,
                    max_papers=3,  # Limit papers per query
                    sources=sources
                )
                
                print(f"Found {len(papers)} papers for this query")
                all_papers.extend(papers)
                
                # Implement exponential backoff with jitter for rate limiting
                if i < len(search_queries) - 1:
                    base_delay = 1.0  # Start with 1 second base delay
                    max_delay = 8.0   # Maximum delay in seconds
                    retry_count = i + 1  # Use query index as retry count
                    jitter = random.uniform(0, 0.5)  # Add random jitter
                    
                    # Calculate exponential backoff with jitter (min 1 second, max as specified)
                    delay = min(base_delay * (2 ** (retry_count - 1)) + jitter, max_delay)
                    print(f"Waiting {delay:.2f} seconds before next query...")
                    time.sleep(delay)
                    
            except Exception as e:
                print(f"Warning: Search failed for query '{search_query}': {e}")
        
        print(f"\nTotal initial papers retrieved: {len(all_papers)}")
        
        if not all_papers:
            print("\nWarning: No initial papers found. This might be due to:")
            print("1. Very specific or narrow search query")
            print("2. API rate limiting")
            print("3. Network connectivity issues")
            return
        
        # STEP 4-5: Combined screening of papers by title AND abstract
        print("\n[STEP 4-5] Screening papers by title and abstract...")
        relevant_papers = prompt_engineer.screen_papers_by_title_and_abstract(
            all_papers,
            research_question=research_def['research_question']
        )
        print(f"Combined screening: {len(relevant_papers)}/{len(all_papers)} papers passed")
        
        if not relevant_papers:
            print("No papers passed combined screening. Try adjusting search terms or criteria.")
            return
            
        # Iterative Search Refinement - human researchers refine based on initial findings
        print("\n[STEP 3b] Refining search based on initial papers...")
        refined_queries = prompt_engineer.generate_refined_search_queries(
            relevant_papers, 
            research_def['research_question']
        )
        
        if refined_queries:
            print("Refined search queries:")
            for query in refined_queries:
                print(f"  - {query}")
                
            # Search with refined queries
            refined_papers = []
            for i, refined_query in enumerate(refined_queries):
                # Don't remove quotes - they're important for precision
                # refined_query = refined_query.replace("\"", "")
                print(f"\nSearching with refined query: '{refined_query}' ({i+1}/{len(refined_queries)})")
                
                try:
                    papers = literature_manager.search(
                        query=refined_query,
                        max_papers=3,  # Limit per query
                        sources=sources
                    )
                    
                    print(f"Found {len(papers)} papers for this refined query")
                    
                    # Add only unique papers
                    for paper in papers:
                        if paper not in all_papers and paper not in refined_papers:
                            refined_papers.append(paper)
                            
                    # Apply exponential backoff between queries
                    if i < len(refined_queries) - 1:
                        base_delay = 1.0
                        max_delay = 8.0
                        retry_count = i + 1
                        jitter = random.uniform(0, 0.5)
                        delay = min(base_delay * (2 ** (retry_count - 1)) + jitter, max_delay)
                        print(f"Waiting {delay:.2f} seconds before next query...")
                        time.sleep(delay)
                        
                except Exception as e:
                    print(f"Warning: Refined search failed for query '{refined_query}': {e}")
            
            print(f"\nFound {len(refined_papers)} additional papers through refined searches")
            
            # Screen refined papers
            if refined_papers:
                # Combined screening for refined papers
                refined_passed = prompt_engineer.screen_papers_by_title_and_abstract(
                    refined_papers,
                    research_question=research_def['research_question']
                )
                print(f"Combined screening for refined papers: {len(refined_passed)}/{len(refined_papers)} passed")
                
                # Add passed papers to our collection
                for paper in refined_passed:
                    if paper not in relevant_papers:
                        relevant_papers.append(paper)
                        
                print(f"Total relevant papers after refinement: {len(relevant_papers)}")
        
        # Follow citation trails
        print("\n[STEP 3c] Following citation trails...")
        citation_papers = prompt_engineer.follow_citation_trail(
            relevant_papers,
            research_def['research_question'],
            literature_manager
        )
        
        if citation_papers:
            print(f"Found {len(citation_papers)} relevant papers from citation trails")
            
            # Combined screening for citation papers
            citation_passed = prompt_engineer.screen_papers_by_title_and_abstract(
                citation_papers,
                research_def['research_question']
            )
            print(f"Combined screening for citation papers: {len(citation_passed)}/{len(citation_papers)} passed")
            
            # Add unique papers to our collection
            for paper in citation_passed:
                if paper not in relevant_papers:
                    relevant_papers.append(paper)
                    
            print(f"Total relevant papers after following citation trails: {len(relevant_papers)}")
        
        # STEP 6: Full-text review and STEP 7: Critical analysis and synthesis
        print("\n[STEP 6-7] Extracting findings, analyzing methodologies, and synthesizing literature...")
        synthesis = prompt_engineer.synthesize_findings_with_critical_analysis(
            relevant_papers,
            research_def['research_question']
        )
        
        # STEP 8: Construct final answer with proper citations
        print("\n[STEP 8] Constructing final answer with proper citations...")
        final_response = prompt_engineer.integrate_literature(
            papers=relevant_papers,
            query=query,  # Pass the original query for strong domain anchoring
            structured_synthesis=synthesis  # Pass the structured synthesis from STEP 6-7
        )
        
        # Output final results
        print("\nFinal Summary:")
        print(final_response.final_summary)
        
        print("\nCitations:")
        
        # Ensure citations are properly formatted and consistent with in-text references
        
        # First, extract all citation numbers from the final summary
        in_text_citations = re.findall(r'\[(\d+)\]', final_response.final_summary)
        used_citation_numbers = set([int(num) for num in in_text_citations])
        
        # Process and write citations
        citation_dict = {}
        
        for citation in final_response.citations:
            # Extract the citation number if present
            num_match = re.match(r'^(\d+)\.\s+', citation)
            if num_match:
                num = int(num_match.group(1))
                # Only include citations that are referenced in the text
                if num in used_citation_numbers:
                    citation_dict[num] = citation
            else:
                # For unnumbered citations, add them to the end
                max_num = max(used_citation_numbers) if used_citation_numbers else 0
                citation_dict[max_num + 1] = f"{max_num + 1}. {citation}"
        
        # Write citations in numerical order
        for num in sorted(citation_dict.keys()):
            print(f"{citation_dict[num]}")
        
        # Save results to file
        output_dir = Path(__file__).parent.parent / "outputs"
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"research_output_human_{timestamp}.txt"
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"Research Query: {query}\n\n")
            
            # Include clarification details if available
            if 'clarifications' in locals() and clarifications:
                f.write("=== STEP 0: Query Clarification ===\n")
                f.write(f"Original Query: {original_query}\n\n")
                f.write("Clarification Questions and Answers:\n")
                for question, answer in clarifications.items():
                    f.write(f"Q: {question}\n")
                    f.write(f"A: {answer}\n\n")
                f.write(f"Refined Query: {query}\n\n")
            
            f.write("=== STEP 1: Research Question & Scope ===\n")
            f.write(f"Research Question: {research_def['research_question']}\n")
            f.write("Inclusion Criteria:\n")
            for criteria in research_def['inclusion_criteria']:
                f.write(f"  - {criteria}\n")
            f.write("Exclusion Criteria:\n")
            for criteria in research_def['exclusion_criteria']:
                f.write(f"  - {criteria}\n")
            f.write("Key Terms:\n")
            for term in research_def.get('primary_keyphrases', []):
                f.write(f"  - {term}\n")
            if 'time_frame' in research_def:
                f.write(f"Time Frame: {research_def['time_frame']}\n")
            
            f.write("\n=== STEP 2: Sources ===\n")
            f.write(f"Selected sources: {', '.join(sources)}\n")
            
            f.write("\n=== STEP 4-5: Retrieved Papers ===\n")
            for i, paper in enumerate(relevant_papers):
                f.write(f"\n{i+1}. {paper.title}\n")
                f.write(f"   Authors: {', '.join(paper.authors)}\n")
                f.write(f"   Year: {paper.year}\n")
                f.write(f"   Source: {paper.source_api}\n")
            
            f.write("\n=== STEP 6-7: Critical Analysis ===\n")
            f.write(f"{synthesis}\n\n")
            
            f.write("\n=== STEP 8: Final Synthesis ===\n")
            f.write("**Part 1: Final Summary**\n\n")
            f.write(f"{final_response.final_summary}\n\n")
            
            f.write("**Part 2: Citations**\n")
            
            # Ensure citations are properly formatted and consistent with in-text references
            
            # First, extract all citation numbers from the final summary
            in_text_citations = re.findall(r'\[(\d+)\]', final_response.final_summary)
            used_citation_numbers = set([int(num) for num in in_text_citations])
            
            # Process and write citations
            citation_dict = {}
            
            for citation in final_response.citations:
                # Extract the citation number if present
                num_match = re.match(r'^(\d+)\.\s+', citation)
                if num_match:
                    num = int(num_match.group(1))
                    # Only include citations that are referenced in the text
                    if num in used_citation_numbers:
                        citation_dict[num] = citation
                else:
                    # For unnumbered citations, add them to the end
                    max_num = max(used_citation_numbers) if used_citation_numbers else 0
                    citation_dict[max_num + 1] = f"{max_num + 1}. {citation}"
            
            # Write citations in numerical order
            for num in sorted(citation_dict.keys()):
                f.write(f"{citation_dict[num]}\n")
        
        print(f"\nResults saved to {output_file}")
        
    except Exception as e:
        print(f"\nError processing query: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function to test the human planning approach."""
    print("\n======================================================")
    print("  DEEP RESEARCHER - HUMAN PLANNING APPROACH")
    print("======================================================")
    print("\nThis script demonstrates a literature review following")
    print("the step-by-step process a human researcher would use.")
    
    # Check environment
    check_environment()
    
    # Set up components
    print("\nSetting up components...")
    ollama, literature_manager, prompt_engineer = setup_components()
    if not all((ollama, literature_manager, prompt_engineer)):
        return
    
    # Get query from user
    print("\nEnter your research query (or press Enter to use the default query):")
    user_query = input().strip()
    
    if not user_query:
        user_query = "What are the latest developments in biomimetic materials for sustainable architecture?"
        print(f"\nUsing default query: {user_query}")
    
    # Process the query
    process_query(user_query, prompt_engineer, literature_manager)

if __name__ == "__main__":
    main() 