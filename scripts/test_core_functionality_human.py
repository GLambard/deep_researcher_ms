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
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from the project root's .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

from src.search.literature_manager import LiteratureManager
from src.ollama_client import OllamaClient
from src.prompt_engineering_human import PromptEngineer

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
        for term in research_def['key_terms']:
            print(f"  - {term}")
        if 'time_frame' in research_def:
            print(f"Time Frame: {research_def['time_frame']}")
        
        # STEP 2: Identify Sources and Databases
        print("\n[STEP 2] Identifying appropriate sources...")
        sources = prompt_engineer.identify_sources(research_def)
        print(f"Selected sources: {', '.join(sources)}")
        
        # Break down the query into components
        print("\n[STEP 3] Breaking down query into components...")
        components = prompt_engineer.process_query(query)
        for comp in components:
            print(f"\nTopic: {comp.topic}")
            for sub in comp.subtopics:
                print(f"  - {sub}")
            if comp.year_range:
                print(f"  - Year range: {comp.year_range[0]}-{comp.year_range[1]}")
        
        # Generate initial response
        print("\n[STEP 3] Generating initial assessment...")
        initial_response = prompt_engineer.generate_initial_response(query, components)
        print("\nInitial Assessment:")
        print(initial_response)
        
        # Generate search queries
        print("\n[STEP 3] Generating search queries...")
        search_queries = prompt_engineer.generate_search_queries(components)
        print("Search queries:")
        for q in search_queries[:5]:  # Show just first 5 queries if there are many
            print(f"  - {q}")
        if len(search_queries) > 5:
            print(f"  - ...and {len(search_queries) - 5} more")
        
        # STEP 3: Initial Search and Retrieval
        print("\n[STEP 3] Searching for papers...")
        all_papers = []
        for i, search_query in enumerate(search_queries):
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
                
                # Small delay to prevent rate limiting
                if i < len(search_queries) - 1:
                    time.sleep(1)
                    
            except Exception as e:
                print(f"Warning: Search failed for query '{search_query}': {e}")
        
        print(f"\nTotal papers retrieved: {len(all_papers)}")
        
        if not all_papers:
            print("\nWarning: No papers found. This might be due to:")
            print("1. Very specific or narrow search query")
            print("2. API rate limiting")
            print("3. Network connectivity issues")
            print("\nSuggestions:")
            print("1. Try a broader search query")
            print("2. Check your API keys and rate limits")
            print("3. Verify your internet connection")
            return
        
        # STEP 4 & 5: Title and Abstract Screening
        print("\n[STEP 4-5] Screening papers by title and abstract...")
        # The screening happens within the integrate_literature method
        
        # STEP 6, 7, 8: Full-text review, data extraction, synthesis, and final answer
        print("\n[STEP 6-8] Integrating literature findings...")
        final_response = prompt_engineer.integrate_literature(
            initial_response=initial_response,
            papers=all_papers
        )
        
        # Output final results
        print("\nFinal Summary:")
        print(final_response.final_summary)
        
        print("\nCitations:")
        for citation in final_response.citations:
            print(citation)
        
        # Save results to file
        output_dir = Path(__file__).parent.parent / "outputs"
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"research_output_human_{timestamp}.txt"
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"Research Query: {query}\n\n")
            
            f.write("=== STEP 1: Research Question & Scope ===\n")
            f.write(f"Research Question: {research_def['research_question']}\n")
            f.write("Inclusion Criteria:\n")
            for criteria in research_def['inclusion_criteria']:
                f.write(f"  - {criteria}\n")
            f.write("Exclusion Criteria:\n")
            for criteria in research_def['exclusion_criteria']:
                f.write(f"  - {criteria}\n")
            f.write("Key Terms:\n")
            for term in research_def['key_terms']:
                f.write(f"  - {term}\n")
            if 'time_frame' in research_def:
                f.write(f"Time Frame: {research_def['time_frame']}\n")
            
            f.write("\n=== STEP 2: Sources ===\n")
            f.write(f"Selected sources: {', '.join(sources)}\n")
            
            f.write("\n=== STEP 3: Initial Assessment ===\n")
            f.write(f"{initial_response}\n\n")
            
            f.write("=== STEP 4-5: Retrieved Papers ===\n")
            for i, paper in enumerate(all_papers):
                f.write(f"\n{i+1}. {paper.title}\n")
                f.write(f"   Authors: {', '.join(paper.authors)}\n")
                f.write(f"   Year: {paper.year}\n")
                f.write(f"   Source: {paper.source_api}\n")
            
            f.write("\n=== STEP 6-8: Final Synthesis ===\n")
            f.write(f"{final_response.final_summary}\n\n")
            
            f.write("=== Citations ===\n")
            for citation in final_response.citations:
                f.write(f"{citation}\n")
        
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