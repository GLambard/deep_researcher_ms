#!/usr/bin/env python3
"""
Test script for Deep Researcher using the OpenAI Deep Research Planning approach.

This script demonstrates a state-based research process following the OpenAI
planning algorithm, managing a finite research budget and context window
constraints while producing high-quality research answers.
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
from src.prompt_engineering_openai import PromptEngineer

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

def process_query(query: str, prompt_engineer: PromptEngineer, literature_manager: LiteratureManager, max_external_calls: int = 10):
    """
    Process a research query following the OpenAI deep research planning approach.
    
    This function implements the multi-phase research process with state transitions:
    1. PLAN_PHASE: Interpret user query and plan the approach
    2. RESEARCH_PHASE: Identify knowledge gaps and gather information
    3. COMPOSING_PHASE: Create draft sections based on gathered information
    4. REVIEW_PHASE: Analyze the draft for issues
    5. REVISION_PHASE: Improve identified issues
    6. FINALIZE_PHASE: Polish the final output
    
    Parameters:
    -----------
    query: The research query
    prompt_engineer: Instance of PromptEngineer
    literature_manager: Instance of LiteratureManager
    max_external_calls: Maximum number of external API calls to make
    """
    try:
        # Configure the max external calls limit
        prompt_engineer.max_external_calls = max_external_calls
        
        # Initialize state
        state = "PLAN_PHASE"
        external_calls_used = 0
        
        print(f"\nState: {state}")
        print("\n1. PLAN_PHASE: Interpreting query and planning approach...")
        
        # PLAN_PHASE: Break down the query into components
        print("\nAnalyzing query and creating initial outline...")
        components = prompt_engineer.process_query(query)
        
        # Display the components
        print("\nQuery Components:")
        for i, comp in enumerate(components):
            print(f"\nComponent {i+1}: {comp.topic}")
            for j, subtopic in enumerate(comp.subtopics):
                print(f"  • Subtopic {j+1}: {subtopic}")
        
        # Transition to RESEARCH_PHASE
        state = "RESEARCH_PHASE"
        print(f"\nState: {state}")
        print("\n2. RESEARCH_PHASE: Identifying knowledge gaps and gathering information...")
        
        # Identify knowledge gaps
        print("\nIdentifying knowledge gaps...")
        knowledge_gaps = prompt_engineer.identify_knowledge_gaps(components)
        
        # Display knowledge gaps
        print("\nKnowledge Gaps:")
        for i, gap in enumerate(knowledge_gaps):
            print(f"\nGap {i+1}: {gap['question']}")
            print(f"  Related to: {gap['topic']}")
            print(f"  Search terms: {', '.join(gap['search_terms'])}")
            print(f"  Importance: {gap['importance']}")
        
        # Generate search queries
        print("\nGenerating search queries...")
        search_queries = prompt_engineer.generate_search_queries(components)
        
        # Display search queries (limit to first 5 if there are many)
        print("\nSearch Queries:")
        for i, query in enumerate(search_queries[:5]):
            print(f"  {i+1}. {query}")
        if len(search_queries) > 5:
            print(f"  ...and {len(search_queries) - 5} more")
            
        # Track external calls
        external_calls_used = len(search_queries)
        print(f"\nExternal calls budget: {external_calls_used}/{max_external_calls}")
        
        # Generate initial response
        print("\nGenerating initial response...")
        initial_response = prompt_engineer.generate_initial_response(query, components)
        
        print("\nInitial Response:")
        # Print just the first few lines of the initial response
        initial_lines = initial_response.split('\n')[:5]
        for line in initial_lines:
            print(f"  {line}")
        if len(initial_lines) < len(initial_response.split('\n')):
            print("  ...")
        
        # Search for papers
        print("\nSearching for papers to fill knowledge gaps...")
        all_papers = []
        
        for i, search_query in enumerate(search_queries):
            print(f"\nSearching for: '{search_query}' ({i+1}/{len(search_queries)})")
            
            try:
                # Set limit lower for multiple queries to avoid overwhelming the system
                papers = literature_manager.search(
                    query=search_query,
                    max_papers=2  # Limit papers per query
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
            
        # Process and chunk papers
        print("\nProcessing papers to fit context window constraints...")
        paper_chunks = prompt_engineer.chunk_and_summarize(all_papers)
        print(f"Divided {len(all_papers)} papers into {len(paper_chunks)} manageable chunks")
        
        # Transition to COMPOSING_PHASE
        state = "COMPOSING_PHASE"
        print(f"\nState: {state}")
        print("\n3. COMPOSING_PHASE: Creating draft sections based on research...")
        
        # Compose sections for each topic
        print("\nComposing draft sections...")
        topics = [comp.topic for comp in components]
        
        for i, topic in enumerate(topics):
            print(f"  Composing section {i+1}/{len(topics)}: {topic}")
            # The actual composition happens in integrate_literature
        
        # Transition to REVIEW_PHASE, REVISION_PHASE, and FINALIZE_PHASE happen within integrate_literature
        print("\nMoving to REVIEW_PHASE, REVISION_PHASE, and FINALIZE_PHASE...")
        print("Integrating literature findings and refining draft...")
        
        # Complete the research process
        final_response = prompt_engineer.integrate_literature(
            initial_response=initial_response,
            papers=all_papers
        )
        
        # Output final results
        print("\nFinal Summary:")
        # Print just the first few lines
        final_lines = final_response.final_summary.split('\n')[:10]
        for line in final_lines:
            print(f"  {line}")
        if len(final_lines) < len(final_response.final_summary.split('\n')):
            print("  ...")
        
        print("\nCitations:")
        for citation in final_response.citations[:3]:  # Show first 3 citations
            print(f"  {citation}")
        if len(final_response.citations) > 3:
            print(f"  ...and {len(final_response.citations) - 3} more")
        
        # Save results to file
        output_dir = Path(__file__).parent.parent / "outputs"
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"research_output_openai_{timestamp}.txt"
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"Research Query: {query}\n\n")
            
            f.write("=== 1. PLAN_PHASE ===\n")
            f.write("Query Components:\n")
            for i, comp in enumerate(components):
                f.write(f"\nComponent {i+1}: {comp.topic}\n")
                for j, subtopic in enumerate(comp.subtopics):
                    f.write(f"  • Subtopic {j+1}: {subtopic}\n")
            
            f.write("\n=== 2. RESEARCH_PHASE ===\n")
            f.write("Knowledge Gaps:\n")
            for i, gap in enumerate(knowledge_gaps):
                f.write(f"\nGap {i+1}: {gap['question']}\n")
                f.write(f"  Related to: {gap['topic']}\n")
                f.write(f"  Search terms: {', '.join(gap['search_terms'])}\n")
                f.write(f"  Importance: {gap['importance']}\n")
                
            f.write("\nInitial Response:\n")
            f.write(f"{initial_response}\n\n")
            
            f.write("Retrieved Papers:\n")
            for i, paper in enumerate(all_papers):
                f.write(f"\n{i+1}. {paper.title}\n")
                f.write(f"   Authors: {', '.join(paper.authors)}\n")
                f.write(f"   Year: {paper.year}\n")
                f.write(f"   Source: {paper.source_api}\n")
            
            f.write("\n=== 3-6. COMPOSING, REVIEW, REVISION, FINALIZE PHASES ===\n")
            f.write("Final Summary:\n")
            f.write(f"{final_response.final_summary}\n\n")
            
            f.write("Citations:\n")
            for citation in final_response.citations:
                f.write(f"{citation}\n")
        
        print(f"\nResults saved to {output_file}")
        
    except Exception as e:
        print(f"\nError processing query: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function to test the OpenAI deep research planning approach."""
    print("\n======================================================")
    print("  DEEP RESEARCHER - OPENAI PLANNING APPROACH")
    print("======================================================")
    print("\nThis script demonstrates a state-based research process")
    print("with context window management and finite search budget.")
    
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
        user_query = "What are the recent advances in quantum machine learning algorithms and their potential applications?"
        print(f"\nUsing default query: {user_query}")
    
    # Get external calls limit
    print("\nEnter maximum number of API calls to make (1-20, or press Enter for default 10):")
    max_calls_input = input().strip()
    
    try:
        max_calls = int(max_calls_input) if max_calls_input else 10
        max_calls = max(1, min(20, max_calls))  # Clamp between 1 and 20
    except ValueError:
        max_calls = 10
        print(f"\nUsing default call limit: {max_calls}")
    
    # Process the query
    process_query(user_query, prompt_engineer, literature_manager, max_calls)

if __name__ == "__main__":
    main() 