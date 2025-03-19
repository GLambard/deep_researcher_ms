"""
Test script for core functionality of Deep Researcher.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from the project root's .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

from src.search.literature_manager import LiteratureManager
from src.ollama_client import OllamaClient
from src.prompt_engineering import PromptEngineer

def check_environment():
    """Check if all required components are set up."""
    # Check Tavily API key
    if not os.getenv('TAVILY_API_KEY'):
        print("\nError: Tavily API key not found!")
        print("Please add your Tavily API key to the .env file:")
        print("TAVILY_API_KEY=your-api-key")
        return False
    
    return True

def setup_components():
    """Set up all necessary components."""
    try:
        # Initialize Ollama client
        print("\nInitializing Ollama client...")
        ollama = OllamaClient(
            model="deepseek-r1:8b",  # Changed to deepseek-r1:8b
            temperature=0.7
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

## TODO: 
# - Update the research_output.txt naming with a unique id for each query, e.g. research_output_<date_time>.txt
# 
def process_query(query: str, prompt_engineer: PromptEngineer, literature_manager: LiteratureManager):
    """Process a research query."""
    try:
        # Break down the query into components
        print("\nBreaking down query into components...")
        components = prompt_engineer.process_query(query)
        for comp in components:
            print(f"\nTopic: {comp.topic}")
            for sub in comp.subtopics:
                print(f"  - {sub}")
        
        # Generate initial response
        print("\nGenerating initial response...")
        initial_response = prompt_engineer.generate_initial_response(query, components)
        print("\nInitial Response:")
        print(initial_response)
        
        # Generate search queries
        print("\nGenerating search queries...")
        search_queries = prompt_engineer.generate_search_queries(components)
        print("Search queries:")
        for q in search_queries:
            print(f"  - {q}")
        
        # Search for papers
        print("\nSearching for relevant papers...")
        all_papers = []
        for search_query in search_queries:
            try:
                papers = literature_manager.search(
                    query=search_query,
                    max_papers=10  # Limit papers per query
                )
                all_papers.extend(papers)
            except Exception as e:
                print(f"Warning: Search failed for query '{search_query}': {e}")
        
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
        
        print(f"\nFound {len(all_papers)} papers:")
        for paper in all_papers:
            print(f"\nTitle: {paper.title}")
            print(f"Authors: {', '.join(paper.authors)}")
            print(f"Year: {paper.year}")
            print(f"Source: {paper.source_api}")
        
        # Integrate literature findings
        print("\nIntegrating literature findings...")
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
        
        output_file = output_dir / "research_output.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"Query: {query}\n\n")
            f.write("Initial Response:\n")
            f.write(f"{initial_response}\n\n")
            f.write("Found Papers:\n")
            for paper in all_papers:
                f.write(f"\nTitle: {paper.title}\n")
                f.write(f"Authors: {', '.join(paper.authors)}\n")
                f.write(f"Year: {paper.year}\n")
                f.write(f"Source: {paper.source_api}\n")
            f.write("\nFinal Summary:\n")
            f.write(f"{final_response.final_summary}\n\n")
            f.write("Citations:\n")
            for citation in final_response.citations:
                f.write(f"{citation}\n")
        
        print(f"\nResults saved to {output_file}")
        
    except Exception as e:
        print(f"\nError processing query: {e}")

def main():
    """Main function to test core functionality."""
    # Check environment
    if not check_environment():
        return
    
    # Set up components
    print("Setting up components...")
    ollama, literature_manager, prompt_engineer = setup_components()
    if not all((ollama, literature_manager, prompt_engineer)):
        return
    
    # Get query from user
    print("\nEnter your research query (or press Enter to use the default query):")
    user_query = input().strip()
    
    if not user_query:
        user_query = "What are the latest developments in single-cell RNA sequencing analysis methods for cancer research?"
        print(f"\nUsing default query: {user_query}")
    
    # Process the query
    process_query(user_query, prompt_engineer, literature_manager)

if __name__ == "__main__":
    main() 