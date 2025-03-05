# Deep Researcher scRNA

A powerful research assistant that leverages the Tavily API for academic literature search and local LLMs through Ollama for text generation and analysis. Specifically designed for single-cell RNA sequencing research.

## Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running locally
- Tavily API key (get one from [Tavily](https://tavily.com/))

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd deep_researcher_scrna
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.template .env
```
Edit `.env` and add your Tavily API key:
```
TAVILY_API_KEY=your-api-key-here
```

4. Start Ollama server:
- Windows:
  ```cmd
  ollama serve
  ```
- Linux/MacOS:
  ```bash
  ollama serve
  ```

## System Architecture

Deep Researcher uses a multi-agent system architecture to provide comprehensive research assistance:

### 1. Query Processing Agent (Prompt Engineer)
- Breaks down complex research queries into manageable components
- Identifies key topics and subtopics
- Generates targeted search queries for literature search
- Uses deepseek-r1:8b model for natural language understanding

### 2. Literature Search Agent (Literature Manager)
- Interfaces with Tavily API for academic paper searches
- Filters and ranks papers based on relevance
- Extracts key metadata (authors, year, title)
- Focuses on academic and research-oriented sources

### 3. Integration Agent (Ollama Client)
- Synthesizes information from multiple sources
- Generates coherent summaries using deepseek-r1:8b model
- Maintains academic rigor with temperature=0.3
- Provides citations and references

## Information Flow

1. User Query → Query Processing Agent
   - Query decomposition
   - Topic identification
   - Search query generation

2. Search Queries → Literature Search Agent
   - Academic paper search
   - Metadata extraction
   - Relevance ranking

3. Search Results → Integration Agent
   - Information synthesis
   - Summary generation
   - Citation compilation

## Testing the System

Run the test script to verify everything is working:

```bash
python scripts/test_core_functionality.py
```

The script will:
1. Check if all required components are set up correctly
2. Initialize the Ollama client with the deepseek-r1:8b model
3. Set up the literature manager for academic paper searches
4. Allow you to input a research query or use the default one
5. Generate an initial response
6. Search for relevant papers
7. Integrate the findings into a final summary
8. Save the results to `outputs/research_output.txt`

### Example Query and Output

The default query is:
```
What are the latest developments in single-cell RNA sequencing analysis methods for cancer research?
```

Example output structure:
```
Query: What are the latest developments in single-cell RNA sequencing analysis methods for cancer research?

Initial Response:
Recent developments in single-cell RNA sequencing (scRNA-seq) analysis for cancer research have focused on improving...

Found Papers:
Title: Advances in single-cell RNA sequencing and its applications in cancer research
Authors: Smith J., Johnson M., et al.
Year: 2023
Source: Tavily

Title: Integration of spatial transcriptomics with scRNA-seq in tumor analysis
Authors: Zhang L., Williams K., et al.
Year: 2023
Source: Tavily

[Additional papers...]

Final Summary:
Recent advances in scRNA-seq analysis methods for cancer research have made significant strides in several key areas:
1. Enhanced clustering algorithms for tumor heterogeneity analysis
2. Integration of spatial information with transcriptomic data
3. Development of machine learning approaches for cell type identification
4. Improved methods for trajectory inference in cancer progression

Citations:
1. Smith et al. (2023) demonstrated novel clustering approaches...
2. Zhang and Williams (2023) integrated spatial transcriptomics...
[Additional citations...]
```

## System Components

- **Ollama Client**: Uses the `deepseek-r1:8b` model with a temperature of 0.3 for focused and consistent responses
- **Literature Manager**: Interfaces with Tavily API for academic paper searches
- **Prompt Engineer**: Processes queries and integrates literature findings

## Future Development

### Planned Features

1. **Web Interface**
   - Local web UI for more interactive user experience
   - Real-time query processing and results visualization
   - Interactive paper exploration and summary generation

### Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Troubleshooting

1. **Ollama Server Issues**
   - Ensure Ollama is installed
   - Check if the server is running with `ollama serve`
   - Verify the model is downloaded (will be automatic on first run)

2. **API Issues**
   - Verify your Tavily API key is correctly set in `.env`
   - Check your internet connection
   - Ensure you're not hitting API rate limits

## License

GNU GENERAL PUBLIC LICENSE