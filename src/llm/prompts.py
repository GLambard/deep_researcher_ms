"""
LLM prompts for research analysis and summarization.
"""

RESEARCH_PROMPTS = {
    "comprehensive_summary": """
    Please provide a comprehensive summary of the following research papers. 
    Focus on synthesizing the key findings, methodologies, and conclusions across all papers.
    Include specific references to papers when discussing their findings.
    
    Papers:
    {papers}
    
    Please structure your response as follows:
    1. Overview
    2. Key Findings
    3. Methodologies Used
    4. Conclusions and Implications
    5. References
    
    Make sure to:
    - Highlight connections between different papers
    - Note any conflicting findings
    - Identify emerging trends or patterns
    - Include specific citations when discussing findings
    """,
    
    "methodology_analysis": """
    Analyze the methodologies used across these papers:
    {papers}
    
    Please provide:
    1. A comparison of different approaches used
    2. Strengths and limitations of each method
    3. Recommendations for future research
    """,
    
    "findings_synthesis": """
    Synthesize the key findings from these papers:
    {papers}
    
    Please provide:
    1. Main findings and their significance
    2. Supporting evidence for each finding
    3. Any contradictions or debates in the field
    4. Potential implications for future research
    """,
    
    "research_gaps": """
    Analyze these papers to identify research gaps:
    {papers}
    
    Please provide:
    1. Current limitations in the field
    2. Areas that need further investigation
    3. Potential future research directions
    4. Recommendations for addressing these gaps
    """
} 