#!/usr/bin/env python
"""
Quick test script for query validation only.
"""

import sys
import os
import re

# Add the project root directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def validate_and_fix_query(query: str) -> str:
    """
    Validate and fix the query to ensure it's properly formatted.
    """
    # Check for truncated or unbalanced parentheses
    open_count = query.count('(')
    close_count = query.count(')')
    
    # If unbalanced, attempt to fix
    if open_count != close_count:
        # If more opening parentheses, add closing ones
        if open_count > close_count:
            query = query + ')' * (open_count - close_count)
        # If more closing parentheses, add opening ones at appropriate locations
        else:
            # This is a simplification - in practice, we'd need more sophisticated parsing
            query = '(' * (close_count - open_count) + query
    
    # Check for AND/OR operators in all caps for proper Boolean syntax
    query = re.sub(r'\b(and)\b', 'AND', query, flags=re.IGNORECASE)
    query = re.sub(r'\b(or)\b', 'OR', query, flags=re.IGNORECASE)
    
    # Preserve already properly quoted terms
    # First, temporarily replace quoted terms with placeholders
    quoted_terms = []
    
    def replace_quoted(match):
        quoted_terms.append(match.group(0))
        return f"__QUOTED_TERM_{len(quoted_terms)-1}__"
    
    # Replace quoted terms with placeholders
    temp_query = re.sub(r'"[^"]*"', replace_quoted, query)
    
    # Now quote unquoted terms
    def ensure_quoted(match):
        term = match.group(1)
        # Skip placeholders and operators
        if term.startswith("__QUOTED_TERM_") or term.upper() in ["AND", "OR"]:
            return term
        return f'"{term}"'
    
    # Quote terms that aren't already quoted and aren't operators
    temp_query = re.sub(r'\b([^\s"()AND|OR]+)\b', ensure_quoted, temp_query)
    
    # Restore original quoted terms
    for i, term in enumerate(quoted_terms):
        temp_query = temp_query.replace(f"__QUOTED_TERM_{i}__", term)
    
    # Fix double spaces and trim
    query = re.sub(r'\s+', ' ', temp_query).strip()
    
    return query

def main():
    """Test the query validation function."""
    test_queries = [
        '("Conformal Prediction" AND "reaction kinetics") AND ("metal',
        '(large language models AND materials science',
        'conformal prediction or bayesian optimization',
        'machine learning drug discovery',
        # Add test cases with already quoted terms
        '"Conformal Prediction" AND reaction kinetics AND "metal catalysis"',
        '("quantum computing" OR "quantum machine learning") AND optimization',
    ]
    
    print("\n=== TESTING QUERY VALIDATION ===")
    for i, query in enumerate(test_queries):
        fixed_query = validate_and_fix_query(query)
        print(f"\nOriginal Query {i+1}: {query}")
        print(f"Fixed Query {i+1}: {fixed_query}")
        
    print("\nDemonstrating that the function preserves legitimate queries:")
    proper_query = '("Conformal Prediction" AND "reaction kinetics") AND ("metal catalysis")'
    fixed_proper = validate_and_fix_query(proper_query)
    print(f"Original: {proper_query}")
    print(f"Fixed: {fixed_proper}")
    
    print("\nDemonstrating fixing a truncated query from the example:")
    truncated_query = '("Conformal Prediction" AND "reaction kinetics") AND ("metal'
    fixed_truncated = validate_and_fix_query(truncated_query)
    print(f"Original (truncated): {truncated_query}")
    print(f"Fixed: {fixed_truncated}")

if __name__ == "__main__":
    main() 