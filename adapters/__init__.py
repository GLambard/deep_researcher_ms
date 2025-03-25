"""
Adapter modules for different scholarly literature sources.
"""
from .semantic_scholar import SemanticScholarAdapter
from .open_alex import OpenAlexAdapter
from .arxiv import ArXivAdapter
from .chemrxiv import ChemRxivAdapter

__all__ = ["SemanticScholarAdapter", "OpenAlexAdapter", "ArXivAdapter", "ChemRxivAdapter"]
