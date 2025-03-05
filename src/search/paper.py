"""
Paper data model for standardized representation across different APIs.
"""

from typing import List, Optional
from dataclasses import dataclass
import hashlib

@dataclass
class Paper:
    """
    Standardized paper representation across different APIs.
    """
    title: str
    abstract: str
    authors: List[str]
    year: Optional[int]
    doi: Optional[str]
    url: Optional[str]
    source_api: str
    venue: Optional[str] = None  # Journal or conference name
    
    def get_hash(self) -> str:
        """Get a unique hash for the paper based on title and authors."""
        content = f"{self.title.lower()}{''.join(sorted(self.authors))}".encode('utf-8')
        return hashlib.md5(content).hexdigest() 