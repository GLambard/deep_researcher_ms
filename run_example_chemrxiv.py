#!/usr/bin/env python
"""
Launcher script for the ChemRxiv adapter example.

This script ensures the example can be run from the project root directory
by adding the project root to the Python path.
"""

import os
import sys
import asyncio

# Add the project root to the Python path to enable absolute imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import the main function from the example script
from src.example_chemrxiv import main

if __name__ == "__main__":
    # Run the main function with asyncio
    asyncio.run(main()) 