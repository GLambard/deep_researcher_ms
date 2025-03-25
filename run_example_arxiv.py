#!/usr/bin/env python3
"""
Launcher script for the arXiv example application.

This script serves as an entry point to run the example_arxiv.py script from 
the project root directory. It addresses Python import path issues by adding the
project root to sys.path, allowing the example script to use absolute imports.

Without this launcher, executing src/example_arxiv.py directly would result
in a ModuleNotFoundError due to Python import system limitations.

Usage:
    Run this script from the project root directory:
    $ python run_example_arxiv.py
"""
import asyncio
import os
import sys

# Add the project root to the Python path to enable imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the main function from the example script
from src.example_arxiv import main

if __name__ == "__main__":
    # Run the async main function with asyncio.run
    # This is the proper way to run an asynchronous entry point
    asyncio.run(main()) 