#!/usr/bin/env python3
"""
Verify Deep Researcher installation and dependencies.
"""

import sys
import requests
import importlib
import subprocess
from typing import List, Tuple

def check_python_version() -> Tuple[bool, str]:
    """Check Python version."""
    version = sys.version_info
    required = (3, 9)
    if version >= required:
        return True, f"Python version {version.major}.{version.minor} OK"
    return False, f"Python version {required[0]}.{required[1]} or higher required"

def check_package(package: str) -> Tuple[bool, str]:
    """Check if a package is installed."""
    try:
        importlib.import_module(package)
        return True, f"Package {package} OK"
    except ImportError:
        return False, f"Package {package} not found"

def check_ollama() -> Tuple[bool, str]:
    """Check if Ollama is running and model is available."""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            if any(m["name"] == "mistral" for m in models):
                return True, "Ollama running and mistral model available"
            return False, "Ollama running but mistral model not found"
        return False, "Ollama not responding"
    except:
        return False, "Ollama not running"

def main():
    """Run all checks."""
    required_packages = [
        "requests",
        "pandas",
        "numpy",
        "pytest",
        "black",
        "flake8",
        "tqdm",
        "responses",
        "dotenv"
    ]
    
    checks = []
    
    # Check Python version
    checks.append(check_python_version())
    
    # Check packages
    for package in required_packages:
        checks.append(check_package(package))
    
    # Check Ollama
    checks.append(check_ollama())
    
    # Print results
    print("\nDeep Researcher Setup Verification")
    print("=================================")
    
    all_ok = True
    for success, message in checks:
        status = "✓" if success else "✗"
        print(f"{status} {message}")
        if not success:
            all_ok = False
    
    print("\nVerification Summary")
    print("===================")
    if all_ok:
        print("✓ All checks passed! Deep Researcher is ready to use.")
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 