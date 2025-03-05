.PHONY: setup test lint clean verify

# Setup environment
setup:
	conda env create -f environment.yml
	conda activate deep_researcher
	python scripts/verify_setup.py

# Run tests
test:
	PYTHONPATH=. pytest tests/ -v

# Run tests with coverage
coverage:
	PYTHONPATH=. pytest tests/ --cov=src --cov-report=term-missing

# Run style checks
lint:
	black src/ tests/ --check
	flake8 src/ tests/

# Format code
format:
	black src/ tests/

# Clean up
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +
	find . -type d -name "*.egg" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name ".coverage" -exec rm -r {} +

# Verify installation
verify:
	python scripts/verify_setup.py

# Run all checks
check: verify lint test

# Help
help:
	@echo "Available commands:"
	@echo "  make setup     - Create and setup conda environment"
	@echo "  make test      - Run tests"
	@echo "  make coverage  - Run tests with coverage report"
	@echo "  make lint      - Check code style"
	@echo "  make format    - Format code with black"
	@echo "  make clean     - Clean up temporary files"
	@echo "  make verify    - Verify installation"
	@echo "  make check     - Run all checks" 