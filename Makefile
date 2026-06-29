# TreeXplain Makefile
# Common development tasks

.PHONY: help install test lint format clean train serve notebook docker

# Default target
help:
	@echo "TreeXplain Makefile - Common Tasks"
	@echo "==================================="
	@echo ""
	@echo "Development:"
	@echo "  make install    - Install dependencies"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run linter"
	@echo "  make format     - Format code"
	@echo "  make clean      - Clean build artifacts"
	@echo ""
	@echo "Training & Running:"
	@echo "  make train      - Train the model"
	@echo "  make serve      - Start API server"
	@echo "  make notebook   - Start Jupyter notebook"
	@echo ""
	@echo "Deployment:"
	@echo "  make docker     - Build Docker image"
	@echo "  make docker-run - Run Docker container"
	@echo ""

# Install dependencies
install:
	uv pip install -e ".[dev]"
	uv pip install pre-commit
	pre-commit install

# Run tests
test:
	pytest tests/ -v

# Run linter
lint:
	ruff check src/
	mypy src/

# Format code
format:
	black src/ tests/
	ruff check --fix src/ tests/

# Clean
clean:
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .venv/
	rm -rf data/
	rm -rf models/
	rm -rf output/

# Train model
train:
	python scripts/train.py --sample-data --epochs 10

# Start API server
serve:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Start Jupyter notebook
notebook:
	jupyter notebook notebooks/ --ip=0.0.0.0 --port=8888 --no-browser

# Docker build
docker:
	docker build -t treexplain .

# Docker run
docker-run:
	docker run -p 8000:8000 -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models treexplain

# Full setup
setup:
	./scripts/setup.sh

# Pre-commit checks
precommit:
	pre-commit run --all-files

# Generate requirements
gen-requirements:
	uv pip compile pyproject.toml -o requirements.txt
