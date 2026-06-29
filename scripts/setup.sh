#!/bin/bash

# TreeXplain Setup Script
# This script sets up the development environment

set -e

echo "=========================================="
echo "TreeXplain Setup Script"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: Please run this script from the project root directory."
    exit 1
fi

# Step 1: Install uv if not already installed
echo "Step 1: Checking uv installation..."
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -Ls https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "uv is already installed."
fi

# Step 2: Create virtual environment
echo ""
echo "Step 2: Creating virtual environment..."
if [ ! -d ".venv" ]; then
    uv venv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

# Step 3: Install dependencies
echo ""
echo "Step 3: Installing dependencies..."
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    uv pip install -e ".[dev]"
    echo "Dependencies installed."
else
    echo "Error: Could not activate virtual environment."
    exit 1
fi

# Step 4: Install pre-commit hooks
echo ""
echo "Step 4: Installing pre-commit hooks..."
uv pip install pre-commit
pre-commit install
pre-commit run --all-files || echo "Pre-commit checks completed (some files may need fixing)"

# Step 5: Run tests
echo ""
echo "Step 5: Running tests..."
pytest tests/ -v --tb=short || echo "Tests completed (some may have failed)"

# Step 6: Generate requirements.txt
echo ""
echo "Step 6: Generating requirements.txt..."
uv pip compile pyproject.toml -o requirements.txt

# Step 7: Create necessary directories
echo ""
echo "Step 7: Creating directories..."
mkdir -p data models output

# Step 8: Summary
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To start developing:"
echo "  1. Activate the virtual environment:"
echo "     source .venv/bin/activate"
echo ""
echo "  2. Start the API:"
echo "     uvicorn src.api.main:app --reload"
echo ""
echo "  3. Run the notebook:"
echo "     jupyter notebook notebooks/data_exploration.ipynb"
echo ""
echo "  4. Train the model:"
echo "     python scripts/train.py"
echo ""
echo "  5. Run tests:"
echo "     pytest tests/ -v"
echo ""
echo "GitHub Pages will be available at:"
echo "  https://iadicarlo.github.io/treexplain/"
echo ""
