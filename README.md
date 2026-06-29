# TreeXplain - Explainable Deforestation Monitoring

[![CI](https://github.com/iadicarlo/treexplain/actions/workflows/ci.yml/badge.svg)](https://github.com/iadicarlo/treexplain/actions/workflows/ci.yml)
[![GitHub Pages](https://github.com/iadicarlo/treexplain/actions/workflows/deploy.yml/badge.svg)](https://github.com/iadicarlo/treexplain/actions/workflows/deploy.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**TreeXplain** is an explainable deforestation monitoring platform that uses satellite imagery and AI to detect and explain forest changes. Our platform provides transparent insights into deforestation patterns, helping researchers, policymakers, and conservationists understand and address forest loss.

## Features

- **Satellite Data Integration**: Fetch and process Sentinel-2, Landsat, and other satellite data
- **AI-Powered Detection**: U-Net based model for accurate deforestation detection
- **Explainable AI**: Multiple XAI methods (SHAP, GradCAM, Integrated Gradients) for transparent predictions
- **Interactive Visualization**: Web-based interface with Leaflet maps
- **API Backend**: FastAPI backend for model inference and explanations
- **Automated CI/CD**: GitHub Actions workflows for testing and deployment

## Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/iadicarlo/treexplain.git
cd treexplain

# Install uv (if not already installed)
curl -Ls https://astral.sh/uv/install.sh | sh

# Install dependencies
uv pip install -e ".[dev]"

# Set up pre-commit hooks
uv pip install pre-commit
pre-commit install
```

### 2. Run Tests

```bash
pytest tests/ -v
```

### 3. Start the API

```bash
uvicorn src.api.main:app --reload
```

The API will be available at `http://localhost:8000`. Try the health endpoint:

```bash
curl http://localhost:8000/health
```

### 4. View the Frontend

The GitHub Pages frontend is automatically deployed and available at:
`https://iadicarlo.github.io/treexplain/`

## Project Structure

```
treexplain/
├── .github/
│   └── workflows/
│       ├── ci.yml          # Continuous Integration
│       └── deploy.yml      # GitHub Pages deployment
├── docs/
│   └── index.html          # GitHub Pages frontend
├── src/
│   ├── __init__.py
│   ├── config.py           # Project configuration
│   ├── api/
│   │   └── main.py         # FastAPI backend
│   ├── data/
│   │   ├── __init__.py
│   │   └── fetcher.py       # Satellite data fetching
│   ├── model/
│   │   ├── __init__.py
│   │   └── deforestation.py # U-Net model
│   └── xai/
│       ├── __init__.py
│       └── explainer.py     # XAI methods
├── tests/
│   ├── __init__.py
│   ├── test_model.py       # Model tests
│   └── test_xai.py         # XAI tests
├── .gitignore
├── .pre-commit-config.yaml
├── pyproject.toml          # Project configuration
└── README.md
```

## Usage

### Fetch Satellite Data

```python
from treexplain.data.fetcher import fetch_sentinel2

# Fetch Sentinel-2 data for a bounding box
bbox = [-62.2159, -3.4653, -62.0, -3.2]  # Amazon region
items = fetch_sentinel2(bbox, days_back=7, limit=5)
```

### Make Predictions

```python
import torch
from treexplain.model.deforestation import DeforestationModel

# Load model
model = DeforestationModel()

# Create dummy input (batch, channels, height, width)
input_tensor = torch.randn(1, 6, 256, 256)

# Get prediction
with torch.no_grad():
    output = model(input_tensor)
    probability = output.item()
    prediction = 1 if probability > 0.5 else 0
```

### Generate Explanations

```python
from treexplain.xai.explainer import TreeXplainExplainer

# Create explainer
explainer = TreeXplainExplainer(model)

# Set background for SHAP
explainer.set_background(torch.zeros(5, 6, 256, 256))

# Generate comprehensive report
report = explainer.generate_report(
    input_tensor, 
    methods=["shap", "gradcam", "integrated_gradients"]
)
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/predict` | Make prediction with explanation |

### Predict Endpoint

**Request:**
```json
{
    "image": "base64_encoded_numpy_array",
    "explain": true,
    "methods": ["shap", "gradcam"]
}
```

**Response:**
```json
{
    "prediction": 1.0,
    "probability": 0.95,
    "explanation": {
        "shap": [...],
        "gradcam": [...]
    },
    "message": "Prediction successful"
}
```

## Development

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest --cov=src --cov-report=html tests/
```

### Code Quality

```bash
# Run linter
ruff check src/

# Format code
black src/

# Run pre-commit checks
pre-commit run --all-files
```

### Adding Dependencies

```bash
# Add a new dependency
uv pip add package_name

# Add development dependency
uv pip add -D package_name

# Update pyproject.toml
uv pip compile pyproject.toml -o requirements.txt
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Planetary Computer](https://planetarycomputer.microsoft.com/) for satellite data
- [PyTorch](https://pytorch.org/) for deep learning framework
- [FastAPI](https://fastapi.tiangolo.com/) for API backend
- [Leaflet](https://leafletjs.com/) for interactive maps
- [SHAP](https://github.com/slundberg/shap) and [Captum](https://captum.ai/) for explainable AI

## Contact

For questions or feedback, please open an issue on GitHub.
