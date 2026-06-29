# TreeXplain - Explainable Deforestation Monitoring

[![CI](https://github.com/iadicarlo/treexplain/actions/workflows/ci.yml/badge.svg)](https://github.com/iadicarlo/treexplain/actions/workflows/ci.yml)
[![GitHub Pages](https://github.com/iadicarlo/treexplain/actions/workflows/deploy.yml/badge.svg)](https://github.com/iadicarlo/treexplain/actions/workflows/deploy.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**TreeXplain** is an explainable deforestation monitoring platform that uses satellite imagery and AI to detect and explain forest changes. Our platform provides transparent insights into deforestation patterns, helping researchers, policymakers, and conservationists understand and address forest loss.

## Features

- **Satellite Data Integration**: Fetch and process Sentinel-2, Landsat, and other satellite data via STAC API
- **AI-Powered Detection**: U-Net based model for accurate deforestation detection
- **Explainable AI**: Multiple XAI methods (SHAP, GradCAM, Integrated Gradients) for transparent predictions
- **Interactive Visualization**: Web-based interface with Leaflet maps
- **API Backend**: FastAPI backend with comprehensive endpoints
- **Automated CI/CD**: GitHub Actions workflows for testing and deployment
- **Data Pipeline**: Complete data preprocessing and augmentation pipeline
- **Model Training**: Full training infrastructure with early stopping and learning rate scheduling
- **Container Support**: Docker and Docker Compose for easy deployment

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
│       ├── ci.yml              # Continuous Integration
│       └── deploy.yml          # GitHub Pages deployment
├── docs/
│   └── index.html              # GitHub Pages frontend
├── notebooks/
│   └── data_exploration.ipynb   # Data exploration notebook
├── scripts/
│   ├── setup.sh                # Setup script
│   └── train.py                # Training script
├── src/
│   ├── __init__.py
│   ├── config.py               # Project configuration
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py             # FastAPI backend
│   │   └── schemas.py          # Pydantic schemas
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fetcher.py           # STAC API client
│   │   ├── pipeline.py          # Data pipeline
│   │   └── preprocessor.py     # Data preprocessing
│   ├── model/
│   │   ├── __init__.py
│   │   ├── deforestation.py     # U-Net model
│   │   └── trainer.py           # Model trainer
│   └── xai/
│       ├── __init__.py
│       └── explainer.py         # XAI methods
├── tests/
│   ├── __init__.py
│   ├── test_model.py           # Model tests
│   └── test_xai.py             # XAI tests
├── .env.example                # Environment variables template
├── .gitignore
├── .pre-commit-config.yaml    # Pre-commit hooks
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Docker Compose configuration
├── LICENSE                    # MIT License
├── pyproject.toml             # Project configuration
├── README.md
└── requirements.txt           # Dependencies
```

## Usage

### Fetch Satellite Data

```python
from treexplain.data.fetcher import fetch_sentinel2

# Fetch Sentinel-2 data for a bounding box
bbox = [-62.2159, -3.4653, -62.0, -3.2]  # Amazon region
items = fetch_sentinel2(bbox, days_back=7, limit=5)
```

### Preprocess Data

```python
from treexplain.data.pipeline import DataPipeline

pipeline = DataPipeline(batch_size=8)
images, labels = pipeline.get_sample_data(num_samples=20)
pipeline.create_datasets(images, labels)
dataloaders = pipeline.get_dataloaders()
```

### Train the Model

```python
from treexplain.model.trainer import ModelTrainer

trainer = ModelTrainer(
    num_epochs=50,
    patience=5,
    learning_rate=0.001
)

results = trainer.train(
    dataloaders['train'],
    dataloaders['val'],
    dataloaders['test']
)

# Plot training history
trainer.plot_training_history()
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
| GET | `/` | API information |
| GET | `/health` | Health check |
| GET | `/model` | Model information |
| POST | `/predict` | Single prediction with explanation |
| POST | `/predict/batch` | Batch predictions |
| POST | `/stac/search` | Search STAC catalog |
| GET | `/stac/search` | Search STAC catalog (GET) |

### Predict Endpoint

**Request:**
```json
{
    "image": "base64_encoded_numpy_array",
    "explain": true,
    "methods": ["shap", "gradcam"],
    "threshold": 0.5
}
```

**Response:**
```json
{
    "prediction": 1,
    "probability": 0.95,
    "explanation": {
        "shap": [...],
        "gradcam": [...]
    },
    "model_version": "0.1.0",
    "processing_time": 0.123,
    "message": "Prediction successful"
}
```

## Command Line Tools

### Training

```bash
# Train with sample data
python scripts/train.py --sample-data --epochs 10

# Train with custom parameters
python scripts/train.py --epochs 50 --batch-size 16 --learning-rate 0.001
```

### Setup

```bash
# Full setup (dependencies, pre-commit, tests)
chmod +x scripts/setup.sh
./scripts/setup.sh
```

## Docker Deployment

### Build and Run

```bash
# Build the image
docker build -t treexplain .

# Run the container
docker run -p 8000:8000 -v $(pwd)/data:/app/data treexplain
```

### Docker Compose

```bash
# Start all services
docker-compose up -d

# Stop services
docker-compose down

# With development profile (includes Jupyter)
docker-compose --profile dev up -d
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

# Update requirements.txt
uv pip compile pyproject.toml -o requirements.txt
```

## Environment Variables

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

Key variables:
- `STAC_API_URL`: STAC API endpoint
- `API_HOST`: API host address
- `API_PORT`: API port
- `MODEL_PATH`: Path to trained model
- `LOG_LEVEL`: Logging level

## Project Checklist

- [x] Repository created on GitHub
- [x] uv installed and configured
- [x] Project structure created
- [x] Core Python modules implemented
- [x] GitHub Pages frontend created
- [x] GitHub Actions workflows configured
- [x] Unit tests written
- [x] Pre-commit hooks configured
- [x] First commits made with proper attribution
- [x] CI/CD pipeline ready
- [x] GitHub Pages deployment configured
- [x] Data pipeline implemented
- [x] Model training infrastructure
- [x] API endpoints extended
- [x] Docker support added
- [x] Notebooks for exploration
- [x] Command-line tools

## Next Steps

### Priority 1: Train with Real Data

1. **Get STAC API Access**: Sign up for Planetary Computer or other STAC provider
2. **Collect Labeled Data**: Gather satellite images with deforestation labels
3. **Train the Model**: Run `python scripts/train.py` with real data
4. **Evaluate Performance**: Test on validation set and tune hyperparameters

### Priority 2: Deploy Backend

1. **Choose Hosting**: Render, Railway, Fly.io, or AWS
2. **Configure Environment**: Set up environment variables and secrets
3. **Deploy**: Push your trained model and start the API service
4. **Monitor**: Set up logging and monitoring

### Priority 3: Enhance Frontend

1. **Connect to API**: Update frontend to call your deployed API
2. **Add Upload**: Allow users to upload their own images
3. **Visualize Results**: Display predictions and explanations interactively
4. **Add Maps**: Integrate with Mapbox or Google Maps for better visualization

### Priority 4: Advanced Features

1. **Time Series Analysis**: Track deforestation over time
2. **Multi-Model Support**: Add different models for comparison
3. **Ensemble Methods**: Combine multiple models for better accuracy
4. **Advanced XAI**: Add LIME, attention mechanisms, etc.

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
- [uv](https://astral.sh/uv) for Python package management

## Contact

For questions or feedback, please open an issue on GitHub.
