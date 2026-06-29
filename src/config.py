"""Configuration for TreeXplain."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
INPUT_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
IMAGE_SIZE = 256
