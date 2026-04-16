from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
TEMPLATES_DIR = BASE_DIR / "app" / "templates"
STATIC_DIR = BASE_DIR / "app" / "static"

DATA_FILE = DATA_DIR / "data_preprocessed.csv"
MODEL_FILE = MODELS_DIR / "laptop_price_bundle.joblib"
METADATA_FILE = MODELS_DIR / "laptop_price_metadata.json"

APP_TITLE = "Algerian Laptop Price Predictor"
APP_DESCRIPTION = "FastAPI laptop price prediction website"
APP_VERSION = "1.0.0"
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 1234
