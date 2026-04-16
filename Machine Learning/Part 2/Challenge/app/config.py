from __future__ import annotations

from pathlib import Path

APP_TITLE = "Global Network Anomaly Monitor"
APP_DESCRIPTION = "Real-time network flow monitor with automatic anomaly flagging"
APP_VERSION = "1.0.0"
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 1240

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "connections.csv"
MODELS_DIR = BASE_DIR / "models"
MODEL_FILE = MODELS_DIR / "isolation_forest.joblib"
MODEL_META_FILE = MODELS_DIR / "model_metadata.json"

TEMPLATES_DIR = BASE_DIR / "app" / "templates"
STATIC_DIR = BASE_DIR / "app" / "static"
