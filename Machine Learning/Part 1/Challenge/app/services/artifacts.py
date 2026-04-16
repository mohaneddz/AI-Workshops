from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import joblib

from app.config import METADATA_FILE, MODEL_FILE, MODELS_DIR


def save_bundle(bundle: Dict[str, Any]) -> None:
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_FILE)
    metadata = {
        "model_name": bundle.get("model_name", "LinearRegression"),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": bundle.get("metrics", {}),
        "feature_columns": bundle.get("feature_columns", []),
        "feature_numeric": bundle.get("feature_numeric", []),
        "feature_binary": bundle.get("feature_binary", []),
        "feature_categorical": bundle.get("feature_categorical", []),
        "catalog": bundle.get("catalog", {}),
    }
    METADATA_FILE.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")


@lru_cache(maxsize=1)
def load_bundle() -> Dict[str, Any]:
    if not MODEL_FILE.exists():
        raise FileNotFoundError("Model bundle does not exist yet")
    return joblib.load(MODEL_FILE)


def load_model_bundle(model_key: str) -> Dict[str, Any]:
    path = MODELS_DIR / f"bundle_{model_key}.joblib"
    if not path.exists():
        # Fallback to default
        return load_bundle()
    return joblib.load(path)


def load_model_registry() -> Dict[str, Any]:
    registry_path = MODELS_DIR / "model_registry.json"
    if not registry_path.exists():
        return {}
    return json.loads(registry_path.read_text(encoding="utf-8"))


def bundle_exists() -> bool:
    return MODEL_FILE.exists()


def any_bundle_exists() -> bool:
    """Check if at least one model bundle exists."""
    if MODEL_FILE.exists():
        return True
    return any(MODELS_DIR.glob("bundle_*.joblib"))


def clear_bundle_cache() -> None:
    load_bundle.cache_clear()
