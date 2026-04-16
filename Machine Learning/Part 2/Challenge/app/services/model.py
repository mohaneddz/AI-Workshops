from __future__ import annotations

import json
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from app.config import DATA_FILE, MODEL_META_FILE, MODELS_DIR

FEATURE_COLUMNS = ["time_s", "amount", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8"]
DEFAULT_MODEL_KEY = "isolation_forest"
MODEL_REGISTRY = {
    "z_score": {"name": "Z-Score", "description": "Statistical distance using feature z-scores"},
    "isolation_forest": {"name": "Isolation Forest", "description": "Tree-based anomaly isolation"},
    "lof": {"name": "Local Outlier Factor", "description": "Local neighborhood density outlier detection"},
}


def _load_training_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing data file: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    required = set(FEATURE_COLUMNS + ["id"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in data: {sorted(missing)}")
    return df


def _model_path(model_key: str):
    return MODELS_DIR / f"{model_key}.joblib"


def _safe_std(series: pd.Series) -> float:
    value = float(series.std(ddof=0))
    return value if value > 1e-12 else 1.0


def _fit_z_score(X: pd.DataFrame) -> Dict[str, Any]:
    means = {col: float(X[col].mean()) for col in FEATURE_COLUMNS}
    stds = {col: _safe_std(X[col]) for col in FEATURE_COLUMNS}
    z = np.abs((X - pd.Series(means)) / pd.Series(stds))
    scores = z.max(axis=1).to_numpy()
    threshold = float(np.quantile(scores, 0.98))
    return {
        "model_key": "z_score",
        "feature_columns": FEATURE_COLUMNS,
        "mean": means,
        "std": stds,
        "score_threshold": threshold,
    }


def _fit_isolation_forest(X: pd.DataFrame, random_state: int) -> Dict[str, Any]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = IsolationForest(n_estimators=250, contamination=0.01, random_state=random_state)
    model.fit(X_scaled)
    scores = -model.decision_function(X_scaled)
    preds = model.predict(X_scaled)
    anomaly_scores = scores[preds == -1]
    threshold = float(anomaly_scores.min()) if len(anomaly_scores) else float(np.quantile(scores, 0.99))
    return {
        "model_key": "isolation_forest",
        "feature_columns": FEATURE_COLUMNS,
        "scaler": scaler,
        "model": model,
        "score_threshold": threshold,
    }


def _fit_lof(X: pd.DataFrame) -> Dict[str, Any]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LocalOutlierFactor(n_neighbors=20, contamination=0.01, novelty=True)
    model.fit(X_scaled)
    scores = -model.decision_function(X_scaled)
    preds = model.predict(X_scaled)
    anomaly_scores = scores[preds == -1]
    threshold = float(anomaly_scores.min()) if len(anomaly_scores) else float(np.quantile(scores, 0.99))
    return {
        "model_key": "lof",
        "feature_columns": FEATURE_COLUMNS,
        "scaler": scaler,
        "model": model,
        "score_threshold": threshold,
    }


def train_and_save_models(random_state: int = 42) -> Dict[str, Any]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = _load_training_data()
    X = df[FEATURE_COLUMNS].copy()

    artifacts = {
        "z_score": _fit_z_score(X),
        "isolation_forest": _fit_isolation_forest(X, random_state=random_state),
        "lof": _fit_lof(X),
    }
    for model_key, artifact in artifacts.items():
        joblib.dump(artifact, _model_path(model_key))

    metadata: Dict[str, Any] = {
        "rows": int(len(df)),
        "feature_columns": FEATURE_COLUMNS,
        "default_model": DEFAULT_MODEL_KEY,
        "models": {
            key: {
                "name": MODEL_REGISTRY[key]["name"],
                "description": MODEL_REGISTRY[key]["description"],
                "score_threshold": float(artifacts[key]["score_threshold"]),
            }
            for key in MODEL_REGISTRY
        },
    }
    MODEL_META_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def ensure_models() -> Dict[str, Any]:
    if not MODEL_META_FILE.exists():
        return train_and_save_models()
    meta = json.loads(MODEL_META_FILE.read_text(encoding="utf-8"))
    if "models" not in meta or "default_model" not in meta:
        return train_and_save_models()
    if any(key not in meta["models"] for key in MODEL_REGISTRY):
        return train_and_save_models()
    missing = [key for key in MODEL_REGISTRY if not _model_path(key).exists()]
    if missing:
        return train_and_save_models()
    return meta


def _validate_model_key(model_key: str) -> str:
    return model_key if model_key in MODEL_REGISTRY else DEFAULT_MODEL_KEY


def load_model_artifact(model_key: str) -> Dict[str, Any]:
    ensure_models()
    safe_key = _validate_model_key(model_key)
    return joblib.load(_model_path(safe_key))


def infer_anomalies(model_key: str, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float, str]:
    safe_key = _validate_model_key(model_key)
    artifact = load_model_artifact(safe_key)
    threshold = float(artifact["score_threshold"])

    if safe_key == "z_score":
        means = pd.Series(artifact["mean"])
        stds = pd.Series(artifact["std"]).replace(0.0, 1.0)
        z = np.abs((X[FEATURE_COLUMNS] - means) / stds)
        scores = z.max(axis=1).to_numpy(dtype=float)
    else:
        scaler = artifact["scaler"]
        model = artifact["model"]
        transformed = scaler.transform(X[FEATURE_COLUMNS])
        scores = -model.decision_function(transformed)

    flags = scores >= threshold
    return scores.astype(float), flags.astype(bool), threshold, safe_key


def get_model_options() -> list[Dict[str, str]]:
    return [{"key": key, "name": value["name"], "description": value["description"]} for key, value in MODEL_REGISTRY.items()]
