from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from app.config import DATA_FILE
from app.services.model import FEATURE_COLUMNS, MODEL_REGISTRY, infer_anomalies

GLOBAL_NODES = [
    {"name": "San Francisco", "country": "US", "lat": 37.7749, "lon": -122.4194, "asn": "AS15169"},
    {"name": "New York", "country": "US", "lat": 40.7128, "lon": -74.0060, "asn": "AS7018"},
    {"name": "London", "country": "GB", "lat": 51.5074, "lon": -0.1278, "asn": "AS3356"},
    {"name": "Paris", "country": "FR", "lat": 48.8566, "lon": 2.3522, "asn": "AS3215"},
    {"name": "Frankfurt", "country": "DE", "lat": 50.1109, "lon": 8.6821, "asn": "AS3320"},
    {"name": "Dubai", "country": "AE", "lat": 25.2048, "lon": 55.2708, "asn": "AS5384"},
    {"name": "Algiers", "country": "DZ", "lat": 36.7538, "lon": 3.0588, "asn": "AS36947"},
    {"name": "Johannesburg", "country": "ZA", "lat": -26.2041, "lon": 28.0473, "asn": "AS3741"},
    {"name": "Mumbai", "country": "IN", "lat": 19.0760, "lon": 72.8777, "asn": "AS55836"},
    {"name": "Singapore", "country": "SG", "lat": 1.3521, "lon": 103.8198, "asn": "AS4657"},
    {"name": "Tokyo", "country": "JP", "lat": 35.6762, "lon": 139.6503, "asn": "AS2516"},
    {"name": "Sydney", "country": "AU", "lat": -33.8688, "lon": 151.2093, "asn": "AS1221"},
    {"name": "Sao Paulo", "country": "BR", "lat": -23.5505, "lon": -46.6333, "asn": "AS28573"},
]

RNG = np.random.default_rng(42)


def _load_source_df() -> pd.DataFrame:
    return pd.read_csv(DATA_FILE)


def _pick_node(exclude_idx: int | None = None) -> tuple[int, Dict[str, Any]]:
    idx = int(RNG.integers(0, len(GLOBAL_NODES)))
    if exclude_idx is not None and idx == exclude_idx:
        idx = (idx + 1) % len(GLOBAL_NODES)
    return idx, GLOBAL_NODES[idx]


def generate_events(batch_size: int = 30, model_key: str = "isolation_forest") -> List[Dict[str, Any]]:
    df = _load_source_df()
    sample = df.sample(n=min(batch_size, len(df)), random_state=int(RNG.integers(0, 1_000_000))).copy()

    X = sample[FEATURE_COLUMNS].copy()
    scores, flags, threshold, safe_key = infer_anomalies(model_key=model_key, X=X)
    model_name = MODEL_REGISTRY[safe_key]["name"]

    events: List[Dict[str, Any]] = []
    for (_, row), score, is_anomaly in zip(sample.iterrows(), scores, flags):
        src_idx, src = _pick_node(None)
        _, dst = _pick_node(src_idx)

        is_anomaly = bool(is_anomaly)
        severity = "low"
        if is_anomaly and score > threshold * 1.35:
            severity = "high"
        elif is_anomaly:
            severity = "medium"

        events.append(
            {
                "event_id": int(row["id"]),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": src,
                "target": dst,
                "amount": float(row["amount"]),
                "model_score": float(score),
                "threshold": float(threshold),
                "is_anomaly": is_anomaly,
                "severity": severity,
                "model_key": safe_key,
                "model_name": model_name,
            }
        )

    return events
