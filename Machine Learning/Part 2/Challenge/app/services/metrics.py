from __future__ import annotations

from copy import deepcopy
from threading import Lock
from typing import Any, Dict

from app.services.model import MODEL_REGISTRY


def _empty_counter() -> Dict[str, int]:
    return {
        "total_requests": 0,
        "passed_requests": 0,
        "anomalies_classified": 0,
        "unclassified": 0,
        "good_connections_misclassified": 0,
        "good_connections_classified": 0,
    }


class MetricsStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._global = _empty_counter()
            self._by_model = {key: _empty_counter() for key in MODEL_REGISTRY}

    def record(self, *, model_key: str, actual_type: str, predicted_anomaly: bool) -> None:
        is_bad = actual_type == "bad_request"

        with self._lock:
            counters = [self._global]
            if model_key in self._by_model:
                counters.append(self._by_model[model_key])

            for bucket in counters:
                bucket["total_requests"] += 1

                if not predicted_anomaly:
                    bucket["passed_requests"] += 1

                if is_bad and predicted_anomaly:
                    bucket["anomalies_classified"] += 1
                elif is_bad and not predicted_anomaly:
                    bucket["unclassified"] += 1
                elif not is_bad and predicted_anomaly:
                    bucket["good_connections_misclassified"] += 1
                else:
                    bucket["good_connections_classified"] += 1

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {"global": deepcopy(self._global), "by_model": deepcopy(self._by_model)}


METRICS_STORE = MetricsStore()
