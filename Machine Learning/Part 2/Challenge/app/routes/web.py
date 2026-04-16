from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from app.schemas import IngestRequest
from app.config import TEMPLATES_DIR
from app.services.metrics import METRICS_STORE
from app.services.model import DEFAULT_MODEL_KEY, FEATURE_COLUMNS, ensure_models, get_model_options, infer_anomalies
from app.services.stream import generate_events

router = APIRouter()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@router.get("/", response_class=HTMLResponse)
def home(request: Request, model: str = DEFAULT_MODEL_KEY):
    metadata = ensure_models()
    options = get_model_options()
    valid_keys = {item["key"] for item in options}
    active_model = model if model in valid_keys else metadata.get("default_model", DEFAULT_MODEL_KEY)
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "meta": metadata, "model_options": options, "active_model": active_model},
    )


@router.get("/api/events")
def events(batch: int = 30, model: str = DEFAULT_MODEL_KEY):
    batch = max(5, min(batch, 100))
    data = generate_events(batch_size=batch, model_key=model)
    response = {
        "events": data,
        "count": len(data),
        "model_key": data[0]["model_key"] if data else model,
        "model_name": data[0]["model_name"] if data else model,
        "threshold": data[0]["threshold"] if data else None,
    }
    return JSONResponse(response)


@router.post("/api/ingest")
def ingest(payload: IngestRequest):
    model_key = payload.model_key or DEFAULT_MODEL_KEY
    missing = [col for col in FEATURE_COLUMNS if col not in payload.features]
    if missing:
        return JSONResponse(
            {
                "ok": False,
                "error": f"Missing required features: {missing}",
            },
            status_code=400,
        )

    frame = pd.DataFrame([{col: float(payload.features[col]) for col in FEATURE_COLUMNS}])
    scores, flags, threshold, safe_key = infer_anomalies(model_key=model_key, X=frame)

    predicted_anomaly = bool(flags[0])
    score = float(scores[0])
    METRICS_STORE.record(model_key=safe_key, actual_type=payload.actual_type, predicted_anomaly=predicted_anomaly)
    snapshot = METRICS_STORE.snapshot()

    return JSONResponse(
        {
            "ok": True,
            "model_key": safe_key,
            "actual_type": payload.actual_type,
            "attack_type": payload.attack_type,
            "predicted_anomaly": predicted_anomaly,
            "score": score,
            "threshold": float(threshold),
            "metrics": snapshot,
        }
    )


@router.get("/api/metrics")
def metrics(model: str | None = None):
    snapshot = METRICS_STORE.snapshot()
    if model:
        model_bucket = snapshot["by_model"].get(model)
        if model_bucket is None:
            return JSONResponse({"ok": False, "error": f"Unknown model: {model}"}, status_code=400)
        return JSONResponse({"ok": True, "model": model, "metrics": model_bucket, "global": snapshot["global"]})
    return JSONResponse({"ok": True, "metrics": snapshot})


@router.post("/api/metrics/reset")
def reset_metrics():
    METRICS_STORE.reset()
    return JSONResponse({"ok": True, "metrics": METRICS_STORE.snapshot()})
