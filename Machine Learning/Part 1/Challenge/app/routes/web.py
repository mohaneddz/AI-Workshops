from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.config import TEMPLATES_DIR
from app.schemas import LaptopInput
from app.services.artifacts import any_bundle_exists, load_model_registry
from app.services.predictor import LaptopPricePredictor
from app.services.training import train_all_models, AVAILABLE_MODELS


router = APIRouter()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

DEFAULT_MODEL_KEY = "GradientBoosting"


def ensure_models_trained():
    if not any_bundle_exists():
        train_all_models()


def get_predictor(model_key: str = DEFAULT_MODEL_KEY) -> LaptopPricePredictor:
    ensure_models_trained()
    return LaptopPricePredictor.load(model_key)


def build_model_list():
    registry = load_model_registry()
    if not registry:
        return [
            {
                "key": k,
                "display_name": v["display_name"],
                "description": v["description"],
                "r2": None,
                "mae": None,
            }
            for k, v in AVAILABLE_MODELS.items()
        ]
    return [
        {
            "key": k,
            "display_name": v["display_name"],
            "description": v["description"],
            "r2": v.get("metrics", {}).get("r2_score"),
            "mae": v.get("metrics", {}).get("mae_dzd"),
        }
        for k, v in registry.items()
    ]


@router.get("/", response_class=HTMLResponse)
def home(request: Request, model_key: str = DEFAULT_MODEL_KEY):
    ensure_models_trained()
    predictor = LaptopPricePredictor.load(model_key)
    catalog = predictor.catalog()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "catalog": catalog,
            "defaults": LaptopInput().model_dump(),
            "result": None,
            "error": None,
            "metrics": predictor.bundle.get("metrics", {}),
            "active_model": model_key,
            "model_list": build_model_list(),
        },
    )


@router.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    payload: LaptopInput = Depends(LaptopInput.as_form),
    model_key: str = DEFAULT_MODEL_KEY,
):
    ensure_models_trained()
    predictor = LaptopPricePredictor.load(model_key)
    catalog = predictor.catalog()
    try:
        result = predictor.predict(payload)
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "catalog": catalog,
                "defaults": payload.model_dump(),
                "result": result,
                "error": None,
                "metrics": predictor.bundle.get("metrics", {}),
                "active_model": model_key,
                "model_list": build_model_list(),
            },
        )
    except Exception as exc:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "catalog": catalog,
                "defaults": payload.model_dump(),
                "result": None,
                "error": str(exc),
                "metrics": predictor.bundle.get("metrics", {}),
                "active_model": model_key,
                "model_list": build_model_list(),
            },
        )
