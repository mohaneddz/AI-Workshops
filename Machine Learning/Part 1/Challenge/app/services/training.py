from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.svm import SVR

from app.config import DATA_FILE, MODELS_DIR
from app.services.artifacts import save_bundle


FEATURE_NUMERIC = [
    "RAM_SIZE_GB",
    "SSD_SIZE_GB",
    "cpu_mark",
    "cpu_single_thread",
    "gpu_score",
    "build_quality_tier",
    "condition_value_retention",
    "cpu_generation",
    "storage_perf_score",
    "cpu_cores",
    "inferred_ddr_ordinal",
    "SCREEN_SIZE_NUM",
    "cpu_threads",
]

FEATURE_BINARY = ["is_pro", "is_gaming_series", "has_gpu"]

FEATURE_CATEGORICAL = [
    "series",
    "cpu_family",
    "gpu_tier",
    "condition",
    "gpu_suffix",
    "cpu_suffix",
    "cpu_gen_brand",
    "listing_year",
    "brand",
    "ram_type_class",
    "resolution_class",
]

ALL_FEATURES = FEATURE_NUMERIC + FEATURE_BINARY + FEATURE_CATEGORICAL

# All available models
AVAILABLE_MODELS = {
    "GradientBoosting": {
        "display_name": "Gradient Boosting",
        "description": "Sequential boosting trees",
        "factory": lambda: GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.08, subsample=0.8, random_state=42),
    },
    "SVR": {
        "display_name": "Support Vector Regression",
        "description": "Non-linear margin regression",
        "factory": lambda: SVR(kernel='rbf', C=10.0, epsilon=0.1),
    },
}

# Try importing XGBoost
try:
    from xgboost import XGBRegressor
    AVAILABLE_MODELS["XGBoost"] = {
        "display_name": "XGBoost",
        "description": "Extreme gradient boosting",
        "factory": lambda: XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.06, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1),
    }
except ImportError:
    pass


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imp", SimpleImputer(strategy="mean")), ("scl", StandardScaler())]),
                FEATURE_NUMERIC,
            ),
            ("bin", "passthrough", FEATURE_BINARY),
            ("cat", make_one_hot_encoder(), FEATURE_CATEGORICAL),
        ]
    )


def _safe_str(value: Any, default: str = "Unknown") -> str:
    if pd.isna(value) or value is None:
        return default
    text = str(value).strip()
    return text if text else default


def prepare_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    if "created_at" in frame.columns:
        frame["created_at"] = pd.to_datetime(frame["created_at"], errors="coerce")
        frame = frame.sort_values("created_at").reset_index(drop=True)

    if "is_pro" not in frame.columns:
        frame["is_pro"] = 0
    frame["is_pro"] = frame["is_pro"].fillna(0).astype(int)

    if "cpu_brand" not in frame.columns:
        frame["cpu_brand"] = "Unknown"
    if "cpu_generation" not in frame.columns:
        frame["cpu_generation"] = 0
    frame["cpu_gen_brand"] = frame["cpu_brand"].fillna("Unknown").astype(str) + "_" + frame["cpu_generation"].fillna(0).astype(int).astype(str)

    if "RAM_SIZE_GB" not in frame.columns:
        frame["RAM_SIZE_GB"] = 0
    if "SSD_SIZE_GB" not in frame.columns:
        frame["SSD_SIZE_GB"] = 0
    frame["RAM_SIZE_GB"] = np.log1p(frame["RAM_SIZE_GB"].fillna(0))
    frame["SSD_SIZE_GB"] = np.log1p(frame["SSD_SIZE_GB"].fillna(0))
    if "listing_year" not in frame.columns:
        frame["listing_year"] = 2025
    frame["listing_year"] = frame["listing_year"].fillna(2025).astype(int).astype(str)
    for column in ["condition", "brand", "series", "cpu_family", "gpu_tier", "gpu_suffix", "cpu_suffix", "ram_type_class", "resolution_class"]:
        if column not in frame.columns:
            frame[column] = "Unknown"
        frame[column] = frame[column].fillna("Unknown").astype(str)
    return frame


def _build_lookup(df: pd.DataFrame, group_cols: List[str], value_cols: List[str]) -> Dict[Tuple[Any, ...], Dict[str, float]]:
    lookup: Dict[Tuple[Any, ...], Dict[str, float]] = {}
    if not value_cols:
        return lookup
    grouped = df.groupby(group_cols, dropna=False)[value_cols].median(numeric_only=True).reset_index()
    for _, row in grouped.iterrows():
        key = tuple(_safe_str(row[col], default="") for col in group_cols)
        lookup[key] = {col: float(row[col]) for col in value_cols if pd.notna(row[col])}
    return lookup


def build_catalog(df: pd.DataFrame) -> Dict[str, Any]:
    def take(column: str, limit: int = 40) -> List[str]:
        if column not in df.columns:
            return []
        values = [str(value) for value in df[column].dropna().astype(str).unique().tolist()]
        values = sorted(dict.fromkeys(values))
        return values[:limit]

    # Build brand -> series mapping
    brand_series_map: Dict[str, List[str]] = {}
    if "brand" in df.columns and "series" in df.columns:
        for brand in df["brand"].dropna().unique():
            series_for_brand = df[df["brand"] == brand]["series"].dropna().unique().tolist()
            series_for_brand = sorted(set(str(s) for s in series_for_brand))
            brand_series_map[str(brand)] = series_for_brand[:80]

    # Build cpu_brand -> cpu_family mapping
    cpu_brand_family_map: Dict[str, List[str]] = {}
    if "cpu_brand" in df.columns and "cpu_family" in df.columns:
        for cb in df["cpu_brand"].dropna().unique():
            families = df[df["cpu_brand"] == cb]["cpu_family"].dropna().unique().tolist()
            families = sorted(set(str(f) for f in families))
            cpu_brand_family_map[str(cb)] = families[:40]

    # Build cpu_brand+family -> cpu_suffix mapping
    cpu_family_suffix_map: Dict[str, List[str]] = {}
    if "cpu_brand" in df.columns and "cpu_family" in df.columns and "cpu_suffix" in df.columns:
        tmp = df[["cpu_brand", "cpu_family", "cpu_suffix"]].dropna()
        for (cb, cf), grp in tmp.groupby(["cpu_brand", "cpu_family"]):
            suffixes = sorted(set(str(s) for s in grp["cpu_suffix"].unique()))
            cpu_family_suffix_map[f"{cb}|{cf}"] = suffixes[:20]

    # Build cpu_brand -> generation mapping
    cpu_brand_generation_map: Dict[str, List[int]] = {}
    if "cpu_brand" in df.columns and "cpu_generation" in df.columns:
        for cb in df["cpu_brand"].dropna().unique():
            gens = df[df["cpu_brand"] == cb]["cpu_generation"].dropna().unique().tolist()
            gens = sorted(set(int(g) for g in gens))
            cpu_brand_generation_map[str(cb)] = gens

    return {
        "brand": take("brand"),
        "series": take("series", 80),
        "condition": take("condition"),
        "cpu_brand": take("cpu_brand"),
        "cpu_family": take("cpu_family"),
        "cpu_suffix": take("cpu_suffix"),
        "gpu_suffix": take("gpu_suffix"),
        "ram_type_class": take("ram_type_class"),
        "resolution_class": take("resolution_class"),
        "listing_year": take("listing_year"),
        "screen_size_num": sorted([float(value) for value in df["SCREEN_SIZE_NUM"].dropna().unique().tolist()])[:100] if "SCREEN_SIZE_NUM" in df.columns else [],
        # Cascade maps
        "brand_series_map": brand_series_map,
        "cpu_brand_family_map": cpu_brand_family_map,
        "cpu_family_suffix_map": cpu_family_suffix_map,
        "cpu_brand_generation_map": cpu_brand_generation_map,
    }


def build_shared_artifacts(df: pd.DataFrame) -> Dict[str, Any]:
    """Build lookups and catalog shared across all models."""
    prepared = prepare_training_frame(df)
    split_idx = int(len(prepared) * 0.8)
    train_df = prepared.iloc[:split_idx].copy()

    cpu_lookup = _build_lookup(
        train_df,
        ["cpu_brand", "cpu_family", "cpu_generation", "cpu_suffix", "is_pro"],
        ["cpu_mark", "cpu_single_thread", "cpu_cores", "cpu_threads"],
    )
    gpu_lookup = _build_lookup(train_df, ["gpu_tier", "gpu_suffix", "has_gpu"], ["gpu_score"])
    build_lookup = _build_lookup(train_df, ["brand", "series"], ["build_quality_tier"])
    condition_lookup = _build_lookup(train_df, ["condition"], ["condition_value_retention"])
    ram_lookup = _build_lookup(train_df, ["ram_type_class"], ["inferred_ddr_ordinal"])
    catalog = build_catalog(train_df)

    return {
        "prepared": prepared,
        "train_df": train_df,
        "split_idx": split_idx,
        "cpu_lookup": cpu_lookup,
        "gpu_lookup": gpu_lookup,
        "build_lookup": build_lookup,
        "condition_lookup": condition_lookup,
        "ram_lookup": ram_lookup,
        "catalog": catalog,
    }


def train_single_model(model_key: str, artifacts: Dict[str, Any]) -> Dict[str, Any]:
    prepared = artifacts["prepared"]
    train_df = artifacts["train_df"]
    split_idx = artifacts["split_idx"]
    test_df = prepared.iloc[split_idx:].copy()

    target = np.log1p(prepared["PRICES"].fillna(prepared["PRICES"].median()))
    y_train = target.iloc[:split_idx].copy()
    y_test = prepared["PRICES"].iloc[split_idx:].copy()

    X_train = train_df[ALL_FEATURES].copy()
    X_test = test_df[ALL_FEATURES].copy()

    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    model_info = AVAILABLE_MODELS[model_key]
    model = model_info["factory"]()
    model.fit(X_train_proc, y_train)

    predictions = np.expm1(model.predict(X_test_proc))
    metrics = {
        "r2_score": float(r2_score(y_test, predictions)) if len(y_test) else None,
        "mae_dzd": float(mean_absolute_error(y_test, predictions)) if len(y_test) else None,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
    }

    return {
        "model_key": model_key,
        "model_name": model_info["display_name"],
        "model_description": model_info["description"],
        "model": model,
        "preprocessor": preprocessor,
        "metrics": metrics,
        "feature_columns": ALL_FEATURES,
        "feature_numeric": FEATURE_NUMERIC,
        "feature_binary": FEATURE_BINARY,
        "feature_categorical": FEATURE_CATEGORICAL,
        **{k: v for k, v in artifacts.items() if k not in ("prepared", "train_df", "split_idx")},
    }


def train_all_models(data_path: Path = DATA_FILE) -> None:
    """Train all available models and save each as a separate joblib file."""
    import joblib

    df = pd.read_csv(data_path)
    artifacts = build_shared_artifacts(df)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_registry = {}

    for model_key in AVAILABLE_MODELS:
        try:
            print(f"  Training {model_key}...")
            bundle = train_single_model(model_key, artifacts)
            model_path = MODELS_DIR / f"bundle_{model_key}.joblib"
            joblib.dump(bundle, model_path)
            model_registry[model_key] = {
                "display_name": AVAILABLE_MODELS[model_key]["display_name"],
                "description": AVAILABLE_MODELS[model_key]["description"],
                "metrics": bundle["metrics"],
                "file": str(model_path),
            }
            print(f"    R²={bundle['metrics']['r2_score']:.3f}  MAE={bundle['metrics']['mae_dzd']:,.0f} DZD")
        except Exception as e:
            print(f"    Failed to train {model_key}: {e}")

    registry_path = MODELS_DIR / "model_registry.json"
    import json
    registry_path.write_text(json.dumps(model_registry, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved registry to {registry_path}")

    # Also keep backward-compatible default (best R²)
    best_key = max(model_registry, key=lambda k: model_registry[k]["metrics"].get("r2_score") or 0)
    best_bundle_path = MODELS_DIR / f"bundle_{best_key}.joblib"
    import shutil
    shutil.copy(best_bundle_path, MODELS_DIR / "laptop_price_bundle.joblib")
    print(f"Default model set to: {best_key}")


def train_default_model(data_path: Path = DATA_FILE) -> Dict[str, Any]:
    """Legacy: train LinearRegression only and save as default bundle."""
    df = pd.read_csv(data_path)
    artifacts = build_shared_artifacts(df)
    bundle = train_single_model("LinearRegression", artifacts)
    save_bundle(bundle)
    return bundle
