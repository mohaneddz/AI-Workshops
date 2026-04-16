from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict

import numpy as np
import pandas as pd

from app.schemas import LaptopInput
from app.services.artifacts import load_bundle, load_model_bundle, load_model_registry


INTEGRATED_GPU_KEYWORDS = ("integrated", "no dedicated", "iris", "uhd")

RESOLUTION_CLASS_MAP = {
    "unknown": "Unknown",
    "fhd": "FHD",
    "1920x1080": "FHD",
    "1366x768": "HD",
    "3840x2160": "UHD / 4K+",
    "4k": "UHD / 4K+",
    "qhd": "QHD",
    "2k": "QHD",
}

RAM_DDR_MAP = {
    "ddr3": 3,
    "ddr4": 4,
    "ddr5": 5,
    "lpddr4": 4,
    "lpddr5": 5,
}


def _lookup(lookup: Dict[Any, Dict[str, float]], key: tuple, column: str, default: float) -> float:
    if key in lookup and column in lookup[key]:
        return float(lookup[key][column])
    return float(default)


def _find_lookup_row_cpu(payload: LaptopInput, lookup: Dict[Any, Dict[str, float]]) -> Dict[str, float]:
    brand = payload.cpu_brand.strip().upper()
    family = payload.cpu_family.strip().lower()
    suffix = payload.cpu_suffix.strip().upper()
    generation = float(payload.cpu_generation)
    is_pro = int(bool(payload.is_pro))

    for key, row in lookup.items():
        if len(key) != 5:
            continue
        k_brand, k_family, k_generation, k_suffix, k_is_pro = [str(value).strip() for value in key]
        try:
            if (
                k_brand.upper() == brand
                and k_family.lower() == family
                and k_suffix.upper() == suffix
                and float(k_generation) == generation
                and int(float(k_is_pro)) == is_pro
            ):
                return row
        except (TypeError, ValueError):
            continue
    return {}


def _extract_gpu_tag(gpu_model: str) -> str:
    text = gpu_model.strip().upper()
    if any(keyword in text.lower() for keyword in INTEGRATED_GPU_KEYWORDS):
        return "NONE"

    patterns = (
        r"(RTX\s*\d{3,4})",
        r"(GTX\s*\d{3,4})",
        r"(MX\s*\d{3,4})",
        r"(RX\s*\d{3,4}M?)",
        r"(ARC\s+[A-Z]?\d{3,4})",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return re.sub(r"\s+", " ", match.group(1)).strip()

    # Fallback keeps token shape mostly compatible with training values.
    text = text.replace("NVIDIA GEFORCE", "").replace("AMD RADEON", "").replace("INTEL", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text or "UNKNOWN"


def _infer_condition_retention(condition: str, lookup: Dict[Any, Dict[str, float]]) -> float:
    return _lookup(lookup, (condition,), "condition_value_retention", 0.8)


def _infer_build_quality(brand: str, series: str, lookup: Dict[Any, Dict[str, float]]) -> float:
    return _lookup(lookup, (brand, series), "build_quality_tier", 2.0)


def _infer_cpu_values(payload: LaptopInput, bundle: Dict[str, Any]) -> Dict[str, float]:
    lookup = bundle.get("cpu_lookup", {})
    row = _find_lookup_row_cpu(payload, lookup)
    defaults = {"cpu_mark": 9000.0, "cpu_single_thread": 2500.0, "cpu_cores": 4.0, "cpu_threads": 8.0}
    return {name: float(row.get(name, default)) for name, default in defaults.items()}


def _infer_gpu_values(payload: LaptopInput, bundle: Dict[str, Any]) -> Dict[str, float]:
    gpu_tier = _extract_gpu_tag(payload.gpu_model)
    has_gpu = 0 if gpu_tier == "NONE" else 1
    gpu_suffix = payload.gpu_suffix.strip()
    if gpu_suffix.lower() in {"none/standard", "none", ""}:
        gpu_suffix = "Unknown"

    key = (gpu_tier, gpu_suffix, str(has_gpu))
    lookup = bundle.get("gpu_lookup", {})
    if key not in lookup:
        # Keep classifier feature aligned with training space whenever possible.
        normalized_tier = gpu_tier.upper()
        normalized_suffix = gpu_suffix.upper()
        for k, row in lookup.items():
            if len(k) != 3:
                continue
            k_tier, k_suffix, k_has_gpu = [str(v).strip() for v in k]
            if k_has_gpu != str(has_gpu):
                continue
            if k_tier.upper() == normalized_tier and k_suffix.upper() == normalized_suffix:
                key = k
                break

    default = 0.0 if has_gpu == 0 else 7000.0
    gpu_score = _lookup(lookup, key, "gpu_score", default)
    # Use the bundle's native category when matched to avoid one-hot unknown vectors.
    matched_tier = str(key[0]) if key in lookup else ("NONE" if has_gpu == 0 else gpu_tier)
    matched_suffix = str(key[1]) if key in lookup else gpu_suffix
    return {"gpu_tier": matched_tier, "gpu_score": gpu_score, "has_gpu": has_gpu, "gpu_suffix": matched_suffix}


def _infer_resolution_class(screen_resolution: str) -> str:
    text = screen_resolution.lower().strip()
    for keyword, resolution_class in RESOLUTION_CLASS_MAP.items():
        if keyword in text:
            return resolution_class
    return "Unknown"


def _infer_ram_ordinal(ram_type_class: str, lookup: Dict[Any, Dict[str, float]]) -> float:
    default = RAM_DDR_MAP.get(str(ram_type_class).lower(), 4)
    return _lookup(lookup, (ram_type_class,), "inferred_ddr_ordinal", default)


def _infer_series_gaming(series: str) -> int:
    text = series.lower().strip()
    gaming_keywords = ["rog", "legion", "gaming", "predator", "alienware", "tuf", "omen", "nitro", "geforce"]
    return int(any(keyword in text for keyword in gaming_keywords))


def _infer_storage_score(ssd_size_gb: float, hdd_size_gb: float) -> float:
    # Must stay aligned with training data semantics: 0=no storage, 1=HDD-only, 2=any SSD present.
    if ssd_size_gb > 0:
        return 2.0
    if hdd_size_gb > 0:
        return 1.0
    return 0.0


def build_feature_frame(payload: LaptopInput, bundle: Dict[str, Any]) -> pd.DataFrame:
    brand = payload.brand.upper().strip()
    series = payload.series.upper().strip()
    condition = payload.condition.upper().strip()
    resolution_class = _infer_resolution_class(payload.screen_resolution)
    gpu_values = _infer_gpu_values(payload, bundle)
    cpu_values = _infer_cpu_values(payload, bundle)

    row = {
        "RAM_SIZE_GB": np.log1p(max(payload.ram_size_gb, 0)),
        "SSD_SIZE_GB": np.log1p(max(payload.ssd_size_gb, 0)),
        "cpu_mark": cpu_values["cpu_mark"],
        "cpu_single_thread": cpu_values["cpu_single_thread"],
        "gpu_score": gpu_values["gpu_score"],
        "build_quality_tier": _infer_build_quality(brand, series, bundle.get("build_lookup", {})),
        "condition_value_retention": _infer_condition_retention(condition, bundle.get("condition_lookup", {})),
        "cpu_generation": int(payload.cpu_generation),
        "storage_perf_score": _infer_storage_score(payload.ssd_size_gb, payload.hdd_size_gb),
        "cpu_cores": cpu_values["cpu_cores"],
        "inferred_ddr_ordinal": _infer_ram_ordinal(payload.ram_type_class, bundle.get("ram_lookup", {})),
        "SCREEN_SIZE_NUM": float(payload.screen_size_num),
        "cpu_threads": cpu_values["cpu_threads"],
        "is_pro": int(bool(payload.is_pro)),
        "is_gaming_series": _infer_series_gaming(series),
        "has_gpu": gpu_values["has_gpu"],
        "series": series,
        "cpu_family": payload.cpu_family.lower().strip(),
        "gpu_tier": gpu_values["gpu_tier"],
        "condition": condition,
        "gpu_suffix": gpu_values["gpu_suffix"],
        "cpu_suffix": payload.cpu_suffix.upper().strip(),
        "cpu_gen_brand": f"{payload.cpu_brand.upper().strip()}_{int(payload.cpu_generation)}",
        "listing_year": str(int(payload.listing_year)),
        "brand": brand,
        "ram_type_class": payload.ram_type_class.strip(),
        "resolution_class": resolution_class,
    }

    return pd.DataFrame([row])


@dataclass
class LaptopPricePredictor:
    bundle: Dict[str, Any]

    @classmethod
    def load(cls, model_key: str = "LinearRegression") -> "LaptopPricePredictor":
        try:
            bundle = load_model_bundle(model_key)
        except Exception:
            bundle = load_bundle()
        return cls(bundle=bundle)

    @property
    def model_key(self) -> str:
        return self.bundle.get("model_key", "LinearRegression")

    @property
    def model_name(self) -> str:
        return self.bundle.get("model_name", "LinearRegression")

    def predict(self, payload: LaptopInput) -> Dict[str, Any]:
        frame = build_feature_frame(payload, self.bundle)
        preprocessor = self.bundle["preprocessor"]
        model = self.bundle["model"]
        processed = preprocessor.transform(frame[self.bundle["feature_columns"]])
        predicted = float(np.expm1(model.predict(processed)[0]))
        # Clip to sensible Algerian laptop market range (e.g. max 1,500,000 DZD)
        predicted = np.clip(predicted, 1000.0, 1500000.0)
        metrics = self.bundle.get("metrics", {})
        return {
            "predicted_price_dzd": max(predicted, 0.0),
            "predicted_price_text": f"{max(predicted, 0.0):,.0f} DZD",
            "model_key": self.model_key,
            "model_name": self.model_name,
            "r2_score": metrics.get("r2_score"),
            "mae_dzd": metrics.get("mae_dzd"),
        }

    def catalog(self) -> Dict[str, Any]:
        cat = self.bundle.get("catalog", {}).copy()
        # Strategic 20 screen sizes (Standard market sizes, removing data noise)
        strategic_sizes = [
            10.1, 10.8, 11.6, 12.0, 12.1, 12.4, 12.5, 13.0, 13.3, 13.5,
            14.0, 14.2, 14.5, 15.0, 15.4, 15.6, 16.0, 16.1, 17.3, 18.0
        ]
        cat["screen_size_num"] = strategic_sizes
        return cat
