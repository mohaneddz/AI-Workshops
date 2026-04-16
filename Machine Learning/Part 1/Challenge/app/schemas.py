from __future__ import annotations

from typing import Optional

from fastapi import Form
from pydantic import BaseModel, Field


class LaptopInput(BaseModel):
    brand: str = Field(default="LENOVO")
    series: str = Field(default="IDEAPAD")
    condition: str = Field(default="JAMAIS UTILISÉ")
    listing_year: int = Field(default=2025, ge=2010, le=2035)
    cpu_brand: str = Field(default="INTEL")
    cpu_family: str = Field(default="i5")
    cpu_generation: int = Field(default=11, ge=0, le=30)
    cpu_suffix: str = Field(default="H")
    is_pro: bool = Field(default=False)
    gpu_model: str = Field(default="No Dedicated GPU (integrated)")
    gpu_suffix: str = Field(default="None/Standard")
    ram_size_gb: float = Field(default=8, ge=0)
    ram_type_class: str = Field(default="Unknown")
    ssd_size_gb: float = Field(default=256, ge=0)
    hdd_size_gb: float = Field(default=0, ge=0)
    screen_size_num: float = Field(default=15.6, ge=10, le=25)
    screen_resolution: str = Field(default="Unknown")
    screen_frequency_hz: Optional[float] = Field(default=None, ge=0)

    @classmethod
    def as_form(
        cls,
        brand: str = Form("LENOVO"),
        series: str = Form("IDEAPAD"),
        condition: str = Form("JAMAIS UTILISÉ"),
        listing_year: int = Form(2025),
        cpu_brand: str = Form("INTEL"),
        cpu_family: str = Form("i5"),
        cpu_generation: int = Form(11),
        cpu_suffix: str = Form("H"),
        is_pro: bool = Form(False),
        gpu_model: str = Form("No Dedicated GPU (integrated)"),
        gpu_suffix: str = Form("None/Standard"),
        ram_size_gb: float = Form(8),
        ram_type_class: str = Form("Unknown"),
        ssd_size_gb: float = Form(256),
        hdd_size_gb: float = Form(0),
        screen_size_num: float = Form(15.6),
        screen_resolution: str = Form("Unknown"),
        screen_frequency_hz: Optional[float] = Form(None),
    ) -> "LaptopInput":
        return cls(
            brand=brand,
            series=series,
            condition=condition,
            listing_year=listing_year,
            cpu_brand=cpu_brand,
            cpu_family=cpu_family,
            cpu_generation=cpu_generation,
            cpu_suffix=cpu_suffix,
            is_pro=is_pro,
            gpu_model=gpu_model,
            gpu_suffix=gpu_suffix,
            ram_size_gb=ram_size_gb,
            ram_type_class=ram_type_class,
            ssd_size_gb=ssd_size_gb,
            hdd_size_gb=hdd_size_gb,
            screen_size_num=screen_size_num,
            screen_resolution=screen_resolution,
            screen_frequency_hz=screen_frequency_hz,
        )


class PredictionResult(BaseModel):
    predicted_price_dzd: float
    predicted_price_text: str
    model_name: str
    r2_score: Optional[float] = None
    mae_dzd: Optional[float] = None
