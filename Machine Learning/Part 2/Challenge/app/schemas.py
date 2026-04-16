from __future__ import annotations

from typing import Dict, Literal

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    actual_type: Literal["good_request", "bad_request"] = Field(..., description="Ground truth label")
    features: Dict[str, float]
    payload: str = ""
    attack_type: str = "none"
    model_key: str | None = None
