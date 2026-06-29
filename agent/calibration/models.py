from __future__ import annotations

from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
    name: str
    provider_type: str = Field(description="anthropic | openai_compatible")
    model: str
    base_url: str = ""
    api_key: str = ""


class CalibrationSample(BaseModel):
    market_id: str
    provider_name: str
    model_name: str
    city: str
    date: str
    threshold_temp: float
    threshold_unit: str
    direction: str
    market_yes_price: float
    llm_action: str
    llm_confidence: float
    llm_rationale: str
    llm_raw_output: str
    weather_snapshot_json: str
    settled: bool = False
    actual_outcome: str | None = None  # "YES" | "NO" | None
