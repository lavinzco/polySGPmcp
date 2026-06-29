from __future__ import annotations

from pydantic import BaseModel, Field


class Market(BaseModel):
    id: str = ""
    question: str = ""
    description: str = ""
    condition_id: str = Field(default="", alias="conditionId")
    slug: str = ""
    end_date_iso: str = Field(default="", alias="endDateIso")
    active: bool = True
    closed: bool = False
    liquidity: str = "0"
    volume: str = "0"
    outcomes: str = "[]"
    outcome_prices: str = Field(default="[]", alias="outcomePrices")

    model_config = {"populate_by_name": True}


class Event(BaseModel):
    id: str = ""
    title: str = ""
    slug: str = ""
    active: bool = True
    closed: bool = False
    markets: list[Market] = []

    model_config = {"populate_by_name": True}


class WeatherMarket(BaseModel):
    market: Market
    relevance_score: float = Field(ge=0, le=1.0)
    matched_keywords: list[str] = []


class DayForecastData(BaseModel):
    date: str
    max_temp_c: float
    min_temp_c: float
    avg_temp_c: float
    max_temp_f: float
    min_temp_f: float
    uv_index: int = 0
    sun_hour: float = 0.0
    total_precip_mm: float = 0.0
    avg_humidity: int = 0
    condition: str = ""


class WeatherData(BaseModel):
    location: str
    temp_c: float
    temp_f: float
    humidity: int
    wind_speed_kmph: int
    wind_dir: str
    weather_desc: str
    feels_like_c: float
    pressure_mb: int
    precip_mm: float
    visibility_km: int
    uv_index: int
    forecast: list[DayForecastData] = []
