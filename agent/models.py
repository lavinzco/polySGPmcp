from __future__ import annotations

from pydantic import BaseModel, Field


class WeatherForecast(BaseModel):
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
    forecast_3day: list[DayForecast] = []


class DayForecast(BaseModel):
    date: str
    max_temp_c: float
    min_temp_c: float
    avg_humidity: int
    total_precip_mm: float
    condition: str


class GammaMarket(BaseModel):
    id: str
    question: str
    description: str
    outcome_yes_price: float = Field(ge=0, le=1)
    outcome_no_price: float = Field(ge=0, le=1)
    liquidity_usd: float = 0
    volume_usd: float = 0
    end_date: str = ""
    matched_keywords: list[str] = []
    quality: str = Field(default="high", description="high, medium, or low")


class TradeSignal(BaseModel):
    market_id: str
    action: str = Field(description="buy_yes | buy_no | hold")
    confidence: float = Field(ge=0, le=1)
    suggested_size_usd: float = Field(ge=0)
    rationale: str = ""
    weather_factors: list[str] = []
    quality: str = Field(default="high", description="quality tier of the source market")


class LLMTradeOutput(BaseModel):
    """Schema for raw LLM output — validated before becoming a TradeSignal."""
    action: str = "hold"
    confidence: float = Field(default=0.0, ge=0, le=1)
    suggested_size_usd: float = Field(default=0.0, ge=0)
    rationale: str = ""
    weather_factors: list[str] = []


class PortfolioState(BaseModel):
    total_balance_usd: float = 0
    daily_pnl_usd: float = 0
    open_positions: int = 0
