from __future__ import annotations

import httpx

from common.config import settings
from polymarket.models import DayForecastData, WeatherData


class WeatherClient:
    """Async client for wttr.in weather data."""

    def __init__(self, base_url: str | None = None, client: httpx.AsyncClient | None = None):
        self._base_url = (base_url or settings.weather_api_base_url).rstrip("/")
        self._external_client = client

    async def get_weather(self, location: str) -> WeatherData:
        url = f"{self._base_url}/{location}"
        params = {"format": "j1"}
        headers = {"User-Agent": "hermes-weather-agent/0.1"}

        client = self._external_client or httpx.AsyncClient(timeout=15)
        should_close = self._external_client is None
        try:
            resp = await client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return self._parse(location, data)
        finally:
            if should_close:
                await client.aclose()

    async def get_multi(self, locations: list[str]) -> list[WeatherData]:
        results: list[WeatherData] = []
        for loc in locations:
            results.append(await self.get_weather(loc))
        return results

    @staticmethod
    def _parse(location: str, data: dict) -> WeatherData:
        current = data["current_condition"][0]

        forecast: list[DayForecastData] = []
        for day in data.get("weather", []):
            hourly = day.get("hourly", [])
            avg_hum = 0
            total_precip = 0.0
            condition = ""
            if hourly:
                avg_hum = sum(int(h.get("humidity", 0)) for h in hourly) // len(hourly)
                total_precip = sum(float(h.get("precipMM", 0)) for h in hourly)
                mid = hourly[len(hourly) // 2]
                condition = mid.get("weatherDesc", [{}])[0].get("value", "")

            forecast.append(DayForecastData(
                date=day["date"],
                max_temp_c=float(day["maxtempC"]),
                min_temp_c=float(day["mintempC"]),
                avg_temp_c=float(day.get("avgtempC", 0)),
                max_temp_f=float(day["maxtempF"]),
                min_temp_f=float(day.get("mintempF", 0)),
                uv_index=int(day.get("uvIndex", 0)),
                sun_hour=float(day.get("sunHour", 0)),
                total_precip_mm=total_precip,
                avg_humidity=avg_hum,
                condition=condition,
            ))

        return WeatherData(
            location=location,
            temp_c=float(current["temp_C"]),
            temp_f=float(current["temp_F"]),
            humidity=int(current["humidity"]),
            wind_speed_kmph=int(current["windspeedKmph"]),
            wind_dir=current["winddir16Point"],
            weather_desc=current["weatherDesc"][0]["value"],
            feels_like_c=float(current["FeelsLikeC"]),
            pressure_mb=int(current["pressure"]),
            precip_mm=float(current["precipMM"]),
            visibility_km=int(current["visibility"]),
            uv_index=int(current["uvIndex"]),
            forecast=forecast,
        )
