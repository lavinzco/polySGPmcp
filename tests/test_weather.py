import httpx
import pytest
import respx

from weather_mcp.tools import WeatherClient


@pytest.mark.asyncio
async def test_get_weather(sample_wttr_response):
    with respx.mock:
        respx.get("https://wttr.in/Miami").mock(
            return_value=httpx.Response(200, json=sample_wttr_response)
        )
        client = WeatherClient()
        result = await client.get_weather("Miami")

    assert result.location == "Miami"
    assert result.temp_c == 32.0
    assert result.temp_f == 90.0
    assert result.humidity == 75
    assert result.wind_speed_kmph == 20
    assert result.wind_dir == "SSE"
    assert result.weather_desc == "Partly cloudy"
    assert result.feels_like_c == 36.0
    assert result.precip_mm == 0.5
    assert result.uv_index == 8


@pytest.mark.asyncio
async def test_get_multi(sample_wttr_response):
    with respx.mock:
        respx.get("https://wttr.in/Miami").mock(
            return_value=httpx.Response(200, json=sample_wttr_response)
        )
        respx.get("https://wttr.in/Houston").mock(
            return_value=httpx.Response(200, json=sample_wttr_response)
        )
        client = WeatherClient()
        results = await client.get_multi(["Miami", "Houston"])

    assert len(results) == 2
    assert results[0].location == "Miami"
    assert results[1].location == "Houston"


@pytest.mark.asyncio
async def test_get_weather_http_error():
    with respx.mock:
        respx.get("https://wttr.in/InvalidPlace").mock(
            return_value=httpx.Response(404)
        )
        client = WeatherClient()
        with pytest.raises(httpx.HTTPStatusError):
            await client.get_weather("InvalidPlace")
