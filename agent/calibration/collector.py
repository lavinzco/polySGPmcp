from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from agent.calibration.db import CalibrationDB
from agent.calibration.models import CalibrationSample, ProviderConfig
from agent.models import LLMTradeOutput, WeatherForecast
from agent.prompts import build_strategy_prompt, get_system_prompt_with_schema
from common.llm.base import LLMProvider
from polymarket.temperature import TemperatureMarket

logger = logging.getLogger("hermes.calibration.collector")


def _build_provider_from_config(cfg: ProviderConfig) -> LLMProvider:
    if cfg.provider_type == "anthropic":
        from common.llm.anthropic_provider import AnthropicProvider
        return AnthropicProvider(
            model=cfg.model,
            api_key=cfg.api_key or os.environ.get("ANTHROPIC_API_KEY", ""),
        )
    if cfg.provider_type == "openai_compatible":
        from common.llm.openai_compatible import OpenAICompatibleProvider
        return OpenAICompatibleProvider(
            model=cfg.model,
            base_url=cfg.base_url,
            api_key=cfg.api_key or os.environ.get(f"{cfg.name.upper()}_API_KEY", ""),
        )
    raise ValueError(f"Unknown provider_type '{cfg.provider_type}'")


def _temp_market_to_gamma(tm: TemperatureMarket):
    from agent.models import GammaMarket
    return GammaMarket(
        id=tm.market.id,
        question=tm.market.question,
        description=tm.market.description,
        outcome_yes_price=tm.outcome_yes_price,
        outcome_no_price=tm.outcome_no_price,
        liquidity_usd=float(tm.market.liquidity),
        volume_usd=float(tm.market.volume),
        end_date=tm.market.end_date_iso,
        matched_keywords=[f"temperature:{tm.threshold_temp}{tm.threshold_unit}"],
    )


async def collect_confidence_samples(
    *,
    provider_configs: list[ProviderConfig],
    temperature_markets: list[TemperatureMarket],
    weather_data: dict[str, WeatherForecast],
    db: CalibrationDB,
    n_repeats: int = 1,
    dry_run: bool = False,
) -> list[CalibrationSample]:
    all_samples: list[CalibrationSample] = []
    system_prompt = get_system_prompt_with_schema()

    for tm in temperature_markets:
        weather = weather_data.get(tm.city)
        if weather is None:
            logger.warning(f"No weather data for {tm.city}, skipping market {tm.market.id}")
            continue

        gamma_market = _temp_market_to_gamma(tm)
        user_prompt = build_strategy_prompt(weather, gamma_market)

        for cfg in provider_configs:
            provider = _build_provider_from_config(cfg)

            for repeat_i in range(n_repeats):
                logger.info(
                    f"Collecting: market={tm.market.id} provider={cfg.name} "
                    f"repeat={repeat_i+1}/{n_repeats}"
                )

                if dry_run:
                    logger.info(f"  [DRY-RUN] Would call {cfg.name} ({cfg.model})")
                    logger.info(f"  City: {tm.city}, Date: {tm.date}, "
                                f"Threshold: {tm.threshold_temp}°{tm.threshold_unit}")
                    logger.info(f"  Prompt length: {len(user_prompt)} chars")
                    sample = CalibrationSample(
                        market_id=tm.market.id,
                        provider_name=cfg.name,
                        model_name=cfg.model,
                        city=tm.city,
                        date=tm.date,
                        threshold_temp=tm.threshold_temp,
                        threshold_unit=tm.threshold_unit,
                        direction=tm.direction,
                        market_yes_price=tm.outcome_yes_price,
                        llm_action="hold",
                        llm_confidence=0.0,
                        llm_rationale="[dry-run]",
                        llm_raw_output="[dry-run]",
                        weather_snapshot_json=weather.model_dump_json(),
                    )
                    all_samples.append(sample)
                    continue

                raw_output = ""
                try:
                    raw_output = await provider.complete(user_prompt, system=system_prompt)
                    parsed = LLMProvider._parse_model(raw_output, LLMTradeOutput)
                    if parsed.action not in ("buy_yes", "buy_no", "hold"):
                        parsed = LLMTradeOutput(action="hold", rationale="invalid action")
                except Exception as exc:
                    logger.warning(f"  LLM call failed: {exc}")
                    parsed = LLMTradeOutput(
                        action="hold",
                        rationale=f"parse error: {exc}",
                    )

                sample = CalibrationSample(
                    market_id=tm.market.id,
                    provider_name=cfg.name,
                    model_name=cfg.model,
                    city=tm.city,
                    date=tm.date,
                    threshold_temp=tm.threshold_temp,
                    threshold_unit=tm.threshold_unit,
                    direction=tm.direction,
                    market_yes_price=tm.outcome_yes_price,
                    llm_action=parsed.action,
                    llm_confidence=parsed.confidence,
                    llm_rationale=parsed.rationale,
                    llm_raw_output=raw_output,
                    weather_snapshot_json=weather.model_dump_json(),
                )

                db.insert_sample(sample)
                all_samples.append(sample)
                logger.info(
                    f"  Result: action={parsed.action} confidence={parsed.confidence:.2f}"
                )

    return all_samples


def parse_provider_arg(spec: str) -> ProviderConfig:
    """Parse 'deepseek-chat' or 'anthropic:claude-sonnet-4-6' into a ProviderConfig."""
    _KNOWN = {
        "deepseek-chat": ProviderConfig(
            name="deepseek-chat",
            provider_type="openai_compatible",
            model="deepseek-chat",
            base_url="https://api.deepseek.com/v1",
        ),
        "deepseek-reasoner": ProviderConfig(
            name="deepseek-reasoner",
            provider_type="openai_compatible",
            model="deepseek-reasoner",
            base_url="https://api.deepseek.com/v1",
        ),
    }

    if spec in _KNOWN:
        return _KNOWN[spec]

    if spec.startswith("claude-") or spec.startswith("claude_"):
        return ProviderConfig(
            name=spec,
            provider_type="anthropic",
            model=spec.replace("_", "-"),
        )

    parts = spec.split(":", 1)
    if len(parts) == 2:
        ptype, model = parts
        return ProviderConfig(name=spec, provider_type=ptype, model=model)

    return ProviderConfig(
        name=spec,
        provider_type="openai_compatible",
        model=spec,
        base_url=os.environ.get("CLASSIFICATION_BASE_URL", ""),
    )


def cli_main() -> None:
    parser = argparse.ArgumentParser(description="Hermes calibration collector")
    parser.add_argument("--markets", default="temperature", choices=["temperature"])
    parser.add_argument("--n-repeats", type=int, default=1)
    parser.add_argument(
        "--providers", required=True,
        help="Comma-separated provider specs (e.g. deepseek-chat,claude-sonnet-4-6)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--db", default="hermes_calibration.db")
    parser.add_argument("--max-markets", type=int, default=5)
    args = parser.parse_args()

    from common.logging import logger as _  # noqa: F811 — init logging

    provider_configs = [parse_provider_arg(s.strip()) for s in args.providers.split(",")]

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(_cli_run(args, provider_configs))


async def _cli_run(args, provider_configs: list[ProviderConfig]) -> None:
    from polymarket.client import GammaClient
    from polymarket.temperature import find_temperature_markets
    from weather_mcp.tools import WeatherClient

    logger.info(f"Fetching markets from Gamma API...")
    gamma = GammaClient()
    for attempt in range(3):
        try:
            all_markets = await gamma.get_all_markets(max_pages=5)
            break
        except Exception as exc:
            logger.warning(f"Gamma API attempt {attempt+1} failed: {exc}")
            if attempt == 2:
                raise
            import asyncio
            await asyncio.sleep(2)
    temp_markets = find_temperature_markets(all_markets)
    logger.info(f"Found {len(temp_markets)} temperature markets")

    temp_markets = temp_markets[: args.max_markets]
    if not temp_markets:
        logger.info("No temperature markets found. Exiting.")
        return

    cities = list({tm.city for tm in temp_markets})
    logger.info(f"Fetching weather for cities: {cities}")

    weather_client = WeatherClient()
    weather_data: dict[str, WeatherForecast] = {}
    for city in cities:
        try:
            from agent.models import WeatherForecast
            wd = await weather_client.get_weather(city)
            weather_data[city] = WeatherForecast(
                location=wd.location,
                temp_c=wd.temp_c,
                temp_f=wd.temp_f,
                humidity=wd.humidity,
                wind_speed_kmph=wd.wind_speed_kmph,
                wind_dir=wd.wind_dir,
                weather_desc=wd.weather_desc,
                feels_like_c=wd.feels_like_c,
                pressure_mb=wd.pressure_mb,
                precip_mm=wd.precip_mm,
                visibility_km=wd.visibility_km,
                uv_index=wd.uv_index,
            )
        except Exception as exc:
            logger.warning(f"Failed to fetch weather for {city}: {exc}")

    db = CalibrationDB(args.db)
    try:
        samples = await collect_confidence_samples(
            provider_configs=provider_configs,
            temperature_markets=temp_markets,
            weather_data=weather_data,
            db=db,
            n_repeats=args.n_repeats,
            dry_run=args.dry_run,
        )
        logger.info(f"Collected {len(samples)} samples total")
    finally:
        db.close()


if __name__ == "__main__":
    cli_main()
