from __future__ import annotations

from datetime import datetime, timezone, timedelta
from unittest.mock import patch

import pytest

from agent.manual_input import (
    DailySoundingNote,
    InversionStrength,
    SGT,
    get_todays_sounding_note,
    submit_sounding_note,
)


class TestSubmitSoundingNote:
    def test_submit_and_retrieve(self, tmp_path):
        db = tmp_path / "test.db"
        result = submit_sounding_note(
            db, inversion=InversionStrength.STRONG,
            surface_temp_c=26.5, note="Low-level inversion visible",
        )
        assert result.inversion == InversionStrength.STRONG
        assert result.surface_temp_c == 26.5
        assert result.note == "Low-level inversion visible"
        assert result.target_date == datetime.now(SGT).strftime("%Y-%m-%d")

    def test_upsert_overwrites_same_day(self, tmp_path):
        db = tmp_path / "test.db"
        submit_sounding_note(db, inversion=InversionStrength.STRONG, note="first")
        submit_sounding_note(db, inversion=InversionStrength.WEAK, note="updated")

        note = get_todays_sounding_note(db)
        assert note is not None
        assert note.inversion == InversionStrength.WEAK
        assert note.note == "updated"

    def test_optional_fields(self, tmp_path):
        db = tmp_path / "test.db"
        result = submit_sounding_note(db, inversion=InversionStrength.NONE)
        assert result.surface_temp_c is None
        assert result.note == ""


class TestGetTodaysSoundingNote:
    def test_returns_none_when_empty(self, tmp_path):
        db = tmp_path / "test.db"
        assert get_todays_sounding_note(db) is None

    def test_returns_none_for_nonexistent_db(self, tmp_path):
        db = tmp_path / "nonexistent.db"
        assert get_todays_sounding_note(db) is None

    def test_returns_todays_note(self, tmp_path):
        db = tmp_path / "test.db"
        submit_sounding_note(db, inversion=InversionStrength.STRONG, note="test")
        note = get_todays_sounding_note(db)
        assert note is not None
        assert note.inversion == InversionStrength.STRONG

    def test_cross_day_not_returned(self, tmp_path):
        db = tmp_path / "test.db"
        yesterday = (datetime.now(SGT) - timedelta(days=1)).strftime("%Y-%m-%d")
        submit_sounding_note(
            db, inversion=InversionStrength.STRONG,
            target_date=yesterday,
        )

        note = get_todays_sounding_note(db)
        assert note is None


class TestPromptIntegration:
    def test_prompt_without_sounding(self):
        from agent.models import WeatherForecast, GammaMarket
        from agent.prompts import build_strategy_prompt

        weather = WeatherForecast(
            location="Singapore", temp_c=31, temp_f=87.8, humidity=80,
            wind_speed_kmph=10, wind_dir="S", weather_desc="Partly cloudy",
            feels_like_c=35, pressure_mb=1010, precip_mm=0,
            visibility_km=10, uv_index=10,
        )
        market = GammaMarket(
            id="test", question="Will the highest temperature in Singapore be 33°C on June 28?",
            description="Test", outcome_yes_price=0.3, outcome_no_price=0.7,
        )

        prompt = build_strategy_prompt(weather, market)
        assert "No manual sounding note submitted" in prompt
        assert "solely on real-time observations" in prompt
        assert "STRONG" not in prompt

    def test_prompt_with_sounding(self):
        from agent.models import WeatherForecast, GammaMarket
        from agent.prompts import build_strategy_prompt

        weather = WeatherForecast(
            location="Singapore", temp_c=31, temp_f=87.8, humidity=80,
            wind_speed_kmph=10, wind_dir="S", weather_desc="Partly cloudy",
            feels_like_c=35, pressure_mb=1010, precip_mm=0,
            visibility_km=10, uv_index=10,
        )
        market = GammaMarket(
            id="test", question="Will the highest temperature in Singapore be 33°C on June 28?",
            description="Test", outcome_yes_price=0.3, outcome_no_price=0.7,
        )

        note = DailySoundingNote(
            target_date="2026-06-28",
            inversion=InversionStrength.STRONG,
            surface_temp_c=26.5,
            note="Low-level inversion clearly visible",
            submitted_at="2026-06-28T08:00:00+08:00",
        )

        prompt = build_strategy_prompt(weather, market, sounding_note=note)
        assert "Daily Atmospheric Prior" in prompt
        assert "STRONG low-level inversion" in prompt
        assert "26.5°C" in prompt
        assert "Low-level inversion clearly visible" in prompt
        assert "qualitative human judgment" in prompt
        assert "No manual sounding note" not in prompt
