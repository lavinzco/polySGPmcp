from __future__ import annotations

import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from common.llm.base import LLMProvider
from common.llm.cache import LLMCache
from common.llm.router import LLMRouter, TaskType


class DummyResult(BaseModel):
    label: str = ""
    score: float = 0.0


# --- base.py ---


class TestExtractJson:
    def test_plain_json(self):
        assert LLMProvider._extract_json('{"a": 1}') == '{"a": 1}'

    def test_fenced_json(self):
        text = '```json\n{"a": 1}\n```'
        assert LLMProvider._extract_json(text) == '{"a": 1}'

    def test_json_with_surrounding_text(self):
        text = 'Here is the result: {"a": 1} hope that helps'
        assert LLMProvider._extract_json(text) == '{"a": 1}'


class TestParseModel:
    def test_valid(self):
        result = LLMProvider._parse_model('{"label": "cat", "score": 0.9}', DummyResult)
        assert result.label == "cat"
        assert result.score == 0.9

    def test_from_fenced(self):
        text = '```json\n{"label": "dog", "score": 0.8}\n```'
        result = LLMProvider._parse_model(text, DummyResult)
        assert result.label == "dog"


# --- anthropic_provider.py ---


@pytest.mark.asyncio
async def test_anthropic_complete():
    mock_anthropic_mod = MagicMock()
    mock_client = AsyncMock()
    mock_anthropic_mod.AsyncAnthropic.return_value = mock_client

    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text="Hello world")]
    mock_client.messages.create = AsyncMock(return_value=mock_msg)

    with patch.dict("sys.modules", {"anthropic": mock_anthropic_mod}):
        import importlib
        import common.llm.anthropic_provider as ap_mod

        importlib.reload(ap_mod)
        provider = ap_mod.AnthropicProvider(model="claude-opus-4-7", api_key="test-key")
        result = await provider.complete("Say hi")

    assert result == "Hello world"
    mock_client.messages.create.assert_called_once()
    assert mock_client.messages.create.call_args.kwargs["model"] == "claude-opus-4-7"


@pytest.mark.asyncio
async def test_anthropic_complete_json():
    mock_anthropic_mod = MagicMock()
    mock_client = AsyncMock()
    mock_anthropic_mod.AsyncAnthropic.return_value = mock_client

    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text='{"label": "weather", "score": 0.95}')]
    mock_client.messages.create = AsyncMock(return_value=mock_msg)

    with patch.dict("sys.modules", {"anthropic": mock_anthropic_mod}):
        import importlib
        import common.llm.anthropic_provider as ap_mod

        importlib.reload(ap_mod)
        provider = ap_mod.AnthropicProvider(model="claude-opus-4-7", api_key="test-key")
        result = await provider.complete_json("Classify this", DummyResult)

    assert isinstance(result, DummyResult)
    assert result.label == "weather"
    assert result.score == 0.95


# --- openai_compatible.py ---


def _make_openai_response(content: str):
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    return resp


@pytest.mark.asyncio
async def test_openai_complete():
    mock_openai_mod = MagicMock()
    mock_client = AsyncMock()
    mock_openai_mod.AsyncOpenAI.return_value = mock_client
    mock_client.chat.completions.create = AsyncMock(
        return_value=_make_openai_response("Hi there")
    )

    with patch.dict("sys.modules", {"openai": mock_openai_mod}):
        import importlib
        import common.llm.openai_compatible as oc_mod

        importlib.reload(oc_mod)
        provider = oc_mod.OpenAICompatibleProvider(
            model="deepseek-chat", base_url="https://api.deepseek.com/v1", api_key="test-key"
        )
        result = await provider.complete("Say hi")

    assert result == "Hi there"


@pytest.mark.asyncio
async def test_openai_complete_json_native():
    mock_openai_mod = MagicMock()
    mock_client = AsyncMock()
    mock_openai_mod.AsyncOpenAI.return_value = mock_client
    mock_client.chat.completions.create = AsyncMock(
        return_value=_make_openai_response('{"label": "rain", "score": 0.8}')
    )

    with patch.dict("sys.modules", {"openai": mock_openai_mod}):
        import importlib
        import common.llm.openai_compatible as oc_mod

        importlib.reload(oc_mod)
        provider = oc_mod.OpenAICompatibleProvider(
            model="deepseek-chat", base_url="https://api.deepseek.com/v1", api_key="test-key"
        )
        result = await provider.complete_json("Classify", DummyResult)

    assert result.label == "rain"
    assert result.score == 0.8
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_openai_complete_json_fallback():
    """Provider doesn't support response_format — falls back to prompt constraint."""
    mock_openai_mod = MagicMock()
    mock_client = AsyncMock()
    mock_openai_mod.AsyncOpenAI.return_value = mock_client

    call_count = 0

    async def side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1 and "response_format" in kwargs:
            raise Exception("response_format is not supported for this model")
        return _make_openai_response('{"label": "snow", "score": 0.7}')

    mock_client.chat.completions.create = AsyncMock(side_effect=side_effect)

    with patch.dict("sys.modules", {"openai": mock_openai_mod}):
        import importlib
        import common.llm.openai_compatible as oc_mod

        importlib.reload(oc_mod)
        provider = oc_mod.OpenAICompatibleProvider(
            model="some-model", base_url="https://api.example.com/v1", api_key="test-key"
        )
        result = await provider.complete_json("Classify", DummyResult)

    assert result.label == "snow"
    assert result.score == 0.7
    assert call_count == 2


# --- cache.py ---


class TestLLMCache:
    def test_put_and_get(self):
        cache = LLMCache()
        cache.put("classification", "hurricane market", "weather")
        assert cache.get("classification", "hurricane market") == "weather"

    def test_miss(self):
        cache = LLMCache()
        assert cache.get("classification", "unknown") is None

    def test_ttl_expiry(self):
        cache = LLMCache(ttl_seconds=1)
        cache.put("classification", "test", "value")
        assert cache.get("classification", "test") == "value"

        cache._store[cache._make_key("classification", "test")].expires_at = time.time() - 1
        assert cache.get("classification", "test") is None

    def test_different_task_types(self):
        cache = LLMCache()
        cache.put("classification", "content", "result_a")
        cache.put("strategy", "content", "result_b")
        assert cache.get("classification", "content") == "result_a"
        assert cache.get("strategy", "content") == "result_b"

    def test_clear(self):
        cache = LLMCache()
        cache.put("a", "b", "c")
        cache.clear()
        assert cache.size == 0


# --- router.py ---


class TestLLMRouter:
    def test_routes_to_openai_compatible(self):
        env = {
            "CLASSIFICATION_PROVIDER": "openai_compatible",
            "CLASSIFICATION_BASE_URL": "https://api.deepseek.com/v1",
            "CLASSIFICATION_MODEL": "deepseek-chat",
            "CLASSIFICATION_API_KEY": "sk-test",
        }
        mock_openai_mod = MagicMock()
        mock_openai_mod.AsyncOpenAI.return_value = AsyncMock()
        with patch.dict(os.environ, env, clear=False):
            with patch.dict("sys.modules", {"openai": mock_openai_mod}):
                import importlib
                import common.llm.openai_compatible as oc_mod

                importlib.reload(oc_mod)
                router = LLMRouter()
                provider = router.get(TaskType.CLASSIFICATION)
                assert isinstance(provider, oc_mod.OpenAICompatibleProvider)
                assert provider.model == "deepseek-chat"

    def test_routes_to_anthropic(self):
        env = {
            "STRATEGY_PROVIDER": "anthropic",
            "STRATEGY_MODEL": "claude-opus-4-7",
            "ANTHROPIC_API_KEY": "sk-ant-test",
        }
        mock_anthropic_mod = MagicMock()
        mock_anthropic_mod.AsyncAnthropic.return_value = AsyncMock()
        with patch.dict(os.environ, env, clear=False):
            with patch.dict("sys.modules", {"anthropic": mock_anthropic_mod}):
                import importlib
                import common.llm.anthropic_provider as ap_mod

                importlib.reload(ap_mod)
                router = LLMRouter()
                provider = router.get(TaskType.STRATEGY)
                assert isinstance(provider, ap_mod.AnthropicProvider)
                assert provider.model == "claude-opus-4-7"

    def test_unknown_provider_raises(self):
        env = {"CLASSIFICATION_PROVIDER": "unknown_thing", "CLASSIFICATION_MODEL": "x"}
        with patch.dict(os.environ, env, clear=False):
            router = LLMRouter()
            with pytest.raises(ValueError, match="Unknown provider"):
                router.get(TaskType.CLASSIFICATION)

    def test_caches_provider_instance(self):
        env = {
            "CLASSIFICATION_PROVIDER": "openai_compatible",
            "CLASSIFICATION_BASE_URL": "https://api.deepseek.com/v1",
            "CLASSIFICATION_MODEL": "deepseek-chat",
            "CLASSIFICATION_API_KEY": "sk-test",
        }
        mock_openai_mod = MagicMock()
        mock_openai_mod.AsyncOpenAI.return_value = AsyncMock()
        with patch.dict(os.environ, env, clear=False):
            with patch.dict("sys.modules", {"openai": mock_openai_mod}):
                import importlib
                import common.llm.openai_compatible as oc_mod

                importlib.reload(oc_mod)
                router = LLMRouter()
                p1 = router.get(TaskType.CLASSIFICATION)
                p2 = router.get(TaskType.CLASSIFICATION)
                assert p1 is p2
