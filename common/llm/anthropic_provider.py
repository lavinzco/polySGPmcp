from __future__ import annotations

from typing import Any, TypeVar

import anthropic
from pydantic import BaseModel

from common.llm.base import JSON_SYSTEM_SUFFIX, LLMProvider

T = TypeVar("T", bound=BaseModel)


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str, *, api_key: str, max_tokens: int = 1024, **kwargs: Any):
        super().__init__(model)
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._max_tokens = max_tokens

    async def complete(self, prompt: str, *, system: str = "") -> str:
        msg = await self._client.messages.create(
            model=self.model,
            max_tokens=self._max_tokens,
            system=system or "You are a helpful assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

    async def complete_json(
        self, prompt: str, schema: type[T], *, system: str = ""
    ) -> T:
        full_system = (system or "You are a helpful assistant.") + JSON_SYSTEM_SUFFIX
        schema_hint = f"\n\nExpected JSON schema:\n{schema.model_json_schema()}"
        raw = await self.complete(prompt + schema_hint, system=full_system)
        return self._parse_model(raw, schema)
