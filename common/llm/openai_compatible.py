from __future__ import annotations

import logging
from typing import Any, TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel

from common.llm.base import JSON_SYSTEM_SUFFIX, LLMProvider

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger("hermes.llm.openai")


class OpenAICompatibleProvider(LLMProvider):
    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        api_key: str,
        max_tokens: int = 1024,
        **kwargs: Any,
    ):
        super().__init__(model)
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._max_tokens = max_tokens

    async def complete(self, prompt: str, *, system: str = "") -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self._max_tokens,
        )
        return resp.choices[0].message.content or ""

    async def complete_json(
        self, prompt: str, schema: type[T], *, system: str = ""
    ) -> T:
        schema_hint = f"\n\nExpected JSON schema:\n{schema.model_json_schema()}"
        full_prompt = prompt + schema_hint

        try:
            return await self._complete_json_native(full_prompt, schema, system=system)
        except Exception as exc:
            if self._is_json_mode_unsupported(exc):
                logger.info("JSON mode not supported, falling back to prompt constraint")
                return await self._complete_json_fallback(full_prompt, schema, system=system)
            raise

    async def _complete_json_native(
        self, prompt: str, schema: type[T], *, system: str = ""
    ) -> T:
        messages: list[dict[str, str]] = []
        sys_content = (system or "You are a helpful assistant.") + JSON_SYSTEM_SUFFIX
        messages.append({"role": "system", "content": sys_content})
        messages.append({"role": "user", "content": prompt})
        resp = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self._max_tokens,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or "{}"
        return self._parse_model(raw, schema)

    async def _complete_json_fallback(
        self, prompt: str, schema: type[T], *, system: str = ""
    ) -> T:
        full_system = (system or "You are a helpful assistant.") + JSON_SYSTEM_SUFFIX
        raw = await self.complete(prompt, system=full_system)
        return self._parse_model(raw, schema)

    @staticmethod
    def _is_json_mode_unsupported(exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(
            kw in msg
            for kw in ["response_format", "json_object", "not supported", "invalid", "unknown"]
        )
