from __future__ import annotations

import os
from enum import Enum

from common.llm.base import LLMProvider


class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    STRATEGY = "strategy"


_TASK_ENV_MAP: dict[TaskType, str] = {
    TaskType.CLASSIFICATION: "CLASSIFICATION",
    TaskType.STRATEGY: "STRATEGY",
}


def _build_provider(prefix: str) -> LLMProvider:
    provider_type = os.environ.get(f"{prefix}_PROVIDER", "").lower()
    model = os.environ.get(f"{prefix}_MODEL", "")

    if provider_type == "anthropic":
        from common.llm.anthropic_provider import AnthropicProvider

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        return AnthropicProvider(model=model, api_key=api_key)

    if provider_type == "openai_compatible":
        from common.llm.openai_compatible import OpenAICompatibleProvider

        base_url = os.environ.get(f"{prefix}_BASE_URL", "")
        api_key = os.environ.get(f"{prefix}_API_KEY", "")
        return OpenAICompatibleProvider(model=model, base_url=base_url, api_key=api_key)

    raise ValueError(f"Unknown provider type '{provider_type}' for {prefix}")


class LLMRouter:
    def __init__(self) -> None:
        self._providers: dict[TaskType, LLMProvider] = {}

    def get(self, task_type: TaskType) -> LLMProvider:
        if task_type not in self._providers:
            prefix = _TASK_ENV_MAP[task_type]
            self._providers[task_type] = _build_provider(prefix)
        return self._providers[task_type]
