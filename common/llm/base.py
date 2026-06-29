from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

JSON_SYSTEM_SUFFIX = (
    "\n\nYou MUST respond with valid JSON only. "
    "No markdown fences, no explanation, no trailing text. "
    "Output a single JSON object."
)


class LLMProvider(ABC):
    def __init__(self, model: str, **kwargs: Any):
        self.model = model

    @abstractmethod
    async def complete(self, prompt: str, *, system: str = "") -> str:
        ...

    @abstractmethod
    async def complete_json(
        self, prompt: str, schema: type[T], *, system: str = ""
    ) -> T:
        ...

    @staticmethod
    def _extract_json(text: str) -> str:
        fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if fenced:
            return fenced.group(1).strip()
        braces = re.search(r"\{[\s\S]*\}", text)
        if braces:
            return braces.group(0)
        return text.strip()

    @classmethod
    def _parse_model(cls, text: str, schema: type[T]) -> T:
        cleaned = cls._extract_json(text)
        data = json.loads(cleaned)
        return schema.model_validate(data)
