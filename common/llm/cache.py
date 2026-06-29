from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field


@dataclass
class _CacheEntry:
    value: str
    expires_at: float


@dataclass
class LLMCache:
    ttl_seconds: float = 7 * 24 * 3600  # 7 days
    _store: dict[str, _CacheEntry] = field(default_factory=dict)

    def _make_key(self, task_type: str, content: str) -> str:
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        return f"{task_type}:{content_hash}"

    def get(self, task_type: str, content: str) -> str | None:
        key = self._make_key(task_type, content)
        entry = self._store.get(key)
        if entry is None:
            return None
        if time.time() > entry.expires_at:
            del self._store[key]
            return None
        return entry.value

    def put(self, task_type: str, content: str, value: str) -> None:
        key = self._make_key(task_type, content)
        self._store[key] = _CacheEntry(value=value, expires_at=time.time() + self.ttl_seconds)

    def clear(self) -> None:
        self._store.clear()

    @property
    def size(self) -> int:
        return len(self._store)
