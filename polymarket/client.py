from __future__ import annotations

import httpx

from common.config import settings
from polymarket.models import Event, Market


WEATHER_TAG_ID = 84


class GammaClient:
    """Thin async wrapper around the Polymarket Gamma API (read-only)."""

    def __init__(self, base_url: str | None = None, client: httpx.AsyncClient | None = None):
        self._base_url = (base_url or settings.gamma_api_base_url).rstrip("/")
        self._external_client = client

    async def _client(self) -> httpx.AsyncClient:
        if self._external_client is not None:
            return self._external_client
        return httpx.AsyncClient(
            base_url=self._base_url,
            timeout=30,
            headers={"Accept-Encoding": "gzip, deflate"},
        )

    async def get_markets(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        active: bool = True,
        closed: bool = False,
    ) -> list[Market]:
        params: dict = {
            "limit": limit,
            "offset": offset,
            "active": active,
            "closed": closed,
        }
        client = await self._client()
        should_close = self._external_client is None
        try:
            resp = await client.get(f"{self._base_url}/markets", params=params)
            resp.raise_for_status()
            return [Market.model_validate(m) for m in resp.json()]
        finally:
            if should_close:
                await client.aclose()

    async def get_all_markets(self, *, max_pages: int = 5, page_size: int = 100) -> list[Market]:
        all_markets: list[Market] = []
        for page in range(max_pages):
            batch = await self.get_markets(limit=page_size, offset=page * page_size)
            all_markets.extend(batch)
            if len(batch) < page_size:
                break
        return all_markets

    async def get_events_by_tag(
        self,
        tag_id: int = WEATHER_TAG_ID,
        *,
        active: bool = True,
        closed: bool = False,
        max_pages: int = 10,
        page_size: int = 100,
    ) -> list[Event]:
        all_events: list[Event] = []
        client = await self._client()
        should_close = self._external_client is None
        try:
            for page in range(max_pages):
                params: dict = {
                    "tag_id": tag_id,
                    "limit": page_size,
                    "offset": page * page_size,
                    "active": active,
                    "closed": closed,
                }
                resp = await client.get(f"{self._base_url}/events", params=params)
                resp.raise_for_status()
                raw = resp.json()
                items = raw if isinstance(raw, list) else []
                for item in items:
                    all_events.append(Event.model_validate(item))
                if len(items) < page_size:
                    break
        finally:
            if should_close:
                await client.aclose()
        return all_events
