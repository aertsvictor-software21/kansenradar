"""Web search providers (all free). Pluggable; 'none' keeps the platform usable.

  - none:    no web search; research uses open data and URLs supplied by the user.
  - searxng: self-hosted metasearch (free, local; e.g. via Docker). KR_SEARXNG_URL.
  - tavily:  free tier (1,000 credits/month, no credit card; status 2026). TAVILY_API_KEY.

Brave Search is deliberately not included: since February 2026 it requires a credit card.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Protocol

import httpx

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchHit:
    title: str
    url: str
    snippet: str


class SearchProvider(Protocol):
    name: str

    def search(self, query: str, max_results: int = 5) -> list[SearchHit]: ...


class NoSearch:
    name = "geen"

    def search(self, query: str, max_results: int = 5) -> list[SearchHit]:
        return []


class SearxngSearch:
    name = "searxng"

    def __init__(self, base_url: str, http: httpx.Client | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.http = http or httpx.Client(timeout=30)

    def search(self, query: str, max_results: int = 5) -> list[SearchHit]:
        try:
            r = self.http.get(f"{self.base_url}/search",
                              params={"q": query, "format": "json", "language": "nl"})
            r.raise_for_status()
        except httpx.HTTPError as exc:
            log.warning("SearXNG-zoekopdracht mislukt: %s", exc)
            return []
        return [SearchHit(h.get("title", ""), h["url"], h.get("content", ""))
                for h in r.json().get("results", [])[:max_results] if h.get("url")]


class TavilySearch:
    name = "tavily"
    URL = "https://api.tavily.com/search"

    def __init__(self, api_key: str, http: httpx.Client | None = None) -> None:
        self.api_key = api_key
        self.http = http or httpx.Client(timeout=30)

    def search(self, query: str, max_results: int = 5) -> list[SearchHit]:
        try:
            r = self.http.post(self.URL, json={"query": query, "max_results": max_results,
                                               "search_depth": "basic"},
                               headers={"Authorization": f"Bearer {self.api_key}"})
            r.raise_for_status()
        except httpx.HTTPError as exc:
            log.warning("Tavily-zoekopdracht mislukt: %s", exc)
            return []
        return [SearchHit(h.get("title", ""), h["url"], h.get("content", ""))
                for h in r.json().get("results", [])[:max_results] if h.get("url")]


def build_search(settings) -> SearchProvider:
    name = settings.search_provider.lower()
    if name == "searxng":
        return SearxngSearch(settings.searxng_url)
    if name == "tavily":
        key = os.environ.get("TAVILY_API_KEY")
        if key:
            return TavilySearch(key)
        log.warning("Geen TAVILY_API_KEY ingesteld; verder zonder webzoekfunctie.")
    elif name not in ("none", "geen"):
        log.warning("Onbekende zoekprovider '%s'; verder zonder webzoekfunctie.", name)
    return NoSearch()
