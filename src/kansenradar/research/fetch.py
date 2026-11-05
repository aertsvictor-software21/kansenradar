"""Polite page fetching: robots.txt respected, size cap, HTML -> plain text. Free, local."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx

log = logging.getLogger(__name__)

USER_AGENT = "Kansenradar/0.1 (marktonderzoek; respecteert robots.txt)"
MAX_BYTES = 2_000_000


@dataclass
class FetchedPage:
    url: str
    title: str
    text: str
    fetched_at: datetime
    published_at: datetime | None = None


class FetchSkipped(Exception):
    """Page not fetched (robots.txt, error, wrong content type). Message explains why."""


class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "nav", "footer", "header", "form", "iframe"}
    BLOCK = {"p", "div", "li", "br", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section",
             "article", "td", "th", "dd", "dt", "blockquote"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._skip = 0
        self._in_title = False
        self.meta: dict[str, str] = {}

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag == "meta":
            a = dict(attrs)
            key = (a.get("property") or a.get("name") or "").lower()
            if key and a.get("content"):
                self.meta[key] = a["content"]
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        elif tag == "title":
            self._in_title = False
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> tuple[str, str, dict[str, str]]:
    p = _TextExtractor()
    p.feed(html)
    text = re.sub(r"[ \t\r\f\v]+", " ", "".join(p.parts))
    text = re.sub(r"\n\s*\n+", "\n", text).strip()
    return p.title.strip(), text, p.meta


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
    except ValueError:
        return None


class PageFetcher:
    def __init__(self, http: httpx.Client | None = None, respect_robots: bool = True) -> None:
        self.http = http or httpx.Client(timeout=20, follow_redirects=True,
                                         headers={"User-Agent": USER_AGENT})
        self.respect_robots = respect_robots
        self._robots: dict[str, RobotFileParser | None] = {}
        self._cache: dict[str, FetchedPage] = {}

    def _allowed(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        parts = urlsplit(url)
        base = f"{parts.scheme}://{parts.netloc}"
        if base not in self._robots:
            rp = RobotFileParser()
            try:
                r = self.http.get(f"{base}/robots.txt")
                rp.parse(r.text.splitlines() if r.status_code == 200 else [])
                self._robots[base] = rp
            except httpx.HTTPError:
                self._robots[base] = None  # unreachable robots.txt: allow, as most crawlers do
        rp = self._robots[base]
        return True if rp is None else rp.can_fetch(USER_AGENT, url)

    def fetch(self, url: str) -> FetchedPage:
        if url in self._cache:
            return self._cache[url]
        if urlsplit(url).scheme not in ("http", "https"):
            raise FetchSkipped("geen http(s)-adres")
        if not self._allowed(url):
            raise FetchSkipped("robots.txt verbiedt ophalen")
        try:
            r = self.http.get(url)
        except httpx.HTTPError as exc:
            raise FetchSkipped(f"niet bereikbaar: {exc}") from exc
        if r.status_code != 200:
            raise FetchSkipped(f"HTTP {r.status_code}")
        ctype = r.headers.get("content-type", "")
        if "html" not in ctype and "text/plain" not in ctype:
            raise FetchSkipped(f"geen HTML ({ctype or 'onbekend type'})")
        body = r.content[:MAX_BYTES].decode(r.encoding or "utf-8", errors="replace")
        if "html" in ctype:
            title, text, meta = html_to_text(body)
        else:
            title, text, meta = "", body, {}
        published = _parse_date(meta.get("article:published_time") or meta.get("date")
                                or meta.get("dc.date"))
        page = FetchedPage(url=str(r.url), title=title, text=text,
                           fetched_at=datetime.now(UTC), published_at=published)
        self._cache[url] = page
        return page
