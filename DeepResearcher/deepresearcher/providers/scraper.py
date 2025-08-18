import os
import asyncio
import json
from firecrawl import FirecrawlApp
import structlog

from ..utils.utils import hash_key, canonicalize

log = structlog.get_logger(__name__)

class Scraper:
    def __init__(self, client: FirecrawlApp, cache_dir: str, run_id: str):
        self.client = client
        self.dir = os.path.join(cache_dir, "scrape")
        os.makedirs(self.dir, exist_ok=True)
        self._lock = asyncio.Lock()
        self.run_id = run_id

    def _path(self, url: str) -> str:
        return os.path.join(self.dir, f"{hash_key(canonicalize(url))}.json")

    async def scrape(self, url: str, timeout_s: int) -> dict:
        path = self._path(url)
        async with self._lock:
            if os.path.exists(path):
                with open(path, "r") as f:
                    return json.load(f)
        try:
            data = self.client.scrape_url(url)
            log.bind(event="firecrawl_scrape", run_id=self.run_id, url=url).info("scraper.invoked")
            # Handle both dict and ScrapeResponse object formats
            if hasattr(data, 'content'):
                # ScrapeResponse object
                content = getattr(data, 'content', '') or ''
                metadata = getattr(data, 'metadata', {}) or {}
                if hasattr(metadata, 'get'):
                    publisher = metadata.get('sourceURL', '')
                    pub_date = metadata.get('publishedTime', None)
                else:
                    publisher = getattr(metadata, 'sourceURL', '') if metadata else ''
                    pub_date = getattr(metadata, 'publishedTime', None) if metadata else None
            else:
                # Dict format
                content = data.get("content", "") if isinstance(data, dict) else ""
                metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
                publisher = metadata.get("sourceURL", "") if isinstance(metadata, dict) else ""
                pub_date = metadata.get("publishedTime", None) if isinstance(metadata, dict) else None
            
            res = {
                "content": content,
                "url": url,
                "publisher": publisher,
                "pub_date": pub_date,
            }
            async with self._lock:
                with open(path, "w") as f:
                    json.dump(res, f)
            return res
        except Exception as e:
            log.bind(event="firecrawl_error", run_id=self.run_id, url=url, error=str(e)).error("scraper.failed")
            return {"error": str(e), "url": url, "content": ""}
