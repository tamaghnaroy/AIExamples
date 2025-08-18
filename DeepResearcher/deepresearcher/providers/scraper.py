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
            data = await asyncio.wait_for(self.client.scrape_url(url), timeout=timeout_s)
            log.bind(event="firecrawl_scrape", run_id=self.run_id, url=url).info("scraper.invoked")
            # Firecrawl returns a dict with 'content', 'markdown', 'metadata'
            res = {
                "content": data.get("content", ""),
                "url": url,
                "publisher": data.get("metadata", {}).get("sourceURL", ""),
                "pub_date": data.get("metadata", {}).get("publishedTime"),
            }
            async with self._lock:
                with open(path, "w") as f:
                    json.dump(res, f)
            return res
        except Exception as e:
            log.bind(event="firecrawl_error", run_id=self.run_id, url=url, error=str(e)).error("scraper.failed")
            return {"error": str(e), "url": url, "content": ""}
