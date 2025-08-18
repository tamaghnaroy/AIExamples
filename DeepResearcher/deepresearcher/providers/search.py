from tavily import TavilyClient
import structlog

log = structlog.get_logger(__name__)

class Searcher:
    def __init__(self, client: TavilyClient, run_id: str):
        self.client = client
        self.run_id = run_id

    async def search(self, query: str, max_results: int = 7) -> list:
        try:
            res = self.client.search(query=query, max_results=max_results, search_depth="advanced")
            log.bind(event="tavily_search", run_id=self.run_id, query=query).info("search.invoked")
            return res.get("results", [])
        except Exception as e:
            log.bind(event="tavily_error", run_id=self.run_id, error=str(e)).error("search.failed")
            return []
