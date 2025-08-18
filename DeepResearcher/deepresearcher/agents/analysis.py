import asyncio
from typing import List, Tuple, Optional

import structlog

from ..core.models import QueryBlock, EnrichedSource, ExtractedData
from ..providers.llm import LLM

log = structlog.get_logger(__name__)

class AnalysisAgent:
    def __init__(self, llm: LLM, run_id: str):
        self.llm = llm
        self.run_id = run_id

    async def extract_from_source(self, url: str, summary: str) -> ExtractedData:
        system = "You are an information extraction system. Output strictly valid JSON with fields."
        user = (
            "From the neutral notes below, extract:\n"
            "- claims: list of {text, polarity: support|refute|neutral}\n"
            "- stats: list of {metric, value (number if possible), unit (optional), n (optional), ci (optional [lo,hi])}\n"
            "- entities: list of {type, name}\n"
            "- dates: list of {type, value}\n\n"
            f"URL: {url}\nNOTES:\n{summary}\n\n"
            'Return JSON with keys {"claims":[], "stats":[], "entities":[], "dates":[]}'
        )
        data = await self.llm.json_schema_complete(system, user, ExtractedData)
        data.url = url
        log.bind(event="tool_call", run_id=self.run_id, tool="extract", url=url,
                 claims=len(data.claims), stats=len(data.stats)).info("analysis.extract_from_source")
        return data

    async def enrich(self, blocks: List[QueryBlock], sem: asyncio.Semaphore) -> List[QueryBlock]:
        tasks = []
        for b in blocks:
            for s in b.sources:
                if not s.error and s.summary and not s.extracted:
                    async def _ext(src: EnrichedSource) -> Tuple[EnrichedSource, Optional[ExtractedData]]:
                        async with sem:
                            try:
                                return src, await self.extract_from_source(src.url, src.summary)
                            except Exception as e:
                                return src, None
                    tasks.append(_ext(s))
        results = await asyncio.gather(*tasks) if tasks else []
        for src, data in results:
            if data:
                src.extracted = data
        return blocks
