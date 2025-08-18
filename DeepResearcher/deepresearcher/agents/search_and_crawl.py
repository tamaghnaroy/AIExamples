import asyncio
import math
import time
from typing import List, Tuple, Dict

import numpy as np
import structlog

from ..core.config import Config
from ..core.models import QueryBlock, EnrichedSource
from ..providers.embedding import EmbeddingsProvider
from ..providers.scraper import Scraper
from ..providers.search import Searcher
from ..utils.utils import (
    cosine_sim,
    tokenize,
    domain,
    canonicalize,
    bias_level,
    authority_weight,
)

log = structlog.get_logger(__name__)

def _bm25_prepare(docs: List[List[str]]) -> Tuple[Dict[str, int], List[Dict[str, int]], float]:
    dfs: Dict[str, int] = {}
    tfs: List[Dict[str, int]] = []
    lengths = []
    for tokens in docs:
        tf: Dict[str, int] = {}
        for w in tokens:
            tf[w] = tf.get(w, 0) + 1
        tfs.append(tf)
        lengths.append(len(tokens))
        for w in tf:
            dfs[w] = dfs.get(w, 0) + 1
    avgdl = sum(lengths) / max(1, len(lengths))
    return dfs, tfs, avgdl

def _bm25_score(qtoks: List[str], tf: Dict[str, int], dfs: Dict[str, int], N: int, dl: int, avgdl: float, k1: float, b: float) -> float:
    score = 0.0
    for w in qtoks:
        df = dfs.get(w)
        if not df:
            continue
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
        f = tf.get(w, 0)
        denom = f + k1 * (1 - b + b * dl / max(1e-9, avgdl))
        score += idf * (f * (k1 + 1)) / max(1e-9, denom)
    return score

def _rank_to_dict(sorted_pairs: List[Tuple[float, int]], items: List[str]) -> Dict[str, int]:
    ranks: Dict[str, int] = {}
    rank = 1
    for _, idx in sorted_pairs:
        u = items[idx]
        if u and u not in ranks:
            ranks[u] = rank
            rank += 1
    return ranks

def rrf_fuse(bm25_ranks: Dict[str, int], sem_ranks: Dict[str, int], k: int) -> List[str]:
    scores: Dict[str, float] = {}
    for url, r in bm25_ranks.items():
        scores[url] = scores.get(url, 0.0) + 1.0 / (k + r)
    for url, r in sem_ranks.items():
        scores[url] = scores.get(url, 0.0) + 1.0 / (k + r)
    return [u for u, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

class SearchAndCrawlAgent:
    def __init__(self, searcher: Searcher, scraper: Scraper, embed: EmbeddingsProvider, sem: asyncio.Semaphore, timeout: int, cfg: Config, run_id: str):
        self.searcher = searcher
        self.scraper = scraper
        self.embed = embed
        self.sem = sem
        self.timeout = timeout
        self.cfg = cfg
        self.run_id = run_id

    async def _semantic_rank(self, query: str, docs: List[str], urls: List[str]) -> Dict[str, int]:
        qv = await self.embed.embed(query)
        async def _emb(i: int, txt: str) -> Tuple[int, np.ndarray]:
            async with self.sem:
                return i, await self.embed.embed(txt)
        tasks = [_emb(i, d) for i, d in enumerate(docs)]
        embs = await asyncio.gather(*tasks)
        scores = [(cosine_sim(qv, v), i) for i, v in embs]
        scores.sort(key=lambda x: x[0], reverse=True)
        return _rank_to_dict(scores, urls)

    def _bm25_rank(self, query: str, titles: List[str], snippets: List[str], urls: List[str]) -> Dict[str, int]:
        docs_tokens = [tokenize((t or "") + " " + (s or "")) for t, s in zip(titles, snippets)]
        dfs, tfs, avgdl = _bm25_prepare(docs_tokens)
        q_tokens = tokenize(query)
        N = max(1, len(docs_tokens))
        scored: List[Tuple[float, int]] = []
        for idx, tokens in enumerate(docs_tokens):
            tf = tfs[idx]
            score = _bm25_score(q_tokens, tf, dfs, N, len(tokens), avgdl, self.cfg.bm25_k1, self.cfg.bm25_b)
            url = urls[idx] or ""
            if url.startswith("https://"):
                score += 0.3
            if len(url) > 120:
                score -= 0.1
            scored.append((score, idx))
        scored.sort(key=lambda x: x[0], reverse=True)
        return _rank_to_dict(scored, urls)

    async def run_for_questions(self, questions: List[str], k_per_query: int = 2) -> List[QueryBlock]:
        blocks: List[QueryBlock] = []
        for q in questions:
            t0 = time.time()
            results = await self.searcher.search(query=q, max_results=12)
            titles = [(r.get("title") or "") for r in results]
            snippets = [(r.get("content") or r.get("snippet") or "") for r in results]
            urls = [(r.get("url") or "") for r in results]
            docs_for_sem = [f"{t}\n\n{sn}" for t, sn in zip(titles, snippets)]

            bm25_ranks = self._bm25_rank(q, titles, snippets, urls)
            sem_ranks = await self._semantic_rank(q, docs_for_sem, urls)
            fused = rrf_fuse(bm25_ranks, sem_ranks, k=self.cfg.rrf_k)

            seen = set()
            chosen: List[str] = []
            for u in fused:
                d = domain(u)
                if not u or not d or d in seen:
                    continue
                seen.add(d)
                chosen.append(u)
                if len(chosen) >= k_per_query:
                    break

            async def _scr(u: str) -> EnrichedSource:
                async with self.sem:
                    try:
                        content = await self.scraper.scrape(u, self.timeout)
                        src = EnrichedSource(
                            url=canonicalize(u),
                            content=content,
                            publisher=domain(u),
                            evidence_type=("paper" if u.lower().endswith(".pdf") else "doc"),
                            bias=bias_level(u),
                            authority=authority_weight(u, None, None),
                        )
                        return src
                    except Exception as e:
                        return EnrichedSource(url=u, content="", error=str(e))
            sources = await asyncio.gather(*[_scr(u) for u in chosen])
            latency = time.time() - t0
            log.bind(event="tool_call", run_id=self.run_id, tool="gather", query=q,
                     sources=sum(1 for s in sources if not s.error), latency_s=round(latency,3)).info("search_and_crawl")
            blocks.append(QueryBlock(query=q, sources=sources))
        return blocks
