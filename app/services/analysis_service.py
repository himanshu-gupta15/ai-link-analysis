"""Orchestrates: validate -> extract -> (cache) -> LLM -> enrich -> response."""

from __future__ import annotations

import logging
import time
from collections import OrderedDict

import httpx

from app.config import Settings
from app.errors import BlockedURL, InvalidURL, WebsiteUnreachable
from app.schemas import Analysis, LinkAnalysisResponse, Metadata
from app.services.competitor_service import CompetitorService
from app.services.llm_service import LLMService
from app.services.website_extractor import WebsiteExtractor
from app.utils.validators import BlockedURLError, DNSResolutionError, InvalidURLError, ensure_public_url

logger = logging.getLogger("analysis")


class TTLCache:
    """Small in-memory LRU cache with expiry (per process).

    Re-analyzing the same website within the TTL costs nothing. For multiple
    workers/instances, swap for Redis.
    """

    def __init__(self, ttl_seconds: int, max_entries: int) -> None:
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self._data: OrderedDict[str, tuple[float, LinkAnalysisResponse]] = OrderedDict()

    def get(self, key: str) -> LinkAnalysisResponse | None:
        item = self._data.get(key)
        if not item:
            return None
        expires, value = item
        if expires < time.monotonic():
            self._data.pop(key, None)
            return None
        self._data.move_to_end(key)
        return value

    def set(self, key: str, value: LinkAnalysisResponse) -> None:
        if self.ttl <= 0:
            return
        self._data[key] = (time.monotonic() + self.ttl, value)
        self._data.move_to_end(key)
        while len(self._data) > self.max_entries:
            self._data.popitem(last=False)


def cache_key(url: str) -> str:
    return url.lower().rstrip("/")


class AnalysisService:
    def __init__(
        self,
        settings: Settings,
        http_client: httpx.AsyncClient,
        llm: LLMService,
        cache: TTLCache,
    ) -> None:
        self.settings = settings
        self.extractor = WebsiteExtractor(settings, http_client)
        self.competitors = CompetitorService(settings, http_client)
        self.llm = llm
        self.cache = cache

    async def analyze(self, raw_url: str) -> LinkAnalysisResponse:
        started = time.perf_counter()
        url = await self._validate(raw_url)

        cached = self.cache.get(cache_key(url))
        if cached:
            logger.info("Cache hit for %s", url)
            return cached.model_copy(update={"metadata": cached.metadata.model_copy(update={"cached": True})})

        _ = self.llm.client  # fail fast (before scraping) if the LLM is not configured
        website = await self.extractor.extract(url)
        logger.info(
            "Extracted %s: %d pages, %d chars (truncated=%s)",
            website.final_url, len(website.pages), len(website.content), website.content_truncated,
        )

        llm_result = await self.llm.analyze(website.final_url, website)
        result = llm_result.analysis
        competitor_analysis = await self.competitors.enrich(result.competitor_analysis, result.brand, website.final_url)

        response = LinkAnalysisResponse(
            website=website.final_url,
            brand=result.brand,
            analysis=Analysis(
                competitor_analysis=competitor_analysis,
                target_search_queries=result.target_search_queries,
                seo_improvements=result.seo_improvements,
                top_services=result.top_services,
                target_audience=result.target_audience,
            ),
            seo_audit=website.seo_audit,
            metadata=Metadata(
                model=llm_result.model,
                input_tokens=llm_result.input_tokens,
                output_tokens=llm_result.output_tokens,
                cache_read_tokens=llm_result.cache_read_tokens,
                estimated_cost_usd=llm_result.estimated_cost_usd,
                llm_latency_ms=llm_result.latency_ms,
                total_latency_ms=int((time.perf_counter() - started) * 1000),
                pages_analyzed=website.page_urls,
                content_chars_sent=len(website.content),
                content_truncated=website.content_truncated,
                cached=False,
                data_quality=result.data_quality,
            ),
        )
        self.cache.set(cache_key(url), response)
        # Cache under the final URL too (after redirects).
        self.cache.set(cache_key(website.final_url), response)
        return response

    async def _validate(self, raw_url: str) -> str:
        try:
            return await ensure_public_url(raw_url, self.settings.allowed_port_set)
        except InvalidURLError as exc:
            raise InvalidURL(str(exc)) from None
        except BlockedURLError as exc:
            raise BlockedURL(str(exc)) from None
        except DNSResolutionError as exc:
            raise WebsiteUnreachable(str(exc), code="DNS_RESOLUTION_FAILED") from None
