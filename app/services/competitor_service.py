"""Post-process LLM competitor output.

The LLM's competitor list is a suggestion, so this service:
- drops the analyzed brand itself and duplicate competitors,
- checks each suggested competitor website (SSRF-safe HEAD/GET) so the response
  says whether the site really exists (``website_verified``),
- builds the "Your Brand vs Competitor" comparison table,
- attaches a clear note on where competitor data comes from.
"""

from __future__ import annotations

import asyncio
import logging
from urllib.parse import urlsplit

import httpx

from app.config import Settings
from app.schemas import (
    Brand,
    CompetitorAnalysis,
    CompetitorAnalysisOut,
    CompetitorOut,
    ComparisonRow,
)
from app.utils.validators import ensure_public_url

logger = logging.getLogger("competitors")

SOURCE_NOTE = (
    "Competitors are AI-generated suggestions based on the website's category, services and market "
    "(basis='ai_suggestion'), unless named on the website itself (basis='mentioned_on_website'). "
    "website_verified=true means the suggested competitor homepage responded publicly; it does not verify "
    "the strengths/weaknesses, which should be validated with market research before use."
)


def _host(url: str | None) -> str:
    if not url:
        return ""
    if "://" not in url:
        url = "https://" + url
    return urlsplit(url).netloc.lower().removeprefix("www.")


class CompetitorService:
    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self.settings = settings
        self.client = client

    async def enrich(self, data: CompetitorAnalysis, brand: Brand, website_url: str) -> CompetitorAnalysisOut:
        own_host = _host(website_url)
        seen: set[str] = set()
        competitors = []
        for c in data.competitors:
            key = c.name.strip().lower()
            if not key or key == brand.name.strip().lower() or key in seen:
                continue
            if c.website and _host(c.website) == own_host:
                continue
            seen.add(key)
            competitors.append(c)

        verified = await asyncio.gather(*(self._verify(c.website) for c in competitors))
        out = [CompetitorOut(**c.model_dump(), website_verified=v) for c, v in zip(competitors, verified)]

        return CompetitorAnalysisOut(
            **data.model_dump(exclude={"competitors"}),
            competitors=out,
            comparison_table=build_comparison_table(brand, data, out),
            source_note=SOURCE_NOTE,
        )

    async def _verify(self, url: str | None) -> bool | None:
        if not url or not self.settings.verify_competitor_websites:
            return None
        try:
            safe_url = await ensure_public_url(url, self.settings.allowed_port_set)
            resp = await self.client.head(
                safe_url,
                headers={"User-Agent": self.settings.user_agent},
                follow_redirects=False,
                timeout=6,
            )
            if resp.status_code in (405, 501):  # HEAD not supported
                resp = await self.client.get(
                    safe_url, headers={"User-Agent": self.settings.user_agent}, follow_redirects=False, timeout=6
                )
            # Redirects (e.g. http->https, locale) and bot-blocks still prove the site exists.
            return resp.status_code < 400 or resp.status_code in (401, 403, 429)
        except Exception as exc:  # noqa: BLE001 - verification is best effort
            logger.info("Competitor website check failed for %s: %s", url, exc)
            return False


def build_comparison_table(
    brand: Brand, data: CompetitorAnalysis, competitors: list[CompetitorOut]
) -> list[ComparisonRow]:
    profile = data.brand_profile
    rows = [
        ("Brand Name", brand.name, lambda c: c.name),
        ("Main Service", profile.main_service, lambda c: c.main_service),
        ("Target Market", profile.target_market, lambda c: c.target_market),
        ("Key Strength", profile.key_strength, lambda c: c.key_strength),
        ("Opportunity", profile.opportunity, lambda c: c.opportunity_for_brand),
    ]
    return [
        ComparisonRow(comparison=label, your_brand=mine, competitors={c.name: getter(c) for c in competitors})
        for label, mine, getter in rows
    ]
