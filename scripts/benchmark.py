"""Measure real latency, token usage and cost per model on real websites.

Usage (spends real API credits – roughly $0.01–$0.15 per URL per model):

    ANTHROPIC_API_KEY=... python -m scripts.benchmark \
        --models claude-opus-5-5 claude-sonnet-5-5 claude-haiku-5-5 \
        --urls https://www.python.org https://stripe.com https://www.shopify.com

Writes a Markdown table to docs/benchmark.md so measured numbers can replace
the estimates in the README.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx

from app.config import get_settings
from app.services.llm_service import LLMService
from app.services.website_extractor import WebsiteExtractor


async def main(models: list[str], urls: list[str], effort: str) -> None:
    base = get_settings()
    rows, samples = [], {}
    async with httpx.AsyncClient(trust_env=False) as http:
        extractor = WebsiteExtractor(base, http)
        sites = {}
        for url in urls:
            try:
                sites[url] = await extractor.extract(url)
            except Exception as exc:  # noqa: BLE001
                print(f"skip {url}: {exc}")

        for model in models:
            settings = base.model_copy(update={"llm_model": model, "llm_fallback_model": "", "llm_effort": effort})
            llm = LLMService(settings)
            lat, cost, tin, tout, ok = [], [], [], [], 0
            for url, site in sites.items():
                t = time.perf_counter()
                try:
                    r = await llm.analyze(site.final_url, site)
                except Exception as exc:  # noqa: BLE001
                    print(f"{model} {url}: FAILED {exc}")
                    continue
                ok += 1
                lat.append(time.perf_counter() - t)
                cost.append(r.estimated_cost_usd or 0)
                tin.append(r.input_tokens)
                tout.append(r.output_tokens)
                samples[f"{model} | {url}"] = {
                    "brand": r.analysis.brand.name,
                    "competitors": [c.name for c in r.analysis.competitor_analysis.competitors],
                    "primary_keyword": r.analysis.seo_improvements.primary_keyword,
                }
                print(f"{model} {url}: {lat[-1]:.1f}s in={r.input_tokens} out={r.output_tokens} ${cost[-1]:.4f}")
            if ok:
                rows.append(
                    f"| {model} | {effort} | {ok}/{len(sites)} | {statistics.median(lat):.1f}s | {max(lat):.1f}s | "
                    f"{int(statistics.mean(tin))} | {int(statistics.mean(tout))} | ${statistics.mean(cost):.4f} |"
                )

    out = Path("docs/benchmark.md")
    out.parent.mkdir(exist_ok=True)
    out.write_text(
        "# Measured benchmark\n\n"
        f"Run: {time.strftime('%Y-%m-%d %H:%M')} · URLs: {', '.join(sites)}\n\n"
        "| Model | Effort | OK | Median latency | Max latency | Avg input tok | Avg output tok | Avg cost |\n"
        "|---|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n\n"
        "## Sample outputs (for side-by-side quality review)\n\n```json\n"
        + json.dumps(samples, indent=2) + "\n```\n"
    )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"])
    p.add_argument("--urls", nargs="+", required=True)
    p.add_argument("--effort", default="low")
    a = p.parse_args()
    asyncio.run(main(a.models, a.urls, a.effort))
