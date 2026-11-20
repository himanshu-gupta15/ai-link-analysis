# AI-Powered Website Link Analysis API

A FastAPI service that takes a website URL, fetches and cleans the site's content (homepage plus key internal pages), and uses an LLM (Anthropic Claude) to return a structured JSON analysis in five sections:

1. **Competitor vs your brand**: brand profile, 3–5 competitors, strengths, gaps, comparison table
2. **Top 5 target search queries**: Google, Facebook, Instagram, LinkedIn and X, each with intent and audience
3. **SEO improvements**: keywords, SEO title, meta description, supporting content, technical fixes
4. **Top 5 services**: each labelled as found on the website or inferred
5. **Target audience**: primary and secondary audiences, industries, profiles, needs, segments

```
POST http://localhost:4000/link-analysis   {"url": "https://example.com"}
```

---

## Contents
- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [Setup and run](#setup--run)
- [API: request, response, errors](#api)
- [Security: SSRF protection](#security--ssrf-protection)
- [LLM platform and model research](#llm-platform--model-research)
- [Cost optimization](#cost-optimization)
- [Final LLM recommendation](#final-llm-recommendation)
- [Deployment](#deployment)
- [Tests](#tests)

---

## How it works

```
URL ─► validate + SSRF check ─► fetch homepage (manual redirects, each hop re-checked, 3 MB cap)
    ─► parse HTML (BeautifulSoup + lxml): title, meta, Open Graph, JSON-LD, nav, headings, main text
    ─► choose up to 3 internal pages (services/about/products/pricing…) ─► fetch them concurrently
    ─► clean: drop scripts/styles/nav/footer/cookie banners, de-duplicate blocks, size budget
    ─► measured on-page SEO audit (no LLM)
    ─► ONE Claude call with a structured-output JSON schema (all 5 sections)
    ─► post-process: remove duplicate or self-named competitors, check competitor websites, build comparison table
    ─► cache the result (1 h) ─► JSON response
```

Design choices:

| Area | Decision |
|---|---|
| **Structured output** | The Pydantic model `LLMAnalysis` is passed as Claude's structured-output schema (`output_format=`). The API constrains generation to that schema and the SDK validates the result, so no regex or "fix the JSON" retries are needed. |
| **Grounding** | The prompt contains only the extracted website content, measured site facts and the SEO audit. The model must use the site's own wording, label each service `website_content` or `inferred`, and label each competitor `mentioned_on_website` or `ai_suggestion`. |
| **Competitors** | These are LLM suggestions, and the response says so. The service checks that each suggested competitor website exists (`website_verified`). It does **not** claim the strengths or weaknesses are verified. |
| **Search queries** | Five queries with platforms, intent, relevance and audience. **No search volumes**, because no volume data source is connected (see `metadata.search_volume_note`). |
| **SEO** | Real HTML is measured: title and meta lengths, H1 count, alt text, canonical, viewport, noindex, JSON-LD, Open Graph, word count. Those facts are returned in `seo_audit`, and the LLM's `technical_fixes` must address them. |
| **Prompt injection** | Scraped text goes inside `<website_content>` tags, and the system prompt tells the model to treat it as untrusted data. |
| **Resilience** | The SDK retries 429 and 5xx errors. If the primary model is overloaded or down, the service switches to a fallback model (Sonnet 5.5). Refusals use Anthropic's server-side `fallbacks="default"`. |

## Project structure

```
ai-link-analysis/
├── app/
│   ├── main.py                    # FastAPI app, routes, error handlers
│   ├── config.py                  # Settings (env vars), model price table
│   ├── schemas.py                 # Request, LLM output schema, API response
│   ├── errors.py                  # Typed errors → HTTP status + error code
│   ├── services/
│   │   ├── website_extractor.py   # Safe fetching, parsing, cleaning, SEO audit
│   │   ├── llm_service.py         # Claude call, prompts, fallbacks, cost
│   │   ├── competitor_service.py  # Dedupe, website checks, comparison table
│   │   └── analysis_service.py    # Orchestration + TTL cache
│   └── utils/validators.py        # URL normalization + SSRF protection
├── tests/test_link_analysis.py    # 48 tests (no network / no LLM spend)
├── scripts/benchmark.py           # Measures latency/tokens/cost per model
├── Dockerfile, render.yaml        # Deployment
├── .env.example, requirements.txt
└── README.md
```

## Setup and run

```bash
python3 -m venv .venv && source .venv/bin/activate     # Python 3.10+
pip install -r requirements.txt
cp .env.example .env                                   # add ANTHROPIC_API_KEY
uvicorn app.main:app --host 0.0.0.0 --port 4000
```

- Swagger UI: http://localhost:4000/docs
- Health check: `GET /health` (shows whether the LLM is configured)

## API

### Request

```bash
curl -X POST http://localhost:4000/link-analysis \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.python.org"}'
```

You can leave out the scheme: `"python.org"` becomes `https://python.org/`.

### Response `200`: shape

The analysis values below are shortened. Every field in the schema is always present.

```json
{
  "success": true,
  "website": "https://www.python.org/",
  "brand": {
    "name": "Python Software Foundation (Python.org)",
    "category": "Programming Language / Open-Source Software",
    "summary": "Official home of the Python programming language …",
    "location": null
  },
  "analysis": {
    "competitor_analysis": {
      "brand_overview": "…",
      "main_category": "…",
      "main_products_services": ["…"],
      "brand_profile": {"main_service": "…", "target_market": "…", "key_strength": "…", "opportunity": "…"},
      "competitors": [
        {
          "name": "…", "website": "https://…", "main_service": "…", "target_market": "…",
          "key_strength": "…", "strengths": ["…"], "weaknesses_or_gaps": ["…"],
          "opportunity_for_brand": "…", "basis": "ai_suggestion", "website_verified": true
        }
      ],
      "competitive_advantages": ["…"],
      "positioning_opportunities": ["…"],
      "comparison_table": [
        {"comparison": "Brand Name", "your_brand": "…", "competitors": {"Competitor A": "…"}},
        {"comparison": "Main Service", "your_brand": "…", "competitors": {"Competitor A": "…"}},
        {"comparison": "Target Market", "your_brand": "…", "competitors": {"Competitor A": "…"}},
        {"comparison": "Key Strength", "your_brand": "…", "competitors": {"Competitor A": "…"}},
        {"comparison": "Opportunity", "your_brand": "…", "competitors": {"Competitor A": "…"}}
      ],
      "source_note": "Competitors are AI-generated suggestions …"
    },
    "target_search_queries": [
      {"query": "…", "platforms": ["Google", "LinkedIn"], "intent": "Service Search",
       "business_relevance": "…", "target_audience": "…"}
    ],
    "seo_improvements": {
      "primary_keyword": "…", "secondary_keywords": ["…"], "long_tail_keywords": ["…"],
      "business_keywords": ["…"], "recommended_title": "…", "title_structure": "Primary Keyword | Benefit | Brand",
      "meta_description": "…",
      "supporting_content": [{"type": "faq", "title": "…", "description": "…", "target_keyword": "…"}],
      "technical_fixes": ["…"]
    },
    "top_services": [
      {"service_name": "…", "description": "…", "target_industry": "…", "target_customer": "…",
       "main_benefit": "…", "basis": "website_content"}
    ],
    "target_audience": {
      "primary_audience": ["…"], "secondary_audience": ["…"], "industries": ["…"], "business_types": ["…"],
      "customer_profiles": ["…"], "geographic_focus": null, "customer_needs": ["…"],
      "customer_challenges": ["…"], "customer_interests": ["…"],
      "recommended_segments": [{"audience": "…", "business_type": "…", "requirement": "…"}]
    }
  },
  "seo_audit": {
    "title": "Welcome to Python.org", "title_length": 21,
    "meta_description": "The official home of the Python Programming Language", "meta_description_length": 52,
    "h1": ["Functions Defined", "All the Flow You’d Expect", "…"], "h1_count": 5, "word_count": 1126,
    "images_total": 1, "images_missing_alt": 0,
    "canonical": null, "has_viewport_meta": true, "noindex": false, "lang": "en",
    "structured_data_types": ["WebSite"], "open_graph": true,
    "issues": [
      "Title is only 21 characters (too short to be descriptive)",
      "Meta description is 52 characters (recommended 120–160)",
      "5 <h1> headings (recommended exactly 1)",
      "No canonical URL"
    ]
  },
  "metadata": {
    "model": "claude-opus-5-5", "input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0,
    "estimated_cost_usd": 0.0, "llm_latency_ms": 0, "total_latency_ms": 0,
    "pages_analyzed": ["https://www.python.org/", "https://www.python.org/about/", "https://www.python.org/about/apps/"],
    "content_chars_sent": 6553, "content_truncated": false, "cached": false,
    "data_quality": {"content_sufficient": true, "notes": "…"},
    "search_volume_note": "Search volumes are not included: no search-volume data source is connected …"
  }
}
```

> The `seo_audit` values and `pages_analyzed` above are **real output** from the extractor on python.org (2026-11-20). The LLM fields are shown as `…` because no live LLM run was made while writing this README. Replace this block with a real response once you have one (see [Screenshots and logs](#screenshots--logs)).

### Errors

Every error has the same shape: `{"success": false, "error": {"code": "...", "message": "..."}}`

| HTTP | `code` | When |
|---|---|---|
| 422 | `VALIDATION_ERROR` | Body missing or wrong type |
| 422 | `INVALID_URL` | Bad scheme (ftp, javascript), credentials in URL, invalid host or port |
| 400 | `URL_NOT_ALLOWED` | Private, loopback, link-local or metadata address, internal host, or a port not on the allow list |
| 502 | `DNS_RESOLUTION_FAILED` | Domain does not resolve |
| 502 | `WEBSITE_UNREACHABLE` | Timeout, connection refused, too many redirects |
| 502 | `WEBSITE_ACCESS_RESTRICTED` | 401, 403, 429 or 451, or a Cloudflare challenge (bot protection or login wall) |
| 502 | `WEBSITE_NOT_FOUND` / `WEBSITE_HTTP_ERROR` | 404 / other 4xx or 5xx from the site |
| 422 | `UNSUPPORTED_CONTENT` | URL is not HTML (PDF, image, …) |
| 422 | `INSUFFICIENT_CONTENT` | Page has almost no readable text (for example, a JavaScript-only app) |
| 503 | `LLM_NOT_CONFIGURED` | API key missing or invalid, or unknown model |
| 503 | `LLM_UNAVAILABLE` | Primary and fallback models both failed (overload, 5xx, timeout) |
| 502 | `LLM_REFUSED` / `LLM_ERROR` | Model declined, output was cut off, or output did not match the schema |

Real log from the local server (no API key configured):

```
INFO httpx: HTTP Request: GET https://www.python.org/ "HTTP/1.1 200 OK"
INFO httpx: HTTP Request: GET https://www.python.org/about/apps "HTTP/1.1 301 Moved Permanently"
INFO httpx: HTTP Request: GET https://www.python.org/about/ "HTTP/1.1 200 OK"
INFO httpx: HTTP Request: GET https://www.python.org/about/apps/ "HTTP/1.1 200 OK"
INFO analysis: Extracted https://www.python.org/: 3 pages, 6553 chars (truncated=False)
WARNING app: URL_NOT_ALLOWED: Access to private or internal network addresses is not allowed (169.254.169.254 -> 169.254.169.254)
WARNING app: INVALID_URL: 'not a url' is not a valid public domain name
WARNING app: DNS_RESOLUTION_FAILED: Could not resolve hostname 'doesnotexist-xyz-12345.com'
```

### Screenshots and logs
Add screenshots of `/docs` and a real `200` response to [`docs/`](docs/), and the output of `scripts/benchmark.py` (it writes `docs/benchmark.md`).

## Security: SSRF protection

`app/utils/validators.py` checks **every** URL the service fetches: the submitted URL, each redirect hop, extra internal pages and competitor websites.

- Only `http` and `https`. No `user:pass@`. Ports limited to 80, 443, 8080 and 8443.
- Hostnames such as `localhost`, `*.local`, `*.internal` and `metadata.google.internal` are rejected.
- **Every** resolved IP address must be globally routable. This blocks private ranges (10/8, 172.16/12, 192.168/16), loopback, link-local (`169.254.169.254` cloud metadata), CGNAT (100.64/10), multicast, reserved and unspecified addresses, and IPv4-mapped IPv6 forms such as `::ffff:127.0.0.1`.
- Redirects are followed **manually**, so an external site cannot redirect the service to an internal address.
- Response size cap (3 MB), timeouts, HTML-only content types, and `trust_env=False` (no proxy picked up from environment variables).
- Secrets come only from environment variables. `.env` is git-ignored.

*Known limitation:* a DNS-rebinding attack between the check and the connection is still possible in theory. To close it, pin the resolved IP in a custom transport or run the service behind an egress proxy that blocks private ranges.

---

## LLM platform and model research

All prices are standard paid-tier API prices in USD per 1M tokens, checked on **2026-11-20** from each provider's official pricing page:
[Anthropic](https://platform.claude.com/docs/en/about-claude/pricing) ·
[OpenAI](https://developers.openai.com/api/docs/pricing) ·
[Google Gemini](https://aistudio.google.com/docs/pricing)

### Workload profile (measured from the extractor)

| Site | Pages | Cleaned content | Prompt size |
|---|---|---|---|
| example.com | 1 | 209 chars | ~0.3K tokens |
| python.org | 3 | 6.5K chars | ~2.3K tokens |
| stripe.com | 4 | 19.1K chars (capped) | ~6.2K tokens |

Adding the system prompt (~0.8K tokens) and schema overhead gives a typical request of **about 3–8K input tokens**. The 24K-character content cap keeps it under ~8K. The JSON output for all five sections is about **3–4K output tokens**, plus a small amount of thinking at `effort=low`.

**The estimates below assume about 6K input and 4K output tokens per analysis.** *These are estimates, not measurements.* Run `scripts/benchmark.py` to get measured numbers.

### A. Comparison

| Provider | Model | Input / Output ($/1M) | Est. cost per analysis | Context window | Structured JSON | Expected speed* | Fit for this app |
|---|---|---|---|---|---|---|---|
| Anthropic | **Claude Opus 5.5** | $4 / $20 (cache read $0.20) | **~$0.10** | 1M | Native structured outputs (JSON-schema-constrained), Pydantic `parse()` in the SDK | Moderate; `effort=low` reduces thinking | **Highest quality tier at a mid price. Recommended primary.** |
| Anthropic | Claude Sonnet 5.5 | $2 / $10 | ~$0.05 | 1M | Same | Faster than Opus | **Fallback.** Strong quality at half the cost. |
| Anthropic | Claude Haiku 5.5 | $0.10 / $0.50 (≤100K prompt) | ~$0.003 | 1M | Same | Fastest Claude | Budget or high-volume mode. Weaker competitor knowledge expected. |
| OpenAI | GPT-6.2 Sol | $2 / $10 | ~$0.05 | >272K (long-context tier exists) | Structured Outputs (`json_schema`, strict) | Not measured | Comparable to Sonnet 5.5 on price |
| OpenAI | GPT-5.4 mini | $0.75 / $4.50 | ~$0.023 | >272K | Same | Not measured | Mid-budget option |
| OpenAI | GPT-6 Luna | $0.10 / $0.50 | ~$0.003 | not verified | Same | Not measured | Budget option |
| Google | Gemini 3.1 Pro (Preview) | $2 / $12 (≤200K) | ~$0.06 | >200K (long-context tier exists) | `responseSchema` JSON mode | Not measured | Preview status is a risk for production |
| Google | Gemini 3.8 Flash | $0.75 / $3.75 until 31 Dec 2026, then $1.50 / $7.50 | ~$0.02 (then ~$0.04) | not verified | `responseSchema` | Not measured | Good value. Price doubles in January. |
| Google | Gemini 3.5 Flash-Lite | $0.30 / $2.50 | ~$0.012 | not verified | `responseSchema` | Not measured | Budget option |

\* **No response times in this table were measured.** No LLM API key was available in the development environment. Speed and accuracy figures for this workload come only from `scripts/benchmark.py`. This README does not claim any model is "most accurate" without that test. Context windows marked "not verified" were not confirmed against provider docs on the research date.

### B. Model selection criteria (and how each is addressed)

1. **Understanding website content.** Content is pre-cleaned and structured (facts, audit, headings, page text), so even mid-tier models get a clear signal. Opus-tier is preferred for nuanced positioning.
2. **Response speed.** One call instead of five, `effort=low`, a capped prompt size, and caching for repeat URLs.
3. **Low cost.** About $0.10 per fresh analysis on Opus 5.5, about $0.05 on Sonnet 5.5, and $0 for a cache hit.
4. **Reliable JSON.** Schema-constrained structured outputs remove JSON parse failures. Pydantic validates the result as well.
5. **Competitor and SEO insight.** This depends on the model's world knowledge of companies and markets, where larger models are expected to be better. This is why a large model is the primary choice rather than a small one.
6. **Large content.** 1M context on every Claude model. The app caps content at about 6K tokens anyway, for cost reasons.

## Cost optimization

What the code does:

| Technique | Where | Effect |
|---|---|---|
| Strip scripts, styles, nav, footer, cookie banners, SVGs and forms | `website_extractor.parse_page` | Removes most of the raw HTML before anything reaches the model |
| De-duplicate repeated text blocks within and across pages | `build_llm_content` | Avoids paying for repeated hero text and CTAs |
| Content budget (24K chars), split between homepage and extra pages | `MAX_CONTENT_CHARS` | Puts a hard ceiling on input tokens. `content_truncated` reports when the cap was hit. |
| Fetch only the 3 most informative internal pages | `select_internal_links` | Scores links by service/about/product keywords and skips login, legal and media links |
| One LLM call for all five sections | `llm_service` | The content is sent once, not five times |
| Prompt caching on the static system prompt | `cache_control` | Cached reads cost 0.05× the input price |
| `effort=low` | `LLM_EFFORT` | Fewer thinking tokens. Output tokens are the main cost driver. |
| Result cache (1 h TTL, LRU) | `TTLCache` | Repeat analysis of the same URL costs $0 and returns in milliseconds |
| Fail fast before calling the LLM | validators and extractor | Invalid, blocked, unreachable and empty sites never reach the model |
| Model routing by config | `LLM_MODEL`, `LLM_FALLBACK_MODEL` | Switch to Sonnet 5.5 (half the cost) or Haiku 5.5 (~1/40 the cost) without code changes |
| Cost tracking per request | `metadata.estimated_cost_usd` | Makes spend visible per analysis |

Further options for production scale:

- Use the Message Batches API (50% cheaper) for bulk or offline analysis.
- Use a shared Redis cache across instances.
- Split the work: a small model (Haiku 5.5) for extracting services and audience, and a large model only for competitors and positioning. Adopt this only if the benchmark shows quality holds.

## Final LLM recommendation

| Item | Recommendation |
|---|---|
| **Provider** | Anthropic |
| **Model** | **Claude Opus 5.5** (`claude-opus-5-5`) with `effort=low` |
| **Estimated input tokens** | ~3–8K per analysis (typical ~6K) |
| **Estimated output tokens** | ~3–4K (JSON) plus a small amount of thinking at low effort |
| **Estimated cost per analysis** | **~$0.08–$0.12** fresh, **$0** when cached |
| **Expected response time** | *Estimate, not measured:* tens of seconds for the LLM step (output-heavy), plus 1–3 s for fetching. Measure with `scripts/benchmark.py`. |
| **Fallback model** | **Claude Sonnet 5.5** (`claude-sonnet-5-5`), used automatically on overload, 5xx or timeout. About half the cost. |
| **Budget mode** | `LLM_MODEL=claude-haiku-5-5` (~$0.003 per analysis) for high-volume or low-stakes use |

**Why this model.** The most valuable and hardest parts of the output are competitor identification, positioning and SEO strategy, which depend on broad, current knowledge of real companies and markets. That favours a top-tier model. Opus 5.5 is Anthropic's current Opus at a lower price than earlier Opus models ($4 / $20). With a capped prompt (~6K tokens), one call and `effort=low`, a fresh analysis costs about $0.10, which is small next to the value of a competitor and SEO report. Native structured outputs mean no JSON failures. A 1M context leaves plenty of room. One provider covers primary, fallback and budget tiers, so a single SDK and schema works across all three.

**Balancing accuracy, speed and cost:**
1. Keep the input small and clean (already done), and cache aggressively.
2. Run `scripts/benchmark.py` on 10–20 representative sites with Opus 5.5, Sonnet 5.5 and Haiku 5.5, and compare outputs side by side (the script saves samples).
3. If Sonnet 5.5's competitor and SEO quality is close enough on your sites, make it the primary model (about 50% saving). Use Haiku 5.5 only where quality is acceptable.
4. Raise `LLM_EFFORT` to `medium` only if the benchmark shows a quality gain worth the extra latency.

---

## Deployment

The repo includes a `Dockerfile` (it reads `$PORT`, default 4000) and a `render.yaml`.

**Render (free tier):**
1. Push the repo to GitHub, then in Render choose **New → Blueprint** and select the repo.
2. Set `ANTHROPIC_API_KEY` in the dashboard.
3. Live endpoints:
   - `https://<your-app>.onrender.com/docs`
   - `https://<your-app>.onrender.com/link-analysis`

**Any Docker host (Railway, Fly.io, a VPS):**
```bash
docker build -t ai-link-analysis .
docker run -p 4000:4000 -e ANTHROPIC_API_KEY=sk-ant-... ai-link-analysis
```

## Tests

```bash
pytest -q        # 48 tests, no network and no LLM spend
```

The tests cover URL normalization and SSRF blocking (including private IPs, metadata IPs, IPv4-mapped IPv6, and redirects to internal hosts), extraction and cleaning (script, cookie-banner and duplicate removal, SEO audit, content budget), website error mapping (403, 404, 500, timeouts, non-HTML, empty JavaScript apps), the LLM request shape (schema, prompt caching, refusal fallback), fallback to the second model on overload, auth errors, refusal and max_tokens stop reasons, competitor de-duplication and the comparison table, the API happy path, the cache, and the error contract.
