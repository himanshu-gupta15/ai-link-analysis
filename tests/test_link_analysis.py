"""Tests for URL validation / SSRF, extraction, LLM handling and the API.

No network or LLM calls are made: websites are served by ``httpx.MockTransport``,
DNS is stubbed, and the Anthropic client is replaced by a fake.
"""

import asyncio
from types import SimpleNamespace

import anthropic
import httpx
import httpx2
import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.errors import LLMError, LLMUnavailable
from app.main import app, get_analysis_service
from app.schemas import LLMAnalysis
from app.services.analysis_service import AnalysisService, TTLCache
from app.services.competitor_service import CompetitorService
from app.services.llm_service import LLMService, build_user_prompt
from app.services.website_extractor import WebsiteExtractor, build_llm_content, PageContent
from app.utils import validators
from app.utils.validators import BlockedURLError, InvalidURLError, ensure_public_url, normalize_url

PORTS = {80, 443, 8080, 8443}

HOME_HTML = """<!doctype html><html lang="en"><head>
<title>Acme Web Studio | Web Development for Small Businesses</title>
<meta name="description" content="Acme Web Studio builds fast, affordable websites and online stores for small businesses in Austin, Texas.">
<meta property="og:title" content="Acme Web Studio"><meta property="og:site_name" content="Acme Web Studio">
<meta name="viewport" content="width=device-width">
<link rel="canonical" href="https://acme.example/">
<script type="application/ld+json">{"@type":"Organization","name":"Acme Web Studio","areaServed":"Austin"}</script>
<script>var tracking = "ignore me";</script></head>
<body><nav><a href="/services">Services</a><a href="/about">About us</a><a href="/login">Login</a></nav>
<main><h1>Websites that grow small businesses</h1>
<p>We design and develop custom websites for small businesses, startups and local shops in Austin.</p>
<p>Our e-commerce stores help retailers sell online with secure checkout and inventory sync.</p>
<p>We design and develop custom websites for small businesses, startups and local shops in Austin.</p>
<img src="a.png"><img src="b.png" alt="team"></main>
<div class="cookie-banner"><p>We use cookies to improve your experience on this website.</p></div>
<footer><p>Copyright Acme Web Studio, all rights reserved worldwide.</p></footer></body></html>"""

SERVICES_HTML = """<html><head><title>Services</title></head><body><main><h2>Our services</h2>
<ul><li>Custom website development with modern frameworks</li><li>SEO optimization and local search marketing</li>
<li>Website maintenance and hosting support plans</li></ul></main></body></html>"""


def settings(**kw) -> Settings:
    base = dict(_env_file=None, anthropic_api_key="test-key", verify_competitor_websites=False)
    base.update(kw)
    return Settings(**base)


@pytest.fixture(autouse=True)
def fake_dns(monkeypatch):
    """Resolve *.example to a public IP, internal.example to a private one."""

    async def resolve(host, port):
        if host.startswith("internal."):
            return ["10.0.0.5"]
        if host.startswith("nxdomain."):
            raise validators.DNSResolutionError(f"Could not resolve hostname '{host}'")
        return ["93.184.215.14"]

    monkeypatch.setattr(validators, "resolve_host", resolve)


def site_transport(routes: dict[str, httpx.Response]):
    def handler(request: httpx.Request) -> httpx.Response:
        key = f"{request.url.host}{request.url.path}"
        if key in routes:
            r = routes[key]
            return httpx.Response(r.status_code, headers=r.headers, content=r.content)
        return httpx.Response(404, text="not found")

    return httpx.MockTransport(handler)


def html(body: str, status: int = 200) -> httpx.Response:
    return httpx.Response(status, headers={"content-type": "text/html; charset=utf-8"}, content=body.encode())


def run(coro):
    return asyncio.run(coro)


def sample_analysis(**overrides) -> LLMAnalysis:
    data = {
        "brand": {"name": "Acme Web Studio", "category": "Web Development", "summary": "Builds websites.", "location": "Austin, TX"},
        "competitor_analysis": {
            "brand_overview": "Small-business web studio",
            "main_category": "Web Development",
            "main_products_services": ["Websites", "E-commerce"],
            "brand_profile": {"main_service": "Web Development", "target_market": "Small Businesses",
                              "key_strength": "Affordable", "opportunity": "Improve visibility"},
            "competitors": [
                {"name": "Big Agency", "website": "https://bigagency.example", "main_service": "Web Development",
                 "target_market": "Enterprise", "key_strength": "Established brand", "strengths": ["Brand"],
                 "weaknesses_or_gaps": ["Expensive"], "opportunity_for_brand": "Price", "basis": "ai_suggestion"},
                {"name": "Big Agency", "website": None, "main_service": "x", "target_market": "x", "key_strength": "x",
                 "strengths": [], "weaknesses_or_gaps": [], "opportunity_for_brand": "x", "basis": "ai_suggestion"},
                {"name": "Acme Web Studio", "website": None, "main_service": "x", "target_market": "x", "key_strength": "x",
                 "strengths": [], "weaknesses_or_gaps": [], "opportunity_for_brand": "x", "basis": "ai_suggestion"},
            ],
            "competitive_advantages": ["Local"],
            "positioning_opportunities": ["Case studies"],
        },
        "target_search_queries": [
            {"query": f"query {i}", "platforms": ["Google", "LinkedIn"], "intent": "Service Search",
             "business_relevance": "high", "target_audience": "SMBs"} for i in range(5)
        ],
        "seo_improvements": {
            "primary_keyword": "web development austin", "secondary_keywords": ["web design"],
            "long_tail_keywords": ["affordable web design for small business austin"], "business_keywords": ["website"],
            "recommended_title": "Web Development Austin | Acme Web Studio", "title_structure": "Keyword | Brand",
            "meta_description": "x" * 150,
            "supporting_content": [{"type": "faq", "title": "FAQ", "description": "d", "target_keyword": "k"}],
            "technical_fixes": ["Add alt text"],
        },
        "top_services": [
            {"service_name": f"S{i}", "description": "d", "target_industry": "Retail", "target_customer": "SMB",
             "main_benefit": "b", "basis": "website_content"} for i in range(5)
        ],
        "target_audience": {
            "primary_audience": ["Small businesses"], "secondary_audience": ["Startups"], "industries": ["Retail"],
            "business_types": ["Local shops"], "customer_profiles": ["Owner"], "geographic_focus": "Austin",
            "customer_needs": ["Website"], "customer_challenges": ["Budget"], "customer_interests": ["Growth"],
            "recommended_segments": [{"audience": "A1", "business_type": "SMB", "requirement": "Website"}],
        },
        "data_quality": {"content_sufficient": True, "notes": ""},
    }
    data.update(overrides)
    return LLMAnalysis.model_validate(data)


class FakeMessages:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    async def parse(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def fake_client(*outcomes):
    messages = FakeMessages(outcomes)
    return SimpleNamespace(beta=SimpleNamespace(messages=messages)), messages


def llm_response(analysis=None, stop_reason="end_turn", model="claude-opus-5-5"):
    return SimpleNamespace(
        stop_reason=stop_reason,
        parsed_output=analysis if analysis is not None else sample_analysis(),
        model=model,
        _request_id="req_test",
        usage=SimpleNamespace(input_tokens=5000, output_tokens=3000, cache_read_input_tokens=0),
    )


def api_error(cls, status):
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls("err", response=httpx2.Response(status, request=req), body=None)


# ---------------------------------------------------------------- URL validation
@pytest.mark.parametrize(
    "url, expected",
    [
        ("example.com", "https://example.com/"),
        ("  https://Example.com/path?q=1#frag ", "https://example.com/path?q=1"),
        ("http://example.com:8080", "http://example.com:8080/"),
    ],
)
def test_normalize_url(url, expected):
    assert normalize_url(url) == expected


@pytest.mark.parametrize(
    "url",
    ["", "ftp://example.com", "javascript:alert(1)", "https://user:pass@example.com", "https://localhost_x",
     "https://exa mple.com", "https://nodot", "http://example.com:99999"],
)
def test_invalid_urls(url):
    with pytest.raises(InvalidURLError):
        normalize_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1", "http://10.1.2.3", "http://192.168.1.1", "http://172.16.0.1",
        "http://169.254.169.254/latest/meta-data", "http://[::1]", "http://[::ffff:127.0.0.1]",
        "http://100.64.0.1", "http://0.0.0.0", "http://localhost", "http://app.internal",
        "https://internal.example", "http://example.com:22",
    ],
)
def test_ssrf_blocked(url):
    with pytest.raises((BlockedURLError, InvalidURLError)):
        run(ensure_public_url(url, PORTS))


def test_public_url_allowed():
    assert run(ensure_public_url("https://acme.example", PORTS)) == "https://acme.example/"


# -------------------------------------------------------------------- extraction
def make_extractor(routes, **kw):
    client = httpx.AsyncClient(transport=site_transport(routes))
    return WebsiteExtractor(settings(**kw), client)


def test_extract_homepage_and_service_page():
    ex = make_extractor({"acme.example/": html(HOME_HTML), "acme.example/services": html(SERVICES_HTML)})
    site = run(ex.extract("https://acme.example"))

    assert site.site_name == "Acme Web Studio"
    assert "https://acme.example/services" in site.page_urls
    assert all("login" not in u for u in site.page_urls)
    assert "Custom website development" in site.content
    assert "ignore me" not in site.content            # scripts removed
    assert "cookies" not in site.content               # cookie banner removed
    assert site.content.count("We design and develop custom websites") == 1  # de-duplicated
    audit = site.seo_audit
    assert audit.h1_count == 1 and audit.images_missing_alt == 1 and audit.canonical == "https://acme.example/"
    assert any("alt text" in i for i in audit.issues)


def test_redirect_to_private_address_blocked():
    redirect = httpx.Response(302, headers={"location": "http://internal.example/admin"})
    ex = make_extractor({"acme.example/": redirect})
    with pytest.raises(Exception) as exc:
        run(ex.extract("https://acme.example"))
    assert getattr(exc.value, "code", "") == "URL_NOT_ALLOWED"


@pytest.mark.parametrize(
    "response, code",
    [
        (html("denied", 403), "WEBSITE_ACCESS_RESTRICTED"),
        (html("missing", 404), "WEBSITE_NOT_FOUND"),
        (html("boom", 500), "WEBSITE_HTTP_ERROR"),
        (httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF"), "UNSUPPORTED_CONTENT"),
        (html("<html><body><div id='root'></div></body></html>"), "INSUFFICIENT_CONTENT"),
    ],
)
def test_website_errors(response, code):
    ex = make_extractor({"acme.example/": response})
    with pytest.raises(Exception) as exc:
        run(ex.extract("https://acme.example"))
    assert exc.value.code == code


def test_timeout_is_reported():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    ex = WebsiteExtractor(settings(), httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    with pytest.raises(Exception) as exc:
        run(ex.extract("https://acme.example"))
    assert exc.value.code == "WEBSITE_UNREACHABLE"


def test_content_budget_and_truncation_flag():
    pages = [PageContent(url="u1", text_blocks=[f"block {i} " + "x" * 100 for i in range(50)]),
             PageContent(url="u2", text_blocks=["page two text " + "y" * 50])]
    content, truncated = build_llm_content(pages, max_chars=2000)
    assert truncated and len(content) <= 2200 and "page two text" in content


def test_prompt_wraps_untrusted_content():
    ex = make_extractor({"acme.example/": html(HOME_HTML)}, max_extra_pages=0)
    site = run(ex.extract("https://acme.example"))
    prompt = build_user_prompt(site.final_url, site)
    assert "<website_content>" in prompt and "<seo_audit>" in prompt


# --------------------------------------------------------------------------- LLM
def test_llm_success_and_request_shape():
    client, messages = fake_client(llm_response())
    result = run(LLMService(settings(), client).analyze("https://acme.example/", _site()))
    call = messages.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["output_format"] is LLMAnalysis
    assert call["fallbacks"] == "default"
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert result.estimated_cost_usd == round((5000 * 4 + 3000 * 20) / 1e6, 5)


def test_llm_falls_back_on_overload():
    client, messages = fake_client(api_error(anthropic.InternalServerError, 529), llm_response(model="claude-sonnet-5-5"))
    result = run(LLMService(settings(), client).analyze("https://acme.example/", _site()))
    assert [c["model"] for c in messages.calls] == ["claude-opus-5-5", "claude-sonnet-5-5"]
    assert result.model == "claude-sonnet-5-5"


def test_llm_all_models_down():
    client, _ = fake_client(api_error(anthropic.InternalServerError, 500), api_error(anthropic.InternalServerError, 503))
    with pytest.raises(LLMUnavailable):
        run(LLMService(settings(), client).analyze("https://acme.example/", _site()))


def test_llm_auth_error_not_retried():
    client, messages = fake_client(api_error(anthropic.AuthenticationError, 401))
    with pytest.raises(Exception) as exc:
        run(LLMService(settings(), client).analyze("https://acme.example/", _site()))
    assert exc.value.code == "LLM_NOT_CONFIGURED" and len(messages.calls) == 1


@pytest.mark.parametrize("stop_reason, code", [("refusal", "LLM_REFUSED"), ("max_tokens", "LLM_ERROR")])
def test_llm_bad_stop_reasons(stop_reason, code):
    client, _ = fake_client(llm_response(stop_reason=stop_reason))
    with pytest.raises(LLMError) as exc:
        run(LLMService(settings(), client).analyze("https://acme.example/", _site()))
    assert exc.value.code == code


def _site():
    ex = make_extractor({"acme.example/": html(HOME_HTML)}, max_extra_pages=0)
    return run(ex.extract("https://acme.example"))


# ------------------------------------------------------------------- competitors
def test_competitors_deduplicated_and_table_built():
    a = sample_analysis()
    svc = CompetitorService(settings(), client=None)
    out = run(svc.enrich(a.competitor_analysis, a.brand, "https://acme.example/"))
    assert [c.name for c in out.competitors] == ["Big Agency"]
    table = {row.comparison: row for row in out.comparison_table}
    assert table["Brand Name"].your_brand == "Acme Web Studio"
    assert table["Target Market"].competitors == {"Big Agency": "Enterprise"}
    assert "ai_suggestion" in out.source_note


# --------------------------------------------------------------------------- API
@pytest.fixture
def api():
    s = settings()
    http = httpx.AsyncClient(transport=site_transport({
        "acme.example/": html(HOME_HTML), "acme.example/services": html(SERVICES_HTML),
    }))
    client, messages = fake_client(llm_response(), llm_response())
    llm = LLMService(s, client)
    cache = TTLCache(3600, 10)
    app.dependency_overrides[get_settings] = lambda: s
    app.dependency_overrides[get_analysis_service] = lambda: AnalysisService(s, http, llm, cache)
    yield TestClient(app), messages
    app.dependency_overrides.clear()


def test_api_success_and_cache(api):
    client, messages = api
    resp = client.post("/link-analysis", json={"url": "acme.example"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is True and body["website"] == "https://acme.example/"
    assert body["brand"]["name"] == "Acme Web Studio"
    analysis = body["analysis"]
    assert set(analysis) == {"competitor_analysis", "target_search_queries", "seo_improvements", "top_services", "target_audience"}
    assert len(analysis["target_search_queries"]) == 5 and len(analysis["top_services"]) == 5
    assert body["seo_audit"]["h1_count"] == 1
    assert body["metadata"]["cached"] is False and body["metadata"]["estimated_cost_usd"] > 0

    again = client.post("/link-analysis", json={"url": "https://acme.example/"})
    assert again.json()["metadata"]["cached"] is True
    assert len(messages.calls) == 1  # second request served from cache


@pytest.mark.parametrize(
    "payload, status, code",
    [
        ({}, 422, "VALIDATION_ERROR"),
        ({"url": "ftp://acme.example"}, 422, "INVALID_URL"),
        ({"url": "http://127.0.0.1:8080"}, 400, "URL_NOT_ALLOWED"),
        ({"url": "https://nxdomain.example"}, 502, "DNS_RESOLUTION_FAILED"),
        ({"url": "https://acme.example/missing"}, 502, "WEBSITE_NOT_FOUND"),
    ],
)
def test_api_errors(api, payload, status, code):
    client, _ = api
    resp = client.post("/link-analysis", json=payload)
    assert resp.status_code == status
    assert resp.json() == {"success": False, "error": {"code": code, "message": resp.json()["error"]["message"]}}
