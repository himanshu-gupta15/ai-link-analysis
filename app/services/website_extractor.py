"""Fetch a website safely and turn its HTML into compact, LLM-ready content.

Pipeline:
1. SSRF-checked fetch of the homepage (manual redirects, each hop re-validated,
   size cap, HTML-only).
2. Parse with BeautifulSoup: metadata, headings, JSON-LD, navigation, main text.
3. Pick up to N high-value internal pages (about / services / products ...) and
   fetch them concurrently with the same protections.
4. Clean the text: drop scripts/styles/nav/footer boilerplate, collapse
   whitespace, de-duplicate repeated blocks, cap total size.
5. Compute a deterministic on-page SEO audit (no LLM involved).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlsplit

import httpx
from bs4 import BeautifulSoup, Tag

from app.config import Settings
from app.errors import (
    BlockedURL,
    InsufficientContent,
    InvalidURL,
    UnsupportedContent,
    WebsiteAccessDenied,
    WebsiteUnreachable,
)
from app.schemas import SEOAudit
from app.utils import validators
from app.utils.validators import BlockedURLError, DNSResolutionError, InvalidURLError

logger = logging.getLogger("extractor")

NOISE_TAGS = ["script", "style", "noscript", "svg", "iframe", "template", "canvas", "form", "button", "select"]
BOILERPLATE_SELECTOR = (
    "[class*='cookie'], [id*='cookie'], [class*='consent'], [id*='consent'], [aria-hidden='true'], [hidden]"
)
PRIORITY_KEYWORDS = (
    "service", "solution", "product", "about", "what-we-do", "offering", "pricing",
    "industr", "feature", "platform", "company", "who-we-are", "capabilit",
)
SKIP_LINK_PATTERN = re.compile(
    r"(login|signin|sign-in|signup|register|cart|checkout|account|privacy|terms|cookie|legal|"
    r"careers|jobs|\.pdf$|\.jpg$|\.png$|\.zip$|mailto:|tel:|javascript:|#)",
    re.IGNORECASE,
)
MIN_BLOCK_CHARS = 25


@dataclass
class PageContent:
    url: str
    title: str | None = None
    meta_description: str | None = None
    headings: list[str] = field(default_factory=list)
    text_blocks: list[str] = field(default_factory=list)


@dataclass
class ExtractedWebsite:
    requested_url: str
    final_url: str
    pages: list[PageContent]
    site_name: str | None
    meta_keywords: str | None
    open_graph: dict[str, str]
    json_ld: list[dict]
    nav_items: list[str]
    contact_hints: list[str]
    social_links: list[str]
    seo_audit: SEOAudit
    content: str = ""
    content_truncated: bool = False

    @property
    def page_urls(self) -> list[str]:
        return [p.url for p in self.pages]


class WebsiteExtractor:
    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self.settings = settings
        self.client = client

    # ------------------------------------------------------------------ public
    async def extract(self, url: str) -> ExtractedWebsite:
        final_url, html = await self.fetch_html(url)
        soup = BeautifulSoup(html, "lxml")

        seo_audit = build_seo_audit(soup, final_url)
        json_ld = extract_json_ld(soup)
        og = {
            m.get("property", "")[3:]: (m.get("content") or "").strip()
            for m in soup.find_all("meta", attrs={"property": re.compile(r"^og:")})
            if m.get("content")
        }
        site_name = og.get("site_name") or _org_name(json_ld)
        meta_keywords = _meta(soup, "keywords")
        nav_items = extract_nav_items(soup)
        social_links = extract_social_links(soup)
        candidate_links = select_internal_links(soup, final_url, self.settings.max_extra_pages)

        home = parse_page(soup, final_url)
        extra_pages = await self._fetch_extra_pages(candidate_links)

        website = ExtractedWebsite(
            requested_url=url,
            final_url=final_url,
            pages=[home, *extra_pages],
            site_name=site_name,
            meta_keywords=meta_keywords,
            open_graph=og,
            json_ld=json_ld,
            nav_items=nav_items,
            contact_hints=extract_contact_hints(html),
            social_links=social_links,
            seo_audit=seo_audit,
        )
        website.content, website.content_truncated = build_llm_content(
            website.pages, self.settings.max_content_chars
        )

        total_words = len(website.content.split())
        if total_words < 40 and not (home.title or home.meta_description):
            raise InsufficientContent(
                "The website returned too little readable content to analyze "
                "(it may be rendered entirely with JavaScript or be a placeholder page)."
            )
        return website

    async def fetch_html(self, url: str) -> tuple[str, str]:
        """Fetch ``url`` following redirects manually; every hop is SSRF-checked."""
        current = await self._validate(url)
        for _ in range(self.settings.max_redirects + 1):
            try:
                async with self.client.stream(
                    "GET",
                    current,
                    headers={
                        "User-Agent": self.settings.user_agent,
                        "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.5",
                        "Accept-Language": "en-US,en;q=0.9",
                    },
                    follow_redirects=False,
                    timeout=self.settings.fetch_timeout_seconds,
                ) as resp:
                    if resp.is_redirect:
                        location = resp.headers.get("location")
                        if not location:
                            raise WebsiteUnreachable("Website returned a redirect without a Location header")
                        current = await self._validate(urljoin(current, location))
                        continue
                    self._check_status(resp)
                    content_type = resp.headers.get("content-type", "").lower()
                    if content_type and not any(t in content_type for t in ("html", "xml", "text/plain")):
                        raise UnsupportedContent(
                            f"URL does not point to an HTML page (content-type: {content_type.split(';')[0]})"
                        )
                    body = await self._read_limited(resp)
                    return str(resp.url), _decode(body, resp.charset_encoding)
            except httpx.TimeoutException:
                raise WebsiteUnreachable(
                    f"Website did not respond within {self.settings.fetch_timeout_seconds:.0f} seconds"
                ) from None
            except httpx.TooManyRedirects:
                raise WebsiteUnreachable("Too many redirects") from None
            except httpx.ConnectError as exc:
                raise WebsiteUnreachable(f"Could not connect to the website: {_short(exc)}") from None
            except httpx.HTTPError as exc:
                raise WebsiteUnreachable(f"Network error while fetching the website: {_short(exc)}") from None
        raise WebsiteUnreachable(f"Too many redirects (more than {self.settings.max_redirects})")

    # ----------------------------------------------------------------- helpers
    async def _validate(self, url: str) -> str:
        try:
            return await validators.ensure_public_url(url, self.settings.allowed_port_set)
        except InvalidURLError as exc:
            raise InvalidURL(str(exc)) from None
        except BlockedURLError as exc:
            raise BlockedURL(str(exc)) from None
        except DNSResolutionError as exc:
            raise WebsiteUnreachable(str(exc), code="DNS_RESOLUTION_FAILED") from None

    @staticmethod
    def _check_status(resp: httpx.Response) -> None:
        code = resp.status_code
        if code < 400:
            return
        if code in (401, 403, 429, 451) or (code == 503 and "cloudflare" in resp.headers.get("server", "").lower()):
            raise WebsiteAccessDenied(
                f"Website restricted automated access (HTTP {code}). "
                "It may require login or block bots/crawlers."
            )
        if code == 404:
            raise WebsiteUnreachable("Page not found (HTTP 404)", code="WEBSITE_NOT_FOUND")
        raise WebsiteUnreachable(f"Website returned an error (HTTP {code})", code="WEBSITE_HTTP_ERROR")

    async def _read_limited(self, resp: httpx.Response) -> bytes:
        chunks, size = [], 0
        async for chunk in resp.aiter_bytes():
            chunks.append(chunk)
            size += len(chunk)
            if size >= self.settings.max_response_bytes:
                logger.info("Response truncated at %d bytes for %s", size, resp.url)
                break
        return b"".join(chunks)

    async def _fetch_extra_pages(self, urls: list[str]) -> list[PageContent]:
        async def fetch(u: str) -> PageContent | None:
            try:
                final, html = await self.fetch_html(u)
                return parse_page(BeautifulSoup(html, "lxml"), final)
            except Exception as exc:  # noqa: BLE001 - extra pages are best effort
                logger.info("Skipping extra page %s: %s", u, exc)
                return None

        results = await asyncio.gather(*(fetch(u) for u in urls))
        return [r for r in results if r and r.text_blocks]


# ------------------------------------------------------------------ parsing utils
def _short(exc: Exception) -> str:
    return (str(exc) or type(exc).__name__)[:200]


def _decode(body: bytes, encoding: str | None) -> str:
    for enc in (encoding, "utf-8"):
        if enc:
            try:
                return body.decode(enc)
            except (LookupError, UnicodeDecodeError):
                continue
    return body.decode("utf-8", errors="replace")


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _meta(soup: BeautifulSoup, name: str) -> str | None:
    tag = soup.find("meta", attrs={"name": re.compile(f"^{name}$", re.IGNORECASE)})
    content = _clean(tag.get("content", "")) if isinstance(tag, Tag) else ""
    return content or None


def _org_name(json_ld: list[dict]) -> str | None:
    for item in json_ld:
        if item.get("@type") in ("Organization", "Corporation", "LocalBusiness", "WebSite") and item.get("name"):
            return str(item["name"])
    return None


def extract_json_ld(soup: BeautifulSoup) -> list[dict]:
    """Return JSON-LD objects (flattened ``@graph``), keeping only useful fields."""
    keep = ("@type", "name", "description", "url", "address", "areaServed", "sameAs", "slogan", "knowsAbout")
    items: list[dict] = []
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            data = json.loads(script.string or "")
        except (ValueError, TypeError):
            continue
        stack = data if isinstance(data, list) else [data]
        for obj in stack:
            if not isinstance(obj, dict):
                continue
            for node in obj.get("@graph", [obj]):
                if isinstance(node, dict):
                    slim = {k: node[k] for k in keep if k in node}
                    if slim.get("@type"):
                        items.append(slim)
    return items[:10]


def extract_nav_items(soup: BeautifulSoup) -> list[str]:
    items: list[str] = []
    for nav in soup.find_all(["nav", "header"]):
        for a in nav.find_all("a"):
            text = _clean(a.get_text(" "))
            if 2 <= len(text) <= 40 and text not in items:
                items.append(text)
    return items[:40]


def extract_social_links(soup: BeautifulSoup) -> list[str]:
    domains = ("facebook.com", "instagram.com", "linkedin.com", "twitter.com", "x.com", "youtube.com", "tiktok.com")
    links = []
    for a in soup.find_all("a", href=True):
        host = urlsplit(a["href"]).netloc.lower().removeprefix("www.")
        if host in domains and a["href"] not in links:
            links.append(a["href"])
    return links[:10]


def extract_contact_hints(html: str) -> list[str]:
    emails = set(re.findall(r"[\w.+-]+@[\w-]+\.[\w.-]+", html))
    emails = {e for e in emails if not e.lower().endswith((".png", ".jpg", ".svg", ".webp", ".gif"))}
    return sorted(emails)[:3]


def select_internal_links(soup: BeautifulSoup, base_url: str, limit: int) -> list[str]:
    """Pick the most informative same-site pages (services, about, products ...)."""
    if limit <= 0:
        return []
    base_host = urlsplit(base_url).netloc.lower().removeprefix("www.")
    scored: dict[str, int] = {}
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or SKIP_LINK_PATTERN.search(href.split("?")[0]):
            continue
        url = urljoin(base_url, href).split("#")[0]
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or parts.netloc.lower().removeprefix("www.") != base_host:
            continue
        if parts.path.rstrip("/") in ("", urlsplit(base_url).path.rstrip("/")):
            continue
        haystack = (parts.path + " " + a.get_text(" ")).lower()
        score = sum(3 for kw in PRIORITY_KEYWORDS if kw in haystack)
        score -= parts.path.count("/")  # prefer top-level pages
        if score > 0:
            scored[url] = max(scored.get(url, score), score)
    return [u for u, _ in sorted(scored.items(), key=lambda kv: -kv[1])][:limit]


def parse_page(soup: BeautifulSoup, url: str) -> PageContent:
    """Extract headings and readable text blocks, dropping boilerplate."""
    title = _clean(soup.title.get_text()) if soup.title else None
    meta_description = _meta(soup, "description")

    for tag in soup(NOISE_TAGS):
        tag.decompose()
    for tag in soup.select(BOILERPLATE_SELECTOR):
        tag.decompose()

    root = soup.find("main") or soup.find(attrs={"role": "main"}) or soup.body or soup
    # Keep nav/footer out of the main text (nav items are captured separately).
    if root is soup.body or root is soup:
        for tag in root.find_all(["nav", "footer"]):
            tag.decompose()

    headings = []
    for h in root.find_all(["h1", "h2", "h3"]):
        text = _clean(h.get_text(" "))
        if text and text not in headings and len(text) < 150:
            headings.append(text)

    blocks: list[str] = []
    seen: set[str] = set()
    for el in root.find_all(["p", "li", "h1", "h2", "h3", "h4", "td", "blockquote", "dd"]):
        if el.find(["p", "li"]):  # skip containers – their children are visited
            continue
        text = _clean(el.get_text(" "))
        key = text.lower()
        if len(text) < MIN_BLOCK_CHARS and el.name not in ("h1", "h2", "h3", "h4"):
            continue
        if key in seen or not text:
            continue
        seen.add(key)
        blocks.append(text)

    if not blocks:  # div-only layouts
        text = _clean(root.get_text(" "))
        if text:
            blocks = [text[:5000]]

    return PageContent(url=url, title=title, meta_description=meta_description, headings=headings[:40], text_blocks=blocks)


def build_llm_content(pages: list[PageContent], max_chars: int) -> tuple[str, bool]:
    """Join pages into one compact document, de-duplicating blocks across pages.

    The homepage gets half the budget; extra pages share the rest so that one
    long page cannot crowd out the others.
    """
    seen: set[str] = set()
    sections: list[str] = []
    truncated = False
    budgets = [max_chars // 2] + [max_chars // 2 // max(1, len(pages) - 1)] * (len(pages) - 1)

    for page, budget in zip(pages, budgets):
        lines = [f"### PAGE: {page.url}"]
        if page.title:
            lines.append(f"Title: {page.title}")
        if page.meta_description:
            lines.append(f"Meta description: {page.meta_description}")
        used = sum(len(x) for x in lines)
        for block in page.text_blocks:
            key = block.lower()
            if key in seen:
                continue
            if used + len(block) > budget:
                truncated = True
                break
            seen.add(key)
            lines.append(block)
            used += len(block) + 1
        sections.append("\n".join(lines))
    return "\n\n".join(sections), truncated


def build_seo_audit(soup: BeautifulSoup, url: str) -> SEOAudit:
    """Measure on-page SEO facts before any tags are stripped."""
    title = _clean(soup.title.get_text()) if soup.title else None
    meta_description = _meta(soup, "description")
    h1 = [_clean(h.get_text(" ")) for h in soup.find_all("h1") if _clean(h.get_text(" "))]
    images = soup.find_all("img")
    missing_alt = sum(1 for img in images if not (img.get("alt") or "").strip())
    canonical_tag = soup.find("link", rel=lambda v: v and "canonical" in v)
    canonical = canonical_tag.get("href") if isinstance(canonical_tag, Tag) else None
    robots = (_meta(soup, "robots") or "").lower()
    html_tag = soup.find("html")
    lang = html_tag.get("lang") if isinstance(html_tag, Tag) else None
    body_text = _clean(soup.body.get_text(" ")) if soup.body else ""
    ld_types = sorted({str(i.get("@type")) for i in extract_json_ld(soup)})
    has_og = bool(soup.find("meta", attrs={"property": "og:title"}))
    has_viewport = bool(soup.find("meta", attrs={"name": "viewport"}))

    issues: list[str] = []
    if not title:
        issues.append("Missing <title> tag")
    elif len(title) > 60:
        issues.append(f"Title is {len(title)} characters (recommended ≤ 60)")
    elif len(title) < 25:
        issues.append(f"Title is only {len(title)} characters (too short to be descriptive)")
    if not meta_description:
        issues.append("Missing meta description")
    elif not 70 <= len(meta_description) <= 160:
        issues.append(f"Meta description is {len(meta_description)} characters (recommended 120–160)")
    if len(h1) == 0:
        issues.append("No <h1> heading")
    elif len(h1) > 1:
        issues.append(f"{len(h1)} <h1> headings (recommended exactly 1)")
    if images and missing_alt:
        issues.append(f"{missing_alt} of {len(images)} images have no alt text")
    if not canonical:
        issues.append("No canonical URL")
    if not has_viewport:
        issues.append("No viewport meta tag (mobile friendliness)")
    if "noindex" in robots:
        issues.append("Page is set to noindex")
    if not ld_types:
        issues.append("No structured data (JSON-LD)")
    if not has_og:
        issues.append("No Open Graph tags (poor social sharing previews)")
    if not lang:
        issues.append("Missing <html lang> attribute")
    word_count = len(body_text.split())
    if word_count < 300:
        issues.append(f"Thin content: about {word_count} words on the page")

    return SEOAudit(
        title=title,
        title_length=len(title or ""),
        meta_description=meta_description,
        meta_description_length=len(meta_description or ""),
        h1=h1[:5],
        h1_count=len(h1),
        word_count=word_count,
        images_total=len(images),
        images_missing_alt=missing_alt,
        canonical=canonical,
        has_viewport_meta=has_viewport,
        noindex="noindex" in robots,
        lang=lang,
        structured_data_types=ld_types,
        open_graph=has_og,
        issues=issues,
    )
