"""Pydantic models.

``LLMAnalysis`` is the JSON schema the LLM must produce. It is sent to Claude as
a structured-output schema, so the response is guaranteed to parse into it.
The API response (``LinkAnalysisResponse``) wraps it with measured data
(SEO audit facts, competitor website checks, token usage and cost).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Platform = Literal["Google", "Facebook", "Instagram", "LinkedIn", "X"]
Basis = Literal["website_content", "inferred"]


# ----------------------------------------------------------------------- request
class LinkAnalysisRequest(BaseModel):
    url: str = Field(..., description="Public website URL to analyze", examples=["https://example.com"])


# --------------------------------------------------------------- LLM output schema
class Brand(BaseModel):
    name: str
    category: str = Field(description="Main business category")
    summary: str = Field(description="2-3 sentence business overview based on the website")
    location: str | None = Field(description="Headquarters / service area if stated on the site, else null")


class BrandProfile(BaseModel):
    main_service: str
    target_market: str
    key_strength: str
    opportunity: str


class Competitor(BaseModel):
    name: str
    website: str | None = Field(description="Homepage URL if known with confidence, else null")
    main_service: str
    target_market: str
    key_strength: str
    strengths: list[str]
    weaknesses_or_gaps: list[str]
    opportunity_for_brand: str = Field(description="How the analyzed brand can win against this competitor")
    basis: Literal["mentioned_on_website", "ai_suggestion"] = Field(
        description="'mentioned_on_website' only if the competitor is named in the provided content"
    )


class CompetitorAnalysis(BaseModel):
    brand_overview: str
    main_category: str
    main_products_services: list[str]
    brand_profile: BrandProfile
    competitors: list[Competitor]
    competitive_advantages: list[str]
    positioning_opportunities: list[str]


class SearchQuery(BaseModel):
    query: str
    platforms: list[Platform]
    intent: str = Field(description="e.g. Service Search, Business Discovery, Inspiration, B2B Search, Comparison")
    business_relevance: str
    target_audience: str


class SupportingContent(BaseModel):
    type: Literal[
        "service_landing_page",
        "industry_page",
        "faq",
        "blog_topic",
        "location_page",
        "business_article",
        "case_study",
        "comparison_page",
    ]
    title: str
    description: str
    target_keyword: str


class SEOImprovements(BaseModel):
    primary_keyword: str
    secondary_keywords: list[str]
    long_tail_keywords: list[str]
    business_keywords: list[str]
    recommended_title: str = Field(description="Under 60 characters, contains the primary keyword")
    title_structure: str = Field(description="Pattern used, e.g. 'Primary Keyword | Benefit | Brand'")
    meta_description: str = Field(description="140-160 characters, human friendly, includes keywords")
    supporting_content: list[SupportingContent]
    technical_fixes: list[str] = Field(description="Fixes based on the measured on-page SEO audit")


class Service(BaseModel):
    service_name: str
    description: str
    target_industry: str
    target_customer: str
    main_benefit: str
    basis: Basis = Field(description="'website_content' if the service appears on the site, else 'inferred'")


class AudienceSegment(BaseModel):
    audience: str
    business_type: str
    requirement: str


class TargetAudience(BaseModel):
    primary_audience: list[str]
    secondary_audience: list[str]
    industries: list[str]
    business_types: list[str]
    customer_profiles: list[str]
    geographic_focus: str | None
    customer_needs: list[str]
    customer_challenges: list[str]
    customer_interests: list[str]
    recommended_segments: list[AudienceSegment]


class DataQuality(BaseModel):
    content_sufficient: bool
    notes: str = Field(description="Caveats, e.g. thin content, JS-only site, assumptions made")


class LLMAnalysis(BaseModel):
    brand: Brand
    competitor_analysis: CompetitorAnalysis
    target_search_queries: list[SearchQuery]
    seo_improvements: SEOImprovements
    top_services: list[Service]
    target_audience: TargetAudience
    data_quality: DataQuality


# ------------------------------------------------------------------- API response
class ComparisonRow(BaseModel):
    comparison: str
    your_brand: str
    competitors: dict[str, str]


class CompetitorOut(Competitor):
    website_verified: bool | None = Field(
        None, description="True if the competitor website responded publicly, null if not checked"
    )


class CompetitorAnalysisOut(CompetitorAnalysis):
    competitors: list[CompetitorOut]
    comparison_table: list[ComparisonRow]
    source_note: str


class SEOAudit(BaseModel):
    """Facts measured from the HTML – not generated by the LLM."""

    title: str | None
    title_length: int
    meta_description: str | None
    meta_description_length: int
    h1: list[str]
    h1_count: int
    word_count: int
    images_total: int
    images_missing_alt: int
    canonical: str | None
    has_viewport_meta: bool
    noindex: bool
    lang: str | None
    structured_data_types: list[str]
    open_graph: bool
    issues: list[str]


class Analysis(BaseModel):
    competitor_analysis: CompetitorAnalysisOut
    target_search_queries: list[SearchQuery]
    seo_improvements: SEOImprovements
    top_services: list[Service]
    target_audience: TargetAudience


class Metadata(BaseModel):
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    estimated_cost_usd: float | None
    llm_latency_ms: int
    total_latency_ms: int
    pages_analyzed: list[str]
    content_chars_sent: int
    content_truncated: bool
    cached: bool
    data_quality: DataQuality
    search_volume_note: str = (
        "Search volumes are not included: no search-volume data source is connected, "
        "and volumes are never estimated by the LLM."
    )


class LinkAnalysisResponse(BaseModel):
    success: bool = True
    website: str
    brand: Brand
    analysis: Analysis
    seo_audit: SEOAudit
    metadata: Metadata


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    success: bool = False
    error: ErrorDetail
