"""Application settings loaded from environment variables / `.env`."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

# Anthropic first-party prices in USD per 1M tokens (input, output, cache read).
# Source: https://platform.claude.com/docs/en/about-claude/pricing (checked 2026-11-20).
MODEL_PRICING: dict[str, tuple[float, float, float]] = {
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 0.10),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
}

# Models that accept the server-side refusal fallback (`fallbacks: "default"`).
SERVER_FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-2", "claude-fable-5-1"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM ---------------------------------------------------------------
    anthropic_api_key: str = ""
    llm_model: str = "claude-opus-5-5"
    # Used when the primary model is overloaded / unavailable (5xx, 529, timeouts).
    llm_fallback_model: str = "claude-sonnet-5-5"
    # low | medium | high – thinking depth. "low" keeps latency and cost down.
    llm_effort: str = "low"
    llm_max_tokens: int = 16000
    llm_timeout_seconds: float = 120.0

    # --- Website extraction -------------------------------------------------
    fetch_timeout_seconds: float = 15.0
    max_response_bytes: int = 3_000_000
    max_redirects: int = 5
    # Extra internal pages (about / services / products) fetched besides the homepage.
    max_extra_pages: int = 3
    # Upper bound for cleaned text sent to the LLM (~4 chars per token).
    max_content_chars: int = 24_000
    user_agent: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/130.0 Safari/537.36 LinkAnalysisBot/1.0"
    )
    allowed_ports: str = "80,443,8080,8443"

    # --- Competitor verification -------------------------------------------
    verify_competitor_websites: bool = True

    # --- Cache ----------------------------------------------------------------
    cache_ttl_seconds: int = 3600
    cache_max_entries: int = 200

    @property
    def allowed_port_set(self) -> set[int]:
        return {int(p) for p in self.allowed_ports.split(",") if p.strip()}


@lru_cache
def get_settings() -> Settings:
    return Settings()
