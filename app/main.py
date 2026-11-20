"""FastAPI entry point.

Run:  uvicorn app.main:app --host 0.0.0.0 --port 4000
Docs: http://localhost:4000/docs
"""

import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse

from app.config import Settings, get_settings
from app.errors import AppError
from app.schemas import ErrorDetail, ErrorResponse, LinkAnalysisRequest, LinkAnalysisResponse
from app.services.analysis_service import AnalysisService, TTLCache
from app.services.llm_service import LLMService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app")


def init_state(app: FastAPI) -> None:
    """Create shared clients once. Called from lifespan, and lazily on the first
    request for serverless runtimes (e.g. Vercel) that may skip lifespan events."""
    if getattr(app.state, "http_client", None) is not None:
        return
    settings = get_settings()
    app.state.http_client = httpx.AsyncClient(
        timeout=settings.fetch_timeout_seconds,
        limits=httpx.Limits(max_connections=50),
        trust_env=False,  # never route scraping through env proxies
    )
    app.state.llm = LLMService(settings)
    app.state.cache = TTLCache(settings.cache_ttl_seconds, settings.cache_max_entries)
    if not settings.anthropic_api_key:
        logger.warning("ANTHROPIC_API_KEY is not set – /link-analysis will return LLM_NOT_CONFIGURED")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_state(app)
    yield
    await app.state.http_client.aclose()
    app.state.http_client = None


app = FastAPI(
    title="AI-Powered Website Link Analysis API",
    description=(
        "Analyzes a website URL with an LLM and returns competitor, search-query, SEO, "
        "services and target-audience insights as structured JSON."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def error_response(status: int, code: str, message: str) -> JSONResponse:
    body = ErrorResponse(error=ErrorDetail(code=code, message=message))
    return JSONResponse(status_code=status, content=body.model_dump())


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    logger.warning("%s: %s", exc.code, exc.message)
    return error_response(exc.status_code, exc.code, exc.message)


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    first = exc.errors()[0] if exc.errors() else {}
    field = ".".join(str(p) for p in first.get("loc", ()) if p != "body") or "body"
    return error_response(422, "VALIDATION_ERROR", f"{field}: {first.get('msg', 'invalid input')}")


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error")
    return error_response(500, "INTERNAL_ERROR", "An unexpected error occurred")


def get_analysis_service(request: Request, settings: Settings = Depends(get_settings)) -> AnalysisService:
    init_state(request.app)
    state = request.app.state
    return AnalysisService(settings, state.http_client, state.llm, state.cache)


ERROR_RESPONSES = {
    400: {"model": ErrorResponse, "description": "URL points to a private/internal address (SSRF protection)"},
    422: {"model": ErrorResponse, "description": "Invalid URL, unsupported or insufficient content"},
    502: {"model": ErrorResponse, "description": "Website unreachable / restricted, or LLM error"},
    503: {"model": ErrorResponse, "description": "LLM not configured or unavailable"},
}


@app.post(
    "/link-analysis",
    response_model=LinkAnalysisResponse,
    responses=ERROR_RESPONSES,
    summary="Analyze a website with an LLM",
)
async def link_analysis(
    payload: LinkAnalysisRequest, service: AnalysisService = Depends(get_analysis_service)
) -> LinkAnalysisResponse:
    return await service.analyze(payload.url)


@app.get("/health", summary="Health check")
async def health(settings: Settings = Depends(get_settings)) -> dict:
    return {"status": "ok", "llm_configured": bool(settings.anthropic_api_key), "model": settings.llm_model}


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse("/docs")
