"""Application errors mapped to HTTP responses by ``app.main``."""


class AppError(Exception):
    status_code = 500
    code = "INTERNAL_ERROR"

    def __init__(self, message: str, *, code: str | None = None, status_code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code:
            self.code = code
        if status_code:
            self.status_code = status_code


class InvalidURL(AppError):
    status_code, code = 422, "INVALID_URL"


class BlockedURL(AppError):
    status_code, code = 400, "URL_NOT_ALLOWED"


class WebsiteUnreachable(AppError):
    status_code, code = 502, "WEBSITE_UNREACHABLE"


class WebsiteAccessDenied(AppError):
    status_code, code = 502, "WEBSITE_ACCESS_RESTRICTED"


class UnsupportedContent(AppError):
    status_code, code = 422, "UNSUPPORTED_CONTENT"


class InsufficientContent(AppError):
    status_code, code = 422, "INSUFFICIENT_CONTENT"


class LLMError(AppError):
    status_code, code = 502, "LLM_ERROR"


class LLMUnavailable(AppError):
    status_code, code = 503, "LLM_UNAVAILABLE"


class LLMNotConfigured(AppError):
    status_code, code = 503, "LLM_NOT_CONFIGURED"
