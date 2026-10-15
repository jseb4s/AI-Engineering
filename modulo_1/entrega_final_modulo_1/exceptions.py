class LLMError(Exception):
    """Base class for every provider failure, so callers never see SDK errors."""

    retryable = False

    def __init__(self, message: str, provider: str | None = None) -> None:
        super().__init__(message)
        self.provider = provider


class LLMAuthError(LLMError):
    """The API key is missing, wrong, or not allowed to use the model."""


class LLMRateLimitError(LLMError):
    """Too many requests or the quota ran out."""

    retryable = True


class LLMConnectionError(LLMError):
    """Network problem or timeout before getting a response."""

    retryable = True
