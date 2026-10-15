from collections.abc import AsyncGenerator

from clients import AnthropicClient, BaseLLMClient, GeminiClient, OpenAIClient
from exceptions import LLMError
from schemas import ChatMessage, ErrorInfo, LLMConfig, ModelResponse, Provider

CLIENTS: dict[Provider, type[BaseLLMClient]] = {
    Provider.OPENAI: OpenAIClient,
    Provider.ANTHROPIC: AnthropicClient,
    Provider.GEMINI: GeminiClient,
}


class AsyncLLMManager:
    """Builds the client for the configured provider and exposes one API for all
    of them. Provider failures never escape from here as exceptions."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig.from_env()
        self._client = self._create_client()

    def _create_client(self) -> BaseLLMClient:
        client_class = CLIENTS[self.config.provider]
        return client_class(
            api_key=self.config.api_key.get_secret_value(),
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
        )

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        try:
            return await self._client.generate(messages)
        except LLMError as exc:
            return ModelResponse(
                provider=self.config.provider,
                model=self.config.model,
                error=ErrorInfo(
                    type=type(exc).__name__, message=str(exc), retryable=exc.retryable
                ),
            )

    async def generate_stream(self, messages: list[ChatMessage]) -> AsyncGenerator[str, None]:
        try:
            async for chunk in self._client.generate_stream(messages):
                yield chunk
        except LLMError as exc:
            # a generator can't return a ModelResponse, so the error is sent
            # as one last chunk instead of raising in the middle of the loop
            yield f"\n[{type(exc).__name__}] {exc}"

    async def close(self) -> None:
        await self._client.close()
