from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

import anthropic
import httpx
import openai
from anthropic import AsyncAnthropic
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from openai import AsyncOpenAI

from exceptions import LLMAuthError, LLMConnectionError, LLMError, LLMRateLimitError
from schemas import ChatMessage, ModelResponse, Provider, Usage


class BaseLLMClient(ABC):
    """What every provider client has to implement."""

    provider: Provider

    def __init__(self, model: str, temperature: float | None, max_tokens: int) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        """Send the conversation and wait for the complete answer."""

    @abstractmethod
    def generate_stream(self, messages: list[ChatMessage]) -> AsyncGenerator[str, None]:
        """Yield the answer in pieces as the provider sends them."""

    @abstractmethod
    async def close(self) -> None:
        """Close the HTTP connections."""


def _translate_error(exc: Exception, sdk, provider: Provider) -> LLMError:
    # The openai and anthropic SDKs name their exceptions the same way,
    # so this mapping works for both.
    name = provider.value
    if isinstance(exc, (sdk.AuthenticationError, sdk.PermissionDeniedError)):
        return LLMAuthError(f"{name}: invalid API key or no access to this model", name)
    if isinstance(exc, sdk.RateLimitError):
        return LLMRateLimitError(f"{name}: rate limit or quota exceeded", name)
    if isinstance(exc, sdk.APIConnectionError):  # timeouts are a subclass of this
        return LLMConnectionError(f"{name}: could not reach the API (network error or timeout)", name)
    return LLMError(f"{name}: {_api_message(exc)}", name)


def _api_message(exc: Exception) -> str:
    # Use the readable message from the JSON error body when there is one.
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        detail = body.get("error", body)
        if isinstance(detail, dict) and detail.get("message"):
            return f"{detail['message']} (HTTP {getattr(exc, 'status_code', '?')})"
    return str(exc)


class OpenAIClient(BaseLLMClient):
    provider = Provider.OPENAI

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float | None = None,
        max_tokens: int = 1024,
        timeout: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        super().__init__(model, temperature, max_tokens)
        self._client = AsyncOpenAI(api_key=api_key, timeout=timeout, max_retries=max_retries)

    def _request_args(self, messages: list[ChatMessage]) -> dict:
        args = {
            "model": self.model,
            "messages": [m.model_dump() for m in messages],
            # newer models reject the old max_tokens parameter
            "max_completion_tokens": self.max_tokens,
        }
        if self.temperature is not None:
            args["temperature"] = self.temperature
        return args

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        try:
            completion = await self._client.chat.completions.create(**self._request_args(messages))
        except openai.OpenAIError as exc:
            raise _translate_error(exc, openai, self.provider) from exc

        choice = completion.choices[0]
        usage = None
        if completion.usage:
            usage = Usage(
                input_tokens=completion.usage.prompt_tokens,
                output_tokens=completion.usage.completion_tokens,
            )
        return ModelResponse(
            provider=self.provider,
            model=completion.model,
            content=choice.message.content or "",
            finish_reason=choice.finish_reason,
            usage=usage,
        )

    async def generate_stream(self, messages: list[ChatMessage]) -> AsyncGenerator[str, None]:
        try:
            stream = await self._client.chat.completions.create(
                **self._request_args(messages), stream=True
            )
            async for chunk in stream:
                # some chunks have no text (the first one with the role, the last one)
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except openai.OpenAIError as exc:
            raise _translate_error(exc, openai, self.provider) from exc

    async def close(self) -> None:
        await self._client.close()


class AnthropicClient(BaseLLMClient):
    provider = Provider.ANTHROPIC

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float | None = None,
        max_tokens: int = 1024,
        timeout: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        super().__init__(model, temperature, max_tokens)
        self._client = AsyncAnthropic(api_key=api_key, timeout=timeout, max_retries=max_retries)

    def _request_args(self, messages: list[ChatMessage]) -> dict:
        # Anthropic wants the system prompt in its own field, not in the message list
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        args = {
            "model": self.model,
            "max_tokens": self.max_tokens,  # mandatory here
            "messages": [m.model_dump() for m in messages if m.role != "system"],
        }
        if system:
            args["system"] = system
        if self.temperature is not None:
            # anthropic>=1.0 no longer has a temperature argument (passing it is a
            # TypeError). Older models still read it from the body. Max is 1.0.
            args["extra_body"] = {"temperature": min(self.temperature, 1.0)}
        return args

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        try:
            message = await self._client.messages.create(**self._request_args(messages))
        except anthropic.AnthropicError as exc:
            raise _translate_error(exc, anthropic, self.provider) from exc

        # content is a list of blocks and not all of them are text
        text = "".join(block.text for block in message.content if block.type == "text")
        return ModelResponse(
            provider=self.provider,
            model=message.model,
            content=text,
            finish_reason=message.stop_reason,
            usage=Usage(
                input_tokens=message.usage.input_tokens,
                output_tokens=message.usage.output_tokens,
            ),
        )

    async def generate_stream(self, messages: list[ChatMessage]) -> AsyncGenerator[str, None]:
        try:
            async with self._client.messages.stream(**self._request_args(messages)) as stream:
                async for text in stream.text_stream:
                    yield text
        except anthropic.AnthropicError as exc:
            raise _translate_error(exc, anthropic, self.provider) from exc

    async def close(self) -> None:
        await self._client.close()


class GeminiClient(BaseLLMClient):
    provider = Provider.GEMINI

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float | None = None,
        max_tokens: int = 1024,
        timeout: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        super().__init__(model, temperature, max_tokens)
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=int(timeout * 1000),  # this SDK takes milliseconds
                retry_options=types.HttpRetryOptions(attempts=max_retries + 1),
            ),
        )

    def _request_args(self, messages: list[ChatMessage]) -> dict:
        # Gemini also keeps the system prompt apart, and calls the assistant "model"
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        contents = [
            types.Content(
                role="model" if m.role == "assistant" else "user",
                parts=[types.Part(text=m.content)],
            )
            for m in messages
            if m.role != "system"
        ]
        config = types.GenerateContentConfig(
            max_output_tokens=self.max_tokens,
            temperature=self.temperature,
            system_instruction=system or None,
        )
        return {"model": self.model, "contents": contents, "config": config}

    def _translate(self, exc: Exception) -> LLMError:
        if not isinstance(exc, genai_errors.APIError):
            return LLMConnectionError("gemini: could not reach the API (network error or timeout)", "gemini")
        message = exc.message or str(exc)
        # a wrong key comes back as a 400 here, not a 401
        if exc.code in (401, 403) or "API key" in message:
            return LLMAuthError("gemini: invalid API key or no access to this model", "gemini")
        if exc.code == 429:
            return LLMRateLimitError("gemini: rate limit or quota exceeded", "gemini")
        return LLMError(f"gemini: {message} (HTTP {exc.code})", "gemini")

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        try:
            response = await self._client.aio.models.generate_content(**self._request_args(messages))
        except (genai_errors.APIError, httpx.HTTPError) as exc:
            raise self._translate(exc) from exc

        usage = None
        if response.usage_metadata:
            usage = Usage(
                input_tokens=response.usage_metadata.prompt_token_count or 0,
                output_tokens=response.usage_metadata.candidates_token_count or 0,
            )
        finish_reason = None
        if response.candidates and response.candidates[0].finish_reason:
            finish_reason = response.candidates[0].finish_reason.name.lower()
        return ModelResponse(
            provider=self.provider,
            model=self.model,
            content=response.text or "",
            finish_reason=finish_reason,
            usage=usage,
        )

    async def generate_stream(self, messages: list[ChatMessage]) -> AsyncGenerator[str, None]:
        try:
            stream = await self._client.aio.models.generate_content_stream(
                **self._request_args(messages)
            )
            async for chunk in stream:
                if chunk.text:
                    yield chunk.text
        except (genai_errors.APIError, httpx.HTTPError) as exc:
            raise self._translate(exc) from exc

    async def close(self) -> None:
        await self._client.aio.aclose()
