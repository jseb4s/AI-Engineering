import os
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


DEFAULT_MODELS = {
    Provider.OPENAI: "gpt-6-luna",
    Provider.ANTHROPIC: "claude-haiku-4-5-20251001",
    Provider.GEMINI: "gemini-flash-latest",
}

API_KEY_VARS = {
    Provider.OPENAI: "OPENAI_API_KEY",
    Provider.ANTHROPIC: "ANTHROPIC_API_KEY",
    Provider.GEMINI: "GOOGLE_API_KEY",
}


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class LLMConfig(BaseModel):
    provider: Provider = Provider.OPENAI
    model: str = ""  # empty means "use the default model for the provider"
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None
    # None = don't send it and let the model use its default. Several recent
    # models reject any custom temperature, so it is opt-in.
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int = Field(default=1024, gt=0)
    timeout: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=2, ge=0, le=5)

    @field_validator("openai_api_key", "anthropic_api_key", "google_api_key")
    @classmethod
    def key_must_be_ascii(cls, key: SecretStr | None) -> SecretStr | None:
        # A key with accents or other odd characters can't go in an HTTP header
        # and blows up with an encoding error deep inside the SDK.
        if key is not None and not key.get_secret_value().isascii():
            raise ValueError("API keys can only contain ASCII characters, check for typos")
        return key

    @model_validator(mode="after")
    def set_model_and_check_key(self) -> "LLMConfig":
        if not self.model:
            self.model = DEFAULT_MODELS[self.provider]
        if self.api_key is None:
            raise ValueError(
                f"{API_KEY_VARS[self.provider]} is required to use {self.provider.value}"
            )
        return self

    @property
    def api_key(self) -> SecretStr | None:
        keys = {
            Provider.OPENAI: self.openai_api_key,
            Provider.ANTHROPIC: self.anthropic_api_key,
            Provider.GEMINI: self.google_api_key,
        }
        return keys[self.provider]

    @classmethod
    def from_env(cls, provider: str | None = None) -> "LLMConfig":
        name = (provider or os.getenv("LLM_PROVIDER") or Provider.OPENAI.value).strip().lower()
        env = {
            "provider": name,
            "model": os.getenv(f"{name.upper()}_MODEL"),
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
            "google_api_key": os.getenv("GOOGLE_API_KEY"),
            "temperature": os.getenv("LLM_TEMPERATURE"),
            "max_tokens": os.getenv("LLM_MAX_TOKENS"),
            "timeout": os.getenv("LLM_TIMEOUT"),
            "max_retries": os.getenv("LLM_MAX_RETRIES"),
        }
        # variables that are unset or empty fall back to the defaults above
        return cls(**{k: v.strip() for k, v in env.items() if v and v.strip()})


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class ErrorInfo(BaseModel):
    type: str
    message: str
    retryable: bool = False


class ModelResponse(BaseModel):
    provider: Provider
    model: str
    content: str = ""
    finish_reason: str | None = None
    usage: Usage | None = None
    error: ErrorInfo | None = None

    @property
    def ok(self) -> bool:
        return self.error is None
