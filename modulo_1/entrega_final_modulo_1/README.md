# Unified Async LLM Client

Async client that talks to OpenAI, Anthropic and Gemini through one interface.
The provider is picked from configuration, so switching from one to another is a change in
`.env` and not in the code. It supports normal responses and streaming, validates messages and
settings with Pydantic, and catches API errors so a bad key or a rate limit doesn't crash the program.

Written for Python 3.12.

## Files

| File | What it has |
| --- | --- |
| `schemas.py` | Pydantic models: `ChatMessage`, `LLMConfig`, `ModelResponse` |
| `clients.py` | `BaseLLMClient` (abstract) and `OpenAIClient`, `AnthropicClient`, `GeminiClient` |
| `manager.py` | `AsyncLLMManager`, creates the right client from the config |
| `exceptions.py` | `LLMError` and its subclasses |
| `main.py` | Test script: one question in normal mode and in streaming mode |
| `.env.example` | Template for the environment variables |

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Open `.env` and paste your API keys. Only the key of the provider you use is needed.
Gemini has a free tier, the key is created at https://aistudio.google.com/apikey.

## Environment variables

| Variable | Default | Description |
| --- | --- | --- |
| `LLM_PROVIDER` | `openai` | `openai`, `anthropic` or `gemini` |
| `OPENAI_API_KEY` | | Required when the provider is `openai` |
| `ANTHROPIC_API_KEY` | | Required when the provider is `anthropic` |
| `GOOGLE_API_KEY` | | Required when the provider is `gemini` |
| `OPENAI_MODEL` | `gpt-6-luna` | |
| `ANTHROPIC_MODEL` | `claude-haiku-4-5-20251001` | |
| `GEMINI_MODEL` | `gemini-flash-latest` | |
| `LLM_TEMPERATURE` | not sent | Between 0 and 2. See the notes at the end |
| `LLM_MAX_TOKENS` | `1024` | Maximum length of the answer |
| `LLM_TIMEOUT` | `30` | Seconds before a request is given up |
| `LLM_MAX_RETRIES` | `2` | Retries on network errors, 429 and 5xx |

The config is validated before any request is made. A missing key, an unknown provider or a
temperature out of range stops the script with a short message.

## Running the test script

```bash
python main.py                     # provider from LLM_PROVIDER
python main.py anthropic           # a specific provider
python main.py openai anthropic    # several, one after the other
```

Output:

```
=== openai (gpt-6-luna) ===
Question: What is entropy?

[1] Normal mode
Entropy is a measure of ...
(tokens in/out: 21/58, finish: stop)

[2] Streaming mode
Entropy is a measure of ...
```

In streaming mode the text shows up little by little as the chunks arrive.

To see the error handling, put a wrong key in `.env` and run it again. Instead of a traceback
you get something like:

```
[1] Normal mode
Request failed [LLMAuthError]: openai: invalid API key or no access to this model

[2] Streaming mode

[LLMAuthError] openai: invalid API key or no access to this model
```

## Using it from other code

```python
import asyncio
from dotenv import load_dotenv

from manager import AsyncLLMManager
from schemas import ChatMessage, LLMConfig

async def main():
    load_dotenv()
    manager = AsyncLLMManager(LLMConfig.from_env())
    messages = [ChatMessage(role="user", content="Explain recursion in one sentence.")]

    response = await manager.generate(messages)
    print(response.content if response.ok else response.error.message)

    async for chunk in manager.generate_stream(messages):
        print(chunk, end="", flush=True)

    await manager.close()

asyncio.run(main())
```

The config can also be built by hand, for example
`LLMConfig(provider="gemini", google_api_key="...", max_tokens=300)`.

## Error handling

Each client catches the exceptions of its SDK and raises one of these instead:

| Exception | When | Retryable |
| --- | --- | --- |
| `LLMAuthError` | Wrong key or no access to the model | no |
| `LLMRateLimitError` | Rate limit or quota exceeded | yes |
| `LLMConnectionError` | Network error or timeout | yes |
| `LLMError` | Any other error from the API (bad request, server error) | no |

Network errors, 429 and 5xx are retried with backoff before giving up (`LLM_MAX_RETRIES`).

`AsyncLLMManager` sits on top and never lets those exceptions out:

- `generate()` returns a `ModelResponse` whose `error` field has `type`, `message` and
  `retryable`. `response.ok` tells you if it worked.
- `generate_stream()` can't return an object because it's a generator, so if something fails it
  yields one last chunk with the error, like `[LLMRateLimitError] ...`, and stops.

If you need the exceptions themselves, use the client classes directly.

## Notes

- `temperature` is optional. When it isn't set nothing is sent and the model uses its default.
  This is on purpose: `gpt-6-luna` and the Claude Sonnet 5 models only accept their default
  value and return a 400 for anything else. Older models such as `claude-haiku-4-5` and Gemini accept it.
- Anthropic's range ends at 1.0, so a higher value is sent as 1.0 to that provider.
- Version 1 of the `anthropic` package removed the `temperature` argument from
  `messages.create()`. The client sends it in the request body instead.
- OpenAI gets `max_completion_tokens` because the newer models don't accept `max_tokens`.
- Reasoning models spend part of `max_tokens` thinking. If the answer comes back cut off or
  empty, raise `LLM_MAX_TOKENS`.
