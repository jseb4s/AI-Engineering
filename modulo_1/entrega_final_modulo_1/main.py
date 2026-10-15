import asyncio
import sys

from dotenv import load_dotenv
from pydantic import ValidationError

from manager import AsyncLLMManager
from schemas import ChatMessage, LLMConfig

QUESTION = "What is entropy?"


async def run_test(provider: str | None) -> None:
    try:
        config = LLMConfig.from_env(provider)
    except ValidationError as exc:
        print(f"\nConfiguration error ({provider or 'LLM_PROVIDER'}):")
        for error in exc.errors():
            print(f"  - {error['msg']}")
        return

    manager = AsyncLLMManager(config)
    messages = [
        ChatMessage(role="system", content="Answer in two or three sentences."),
        ChatMessage(role="user", content=QUESTION),
    ]
    print(f"\n=== {config.provider.value} ({config.model}) ===")
    print(f"Question: {QUESTION}")

    try:
        print("\n[1] Normal mode")
        response = await manager.generate(messages)
        if response.ok:
            print(response.content)
            if response.usage:
                print(
                    f"(tokens in/out: {response.usage.input_tokens}/"
                    f"{response.usage.output_tokens}, finish: {response.finish_reason})"
                )
        else:
            print(f"Request failed [{response.error.type}]: {response.error.message}")

        print("\n[2] Streaming mode")
        async for chunk in manager.generate_stream(messages):
            print(chunk, end="", flush=True)
        print()
    finally:
        await manager.close()


async def main() -> None:
    load_dotenv()
    # no arguments: use LLM_PROVIDER. Otherwise test each provider given,
    # e.g. `python main.py openai anthropic`
    for provider in sys.argv[1:] or [None]:
        await run_test(provider)


if __name__ == "__main__":
    asyncio.run(main())
