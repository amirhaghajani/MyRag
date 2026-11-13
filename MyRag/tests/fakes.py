"""Shared test fakes.

The fake OpenAI clients return real openai SDK objects, so llama_index parses them
exactly as it would a live response and its instrumentation emits real usage events.
"""

from __future__ import annotations

from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage

ANSWER = "The rated capacity is 4.5 t/h."


def make_completion(model_id: str = "test/model") -> ChatCompletion:
    """Build a ChatCompletion with the usage the cost tests assert on."""
    return ChatCompletion(
        id="fake-completion",
        created=0,
        model=model_id,
        object="chat.completion",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(role="assistant", content=ANSWER),
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=300,
            completion_tokens=120,
            total_tokens=420,
        ),
    )


class FakeCompletions:
    def __init__(self, model_id: str = "test/model") -> None:
        self.model_id = model_id

    def create(self, **kwargs) -> ChatCompletion:
        return make_completion(self.model_id)


class FakeAsyncCompletions(FakeCompletions):
    async def create(self, **kwargs) -> ChatCompletion:
        return make_completion(self.model_id)


class FakeClient:
    """Stand-in for openai.OpenAI."""

    def __init__(self, model_id: str = "test/model") -> None:
        self.chat = type("Chat", (), {"completions": FakeCompletions(model_id)})()

    def close(self) -> None:
        pass


class FakeAsyncClient(FakeClient):
    """Stand-in for openai.AsyncOpenAI."""

    def __init__(self, model_id: str = "test/model") -> None:
        self.chat = type("Chat", (), {"completions": FakeAsyncCompletions(model_id)})()

    async def close(self) -> None:
        pass