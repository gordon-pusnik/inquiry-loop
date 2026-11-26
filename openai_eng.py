"""OpenAI backend; the agent owns tools and executes their calls."""

import os

from response_types import ModelResponse
import time
from pathlib import Path


class OpenAIModel:
    def __init__(self, *, tool_schemas, model=None):
        from openai import OpenAI
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().with_name(".env"), override=False)
        self.model = model or os.environ.get("AGENT_MODEL", "gpt-4o-mini")
        self.tool_schemas = tool_schemas
        self.client = OpenAI()

    def respond(self, messages, **completion_options):
        self.last_stats = {}
        started = time.perf_counter()
        tools = ({"tools": self.tool_schemas, "tool_choice": "auto", "parallel_tool_calls": False}
                 if self.tool_schemas else {})
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, **tools, **completion_options,
        )
        finish = getattr(response.choices[0], 'finish_reason', None)
        if finish == 'length':
            raise ValueError('OpenAI hit its output-token limit; the partial response and its tool calls were discarded.')
        if finish not in (None, 'stop', 'tool_calls'):
            raise ValueError(f'OpenAI did not complete normally (finish_reason={finish!r}).')
        message = response.choices[0].message
        self.last_stats = {
            "prompt_tokens": getattr(getattr(response, "usage", None), "prompt_tokens", None),
            "generation_tokens": getattr(getattr(response, "usage", None), "completion_tokens", None),
            "generation_seconds": time.perf_counter() - started,
            "speed_kind": "request",
        }
        return ModelResponse.from_parts(list(message.tool_calls or []), message.content or "",
                                        self.last_stats, finish)

    def close(self):
        self.client.close()
