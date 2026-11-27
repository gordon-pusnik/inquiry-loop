"""OpenAI backend; the agent owns tools and executes their calls."""

import os

from inquiry_loop.response_types import ModelResponse
import time

from inquiry_loop.paths import ENV_FILE


def configured_openai_model(model=None):
    from dotenv import load_dotenv

    load_dotenv(ENV_FILE, override=False)
    return model or os.environ.get("AGENT_MODEL", "gpt-4o-mini")


class OpenAIModel:
    def __init__(self, *, tool_schemas, model=None):
        from openai import OpenAI

        self.model = configured_openai_model(model)
        self.tool_schemas = tool_schemas
        self.client = OpenAI()

    def respond(self, messages, *, allow_tools=True, **completion_options):
        self.last_stats = {}
        started = time.perf_counter()
        tools = ({"tools": self.tool_schemas, "tool_choice": "auto", "parallel_tool_calls": False}
                 if allow_tools and self.tool_schemas else {})
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

    def respond_stream(self, messages, **completion_options):
        """Stream one tool-free benchmark completion and retain request metrics."""
        started = time.perf_counter()
        self.last_stats = {}
        if self.tool_schemas or completion_options.get("tools") or completion_options.get("functions"):
            raise ValueError("OpenAI benchmark streaming does not support tools.")
        if completion_options.get("n") not in (None, 1):
            raise ValueError("OpenAI benchmark streaming requires a single completion.")
        completion_options["stream"] = True
        completion_options["stream_options"] = {
            **(completion_options.get("stream_options") or {}), "include_usage": True,
        }

        stats = self.last_stats = {
            "requested_model": self.model,
            "resolved_model": None,
            "system_fingerprint": None,
            "prompt_tokens": None,
            "generation_tokens": None,
            "cached_prompt_tokens": None,
            "ttft_seconds": None,
            "finish_reason": None,
            "speed_kind": "request",
        }
        parts = []
        saw_tool_delta = False
        stream = None
        try:
            stream = self.client.chat.completions.create(
                model=self.model, messages=messages, **completion_options,
            )
            for chunk in stream:
                for choice in chunk.choices:
                    delta = getattr(choice, "delta", None)
                    content = getattr(delta, "content", None)
                    if content is not None:
                        if stats["ttft_seconds"] is None and content.strip():
                            stats["ttft_seconds"] = time.perf_counter() - started
                        parts.append(content)
                    if getattr(delta, "tool_calls", None) or getattr(delta, "function_call", None) is not None:
                        saw_tool_delta = True
                    finish = getattr(choice, "finish_reason", None)
                    if finish is not None:
                        stats["finish_reason"] = finish

                for key, attr in (("resolved_model", "model"), ("system_fingerprint", "system_fingerprint")):
                    if value := getattr(chunk, attr, None):
                        stats[key] = value
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    details = getattr(usage, "prompt_tokens_details", None)
                    for key, count in (
                        ("prompt_tokens", getattr(usage, "prompt_tokens", None)),
                        ("generation_tokens", getattr(usage, "completion_tokens", None)),
                        ("cached_prompt_tokens", getattr(details, "cached_tokens", None)),
                    ):
                        if type(count) is int and count >= 0:
                            stats[key] = count
        finally:
            elapsed = stats["generation_seconds"] = time.perf_counter() - started
            tokens = stats["generation_tokens"]
            stats["generation_tps"] = tokens / elapsed if tokens is not None and elapsed > 0 else None
            if stream is not None:
                stream.close()

        # Finish and tool deltas may precede the usage-only chunk.
        finish = stats["finish_reason"]
        if saw_tool_delta or finish in ("tool_calls", "function_call"):
            raise ValueError("OpenAI returned a tool call in a tool-free benchmark stream.")
        if finish == "length":
            raise ValueError("OpenAI hit its output-token limit; the partial streamed response was discarded.")
        if finish != "stop":
            raise ValueError(f"OpenAI stream did not complete normally (finish_reason={finish!r}).")
        return ModelResponse.from_parts([], "".join(parts), stats, finish)

    def close(self):
        self.client.close()
