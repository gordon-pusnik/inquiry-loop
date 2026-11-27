"""Stateless OpenAI Responses inference with CLI-managed function calls."""

import json
import os
import time

from inquiry_loop.response_types import ModelResponse, REPLAY_ITEMS_KEY


REASONING_EFFORTS = ("low", "medium", "high", "max")
_MODEL_FAMILIES = ("gpt-6.2-sol", "gpt-6.2-astra")


def uses_responses(model):
    return model in _MODEL_FAMILIES or any(model.startswith(f"{family}-20") for family in _MODEL_FAMILIES)


def _input_items(messages):
    items = []
    emitted_calls = set()
    for message in messages:
        attached = message.get(REPLAY_ITEMS_KEY)
        if attached:
            items.extend(attached)
            emitted_calls.update(item["call_id"] for item in attached if item["type"] == "function_call")
            continue
        role = message["role"]
        if role == "tool":
            items.append({"type": "function_call_output", "call_id": message["tool_call_id"],
                          "output": message["content"]})
            continue
        if message.get("content") is not None:
            items.append({"role": "developer" if role == "system" else role, "content": message["content"]})
        for call in message.get("tool_calls", ()):
            call_id = call["id"]
            if call_id not in emitted_calls:
                function = call["function"]
                items.append({"type": "function_call", "call_id": call_id,
                              "name": function["name"], "arguments": function["arguments"]})
                emitted_calls.add(call_id)
    return items


class OpenAIResponsesModel:
    preserves_replay_items = True

    def __init__(self, *, tool_schemas, model):
        from openai import OpenAI

        self.model = model
        self.reasoning_effort = os.environ.get("OPENAI_REASONING_EFFORT", "medium")
        if self.reasoning_effort not in REASONING_EFFORTS:
            raise ValueError("OPENAI_REASONING_EFFORT must be low, medium, high, or max.")
        self.tools = [{"type": "function", **schema["function"], "strict": False}
                      for schema in tool_schemas]
        self.last_stats = {}
        self.client = OpenAI()

    def respond(self, messages, *, allow_tools=True):
        self.last_stats = {}
        started = time.perf_counter()
        tool_options = {"tools": self.tools, "parallel_tool_calls": False} if allow_tools and self.tools else {}
        response = self.client.responses.create(
            model=self.model, input=_input_items(messages), store=False,
            reasoning={"effort": self.reasoning_effort}, include=["reasoning.encrypted_content"],
            **tool_options,
        )
        if response.status == "incomplete":
            reason = getattr(response.incomplete_details, "reason", None) or "unknown"
            raise ValueError(f"OpenAI Responses returned an incomplete result ({reason}). "
                             "Partial text and tool calls were discarded.")
        if response.status != "completed":
            raise ValueError(f"OpenAI Responses did not complete normally (status={response.status!r}).")

        items = [item.to_dict() for item in response.output]
        calls = []
        call_ids = set()
        refusals = []
        for item in items:
            kind = item["type"]
            if kind == "function_call":
                if not tool_options:
                    raise ValueError("OpenAI Responses returned a tool call in a tool-free request.")
                call_id, name, arguments = item.get("call_id"), item.get("name"), item.get("arguments")
                try:
                    valid = isinstance(json.loads(arguments), dict)
                except (ValueError, TypeError):
                    valid = False
                if not (isinstance(call_id, str) and call_id and call_id not in call_ids
                        and isinstance(name, str) and name and valid):
                    raise ValueError("OpenAI Responses returned invalid or duplicate tool calls. No tool was executed.")
                call_ids.add(call_id)
                calls.append({"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}})
            elif kind == "message":
                refusals.extend(part["refusal"] for part in item["content"] if part["type"] == "refusal")
            elif kind != "reasoning":
                raise ValueError(f"OpenAI Responses returned an unsupported item ({kind}). No tool was executed.")

        text = response.output_text or "\n".join(refusals)
        if not calls and not text.strip():
            raise ValueError("OpenAI Responses returned neither a usable answer nor a tool call.")
        self.last_stats = {
            "prompt_tokens": getattr(response.usage, "input_tokens", None),
            "generation_tokens": getattr(response.usage, "output_tokens", None),
            "generation_seconds": time.perf_counter() - started,
            "speed_kind": "request",
        }
        return ModelResponse(calls, text, self.last_stats, "tool_calls" if calls else "stop",
                             replay_items=items)

    def close(self):
        self.client.close()
