"""Scripted offline backend; no model dependencies or API calls."""

import json

from inquiry_loop.response_types import ModelResponse
from typing import Any, Dict, List


class _FakeFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class _FakeCall:
    """Mirrors the shape of an OpenAI tool-call object: call.function.name/.arguments"""
    def __init__(self, call_id: str, name: str, args: Dict[str, Any]):
        self.id = call_id
        self.function = _FakeFunction(name, json.dumps(args))


class StubModel:
    """Scripted model so the loop can be run and understood without an API key."""

    def __init__(self):
        self.script = [
            ([_FakeCall("call_1", "list_dir", {"path": "."})], ""),
            ([_FakeCall("call_2", "read_file", {"path": "agent.py"})], ""),
            ([], "Done. I listed the directory and read agent.py - the loop worked."),
        ]
        self.i = 0

    def respond(self, messages: List[Dict[str, Any]]):
        step = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return ModelResponse.from_parts(*step)


