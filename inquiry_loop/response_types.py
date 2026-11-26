"""Backend-independent model and run results; no inference dependencies."""
from dataclasses import dataclass, field
from enum import Enum
import copy
from typing import Callable


class RunStatus(str, Enum):
    RUNNING = 'running'
    COMPLETED = 'completed'
    STEP_LIMIT = 'step_limit'
    TOOL_ERROR_LIMIT = 'tool_error_limit'
    FAILED = 'failed'
    INTERRUPTED = 'interrupted'


@dataclass
class ModelResponse:
    tool_calls: list[dict]
    text: str
    stats: dict = field(default_factory=dict)
    finish_reason: str | None = None

    @classmethod
    def from_parts(cls, calls, text, stats=None, finish_reason=None):
        normalized = []
        for call in calls:
            if isinstance(call, dict):
                normalized.append(copy.deepcopy(call))
            else:
                normalized.append({'id': call.id, 'type': 'function', 'function': {
                    'name': call.function.name, 'arguments': call.function.arguments}})
        return cls(normalized, text, copy.deepcopy(stats or {}), finish_reason)

    def __iter__(self):
        # Preserve two-value unpacking for existing adapter callers.
        yield self.tool_calls
        yield self.text

    @classmethod
    def coerce(cls, response, stats=None):
        if isinstance(response, cls):
            return response
        calls, text = response  # compatibility with external/scripted adapters
        return cls.from_parts(calls, text, stats)


@dataclass
class RunResult:
    run_id: str
    session_id: str
    status: RunStatus
    answer: str
    elapsed: float
    stats: dict
    error: str | None = None


@dataclass
class RunContext:
    previous_messages: list = field(default_factory=list)
    messages: list = field(default_factory=list)
    status: RunStatus = RunStatus.RUNNING
    emit: Callable[[str, dict], None] = lambda *_: None

    def finish(self, status, answer, messages):
        self.status = status
        self.messages = copy.deepcopy([m for m in messages if m['role'] != 'system'])
        self.messages.append({'role': 'assistant', 'content': answer})
        return answer
