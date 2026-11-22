"""In-process MLX-LM backend with native model templates and tool parsers."""

from __future__ import annotations

from response_types import ModelResponse

import copy
import gc
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

DEFAULT_MODEL = "mlx-community/Qwen3-4B-4bit"


class MLXModel:
    def __init__(self, *, tool_schemas, model=None, model_path=None,
                 max_tokens=None, max_context_tokens=None, prefill_step_size=None,
                 cache_limit_mb=None):
        if sys.platform != "darwin":
            raise RuntimeError("This MLX-LM backend requires an Apple Silicon Mac with Metal access.")
        if model and model_path:
            raise ValueError("Select an MLX model with --model or --model-path, not both.")
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().with_name(".env"), override=False)
        self.max_tokens = int(max_tokens if max_tokens is not None else os.environ.get("MLX_MAX_TOKENS", "2048"))
        self.max_context_tokens = int(max_context_tokens if max_context_tokens is not None else os.environ.get("MLX_MAX_CONTEXT_TOKENS", "16384"))
        self.prefill_step_size = int(prefill_step_size if prefill_step_size is not None else os.environ.get("MLX_PREFILL_STEP_SIZE", "512"))
        cache_mb = int(cache_limit_mb if cache_limit_mb is not None else os.environ.get("MLX_CACHE_LIMIT_MB", "512"))
        if not 0 < self.max_tokens < self.max_context_tokens or self.prefill_step_size <= 0 or cache_mb < 0:
            raise ValueError("Require 0 < MLX_MAX_TOKENS < MLX_MAX_CONTEXT_TOKENS, positive prefill size, and nonnegative cache limit.")
        self.model_id = model or os.environ.get("MLX_MODEL") or DEFAULT_MODEL
        if model_path:
            path = Path(model_path).expanduser()
            if not (path / "config.json").is_file():
                raise ValueError("MLX --model-path must be a model directory containing config.json and MLX weights, not a GGUF file.")
            self.model_id = str(path)
        try:
            import mlx.core as mx
            from mlx_lm import load, stream_generate
            from mlx_lm.sample_utils import make_sampler
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError(f"Cannot initialize MLX-LM: {exc}. Install mlx-lm in this Python environment and run with Metal GPU access.") from exc
        self.mx = mx
        self._stream_generate = stream_generate
        self._sampler = make_sampler(temp=0.0)
        self.tool_schemas = tool_schemas
        self.model = self.tokenizer = None
        self.last_stats = {}
        self._previous_cache_limit = mx.set_cache_limit(cache_mb * 1024 * 1024)
        try:
            # Explicit repository selection may download weights on first use.
            # MLX's native quantized loader is used, not AirLLM's layer loader.
            self.model, self.tokenizer, config = load(self.model_id, return_config=True)
            context = config.get("max_position_embeddings")
            if isinstance(context, int) and context > 0:
                self.max_context_tokens = min(self.max_context_tokens, context)
            if self.max_tokens >= self.max_context_tokens:
                raise ValueError("MLX output budget must be smaller than this model's context limit.")
            if (not self.tokenizer.has_tool_calling or not callable(self.tokenizer.tool_parser)
                    or not self.tokenizer.tool_call_start or not self.tokenizer.tool_call_end):
                raise ValueError("This MLX tokenizer has no supported native tool-call parser with explicit delimiters. Try the default Qwen3 model.")
        except Exception:
            self.close()
            raise

    def _messages(self, messages):
        # HF templates expect argument objects in assistant history, whereas
        # the agent/OpenAI transcript stores JSON argument strings.
        history = copy.deepcopy(messages)
        for message in history:
            # OpenAI permits content to be absent/null on tool-call messages.
            # Native chat templates (including Qwen3) concatenate it as text.
            if message.get("content") is None:
                message["content"] = ""
            for call in message.get("tool_calls") or []:
                function = call["function"]
                if isinstance(function.get("arguments"), str):
                    function["arguments"] = json.loads(function["arguments"])
        return history

    def _parse_response(self, text):
        start, end = self.tokenizer.tool_call_start, self.tokenizer.tool_call_end
        think_start = getattr(self.tokenizer, "think_start", None)
        think_end = getattr(self.tokenizer, "think_end", None)
        calls, answer = [], []
        position = 0
        while position < len(text):
            if text.startswith(start, position):
                stop = text.find(end, position + len(start))
                if stop < 0:
                    raise ValueError("MLX-LM returned an incomplete tool call; no calls from this response were executed.")
                payload = text[position + len(start):stop]
                try:
                    parsed = self.tokenizer.tool_parser(payload, [s["function"] for s in self.tool_schemas])
                    if isinstance(parsed, list) and not parsed:
                        raise ValueError("Empty tool-call list")
                    for call in parsed if isinstance(parsed, list) else [parsed]:
                        if not isinstance(call, dict) or not isinstance(call.get("name"), str) or not call["name"]:
                            raise ValueError("Missing tool name")
                        arguments = call.get("arguments")
                        if isinstance(arguments, str):
                            arguments = json.loads(arguments)
                        if not isinstance(arguments, dict):
                            raise ValueError("Tool arguments must be an object")
                        calls.append({"id": "call_" + uuid4().hex, "type": "function", "function": {
                            "name": call["name"], "arguments": json.dumps(arguments)}})
                except (ValueError, TypeError, KeyError) as exc:
                    raise ValueError("MLX-LM returned invalid tool arguments; no calls from this response were executed.") from exc
                position = stop + len(end)
            elif think_start and think_end and text.startswith(think_start, position):
                stop = text.find(think_end, position + len(think_start))
                if stop < 0:
                    raise ValueError("MLX-LM returned an unfinished reasoning block.")
                position = stop + len(think_end)
            else:
                markers = [text.find(start, position)]
                if think_start and think_end:
                    markers.append(text.find(think_start, position))
                stop = min((n for n in markers if n >= 0), default=len(text))
                answer.append(text[position:stop])
                position = stop
        result = "".join(answer).strip()
        if not calls and not result:
            raise ValueError("MLX-LM returned neither a tool call nor an answer.")
        return calls, result

    def respond(self, messages):
        if self.model is None:
            raise RuntimeError("The MLX model is closed.")
        prompt = self.tokenizer.apply_chat_template(
            self._messages(messages), tools=self.tool_schemas, tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(tokens) + self.max_tokens > self.max_context_tokens:
            raise ValueError("MLX context budget exceeded. Shorten the task or adjust MLX_MAX_CONTEXT_TOKENS within your memory budget.")
        self.last_stats = {}
        self.mx.reset_peak_memory()
        stream = self._stream_generate(
            self.model, self.tokenizer, prompt=tokens, max_tokens=self.max_tokens,
            sampler=self._sampler, prefill_step_size=self.prefill_step_size,
        )
        parts, finish = [], None
        try:
            for response in stream:
                parts.append(response.text)
                finish = response.finish_reason
                if finish is not None:
                    self.last_stats = {name: getattr(response, name, None) for name in (
                        "prompt_tokens", "generation_tokens", "prompt_tps", "generation_tps", "peak_memory")}
        finally:
            stream.close()
            # Each response uses a fresh prompt cache. Return unused buffers to
            # the allocator between agent turns without unloading model weights.
            self.mx.synchronize()
            self.mx.clear_cache()
        if finish != "stop":
            raise ValueError(f"MLX-LM did not finish within its output budget ({self.max_tokens} tokens); partial output was discarded and no partial tool calls were executed. Request a concise assessment or increase MLX_MAX_TOKENS within the context budget.")
        calls, text = self._parse_response("".join(parts))
        return ModelResponse.from_parts(calls, text, self.last_stats, finish)

    def close(self):
        self.model = self.tokenizer = None
        gc.collect()
        self.mx.synchronize()
        self.mx.clear_cache()
        if self._previous_cache_limit is not None:
            self.mx.set_cache_limit(self._previous_cache_limit)
            self._previous_cache_limit = None
