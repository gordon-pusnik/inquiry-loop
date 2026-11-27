"""Run a local GGUF with a private llama.cpp server for the model's lifetime."""

import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import time
from uuid import uuid4

from inquiry_loop.paths import ENV_FILE
from inquiry_loop.response_types import ModelResponse


def resolve_server(model_path=None):
    from dotenv import load_dotenv

    load_dotenv(ENV_FILE, override=False)
    selected = model_path or os.environ.get("LLAMA_MODEL_PATH")
    if not selected:
        raise ValueError("llama.cpp needs a local GGUF file. Set --model-path or LLAMA_MODEL_PATH.")
    path = Path(selected).expanduser()
    if path.suffix.lower() != ".gguf" or not path.is_file():
        raise ValueError(f"llama.cpp needs an existing .gguf model file: {path}")
    configured_server = os.environ.get("LLAMA_SERVER")
    if configured_server:
        binary = shutil.which(os.path.expanduser(configured_server))
    else:
        binary = shutil.which("llama-server") or shutil.which("llama")
    if not binary:
        raise ValueError("Could not find llama-server or llama. Install llama.cpp or set LLAMA_SERVER to its executable.")
    return path, binary


class LlamaModel:
    def __init__(self, *, tool_schemas, model_path=None):
        import httpx

        path, binary = resolve_server(model_path)
        server_command = [binary]
        if Path(binary).stem.lower() == "llama":
            server_command.append("serve")

        self.model_path = path.absolute()
        self.model_id = str(self.model_path.resolve())
        self.tool_schemas = tool_schemas
        self.max_tokens = int(os.environ.get("LLAMA_MAX_TOKENS", "2048"))
        self.max_context_tokens = int(os.environ.get("LLAMA_CONTEXT_SIZE", "8192"))
        gpu_layers = int(os.environ.get("LLAMA_GPU_LAYERS", "999"))
        timeout = int(os.environ.get("LLAMA_TIMEOUT", "300"))
        if not 0 < self.max_tokens < self.max_context_tokens or gpu_layers < 0 or timeout <= 0:
            raise ValueError("Require 0 < LLAMA_MAX_TOKENS < LLAMA_CONTEXT_SIZE, "
                             "nonnegative LLAMA_GPU_LAYERS, and positive LLAMA_TIMEOUT.")

        self.last_stats = {}
        self._process = self._client = None
        self._alias = f"inquiry-loop-{uuid4().hex}"
        key = uuid4().hex
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        env = {name: value for name, value in os.environ.items() if not name.startswith("LLAMA_ARG_")}
        env["LLAMA_API_KEY"] = key
        command = [
            *server_command, "--model", str(self.model_path), "--alias", self._alias, "--jinja", "--no-webui",
            "--host", "127.0.0.1", "--port", str(port), "--parallel", "1",
            "--ctx-size", str(self.max_context_tokens), "--gpu-layers", str(gpu_layers), "--no-context-shift",
            "--log-verbosity", "2",
        ]
        with tempfile.TemporaryFile() as log:
            try:
                self._process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                                                 stdout=log, stderr=subprocess.STDOUT)
                self._client = httpx.Client(
                    base_url=f"http://127.0.0.1:{port}", headers={"Authorization": f"Bearer {key}"},
                    timeout=timeout, trust_env=False,
                )
                self._wait_until_ready(timeout)
            except RuntimeError as exc:
                self.close()
                log.seek(max(0, log.seek(0, os.SEEK_END) - 2048))
                detail = log.read().decode(errors="replace").strip()
                if detail:
                    raise RuntimeError(f"{exc}\nllama-server: {detail.replace(key, '[redacted]')}") from exc
                raise
            except BaseException:
                self.close()
                raise

    def _wait_until_ready(self, timeout):
        import httpx

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self._process.poll()
            if status is not None:
                raise RuntimeError(f"llama-server exited during startup (status {status}). "
                                   "Check that the binary and GGUF are compatible and fit in memory.")
            try:
                if self._client.get("/health", timeout=1).status_code == 200:
                    models = self._client.get("/v1/models", timeout=1)
                    if models.is_success and any(item.get("id") == self._alias
                                                 for item in models.json().get("data", [])):
                        return
                    raise RuntimeError("llama-server did not identify the requested model on its loopback port.")
            except httpx.TransportError:
                pass
            time.sleep(0.1)
        raise RuntimeError("Timed out loading the GGUF in llama-server. Check available memory or increase LLAMA_TIMEOUT.")

    def _complete(self, payload):
        import httpx

        try:
            reply = self._client.post("/v1/chat/completions", json=payload)
        except httpx.TimeoutException as exc:
            raise RuntimeError("llama-server timed out. Increase LLAMA_TIMEOUT or request a shorter response.") from exc
        except httpx.RequestError as exc:
            raise RuntimeError("Could not reach the local llama-server.") from exc
        try:
            body = reply.json()
        except ValueError:
            body = None
        return reply, body

    def _context_exceeded(self, reply, body):
        if not isinstance(body, dict):
            return False
        error = body.get("error")
        if reply.status_code == 400 and isinstance(error, dict):
            if error.get("type") == "exceed_context_size_error":
                return True
            message = error.get("message")
            return (error.get("type") == "invalid_request_error" and isinstance(message, str)
                    and "exceeds the available context size" in message.lower())
        if reply.is_success:
            choices = body.get("choices")
            usage = body.get("usage")
            if (isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict)
                    and choices[0].get("finish_reason") == "length" and isinstance(usage, dict)):
                prompt = usage.get("prompt_tokens")
                generated = usage.get("completion_tokens")
                return (type(prompt) is int and type(generated) is int
                        and prompt + generated >= self.max_context_tokens - 1)
        return False

    def respond(self, messages, *, allow_tools=True):
        self.last_stats = {}
        if self._process is None or self._process.poll() is not None:
            raise RuntimeError("llama-server is not running.")
        payload = {"model": self._alias, "messages": messages, "max_tokens": self.max_tokens,
                   "temperature": 0, "stream": False}
        if allow_tools and self.tool_schemas:
            payload.update(tools=self.tool_schemas, tool_choice="auto", parallel_tool_calls=False)
        started = time.monotonic()
        reply, result = self._complete(payload)
        omitted_messages = omitted_turns = 0
        if self._context_exceeded(reply, result):
            turns = [index for index, message in enumerate(messages) if message["role"] == "user"]
            if len(turns) > 1:
                # Retry with fewer complete prior turns, keeping the current request and its tool results.
                for omitted_turns in sorted({len(turns) // 2, len(turns) - 1}):
                    omitted_messages = turns[omitted_turns] - turns[0]
                    payload["messages"] = messages[:turns[0]] + messages[turns[omitted_turns]:]
                    reply, result = self._complete(payload)
                    if not self._context_exceeded(reply, result):
                        break
            if self._context_exceeded(reply, result):
                raise ValueError("The current request and its tool results exceed the llama.cpp context window. "
                                 "Narrow the task or increase LLAMA_CONTEXT_SIZE.")
        if not reply.is_success:
            raise RuntimeError(f"llama-server rejected the completion (HTTP {reply.status_code}). "
                               "Check the model's chat template and context budget.")
        try:
            choice, = result["choices"]
            finish = choice["finish_reason"]
            message = choice["message"]
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError("llama-server returned an invalid chat completion. No tool was executed.") from exc
        if finish == "length":
            raise ValueError("llama-server reached its output or context limit. "
                             "Partial text and tool calls were discarded. Adjust LLAMA_MAX_TOKENS or LLAMA_CONTEXT_SIZE.")
        try:
            text = message.get("content")
            calls = message.get("tool_calls")
            valid_calls = _validate_calls([] if calls is None else calls)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError("llama-server returned an invalid chat completion. No tool was executed.") from exc
        if finish not in ("stop", "tool_calls") or (finish == "tool_calls" and not valid_calls):
            raise ValueError(f"llama-server did not complete normally (finish_reason={finish!r}).")
        if message.get("function_call") is not None or (valid_calls and "tools" not in payload):
            raise ValueError("llama-server returned an unexpected tool call. No tool was executed.")
        if (text is not None and not isinstance(text, str)) or (not valid_calls and not (text or "").strip()):
            raise ValueError("llama-server returned neither usable text nor tool calls.")

        usage = result.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        timings = result.get("timings")
        timings = timings if isinstance(timings, dict) else {}
        self.last_stats = {"prompt_tokens": usage.get("prompt_tokens"),
                           "generation_tokens": usage.get("completion_tokens"),
                           "generation_seconds": time.monotonic() - started, "speed_kind": "request"}
        if omitted_turns:
            self.last_stats.update(history_turns_omitted=omitted_turns, history_messages_omitted=omitted_messages)
        decoded = timings.get("predicted_n")
        milliseconds = timings.get("predicted_ms")
        if type(decoded) is int and decoded > 0 and type(milliseconds) in (int, float) and milliseconds > 0:
            self.last_stats.update(generation_tokens=decoded, generation_seconds=milliseconds / 1000,
                                   speed_kind="generation")
        return ModelResponse.from_parts(valid_calls, text or "", self.last_stats, finish)

    def close(self):
        client, self._client = self._client, None
        process, self._process = self._process, None
        try:
            if client is not None:
                client.close()
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def _validate_calls(calls):
    if not isinstance(calls, list):
        raise ValueError("Expected a list of tool calls.")
    seen = set()
    for call in calls:
        call_id = call["id"]
        function = call["function"]
        name = function["name"]
        if (call["type"] != "function" or not isinstance(call_id, str) or not call_id
                or call_id in seen or not isinstance(name, str) or not name
                or not isinstance(json.loads(function["arguments"]), dict)):
            raise ValueError("Invalid or duplicate tool call.")
        seen.add(call_id)
    return calls
