"""Client for Strata's local OpenAI-compatible API; no model weights are loaded."""

from __future__ import annotations

from response_types import ModelResponse

import json
import math
import os
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"


class StrataModel:
    def __init__(self, *, tool_schemas, base_url=None, model=None, api_key=None,
                 timeout=None, max_tokens=None, enable_thinking=None, http_client=None):
        # Keep imports lazy so OpenAI, AirLLM, and stub modes stay independent.
        try:
            import httpx
            import openai
            from dotenv import load_dotenv
        except ImportError as exc:
            raise ImportError("Install the Strata client dependencies with python3 -m pip install -r requirements.txt.") from exc

        load_dotenv(Path(__file__).resolve().with_name(".env"), override=False)
        self.base_url = self._base_url(base_url or os.environ.get("STRATA_BASE_URL") or DEFAULT_BASE_URL)
        self.timeout = float(timeout if timeout is not None else os.environ.get("STRATA_TIMEOUT", "300"))
        self.max_tokens = int(max_tokens if max_tokens is not None else os.environ.get("STRATA_MAX_TOKENS", "2048"))
        if not math.isfinite(self.timeout) or self.timeout <= 0 or self.max_tokens <= 0:
            raise ValueError("Strata timeout and max_tokens must be positive finite values.")
        if enable_thinking is None:
            setting = os.environ.get("STRATA_THINKING", "false").lower()
            if setting not in ("true", "false"):
                raise ValueError("STRATA_THINKING must be true or false.")
            enable_thinking = setting == "true"
        self.enable_thinking = enable_thinking
        self.tool_schemas = tool_schemas
        self._openai = openai
        # Explicit URL/credential isolation: never inherit OPENAI_BASE_URL or
        # OPENAI_API_KEY. Local requests must not use ambient HTTP proxies.
        self.client = openai.OpenAI(
            base_url=self.base_url,
            api_key=api_key or os.environ.get("STRATA_API_KEY") or "strata-local",
            organization="", project="", max_retries=0, timeout=self.timeout,
            http_client=http_client if http_client is not None else httpx.Client(trust_env=False),
        )
        try:
            self.model = model or os.environ.get("STRATA_MODEL") or self._discover_model()
        except Exception:
            self.client.close()
            raise

    @staticmethod
    def _base_url(value):
        parts = urlsplit(value.strip())
        if (parts.scheme not in ("http", "https") or not parts.hostname
                or parts.username is not None or parts.password is not None
                or parts.query or parts.fragment):
            raise ValueError("Strata base URL must be an http(s) API address without credentials, query, or fragment.")
        # Validate the port as well; a bare server address is convenient to copy.
        _ = parts.port
        path = parts.path.rstrip("/") or "/v1"
        if not path.endswith("/v1"):
            raise ValueError("Strata base URL must end in /v1, not /models or /chat/completions.")
        return urlunsplit((parts.scheme, parts.netloc, path, "", ""))

    def _request(self, operation):
        try:
            return operation()
        except self._openai.APITimeoutError as exc:
            raise RuntimeError("Strata request timed out. Check the loaded model or increase STRATA_TIMEOUT; no automatic retry was made.") from exc
        except self._openai.APIConnectionError as exc:
            raise RuntimeError("Cannot connect to Strata. Start its API and check the Serving screen's address against --base-url or STRATA_BASE_URL.") from exc
        except self._openai.APIStatusError as exc:
            if exc.status_code in (401, 403):
                message = "Strata rejected authentication. Set STRATA_API_KEY to the key configured in its Serving screen."
            elif exc.status_code == 404:
                message = "Strata endpoint or model was not found. Check the API base URL and model ID in its Serving screen."
            else:
                # Include the server's diagnostic (e.g. context overflow) but
                # avoid SDK request headers, which may contain credentials.
                body = exc.body if isinstance(exc.body, dict) else {}
                error = body.get("error", body)
                detail = error.get("message", "") if isinstance(error, dict) else ""
                message = f"Strata returned HTTP {exc.status_code}. {str(detail)[:500]}".strip()
            raise RuntimeError(message) from exc
        except self._openai.APIError as exc:
            raise RuntimeError("Strata returned an invalid API response.") from exc

    def _discover_model(self):
        response = self._request(self.client.models.list)
        data = getattr(response, "data", None)
        if not isinstance(data, list):
            raise ValueError("Strata returned an invalid model list.")
        ids = list(dict.fromkeys(
            item.id for item in data
            if isinstance(getattr(item, "id", None), str) and item.id.strip()
        ))
        if not ids:
            raise ValueError("Strata reports no models. Load the existing Gyro-S model in Strata first.")
        if len(ids) != 1:
            raise ValueError("Strata reports multiple models. Select the Serving screen's model ID with --model or STRATA_MODEL.")
        return ids[0]

    def respond(self, messages):
        self.last_stats = {}
        response = self._request(lambda: self.client.chat.completions.create(
            model=self.model, messages=messages, tools=self.tool_schemas,
            tool_choice="auto", parallel_tool_calls=False, stream=False,
            max_tokens=self.max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": self.enable_thinking}},
        ))
        choices = getattr(response, "choices", None)
        if not isinstance(choices, list) or not choices:
            raise ValueError("Strata returned no completion choices.")
        choice = choices[0]
        finish = getattr(choice, "finish_reason", None)
        if finish == "length":
            raise ValueError(f"Strata hit its output-token limit ({self.max_tokens} tokens); the partial response was discarded and no tool calls from it were executed. Request a shorter assessment or increase STRATA_MAX_TOKENS within the server's context budget, then restart the interactive session.")
        if finish not in ("stop", "tool_calls"):
            raise ValueError(f"Strata did not complete normally (finish_reason={finish!r}).")
        message = getattr(choice, "message", None)
        raw_calls = getattr(message, "tool_calls", None)
        if raw_calls is not None and not isinstance(raw_calls, list):
            raise ValueError("Strata returned an invalid tool-call list.")
        calls = raw_calls or []
        seen = set()
        for call in calls:
            function = getattr(call, "function", None)
            call_id = getattr(call, "id", None)
            if (getattr(call, "type", None) != "function"
                    or not isinstance(call_id, str) or not call_id or call_id in seen
                    or not isinstance(getattr(function, "name", None), str) or not function.name):
                raise ValueError("Strata returned an invalid or duplicate tool call; no calls from this response were executed.")
            seen.add(call.id)
            try:
                args = json.loads(getattr(function, "arguments", None))
            except (ValueError, TypeError) as exc:
                raise ValueError("Strata returned invalid tool-argument JSON; no calls from this response were executed.") from exc
            if not isinstance(args, dict):
                raise ValueError("Strata tool arguments must be a JSON object.")
        text = getattr(message, "content", None) or ""
        if not isinstance(text, str) or (not calls and not text.strip()):
            raise ValueError("Strata returned neither usable tool calls nor an answer. Check reasoning settings and the output-token budget.")
        if finish == "tool_calls" and not calls:
            raise ValueError("Strata reported tool calls without providing any.")
        timings = getattr(response, "timings", None)
        if isinstance(timings, dict):
            milliseconds = timings.get("predicted_ms")
            self.last_stats = {
                "generation_tokens": timings.get("predicted_n"),
                "generation_tps": timings.get("predicted_per_second"),
                "generation_seconds": milliseconds / 1000 if type(milliseconds) in (int, float) else None,
            }
        usage = getattr(response, 'usage', None)
        for field, key in (('prompt_tokens', 'prompt_tokens'), ('completion_tokens', 'completion_tokens')):
            count = getattr(usage, field, None)
            if type(count) is int and count >= 0:
                self.last_stats[key] = count
        return ModelResponse.from_parts(calls, text, self.last_stats, finish)

    def close(self):
        self.client.close()
