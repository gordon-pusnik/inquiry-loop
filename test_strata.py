"""Strata adapter tests with SDK doubles; no network, API keys, or GPU needed."""

import contextlib
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import agent
from strata_eng import StrataModel


def completion(text="done", calls=None, finish=None):
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=finish or ("tool_calls" if calls else "stop"),
        message=SimpleNamespace(content=text, tool_calls=calls))])


def tool_call(name, args, call_id="call_test"):
    return SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(
        name=name, arguments=json.dumps(args)))


class APIError(Exception):
    pass


class APIConnectionError(APIError):
    pass


class APITimeoutError(APIConnectionError):
    pass


class APIStatusError(APIError):
    def __init__(self, code):
        self.status_code = code
        self.body = {"error": {"message": "context overflow"}}


class StrataTests(unittest.TestCase):
    def setUp(self):
        self.client = Mock()
        self.client.models.list.return_value = SimpleNamespace(data=[SimpleNamespace(id="gyro-s")])
        self.client.chat.completions.create.return_value = completion()
        self.sdk = SimpleNamespace(
            OpenAI=Mock(return_value=self.client), APIError=APIError,
            APIConnectionError=APIConnectionError, APITimeoutError=APITimeoutError,
            APIStatusError=APIStatusError,
        )
        self.httpx = SimpleNamespace(Client=Mock())
        self.dotenv = SimpleNamespace(load_dotenv=Mock())
        env = patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        modules = patch.dict(sys.modules, {"openai": self.sdk, "httpx": self.httpx, "dotenv": self.dotenv})
        modules.start()
        self.addCleanup(modules.stop)

    def make_model(self, **options):
        model = StrataModel(tool_schemas=agent.TOOL_SCHEMAS, **options)
        self.addCleanup(model.close)
        return model

    def test_defaults_discover_model_and_isolate_cloud_settings(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "cloud-secret", "OPENAI_BASE_URL": "https://cloud.invalid/v1",
                                     "OPENAI_ORG_ID": "cloud-org", "OPENAI_PROJECT_ID": "cloud-project"}):
            model = self.make_model()
        self.assertEqual(model.model, "gyro-s")
        self.assertEqual(model.max_tokens, 2048)
        model.respond([])
        self.assertEqual(self.client.chat.completions.create.call_args.kwargs["max_tokens"], 2048)
        options = self.sdk.OpenAI.call_args.kwargs
        self.assertEqual(options["base_url"], "http://127.0.0.1:8080/v1")
        self.assertEqual(options["api_key"], "strata-local")
        self.assertEqual(options["organization"], "")
        self.assertEqual(options["project"], "")
        self.assertEqual(options["max_retries"], 0)
        self.httpx.Client.assert_called_once_with(trust_env=False)
        self.client.models.list.assert_called_once()

    def test_environment_configuration_and_explicit_override(self):
        settings = {"STRATA_BASE_URL": "http://localhost:9000", "STRATA_MODEL": "configured",
                    "STRATA_API_KEY": "local-key", "STRATA_TIMEOUT": "90",
                    "STRATA_MAX_TOKENS": "256", "STRATA_THINKING": "true"}
        with patch.dict(os.environ, settings):
            model = self.make_model(base_url="http://127.0.0.1:9100/v1/", model="override")
        self.assertEqual(tuple(model.respond([])), ([], "done"))
        options = self.sdk.OpenAI.call_args.kwargs
        self.assertEqual(options["base_url"], "http://127.0.0.1:9100/v1")
        self.assertEqual(options["api_key"], "local-key")
        request = self.client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["model"], "override")
        self.assertEqual(request["max_tokens"], 256)
        self.assertTrue(request["extra_body"]["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(model.timeout, 90)
        self.client.models.list.assert_not_called()
        self.assertEqual(StrataModel._base_url(settings["STRATA_BASE_URL"]), "http://localhost:9000/v1")

    def test_reports_native_decode_timings_and_clears_old_stats(self):
        response = completion()
        response.timings = {"predicted_n": 23, "predicted_ms": 2000, "predicted_per_second": 12}
        self.client.chat.completions.create.return_value = response
        model = self.make_model(model="gyro-s")
        model.respond([])
        self.assertEqual(model.last_stats, {
            "generation_tokens": 23, "generation_seconds": 2, "generation_tps": 12})
        self.client.chat.completions.create.return_value = completion()
        model.respond([])
        self.assertEqual(model.last_stats, {})

    def test_model_discovery_empty_ambiguous_and_invalid(self):
        for data, message in (([], "no models"), ([SimpleNamespace(id="one"), SimpleNamespace(id="two")], "multiple models"),
                              (None, "invalid model list")):
            self.client.models.list.return_value = SimpleNamespace(data=data)
            with self.subTest(data=data), self.assertRaisesRegex(ValueError, message):
                self.make_model()
            self.client.close.assert_called()

    def test_two_tool_turns_with_real_files_and_native_transcript(self):
        requests = []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.txt"
            path.write_text("The test value is 42.")
            replies = iter([
                completion(None, [tool_call("list_dir", {"path": directory}, "c1")]),
                completion(None, [tool_call("read_file", {"path": str(path)}, "c2")]),
                completion("The test value is 42."),
            ])

            def respond(**kwargs):
                requests.append(copy.deepcopy(kwargs))
                return next(replies)

            self.client.chat.completions.create.side_effect = respond
            model = self.make_model(model="gyro-s")
            self.assertEqual(agent.run_agent("List then read", model, verbose=False), "The test value is 42.")
        self.assertEqual(json.loads(requests[1]["messages"][-1]["content"])["entries"], ["example.txt"])
        last = requests[2]["messages"][-1]
        self.assertEqual(last["tool_call_id"], "c2")
        self.assertEqual(json.loads(last["content"])["content"], "The test value is 42.")
        self.assertEqual(requests[0]["tools"], agent.TOOL_SCHEMAS)
        self.assertEqual(requests[0]["tool_choice"], "auto")
        self.assertFalse(requests[0]["parallel_tool_calls"])
        self.assertFalse(requests[0]["stream"])
        self.assertFalse(requests[0]["extra_body"]["chat_template_kwargs"]["enable_thinking"])

    def test_http_failures_are_actionable_and_not_retried(self):
        model = self.make_model(model="gyro-s")
        for status, message in ((401, "authentication"), (403, "authentication"), (404, "not found"),
                                (400, "context overflow"), (500, "HTTP 500")):
            self.client.chat.completions.create.reset_mock()
            self.client.chat.completions.create.side_effect = APIStatusError(status)
            with self.subTest(status=status), self.assertRaisesRegex(RuntimeError, message):
                model.respond([])
            self.client.chat.completions.create.assert_called_once()

    def test_discovery_connection_and_timeout_failures(self):
        for error, message in ((APIConnectionError("offline"), "Cannot connect"),
                               (APITimeoutError("slow"), "timed out")):
            self.client.models.list.side_effect = error
            with self.subTest(error=error), self.assertRaisesRegex(RuntimeError, message):
                self.make_model()

    def test_reject_partial_empty_or_malformed_completions(self):
        model = self.make_model(model="gyro-s")
        self.client.chat.completions.create.return_value = completion("unfinished assessment", [tool_call("write_file", {"path": "unused", "content": "partial"})], finish="length")
        with self.assertRaisesRegex(ValueError, r"2048 tokens.*no tool calls.*restart"):
            model.respond([])
        invalid_call = tool_call("read_file", {})
        invalid_call.function.arguments = "{"
        for reply in (completion(finish="length"), completion(finish="content_filter"),
                      completion(""), SimpleNamespace(choices=[]), completion(None, [invalid_call]),
                      completion(None, [tool_call("list_dir", [])]),
                      completion(None, [tool_call("list_dir", {}), tool_call("read_file", {})]),
                      completion("text", finish="tool_calls"),
                      SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=None)])):
            self.client.chat.completions.create.return_value = reply
            with self.subTest(reply=reply), self.assertRaises(ValueError):
                model.respond([])

    def test_invalid_configuration(self):
        for url in ("ftp://localhost/v1", "http://name:secret@localhost/v1", "http://localhost/v1?key=secret",
                    "http://localhost/v1/models", "http://localhost:invalid/v1"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                StrataModel._base_url(url)
        for options in ({"timeout": 0}, {"timeout": float("nan")}, {"max_tokens": 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.make_model(model="gyro-s", **options)

    def test_cli_factory_and_strata_only_options(self):
        with patch("strata_eng.StrataModel") as backend:
            self.assertIs(agent.create_model("strata", base_url="http://localhost:9000/v1", model="gyro-s"), backend.return_value)
            backend.assert_called_once_with(tool_schemas=agent.TOOL_SCHEMAS, base_url="http://localhost:9000/v1", model="gyro-s")
        with patch("agent.create_model") as factory, patch("agent.run_agent", return_value="done"), \
                contextlib.redirect_stdout(io.StringIO()):
            agent.main(["--no-trace", "--llm=strata", "--base-url=http://localhost:9000/v1", "--model=gyro-s", "hello"])
            factory.assert_called_once_with("strata", stub=False, model_path=None,
                                            base_url="http://localhost:9000/v1", model="gyro-s")
            factory.return_value.close.assert_called_once()
        for argv in (["--base-url=http://localhost/v1"], ["--llm=airllm", "--model=gyro-s"],
                     ["--llm=strata", "--model-path=/model"], ["--llm=strata", "--stub", "--model=gyro-s"]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                agent.main(argv)
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
