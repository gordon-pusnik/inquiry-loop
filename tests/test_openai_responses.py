import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import httpx
from openai import OpenAI

from inquiry_loop import agent, cli, research
from inquiry_loop.engine import create_model
from inquiry_loop.session_store import SessionStore


SOL = "gpt-6.2-sol"
ASTRA = "gpt-6.2-astra"
REASONING = {"id": "rs_test", "type": "reasoning", "summary": [],
             "encrypted_content": "opaque-replay-token"}
TOOL_CALL = {"id": "fc_test", "call_id": "call_test", "type": "function_call", "status": "completed",
             "name": "list_dir", "arguments": '{"path":"."}'}


def message(text, item_id="msg_test"):
    return {"type": "message", "id": item_id, "status": "completed", "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": []}]}


class ResponsesTests(unittest.TestCase):
    def setUp(self):
        self.requests = []
        self.replies = []
        self.enterContext(patch.dict(os.environ, {"AGENT_MODEL": SOL}, clear=True))
        self.enterContext(patch("dotenv.load_dotenv"))
        self.enterContext(patch("openai.OpenAI", side_effect=lambda: OpenAI(
            api_key="test-key", base_url="https://example.test/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(self.handle_request)))))

    def handle_request(self, request):
        self.assertEqual(request.url.path, "/v1/responses")
        payload = json.loads(request.content)
        self.assertIs(payload["store"], False)
        self.requests.append(payload)
        output, options = self.replies.pop(0)
        return httpx.Response(200, json={
            "id": "resp_test", "object": "response", "created_at": 0,
            "model": payload["model"], "status": "completed", "output": output,
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            **options,
        })

    def test_code_tool_round_trip_and_resume_preserve_native_items(self):
        self.replies = [
            ([REASONING, TOOL_CALL], {}),
            ([message("Read the fixture.")], {}),
            ([message("I remember the fixture.", "msg_resumed")], {}),
        ]
        directory = Mock(return_value={"entries": ["fixture.txt"]})
        with tempfile.TemporaryDirectory() as folder, patch.dict(agent.TOOLS, {"list_dir": directory}):
            path = str(Path(folder) / "session.sqlite3")
            args = ["--db", path, "--llm=openai"]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(args + ["List this directory."]), 0)
                with contextlib.closing(SessionStore(path)) as store:
                    session, = store.list_sessions()
                    first, = store.runs(session["id"])
                    saved = store.get_session(session["id"])
                    self.assertIn("opaque-replay-token", saved["replay_items_json"])
                    self.assertNotIn("opaque-replay-token", saved["history_json"])
                    self.assertNotIn("opaque-replay-token", str(store.events(first["id"])))
                self.assertEqual(cli.main(args + ["--resume", session["id"], "What did you see?"]), 0)
        directory.assert_called_once_with(path=".")
        first, second, resumed = self.requests
        self.assertFalse(first["parallel_tool_calls"])
        self.assertTrue(all(tool["type"] == "function" and tool["strict"] is False for tool in first["tools"]))
        self.assertEqual(first["reasoning"], {"effort": "medium"})
        self.assertEqual(second["input"], first["input"] + [
            REASONING, TOOL_CALL,
            {"type": "function_call_output", "call_id": "call_test", "output": '{"entries": ["fixture.txt"]}'},
        ])
        self.assertEqual(resumed["input"], second["input"] + [
            message("Read the fixture."), {"role": "user", "content": "What did you see?"},
        ])

    def test_effort_change_stays_with_openai_after_switching(self):
        first_reply = message("First hosted reply.")
        self.replies = [([first_reply], {}), ([message("Second hosted reply.")], {})]
        local = SimpleNamespace(model_id="local", close=Mock())
        commands = ["/chat", "first", "/effort high", "/model mlx", "/effort max", "/model openai",
                    "/status", "second", "/quit"]
        with (patch.dict(os.environ, {"AGENT_MODEL": ASTRA, "OPENAI_REASONING_EFFORT": "low"}),
              patch("inquiry_loop.cli.create_model",
                    side_effect=lambda backend, **options: agent.create_model(backend, **options) if backend == "openai" else local),
              patch("builtins.input", side_effect=commands), contextlib.redirect_stdout(io.StringIO()) as output):
            self.assertEqual(cli.main(["--interactive", "--no-trace"]), 0)
        self.assertEqual([request["reasoning"]["effort"] for request in self.requests], ["low", "high"])
        self.assertTrue(all(request["model"] == ASTRA and "tools" not in request for request in self.requests))
        self.assertIn("Reasoning effort is unavailable for the active model.", output.getvalue())
        self.assertIn(f"Engine: openai\nModel: {ASTRA}\nMode: chat\nReasoning effort: high", output.getvalue())

    def test_switching_to_responses_excludes_local_tool_output(self):
        self.replies = [([message("Hosted reply.")], {})]
        local = SimpleNamespace(model_id="local-model", close=Mock(), respond=Mock(side_effect=[
            ([{"id": "local_call", "function": {"name": "list_dir", "arguments": "{}"}}], ""),
            ([], "Local code reply."),
            ([], "Local follow-up."),
        ]))
        commands = ["private local request", "/model openai", "public hosted request",
                    "/model mlx", "local follow-up", "/quit"]
        with (tempfile.TemporaryDirectory() as folder,
              patch("inquiry_loop.cli.create_model",
                    side_effect=lambda backend, **options: agent.create_model(backend, **options)
                    if backend == "openai" else local),
              patch.dict(agent.TOOLS, {"list_dir": lambda: {"entries": ["local-only-fixture.txt"]}}),
              patch("builtins.input", side_effect=commands), contextlib.redirect_stdout(io.StringIO())):
            self.assertEqual(cli.main(["--llm=mlx", "--interactive", "--db", str(Path(folder) / "sessions.db")]), 0)
        request, = self.requests
        self.assertEqual([item["content"] for item in request["input"] if item["role"] == "user"],
                         ["public hosted request"])
        self.assertNotIn("local-only-fixture.txt", str(request["input"]))
        self.assertIn("local-only-fixture.txt", str(local.respond.call_args.args[0]))

    def test_research_planner_and_synthesis_use_independent_tool_free_requests(self):
        self.replies = [
            ([message('{"gap":"estimate year","query":"Phoenix city census estimate"}')], {}),
            ([message("The city has an estimate [S1].")], {}),
        ]
        observation = {"status": "ok", "results": [
            {"title": "Census", "url": "https://example.test/census", "snippet": "A population estimate."},
        ]}
        with contextlib.closing(agent.create_model("openai")) as model:
            result = research.run_research(
                "What is the population of Phoenix?", model, enforce=True,
                judge=lambda *_: {"status": "ok", "label": "partial"},
                search_fn=Mock(return_value=observation),
            )
        self.assertIn("[S1]", result)
        planner, synthesis = self.requests
        for request in self.requests:
            self.assertNotIn("tools", request)
            self.assertEqual([item["role"] for item in request["input"]], ["developer", "user"])
        self.assertNotIn("Phoenix city census estimate", synthesis["input"][-1]["content"])
        self.assertIn("gap", planner["input"][0]["content"])

    def test_incomplete_response_does_not_dispatch_tools(self):
        self.replies = [([TOOL_CALL], {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}})]
        directory = Mock()
        with (contextlib.closing(agent.create_model("openai")) as model,
              patch.dict(agent.TOOLS, {"list_dir": directory}),
              self.assertRaisesRegex(ValueError, "incomplete result")):
            agent.run_agent("List this directory.", model, verbose=False)
        directory.assert_not_called()

    def test_unsupported_effort_is_rejected_only_for_responses_models(self):
        with patch.dict(os.environ, {"OPENAI_REASONING_EFFORT": "none"}):
            with self.assertRaisesRegex(ValueError, "OPENAI_REASONING_EFFORT"):
                create_model("openai")
            with contextlib.closing(create_model("openai", model="gpt-4o-mini")) as legacy:
                self.assertEqual(legacy.model, "gpt-4o-mini")
