import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import httpx

from inquiry_loop import agent, cli, research
from inquiry_loop.backends.llama_eng import LlamaModel
from inquiry_loop.session_store import SessionStore


def completion(content=None, *, calls=None, finish="stop"):
    return httpx.Response(200, json={
        "choices": [{"finish_reason": finish, "message": {"content": content, "tool_calls": calls}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4},
        "timings": {"predicted_n": 4, "predicted_ms": 200},
    })


class LlamaTests(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.model_path = Path(directory, "model file.gguf")
        self.model_path.touch()
        self.db_path = Path(directory, "session.sqlite3")
        self.replies = []
        self.completions = []
        self.process = Mock()
        self.process.poll.return_value = None
        self.start_server = self.enterContext(patch(
            "inquiry_loop.backends.llama_eng.subprocess.Popen", return_value=self.process))
        self.enterContext(patch("inquiry_loop.backends.llama_eng.shutil.which", return_value="/bin/llama-server"))
        socket = self.enterContext(patch("inquiry_loop.backends.llama_eng.socket.socket"))
        socket.return_value.__enter__.return_value.getsockname.return_value = ("127.0.0.1", 43123)
        self.enterContext(patch.dict(os.environ, {"LLAMA_ARG_HOST": "0.0.0.0"}, clear=True))
        self.enterContext(patch("dotenv.load_dotenv"))
        client = httpx.Client
        self.http_client = self.enterContext(patch("httpx.Client", side_effect=lambda **kwargs: client(
            **kwargs, transport=httpx.MockTransport(self.handle_request))))

    def handle_request(self, request):
        self.assertEqual(request.url.host, "127.0.0.1")
        startup = self.start_server.call_args
        secret = startup.kwargs["env"]["LLAMA_API_KEY"]
        self.assertEqual(request.headers["Authorization"], f"Bearer {secret}")
        command = startup.args[0]
        alias = command[command.index("--alias") + 1]
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": alias}]})
        self.assertEqual(request.url.path, "/v1/chat/completions")
        payload = json.loads(request.content)
        self.assertEqual(payload["model"], alias)
        self.completions.append(payload)
        return self.replies.pop(0)

    def test_runs_cli_tool_call_and_passes_observation_back(self):
        self.replies = [completion(calls=[{
            "id": "call_directory", "type": "function",
            "function": {"name": "list_dir", "arguments": '{"path":"."}'},
        }], finish="tool_calls"), completion("Found example.txt.")]
        directory = Mock(return_value={"entries": ["example.txt"]})
        with patch.dict(agent.TOOLS, {"list_dir": directory}), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(cli.main(["--llm=llama", "--no-trace", "--model-path", str(self.model_path),
                                       "List files"]), 0)
        directory.assert_called_once_with(path=".")
        first, second = self.completions
        self.assertEqual(first["tools"], agent.TOOL_SCHEMAS)
        self.assertEqual(second["messages"][-1], {"role": "tool", "tool_call_id": "call_directory",
                                                  "content": '{"entries": ["example.txt"]}'})
        self.assertIn("Found example.txt.", output.getvalue())
        self.assertIn("tokens per second = 20.00", output.getvalue())
        startup = self.start_server.call_args
        self.assertIn("--jinja", startup.args[0])
        self.assertNotIn("LLAMA_ARG_HOST", startup.kwargs["env"])
        self.assertFalse(self.http_client.call_args.kwargs["trust_env"])
        self.process.terminate.assert_called_once()

    def test_interactive_research_enforce_omits_tools(self):
        answer = "Phoenix has a population estimate [S1]."
        self.replies = [completion('{"gap":"estimate year","query":"Phoenix census estimate year"}'),
                        completion(answer)]
        observation = {"status": "ok", "results": [
            {"title": "Census", "url": "https://example.org/census", "snippet": "A population estimate."},
        ]}
        search = Mock(return_value=observation)
        judge = Mock(configured=True)
        judge.judge.return_value = {"status": "ok", "label": "partial", "model": "jev-example",
                                   "probabilities": {"supported": 0.2, "partial": 0.8, "insufficient": 0}}
        run = research.run_research
        with (patch("builtins.input", side_effect=["/research enforce", "Population of Phoenix?", "/quit"]),
              patch("inquiry_loop.providers.jev_eng.JevClient", return_value=judge),
              patch.object(research, "run_research", side_effect=lambda *a, **k: run(*a, **k, search_fn=search)),
              patch.dict(os.environ, {"LLAMA_MODEL_PATH": str(self.model_path)}),
              contextlib.redirect_stdout(io.StringIO()) as output):
            self.assertEqual(cli.main(["--llm=llama", "--no-trace", "--interactive"]), 0)
        self.assertIn(answer, output.getvalue())
        self.assertEqual([call.args[0] for call in search.call_args_list],
                         ["Population of Phoenix?", "Phoenix census estimate year"])
        for payload in self.completions:
            self.assertNotIn("tools", payload)
            self.assertNotIn("tool_choice", payload)

    def test_bad_completions_are_rejected_before_tool_execution(self):
        model = LlamaModel(model_path=self.model_path, tool_schemas=agent.TOOL_SCHEMAS)
        self.addCleanup(model.close)
        calls = [{"type": "function", "id": "call_bad",
                  "function": {"name": "list_dir", "arguments": "not JSON"}}]
        self.replies = [completion("Incomplete", finish="length"), completion(calls=calls, finish="tool_calls")]
        with self.assertRaisesRegex(ValueError, "output or context limit"):
            model.respond([])
        with self.assertRaisesRegex(ValueError, "No tool was executed"):
            model.respond([])

    def test_unified_cli_uses_serve_with_or_without_explicit_path(self):
        for configured in (None, "~/bin/llama"):
            with self.subTest(configured=configured):
                settings = {"LLAMA_SERVER": configured} if configured else {}
                executable = str(Path("~/bin/llama").expanduser())
                paths = {"llama": executable, executable: executable}
                with (patch.dict(os.environ, settings),
                      patch("inquiry_loop.backends.llama_eng.shutil.which", side_effect=paths.get) as which):
                    model = LlamaModel(model_path=self.model_path, tool_schemas=[])
                    model.close()
                self.assertEqual(self.start_server.call_args.args[0][:3], [executable, "serve", "--model"])
                if configured:
                    which.assert_called_once_with(executable)
                else:
                    self.assertEqual([call.args for call in which.call_args_list], [("llama-server",), ("llama",)])

    def test_status_displays_gguf_path_and_resume_uses_resolved_identity(self):
        blob = self.model_path.with_name("cached-blob-hash")
        self.model_path.rename(blob)
        self.model_path.symlink_to(blob)
        self.replies = [completion("I will remember Juniper."), completion("Juniper.")]
        arguments = ["--llm=llama", "--db", str(self.db_path)]
        with (patch.dict(os.environ, {"LLAMA_MODEL_PATH": str(self.model_path)}),
              patch("builtins.input", side_effect=["/status", "/quit"]),
              contextlib.redirect_stdout(io.StringIO()) as output):
            self.assertEqual(cli.main(arguments + ["Remember Juniper"]), 0)
            with contextlib.closing(SessionStore(self.db_path)) as store:
                session, = store.list_sessions()
            self.assertEqual(cli.main(arguments + ["--resume", session["id"], "--interactive", "What did I say?"]), 0)
        first, second = self.completions
        self.assertNotEqual(first["model"], second["model"])
        self.assertEqual(session["model"], str(blob.resolve()))
        self.assertIn(f"Engine: llama\nModel: {self.model_path}", output.getvalue())
        self.assertNotIn(f"Model: {blob.resolve()}", output.getvalue())
        self.assertEqual(second["messages"][1:], [
            {"role": "user", "content": "Remember Juniper"},
            {"role": "assistant", "content": "I will remember Juniper."},
            {"role": "user", "content": "What did I say?"},
        ])

    def test_failed_start_terminates_process_and_closes_http_client(self):
        self.enterContext(patch.object(LlamaModel, "_wait_until_ready", side_effect=RuntimeError("startup failed")))
        with self.assertRaisesRegex(RuntimeError, "startup failed"):
            LlamaModel(model_path=self.model_path, tool_schemas=[])
        self.process.terminate.assert_called_once()
        self.process.wait.assert_called_once_with(timeout=5)
