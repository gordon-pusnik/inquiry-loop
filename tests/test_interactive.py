import contextlib
import copy
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

from inquiry_loop import agent, cli
from inquiry_loop.session_store import SessionStore


class ReusableModel:
    def __init__(self):
        self.histories = []
        self.tool_permissions = []
        self.close = Mock()

    def respond(self, messages, *, allow_tools=True):
        self.histories.append(copy.deepcopy(messages))
        self.tool_permissions.append(allow_tools)
        self.last_stats = {"generation_tokens": 10, "generation_tps": 5}
        return [], f"Answer to {messages[-1]['content']}"


class InteractiveTests(unittest.TestCase):
    def test_load_once_fresh_tasks_and_separate_timing(self):
        model = ReusableModel()
        output = io.StringIO()
        with patch("inquiry_loop.cli.create_model", return_value=model) as factory, \
                patch("builtins.input", side_effect=["", "first task", "second task", "/quit"]) as prompt, \
                patch("inquiry_loop.cli.time.perf_counter", side_effect=[0, 5, 100, 102, 200, 203]), \
                contextlib.redirect_stdout(output):
            self.assertEqual(cli.main(["--no-trace", "--llm=mlx", "--interactive"]), 0)
        factory.assert_called_once()
        self.assertEqual(prompt.call_args_list, [call("You> ")] * 4)
        model.close.assert_called_once()
        self.assertEqual(len(model.histories), 2)
        self.assertEqual([len(h) for h in model.histories], [2, 2])
        self.assertEqual(model.histories[1][-1]["content"], "second task")
        text = output.getvalue()
        self.assertIn("Initialized backend in 5.00 seconds.", text)
        self.assertIn("mlx processed request in 2.00 seconds, tokens per second = 5.00", text)
        self.assertIn("mlx processed request in 3.00 seconds, tokens per second = 5.00", text)
        self.assertEqual(text.count("processed request"), 2)

    def test_web_search_is_printed_before_execution(self):
        model = ReusableModel()
        call = {"id": "search", "function": {"name": "search_web", "arguments": '{"query":"test query"}'}}
        model.respond = Mock(side_effect=[([call], ""), ([], "Search answer.")])
        output = io.StringIO()
        step = "[step 1] search_web({'query': 'test query'})"

        observed = []

        def search(query):
            observed.append((query, step in output.getvalue()))
            return {"results": []}

        with tempfile.TemporaryDirectory() as directory, \
                patch("inquiry_loop.cli.create_model", return_value=model), \
                patch.dict(agent.TOOLS, {"search_web": search}), \
                patch("builtins.input", side_effect=["search for test query", "/quit"]), \
                contextlib.redirect_stdout(output):
            self.assertEqual(cli.main(["--interactive", "--db", str(Path(directory) / "trace.sqlite3")]), 0)
        self.assertEqual(observed, [("test query", True)])
        self.assertEqual(output.getvalue().count(step), 1)
        self.assertIn("Search answer.", output.getvalue())

    def test_optional_initial_task_runs_before_input(self):
        model = ReusableModel()
        with patch("inquiry_loop.cli.create_model", return_value=model), \
                patch("builtins.input", side_effect=["/exit"]), \
                contextlib.redirect_stdout(io.StringIO()):
            cli.main(["--no-trace", "-i", "--llm=mlx", "initial task"])
        self.assertEqual(model.histories[0][-1]["content"], "initial task")
        model.close.assert_called_once()

    def test_status_shows_active_model_without_calling_it(self):
        model = ReusableModel()
        model.model_id = Path("/models/example.gguf")
        commands = ["/status extra", "/status", "/quit"]
        output = io.StringIO()
        with (patch("inquiry_loop.cli.create_model", return_value=model),
              patch("builtins.input", side_effect=commands),
              contextlib.redirect_stdout(output)):
            self.assertEqual(cli.main(["--llm=llama", "--no-trace", "--interactive"]), 0)
        self.assertIn("Error: Usage: /status", output.getvalue())
        self.assertIn("Engine: llama\nModel: /models/example.gguf", output.getvalue())
        self.assertEqual(model.histories, [])

    def test_model_switch_reuses_parked_local_and_keeps_history_separate(self):
        mlx, hosted = ReusableModel(), ReusableModel()
        mlx.model_id, hosted.model = "local-model", "hosted-model"
        commands = ["local first", "/model openai", "hosted first", "/model mlx",
                    "local second", "/status", "/quit"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sessions.sqlite3"
            with (patch("inquiry_loop.cli.create_model", side_effect=[mlx, hosted]) as factory,
                  patch("builtins.input", side_effect=commands), contextlib.redirect_stdout(io.StringIO()) as output):
                self.assertEqual(cli.main(["--llm=mlx", "--db", str(path), "--interactive"]), 0)
            with contextlib.closing(SessionStore(path)) as store:
                sessions = {row["backend"]: row for row in store.list_sessions()}
                self.assertEqual(len(store.runs(sessions["mlx"]["id"])), 2)
                self.assertEqual(len(store.runs(sessions["openai"]["id"])), 1)
        self.assertEqual(factory.call_args_list, [call("mlx", stub=False, model_path=None, model=None), call("openai")])
        self.assertNotIn("local first", str(hosted.histories))
        self.assertNotIn("hosted first", str(mlx.histories))
        self.assertIn("Answer to local first", str(mlx.histories[1]))
        self.assertIn("Engine: mlx\nModel: local-model", output.getvalue())
        mlx.close.assert_called_once()
        hosted.close.assert_called_once()

    def test_chat_and_code_histories_are_isolated_across_modes_and_backends(self):
        mlx, hosted = ReusableModel(), ReusableModel()
        mlx.model_id, hosted.model = "local-model", "hosted-model"
        hosted.reasoning_effort = "medium"
        commands = ["local code first", "/chat", "local chat first", "/model openai", "/status",
                    "hosted chat first", "/model mlx", "local chat second", "/code", "local code second", "/quit"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sessions.sqlite3"
            with (patch("inquiry_loop.cli.create_model", side_effect=[mlx, hosted]),
                  patch("builtins.input", side_effect=commands), contextlib.redirect_stdout(io.StringIO()) as output):
                self.assertEqual(cli.main(["--llm=mlx", "--db", str(path), "--interactive"]), 0)
            with contextlib.closing(SessionStore(path)) as store:
                sessions = {(row["backend"], row["mode"]): row["id"] for row in store.list_sessions()}
                self.assertEqual({key: len(store.runs(sid)) for key, sid in sessions.items()},
                                 {("mlx", "code"): 2, ("mlx", "chat"): 2, ("openai", "chat"): 1})
                hosted_run, = store.runs(sessions["openai", "chat"])
                event = store.events(hosted_run["id"])[0]
                self.assertIn('"reasoning_effort": "medium"', event["payload_json"])
        self.assertEqual(mlx.tool_permissions, [True, False, False, True])
        self.assertEqual(hosted.tool_permissions, [False])
        self.assertEqual(hosted.histories, [[{"role": "user", "content": "hosted chat first"}]])
        self.assertEqual([message["content"] for message in mlx.histories[2]],
                         ["local chat first", "Answer to local chat first", "local chat second"])
        self.assertIn("local code first", str(mlx.histories[3]))
        self.assertNotIn("local chat", str(mlx.histories[3]))
        self.assertIn("Engine: openai\nModel: hosted-model\nMode: chat", output.getvalue())

    def test_chat_without_tracing_starts_each_turn_fresh(self):
        model = ReusableModel()
        with (patch("inquiry_loop.cli.create_model", return_value=model),
              patch("builtins.input", side_effect=["/chat", "first", "second", "/quit"]),
              contextlib.redirect_stdout(io.StringIO())):
            self.assertEqual(cli.main(["--llm=mlx", "--no-trace", "--interactive"]), 0)
        self.assertEqual(model.tool_permissions, [False, False])
        self.assertEqual(model.histories, [[{"role": "user", "content": text}] for text in ("first", "second")])

    def test_resume_restores_chat_mode_in_interactive_and_one_shot_cli(self):
        model = ReusableModel()
        model.model_id = "local-model"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sessions.sqlite3"
            args = ["--llm=mlx", "--db", str(path)]
            with (patch("inquiry_loop.cli.create_model", return_value=model),
                  patch("builtins.input", side_effect=["/chat", "first", "/quit", "/status", "second", "/quit"]),
                  contextlib.redirect_stdout(io.StringIO()) as output):
                self.assertEqual(cli.main(args + ["--interactive"]), 0)
                with contextlib.closing(SessionStore(path)) as store:
                    session, = store.list_sessions()
                resume = args + ["--resume", session["id"]]
                self.assertEqual(cli.main(resume + ["--interactive"]), 0)
                self.assertEqual(cli.main(resume + ["third"]), 0)
        self.assertIn("Mode: chat", output.getvalue())
        self.assertEqual(model.tool_permissions, [False, False, False])
        self.assertEqual([message["content"] for message in model.histories[2]],
                         ["first", "Answer to first", "second", "Answer to second", "third"])

    def test_local_switch_unloads_previous_backend_before_loading_and_keeps_research(self):
        mlx, llama = ReusableModel(), ReusableModel()
        commands = ["/research off", "/model llama", "public question", "/quit"]

        def load(backend, **options):
            if backend == "mlx":
                return mlx
            mlx.close.assert_called_once()
            return llama

        with (patch("inquiry_loop.cli.create_model", side_effect=load),
              patch("inquiry_loop.backends.llama_eng.resolve_server") as validate,
              patch("inquiry_loop.research.run_research", return_value="research answer") as research,
              patch("builtins.input", side_effect=commands),
              contextlib.redirect_stdout(io.StringIO()) as output):
            self.assertEqual(cli.main(["--llm=mlx", "--no-trace", "--interactive"]), 0)
        validate.assert_called_once()
        self.assertEqual(research.call_args.args, ("public question", llama))
        self.assertIn("llama processed request", output.getvalue())
        llama.close.assert_called_once()
        self.assertEqual(mlx.histories, [])

    def test_failed_local_load_restores_launch_model_and_reports_double_failure(self):
        for restore in (True, False):
            with self.subTest(restore=restore):
                original, reopened = ReusableModel(), ReusableModel()
                commands = ["/model llama", "/status", "after failure", "/quit"]
                restore_result = reopened if restore else RuntimeError("restore failed")
                with (patch("inquiry_loop.cli.create_model",
                            side_effect=[original, RuntimeError("load failed"), restore_result]) as factory,
                      patch("inquiry_loop.backends.llama_eng.resolve_server"),
                      patch("builtins.input", side_effect=commands),
                      contextlib.redirect_stdout(io.StringIO()) as output):
                    self.assertEqual(cli.main(["--llm=mlx", "--model", "launch-model", "--no-trace",
                                               "--interactive"]), 0)
                self.assertEqual(factory.call_args_list[-1], call("mlx", model="launch-model"))
                original.close.assert_called_once()
                self.assertIn("load failed", output.getvalue())
                if restore:
                    self.assertIn("Restored mlx", output.getvalue())
                    self.assertEqual(len(reopened.histories), 1)
                    reopened.close.assert_called_once()
                else:
                    self.assertIn("No active model", output.getvalue())
                    self.assertIn("Could not restore mlx", output.getvalue())

    def test_invalid_gguf_does_not_unload_current_model(self):
        mlx = ReusableModel()
        with (patch("inquiry_loop.cli.create_model", return_value=mlx) as factory,
              patch("inquiry_loop.backends.llama_eng.resolve_server", side_effect=ValueError("missing GGUF")),
              patch("builtins.input", side_effect=["/model llama", "keep going", "/quit"]),
              contextlib.redirect_stdout(io.StringIO()) as output):
            self.assertEqual(cli.main(["--llm=mlx", "--no-trace", "--interactive"]), 0)
        factory.assert_called_once()
        self.assertIn("missing GGUF", output.getvalue())
        self.assertEqual(len(mlx.histories), 1)
        mlx.close.assert_called_once()

    def test_failed_local_switch_keeps_active_hosted_engine(self):
        mlx, hosted = ReusableModel(), ReusableModel()
        commands = ["/model openai", "/model llama", "/status", "keep going", "/quit"]
        with (patch("inquiry_loop.cli.create_model",
                    side_effect=[mlx, hosted, RuntimeError("GGUF load failed")]),
              patch("inquiry_loop.backends.llama_eng.resolve_server"),
              patch("builtins.input", side_effect=commands),
              contextlib.redirect_stdout(io.StringIO()) as output):
            self.assertEqual(cli.main(["--llm=mlx", "--no-trace", "--interactive"]), 0)
        self.assertIn("GGUF load failed", output.getvalue())
        self.assertIn("Engine: openai", output.getvalue())
        self.assertEqual(len(hosted.histories), 1)
        mlx.close.assert_called_once()
        hosted.close.assert_called_once()

    def test_local_switch_retries_cleanup_before_loading_another_model(self):
        mlx, llama = ReusableModel(), ReusableModel()
        mlx.close.side_effect = [RuntimeError("cleanup failed"), None]
        commands = ["/model llama", "/status", "/model llama", "/quit"]

        def load(backend, **options):
            if backend == "mlx":
                return mlx
            self.assertEqual(mlx.close.call_count, 2)
            return llama

        with (patch("inquiry_loop.cli.create_model", side_effect=load),
              patch("inquiry_loop.backends.llama_eng.resolve_server"),
              patch("builtins.input", side_effect=commands),
              contextlib.redirect_stdout(io.StringIO()) as output):
            self.assertEqual(cli.main(["--llm=mlx", "--no-trace", "--interactive"]), 0)
        self.assertIn("cleanup failed", output.getvalue())
        self.assertIn("No active model", output.getvalue())
        llama.close.assert_called_once()

    def test_hub_commands_do_not_call_the_model(self):
        model = ReusableModel()
        output = io.StringIO()
        commands = ["/search Qwen3-30B-A3B-Instruct-2507", "/download mlx-community/example", "/quit"]
        with (patch("builtins.input", side_effect=commands),
              patch("inquiry_loop.providers.hub_models.HfApi") as hub,
              contextlib.redirect_stdout(output)):
            api = hub.return_value
            api.list_models.side_effect = [
                [SimpleNamespace(id="mlx-community/example")],
                [SimpleNamespace(id="owner/example-GGUF")],
            ]
            api.model_info.return_value = SimpleNamespace(sha="revision", siblings=[
                SimpleNamespace(rfilename="model.safetensors"),
            ])
            api.snapshot_download.return_value = "/cache/snapshots/revision"
            cli.run_interactive(model, backend="stub")
        self.assertEqual(api.list_models.call_args_list, [
            call(search="Qwen3-30B-A3B-Instruct-2507", filter="mlx", limit=5, sort="downloads"),
            call(search="Qwen3-30B-A3B-Instruct-2507", filter="gguf", limit=5, sort="downloads"),
        ])
        api.snapshot_download.assert_called_once_with("mlx-community/example", revision="revision")
        self.assertIn("1. mlx-community/example (mlx)", output.getvalue())
        self.assertIn("2. owner/example-GGUF (gguf)", output.getvalue())
        self.assertIn("Cached model: /cache/snapshots/revision", output.getvalue())
        self.assertEqual(model.histories, [])

    def test_request_failure_does_not_end_session_or_reuse_metrics(self):
        model = ReusableModel()
        output = io.StringIO()
        model.respond = Mock(side_effect=[ValueError("bad response"), ([], "recovered")])
        with patch("inquiry_loop.cli.create_model", return_value=model), \
                patch("builtins.input", side_effect=["first", "second", "exit"]), \
                contextlib.redirect_stdout(output):
            cli.main(["--no-trace", "--interactive"])
        self.assertEqual(model.respond.call_count, 2)
        self.assertIn("Error: bad response", output.getvalue())
        self.assertIn("recovered", output.getvalue())
        self.assertEqual(output.getvalue().count("openai processed request"), 1)
        model.close.assert_called_once()

    def test_eof_or_interrupt_at_prompt_closes_backend(self):
        for interruption in (EOFError, KeyboardInterrupt):
            model = ReusableModel()
            with self.subTest(interruption=interruption), patch("inquiry_loop.cli.create_model", return_value=model), \
                    patch("builtins.input", side_effect=interruption), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(["--no-trace", "--interactive"]), 0)
            self.assertEqual(model.histories, [])
            model.close.assert_called_once()

    def test_interrupt_during_task_exits_and_closes_backend(self):
        model = ReusableModel()
        model.respond = Mock(side_effect=KeyboardInterrupt)
        with patch("inquiry_loop.cli.create_model", return_value=model), \
                patch("builtins.input", return_value="task"), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["--no-trace", "--interactive"]), 0)
        model.respond.assert_called_once()
        model.close.assert_called_once()

    def test_stub_replays_tools_for_each_independent_task(self):
        output = io.StringIO()
        with patch("builtins.input", side_effect=["first", "second", "quit"]), \
                contextlib.redirect_stdout(output):
            self.assertEqual(cli.main(["--no-trace", "--stub", "--interactive"]), 0)
        self.assertEqual(output.getvalue().count("[step 1] list_dir"), 2)
        self.assertEqual(output.getvalue().count("[step 2] read_file"), 2)
        self.assertEqual(output.getvalue().count("stub processed request"), 2)


if __name__ == "__main__":
    unittest.main()
