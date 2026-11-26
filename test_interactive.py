import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import agent


class ReusableModel:
    def __init__(self):
        self.histories = []
        self.close = Mock()

    def respond(self, messages):
        self.histories.append(copy.deepcopy(messages))
        self.last_stats = {"generation_tokens": 10, "generation_tps": 5}
        return [], f"Answer to {messages[-1]['content']}"


class InteractiveTests(unittest.TestCase):
    def test_load_once_fresh_tasks_and_separate_timing(self):
        model = ReusableModel()
        output = io.StringIO()
        with patch("agent.create_model", return_value=model) as factory, \
                patch("builtins.input", side_effect=["", "first task", "second task", "/quit"]), \
                patch("agent.time.perf_counter", side_effect=[0, 5, 100, 102, 200, 203]), \
                contextlib.redirect_stdout(output):
            self.assertEqual(agent.main(["--no-trace", "--llm=mlx", "--interactive"]), 0)
        factory.assert_called_once()
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
                patch("agent.create_model", return_value=model), \
                patch.dict(agent.TOOLS, {"search_web": search}), \
                patch("builtins.input", side_effect=["search for test query", "/quit"]), \
                contextlib.redirect_stdout(output):
            self.assertEqual(agent.main(["--interactive", "--db", str(Path(directory) / "trace.sqlite3")]), 0)
        self.assertEqual(observed, [("test query", True)])
        self.assertEqual(output.getvalue().count(step), 1)
        self.assertIn("Search answer.", output.getvalue())

    def test_optional_initial_task_runs_before_input(self):
        model = ReusableModel()
        with patch("agent.create_model", return_value=model), \
                patch("builtins.input", side_effect=["/exit"]), \
                contextlib.redirect_stdout(io.StringIO()):
            agent.main(["--no-trace", "-i", "--llm=mlx", "initial task"])
        self.assertEqual(model.histories[0][-1]["content"], "initial task")
        model.close.assert_called_once()

    def test_request_failure_does_not_end_session_or_reuse_metrics(self):
        model = ReusableModel()
        output = io.StringIO()
        model.respond = Mock(side_effect=[ValueError("bad response"), ([], "recovered")])
        with patch("agent.create_model", return_value=model), \
                patch("builtins.input", side_effect=["first", "second", "exit"]), \
                contextlib.redirect_stdout(output):
            agent.main(["--no-trace", "--interactive"])
        self.assertEqual(model.respond.call_count, 2)
        self.assertIn("Error: bad response", output.getvalue())
        self.assertIn("recovered", output.getvalue())
        self.assertEqual(output.getvalue().count("openai processed request"), 1)
        model.close.assert_called_once()

    def test_eof_or_interrupt_at_prompt_closes_backend(self):
        for interruption in (EOFError, KeyboardInterrupt):
            model = ReusableModel()
            with self.subTest(interruption=interruption), patch("agent.create_model", return_value=model), \
                    patch("builtins.input", side_effect=interruption), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(agent.main(["--no-trace", "--interactive"]), 0)
            self.assertEqual(model.histories, [])
            model.close.assert_called_once()

    def test_interrupt_during_task_exits_and_closes_backend(self):
        model = ReusableModel()
        model.respond = Mock(side_effect=KeyboardInterrupt)
        with patch("agent.create_model", return_value=model), \
                patch("builtins.input", return_value="task"), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(agent.main(["--no-trace", "--interactive"]), 0)
        model.respond.assert_called_once()
        model.close.assert_called_once()

    def test_stub_replays_tools_for_each_independent_task(self):
        output = io.StringIO()
        with patch("builtins.input", side_effect=["first", "second", "quit"]), \
                contextlib.redirect_stdout(output):
            self.assertEqual(agent.main(["--no-trace", "--stub", "--interactive"]), 0)
        self.assertEqual(output.getvalue().count("[step 1] list_dir"), 2)
        self.assertEqual(output.getvalue().count("[step 2] read_file"), 2)
        self.assertEqual(output.getvalue().count("stub processed request"), 2)


if __name__ == "__main__":
    unittest.main()
