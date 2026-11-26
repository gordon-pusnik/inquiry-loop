import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import httpx

from inquiry_loop import cli
from benchmarks import benchmark_research
from inquiry_loop.providers import jev_eng
from inquiry_loop import research
from inquiry_loop.request_metrics import RequestMetrics
from benchmarks.fixtures.research_cases import CASES, GPT_NAMES
from inquiry_loop.response_types import ModelResponse, RunContext, RunStatus
from inquiry_loop.session_store import SessionStore


QUESTION = "Which gpt-oss models did OpenAI announce?"
OBSERVATION = {"status": "ok", "provider": "tavily", "results": [GPT_NAMES]}


class ResearchTests(unittest.TestCase):
    def test_shadow_keeps_baseline_and_ignores_prior_history(self):
        outputs, prompts = [], []
        for judge in (None, Mock(return_value={"status": "ok", "label": "insufficient"}),
                      Mock(return_value={"status": "error", "error_code": "timeout"})):
            events = []
            context = RunContext(previous_messages=[{"role": "user", "content": "private prior turn"}],
                                 emit=lambda kind, payload: events.append((kind, payload)))
            search = Mock(return_value=OBSERVATION)
            model = Mock()
            model.respond.return_value = ModelResponse([], "gpt-oss-120b and gpt-oss-20b [S1].", {})
            metrics = RequestMetrics()
            outputs.append(research.run_research(QUESTION, model, judge=judge, search_fn=search,
                                                 context=context, metrics=metrics, as_of="2025-08-05"))
            prompts.append(model.respond.call_args.args[0])
            search.assert_called_once_with(QUESTION, max_results=5, provider="tavily")
            self.assertEqual(context.status, RunStatus.COMPLETED)
            self.assertEqual(context.messages[0]["content"], "private prior turn")
            self.assertEqual(metrics.calls, 1)
            if judge is not None:
                judge.assert_called_once_with(QUESTION, "2025-08-05", [{"id": "S1", **GPT_NAMES}])
                self.assertEqual(dict(events)["evidence_check"]["route"], "baseline")
            else:
                self.assertNotIn("evidence_check", dict(events))
        self.assertEqual(outputs, [outputs[0]] * 3)
        self.assertEqual(prompts, [prompts[0]] * 3)
        self.assertNotIn("private prior turn", str(prompts))
        self.assertIn(GPT_NAMES["url"], outputs[0])

    def test_search_failure_and_empty_results_skip_judge_and_model(self):
        model, judge = Mock(), Mock()
        events = []
        context = RunContext(emit=lambda kind, payload: events.append((kind, payload)))
        answer = research.run_research(QUESTION, model, judge=judge, context=context,
                                       search_fn=Mock(return_value={"status": "no_results"}))
        self.assertIn("no usable evidence", answer)
        self.assertEqual(dict(events)["evidence_check"]["status"], "skipped")
        with self.assertRaisesRegex(RuntimeError, "Tavily search failed.*auth"):
            research.run_research(QUESTION, model, judge=judge, search_fn=Mock(return_value={
                "status": "error", "error_code": "auth", "error": "Check key."}))
        judge.assert_not_called()
        model.respond.assert_not_called()

    def test_synthesis_rejects_unknown_citations_urls_and_tool_calls(self):
        model = Mock()
        search = Mock(return_value=OBSERVATION)
        for text in ("No citations", "Wrong source [S2].", "[S1] https://example.org", "[S1] [other](/tmp)"):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "citation IDs"):
                model.respond.return_value = ModelResponse([], text, {})
                research.run_research(QUESTION, model, search_fn=search)
        model.respond.return_value = ModelResponse([{"name": "write_file"}], "", {})
        with self.assertRaisesRegex(ValueError, "No tool was executed"):
            research.run_research(QUESTION, model, search_fn=search)

    def test_cli_records_shadow_and_search_events(self):
        model = Mock()
        model.respond.return_value = ModelResponse([], "The two models are named here [S1].", {})
        judge = Mock()
        judge.judge.return_value = {"status": "ok", "label": "partial", "model": "jev-example",
                                   "probabilities": {"supported": 0.26, "partial": 0.73, "insufficient": 0.01}}
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "inquiry_loop.research.sqlite3")
            real_runner = research.run_research
            with (patch("inquiry_loop.cli.create_model", return_value=model), patch("inquiry_loop.providers.jev_eng.JevClient", return_value=judge),
                  patch("inquiry_loop.research.run_research", side_effect=lambda *a, **kw: real_runner(
                      *a, **kw, search_fn=Mock(return_value=OBSERVATION)))):
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(cli.main(["--research", "--db", path, QUESTION]), 0)
                with contextlib.redirect_stdout(io.StringIO()) as no_trace_output:
                    self.assertEqual(cli.main(["--research", "--evidence-check", "shadow", "--no-trace", QUESTION]), 0)
            store = SessionStore(path)
            try:
                run, = store.runs(store.list_sessions()[0]["id"])
                events = {row["kind"]: json.loads(row["payload_json"]) for row in store.events(run["id"])}
            finally:
                store.close()
        self.assertEqual(events["evidence_check"]["label"], "partial")
        self.assertEqual(events["tool_result"]["observation"], OBSERVATION)
        self.assertIn("[completed]", output.getvalue())
        for console in (output, no_trace_output):
            text = console.getvalue()
            self.assertIn("Sources\n\n", text)
            self.assertEqual(text.count("[step 1] search_web("), 1)
            self.assertLess(text.index("[step 1] search_web("), text.index("Sources"))
            self.assertIn("Jev evidence check (shadow) — partial (73%, jev-example)", text)
            self.assertLess(text.index("Sources"), text.index("Jev evidence check"))

    def test_interactive_research_keeps_session_and_new_restores_default(self):
        model = Mock()
        model.respond.side_effect = [ModelResponse([], text, {}) for text in (
            "Normal answer", "Evidence [S1].", "More evidence [S1].", "Fresh answer")]
        judge = Mock()
        judge.judge.return_value = {"status": "ok", "label": "partial", "model": "jev-example",
                                   "probabilities": {"supported": 0.26, "partial": 0.73, "insufficient": 0.01}}
        progress = []

        def verify_progress(query, **kwargs):
            arguments = {"query": query}
            progress.append(f"[step 1] search_web({arguments})" in output.getvalue())
            return OBSERVATION

        search = Mock(side_effect=verify_progress)
        real_runner = research.run_research
        commands = ["private prior turn", "/research", QUESTION, "What was announced?", "/new", "fresh task", "/quit"]
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "inquiry_loop.research.sqlite3")
            with (patch("inquiry_loop.cli.create_model", return_value=model), patch("inquiry_loop.providers.jev_eng.JevClient", return_value=judge),
                  patch("inquiry_loop.research.run_research", side_effect=lambda *a, **kw: real_runner(*a, **kw, search_fn=search)),
                  patch("builtins.input", side_effect=commands), contextlib.redirect_stdout(io.StringIO()) as output):
                self.assertEqual(cli.main(["--interactive", "--db", path]), 0)
            store = SessionStore(path)
            try:
                sessions = {session["title"]: session["id"] for session in store.list_sessions()}
                original, fresh = sessions["private prior turn"], sessions["fresh task"]
                original_runs = store.runs(original)
                self.assertEqual([run["request"] for run in original_runs], commands[:1] + commands[2:4])
                self.assertEqual([run["request"] for run in store.runs(fresh)], ["fresh task"])
                history = json.loads(store.get_session(original)["history_json"])
                self.assertEqual([message["content"] for message in history if message["role"] == "user"],
                                 [run["request"] for run in original_runs])
                for run in original_runs[1:]:
                    self.assertIn("evidence_check", [event["kind"] for event in store.events(run["id"])])
            finally:
                store.close()
        self.assertEqual(search.call_count, 2)
        self.assertEqual(progress, [True, True])
        self.assertEqual(output.getvalue().count("[step 1] search_web("), 2)
        self.assertEqual(judge.judge.call_count, 2)
        self.assertNotIn("private prior turn", str(judge.judge.call_args_list))
        self.assertNotIn("private prior turn", str(model.respond.call_args_list[1:]))
        self.assertEqual(output.getvalue().count("partial (73%, jev-example)\nopenai processed request"), 2)


class JevTests(unittest.TestCase):
    @patch("inquiry_loop.providers.jev_eng._setting", side_effect=lambda name: {"TYPESAFE_API_KEY": "test-only", "JEV_MODEL": ""}[name])
    @patch("httpx.post")
    def test_http_contract_and_sanitized_failures(self, post, setting):
        response = httpx.Response(200, request=httpx.Request("POST", jev_eng.ENDPOINT), json={
            "model": "jev-example", "answers": {jev_eng.QUESTION_ID: {
                "type": "choice", "choice": "supported",
                "probabilities": {"supported": 0.9, "partial": 0.08, "insufficient": 0.02},
            }}, "usage": {"input_tokens": 75, "output_tokens": 3},
        })
        post.return_value = response
        client = jev_eng.JevClient()
        result = client.judge(QUESTION, "2025-08-05", [GPT_NAMES])
        self.assertEqual((result["model"], result["label"], result["usage"]["input_tokens"]),
                         ("jev-example", "supported", 75))
        args, kwargs = post.call_args
        self.assertEqual(args, (jev_eng.ENDPOINT,))
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-only")
        self.assertFalse(kwargs["follow_redirects"])
        self.assertEqual(kwargs["json"]["state"], {"question": QUESTION, "as_of": "2025-08-05",
                                                   "public_excerpts": [GPT_NAMES]})
        post.return_value = httpx.Response(401, request=response.request, text="test-only")
        self.assertEqual(client.judge(QUESTION, "2025-08-05", []), {"status": "error", "error_code": "auth"})
        post.return_value = httpx.Response(200, request=response.request, json={"answers": []})
        self.assertEqual(client.judge(QUESTION, "2025-08-05", [GPT_NAMES])["error_code"], "invalid_response")
        post.side_effect = httpx.ReadTimeout("test-only")
        self.assertEqual(client.judge(QUESTION, "2025-08-05", [GPT_NAMES]), {"status": "error", "error_code": "timeout"})

    def test_presence_report_scores_frozen_cases_without_calls(self):
        report = benchmark_research.replay(CASES, benchmark_research.presence_baseline, evaluator="source-presence")
        self.assertEqual(report["summary"]["cases"], 20)
        self.assertEqual(report["summary"]["skipped_no_evidence"], 2)
        self.assertEqual(report["summary"]["false_supported"], 11)
        self.assertEqual(report["by_split"]["held-out"]["cases"], 12)
