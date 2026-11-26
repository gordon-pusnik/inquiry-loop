import contextlib
import io
import json
import unittest
from unittest.mock import Mock, call, patch

from inquiry_loop import agent, cli
from inquiry_loop import research
from inquiry_loop.request_metrics import RequestMetrics
from inquiry_loop.response_types import ModelResponse, RunContext, RunStatus


QUESTION = "When was Example Model announced, and when did it become available?"
QUERY = "Example Model public availability release date"
AS_OF = "2025-08-05"
INITIAL = {"title": "Announcement", "url": "https://example.org/announce#date",
           "snippet": "Example Model was announced on July 1."}
NOVEL = {"title": "Release notes", "url": "https://example.org/releases#availability",
         "snippet": "Example Model became publicly available on July 15."}
EMPTY = {"status": "no_results", "provider": "tavily", "results": []}
SEARCH_ERROR = {"status": "error", "provider": "tavily", "error_code": "timeout",
                "error": "Search provider timed out", "retryable": True}
JEV_ERROR = {"status": "error", "error_code": "timeout"}
PLAN = json.dumps({"gap": "Date of public availability", "query": QUERY})
BASE_ANSWER = "Example Model was announced on July 1 [S1]."
COMBINED_ANSWER = f"{BASE_ANSWER} It became publicly available on July 15 [S2]."


def found(*sources):
    return {"status": "ok", "provider": "tavily", "results": list(sources)}


def judgment(label, supported, partial, insufficient):
    return {"status": "ok", "label": label, "model": "jev-example", "probabilities": {
        "supported": supported, "partial": partial, "insufficient": insufficient,
    }}


LOW = judgment("supported", 0.59, 0.30, 0.11)
PARTIAL = judgment("partial", 0.10, 0.80, 0.10)
INSUFFICIENT = judgment("insufficient", 0.09, 0.10, 0.81)
HIGH = judgment("supported", 0.99, 0.005, 0.005)


def model_response(text, *, tokens=30, seconds=1.0, tools=()):
    return ModelResponse(list(tools), text, {"generation_tokens": tokens, "generation_seconds": seconds})


class ResearchEnforcementTests(unittest.TestCase):
    def prepare(self, observations, judgments, *responses):
        self.search = Mock(side_effect=observations)
        self.judge = Mock(side_effect=judgments)
        self.model = Mock()
        self.model.respond.side_effect = responses
        self.events = []
        self.context = RunContext(emit=lambda kind, payload: self.events.append((kind, payload)))
        self.on_evidence = Mock()
        self.metrics = RequestMetrics()

    def research(self, max_steps=2):
        return research.run_research(
            QUESTION, self.model, judge=self.judge, enforce=True, supported_cutoff=0.60,
            search_fn=self.search, max_steps=max_steps, context=self.context,
            on_evidence=self.on_evidence, metrics=self.metrics, as_of=AS_OF,
        )

    def payloads(self, kind):
        return [payload for event, payload in self.events if event == kind]

    def assert_searches(self, *queries):
        self.assertEqual(self.search.call_args_list, [call(query, max_results=5, provider="tavily")
                                                      for query in queries])

    def assert_checks(self, *attempts):
        checks = self.payloads("evidence_check")
        self.assertEqual([(check["mode"], check["search_attempt"]) for check in checks],
                         [("enforce", attempt) for attempt in attempts])
        self.assertEqual(self.on_evidence.call_args_list, [call(check) for check in checks])
        return checks

    def assert_completed(self, answer, expected_text):
        self.assertIn(expected_text, answer)
        self.assertEqual(self.context.status, RunStatus.COMPLETED)

    def test_low_supported_partial_and_insufficient_trigger_one_followup(self):
        for initial in (LOW, PARTIAL, INSUFFICIENT):
            with self.subTest(label=initial["label"]):
                self.prepare([found(INITIAL), found(NOVEL)], [initial, PARTIAL],
                             model_response(PLAN, tokens=10, seconds=2.0), model_response(COMBINED_ANSWER))
                answer = self.research()

                self.assert_completed(answer, COMBINED_ANSWER)
                self.assert_searches(QUESTION, QUERY)
                self.assertEqual(self.judge.call_args_list, [
                    call(QUESTION, AS_OF, [{"id": "S1", **INITIAL}]),
                    call(QUESTION, AS_OF, [{"id": "S1", **INITIAL}, {"id": "S2", **NOVEL}]),
                ])
                self.assertEqual([check["label"] for check in self.assert_checks(1, 2)],
                                 [initial["label"], "partial"])
                self.assertTrue(self.payloads("research_followup"))
                stats = self.metrics.as_dict()
                self.assertEqual((stats["calls"], stats["measured_tokens"], stats["measurement_seconds"]),
                                 (2, 40, 3.0))
                self.assertAlmostEqual(stats["tokens_per_second"], 40 / 3)

    def test_empty_initial_search_skips_first_judgment_but_can_recover(self):
        self.prepare([EMPTY, found(NOVEL)], [HIGH], model_response(PLAN),
                     model_response("Example Model became publicly available on July 15 [S1]."))
        answer = self.research()

        self.assert_completed(answer, "July 15 [S1]")
        self.assert_searches(QUESTION, QUERY)
        self.judge.assert_called_once_with(QUESTION, AS_OF, [{"id": "S1", **NOVEL}])
        self.assertEqual([check["status"] for check in self.assert_checks(1, 2)], ["skipped", "ok"])
        self.assertTrue(self.payloads("research_followup"))

    def test_high_supported_and_supported_at_cutoff_skip_followup(self):
        for initial in (HIGH, judgment("supported", 0.60, 0.30, 0.10)):
            with self.subTest(probability=initial["probabilities"]["supported"]):
                self.prepare([found(INITIAL)], [initial], model_response(BASE_ANSWER))
                self.assert_completed(self.research(), BASE_ANSWER)
                self.assert_searches(QUESTION)
                self.judge.assert_called_once()
                self.model.respond.assert_called_once()
                self.assert_checks(1)

    def test_second_search_error_falls_back_to_initial_evidence(self):
        self.prepare([found(INITIAL), SEARCH_ERROR], [PARTIAL],
                     model_response(PLAN), model_response(BASE_ANSWER))
        answer = self.research()

        self.assert_completed(answer, BASE_ANSWER)
        self.assertIn(INITIAL["url"], answer)
        self.assertRegex(answer.lower(), r"fail|timed out|timeout|unavailable")
        self.assert_searches(QUESTION, QUERY)
        self.judge.assert_called_once()
        self.assertEqual(self.payloads("tool_result")[-1]["observation"], SEARCH_ERROR)
        self.assertTrue(self.payloads("research_followup"))

    def test_initial_search_error_is_not_an_empty_result_or_a_followup(self):
        self.prepare([SEARCH_ERROR], [])
        with self.assertRaisesRegex(RuntimeError, r"Tavily search failed.*timeout"):
            self.research()

        self.assert_searches(QUESTION)
        self.judge.assert_not_called()
        self.model.respond.assert_not_called()
        self.on_evidence.assert_not_called()
        self.assertFalse(self.payloads("research_followup"))
        self.assertEqual(self.metrics.calls, 0)

    def test_returned_jev_error_uses_baseline_without_followup(self):
        self.prepare([found(INITIAL)], [JEV_ERROR], model_response(BASE_ANSWER))
        answer = self.research()

        self.assert_completed(answer, BASE_ANSWER)
        self.assertRegex(answer.lower(), r"unassessed|not (?:be )?(?:assessed|checked|verified)|unavailable|timed out|failed")
        self.assert_searches(QUESTION)
        self.model.respond.assert_called_once()
        check, = self.assert_checks(1)
        self.assertEqual((check["status"], check["error_code"]), ("error", "timeout"))

    def test_followup_deduplicates_urls_and_citations_use_retained_ids(self):
        duplicate = {**INITIAL, "url": "https://example.org/announce#other",
                     "snippet": "An alternate excerpt of the same page."}
        for text in (COMBINED_ANSWER, "Wrong source [S3].", "Inline URL [S2] https://example.org"):
            with self.subTest(text=text):
                self.prepare([found(INITIAL), found(INITIAL, duplicate, NOVEL)], [PARTIAL, HIGH],
                             model_response(PLAN), model_response(text))
                if text == COMBINED_ANSWER:
                    answer = self.research()
                    self.assert_completed(answer, COMBINED_ANSWER)
                    self.assertIn(f"[S1](<{INITIAL['url']}>)", answer)
                    self.assertIn(f"[S2](<{NOVEL['url']}>)", answer)
                    self.assertNotIn(duplicate["url"], answer)
                else:
                    with self.assertRaisesRegex(ValueError, "citation IDs"):
                        self.research()
                self.assert_searches(QUESTION, QUERY)
                self.assertEqual(self.judge.call_args.args[2], [{"id": "S1", **INITIAL}, {"id": "S2", **NOVEL}])
                self.assert_checks(1, 2)

    def test_duplicate_only_followup_keeps_initial_judgment(self):
        self.prepare([found(INITIAL), found({**INITIAL, "url": "https://example.org/announce"})],
                     [LOW], model_response(PLAN), model_response(BASE_ANSWER))
        self.assert_completed(self.research(), "no new usable sources")
        self.assert_searches(QUESTION, QUERY)
        self.judge.assert_called_once()
        self.assert_checks(1)

    def test_second_jev_error_keeps_merged_evidence_without_another_search(self):
        self.prepare([found(INITIAL), found(NOVEL)], [LOW, JEV_ERROR],
                     model_response(PLAN), model_response(COMBINED_ANSWER))
        answer = self.research()
        self.assert_completed(answer, COMBINED_ANSWER)
        self.assertIn("coverage is unassessed", answer)
        self.assert_searches(QUESTION, QUERY)
        self.assertEqual([row["status"] for row in self.assert_checks(1, 2)], ["ok", "error"])

    def test_invalid_plans_do_not_trigger_a_second_search(self):
        invalid = {
            "malformed JSON": "{",
            "fenced JSON": f"```json\n{PLAN}\n```",
            "missing gap": json.dumps({"query": QUERY}),
            "extra key": json.dumps({"gap": "date", "query": QUERY, "provider": "instant-answer"}),
            "non-string query": json.dumps({"gap": "date", "query": [QUERY]}),
            "repeated query": json.dumps({"gap": "date", "query": "  " + "  ".join(QUESTION.upper().split())}),
        }
        for reason, text in invalid.items():
            with self.subTest(reason=reason):
                self.prepare([found(INITIAL)], [PARTIAL], model_response(text), model_response(BASE_ANSWER))
                self.assert_completed(self.research(), BASE_ANSWER)
                self.assert_searches(QUESTION)
                self.judge.assert_called_once()
                self.assert_checks(1)
                self.assertTrue(self.payloads("research_followup"))
                self.assertEqual(self.metrics.calls, 2)

    def test_planner_tool_call_is_not_executed_even_with_valid_json(self):
        tool_call = {"id": "unexpected", "name": "write_file", "arguments": {"path": "unexpected.txt"}}
        self.prepare([found(INITIAL)], [PARTIAL], model_response(PLAN, tools=[tool_call]),
                     model_response(BASE_ANSWER))
        tool = Mock(side_effect=AssertionError("research must not dispatch model tools"))
        with patch.dict(agent.TOOLS, {"write_file": tool}):
            self.assert_completed(self.research(), BASE_ANSWER)
        tool.assert_not_called()
        self.assert_searches(QUESTION)
        self.assertEqual(len(self.payloads("tool_started")), 1)
        self.assert_checks(1)

    def test_single_step_budget_is_reserved_for_synthesis(self):
        self.prepare([found(INITIAL)], [LOW], model_response(BASE_ANSWER))
        self.assert_completed(self.research(max_steps=1), BASE_ANSWER)
        self.assert_searches(QUESTION)
        self.judge.assert_called_once()
        self.model.respond.assert_called_once()
        self.assert_checks(1)
        self.assertEqual(self.metrics.calls, 1)


class ResearchEnforcementCliTests(unittest.TestCase):
    def test_interactive_and_one_shot_accept_enforce_with_optional_cutoff(self):
        for interactive in (False, True):
            for explicit in (False, True):
                with self.subTest(interactive=interactive, explicit_cutoff=explicit):
                    argv = ["--no-trace", "--max-steps", "2"]
                    commands = []
                    if interactive:
                        argv.append("--interactive")
                        commands = ["/research enforce" + (" 0.95" if explicit else ""), QUESTION, "/quit"]
                    else:
                        argv.extend(["--research", "--evidence-check", "enforce"])
                        if explicit:
                            argv.extend(["--evidence-threshold", "0.95"])
                        argv.append(QUESTION)
                    initial = judgment("supported", 0.90, 0.09, 0.01) if explicit else PARTIAL
                    judge = Mock(configured=True)
                    judge.judge.side_effect = [initial, HIGH]
                    search = Mock(side_effect=[found(INITIAL), found(NOVEL)])
                    model = Mock()
                    model.respond.side_effect = [model_response(PLAN), model_response(COMBINED_ANSWER)]
                    runner = research.run_research
                    with (patch("inquiry_loop.cli.create_model", return_value=model),
                          patch("inquiry_loop.providers.jev_eng.JevClient", return_value=judge),
                          patch("inquiry_loop.research.run_research", side_effect=lambda *a, **kw: runner(*a, **kw, search_fn=search)),
                          patch("builtins.input", side_effect=commands),
                          contextlib.redirect_stdout(io.StringIO()) as output):
                        self.assertEqual(cli.main(argv), 0)
                    text = output.getvalue()
                    self.assertIn(COMBINED_ANSWER, text)
                    self.assertIn("Research follow-up:", text)
                    self.assertIn("Jev evidence check (enforce) [search 1]", text)
                    self.assertIn("Jev evidence check (enforce) [search 2]", text)
                    self.assertEqual([c.args[0] for c in search.call_args_list], [QUESTION, QUERY])
                    self.assertEqual(judge.judge.call_count, 2)


if __name__ == "__main__":
    unittest.main()
