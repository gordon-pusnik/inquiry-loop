import contextlib
import io
import unittest
from unittest.mock import Mock, patch

import agent
from request_metrics import RequestMetrics


class RequestMetricsTests(unittest.TestCase):
    def test_weighted_generation_rate_across_tool_and_answer_turns(self):
        metrics = RequestMetrics()
        metrics.record({"generation_tokens": 60, "generation_tps": 10})
        metrics.record({"generation_tokens": 20, "generation_tps": 20})
        self.assertEqual(metrics.summary(35), "Processed request in 35.00 seconds, tokens per second = 11.43")

    def test_strata_engine_rate_overrides_different_api_token_count(self):
        metrics = RequestMetrics()
        metrics.record({"generation_tokens": 5, "generation_seconds": 2, "generation_tps": 12})
        self.assertEqual(metrics.summary(35), "Processed request in 35.00 seconds, tokens per second = 12.00")

    def test_request_throughput_is_labeled(self):
        metrics = RequestMetrics()
        metrics.record({"generation_tokens": 50, "generation_seconds": 10, "speed_kind": "request"})
        self.assertEqual(metrics.summary(35), "Processed request in 35.00 seconds, tokens per second = 5.00 (includes prompt processing and request overhead)")

    def test_missing_invalid_or_partial_stats_do_not_invent_a_rate(self):
        for stats in (None, {}, {"generation_tokens": 5, "generation_tps": 0},
                      {"generation_tokens": 5, "generation_tps": float("nan")},
                      {"generation_tokens": 5, "generation_tps": float("inf")},
                      {"generation_tokens": True, "generation_tps": 5}):
            with self.subTest(stats=stats):
                metrics = RequestMetrics()
                metrics.record({"generation_tokens": 50, "generation_tps": 10})
                metrics.record(stats)
                self.assertTrue(metrics.summary(35).endswith("= N/A"))

    def test_cli_keeps_answer_and_appends_one_summary(self):
        output = io.StringIO()
        with patch("agent.create_model", return_value=Mock()) as factory, \
                patch("agent.run_agent", return_value="Original answer\n```\nfile contents\n```"), \
                patch("agent.time.perf_counter", side_effect=[10, 45]), \
                contextlib.redirect_stdout(output):
            self.assertEqual(agent.main(["--no-trace", "--llm=strata", "test"]), 0)
        self.assertEqual(output.getvalue(), "Original answer\n```\nfile contents\n```\nProcessed request in 35.00 seconds, tokens per second = N/A\n")
        factory.return_value.close.assert_called_once()

    def test_agent_records_every_model_call_and_keeps_return_type(self):
        class Model:
            def __init__(self):
                self.calls = 0

            def respond(self, messages):
                self.calls += 1
                self.last_stats = {"generation_tokens": 10, "generation_tps": 5}
                if self.calls == 1:
                    return [{"id": "c1", "function": {"name": "list_dir", "arguments": "{}"}}], ""
                return [], "The final answer"

        metrics = RequestMetrics()
        with patch.dict(agent.TOOLS, {"list_dir": lambda: {"entries": []}}):
            answer = agent.run_agent("test", Model(), verbose=False, metrics=metrics)
        self.assertEqual(answer, "The final answer")
        self.assertEqual(metrics.calls, 2)
        self.assertEqual(metrics.tokens, 20)
        self.assertTrue(metrics.summary(8).endswith("= 5.00"))


if __name__ == "__main__":
    unittest.main()
