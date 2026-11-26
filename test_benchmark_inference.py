"""Offline tests for fixed-evidence inference benchmarking."""

from copy import deepcopy
import io
import json
import unittest
from unittest.mock import Mock, patch

import benchmark_inference as benchmark
from response_types import RunResult, RunStatus


def usage(cached):
    return {"prompt_tokens": 1000, "generation_tokens": 200, "cached_prompt_tokens": cached}


class CostTests(unittest.TestCase):
    def test_standard_and_cached_input_rates(self):
        for cached, dollars in ((0, 0.000270), (400, 0.000240), (1000, 0.000195)):
            with self.subTest(cached=cached):
                self.assertAlmostEqual(benchmark.estimate_cost(usage(cached)), dollars, places=12)
        self.assertEqual(benchmark.estimate_cost(dict(usage(0), prompt_tokens=0, generation_tokens=0)), 0)

    def test_unavailable_or_invalid_usage_has_no_estimate(self):
        for stats in ({}, {"prompt_tokens": 1000, "generation_tokens": 200}, usage(None),
                      usage(-1), usage(1001), usage(False),
                      dict(usage(0), generation_tokens=None),
                      dict(usage(0), prompt_tokens=1000.0)):
            with self.subTest(stats=stats):
                self.assertIsNone(benchmark.estimate_cost(stats))


class SummaryTests(unittest.TestCase):
    def test_latency_uses_only_completed_runs_and_counts_each_available_metric(self):
        def row(state, completed, latency, stats):
            ttft, elapsed, startup = latency
            return {"state": state, "answer": {"status": "diagnosed"} if completed else None,
                    "run_status": "completed" if completed else "failed", "stats": stats,
                    "ttft_seconds": ttft, "end_to_end_seconds": elapsed,
                    "startup_plus_end_to_end_seconds": startup}

        rows = [
            row("cold", True, (1, 4, 10), dict(usage(0), peak_memory_bytes=100)),
            row("cold", True, (None, 8, 16), {}),
            row("cold", False, (100, 200, 300), dict(usage(400), peak_memory_bytes=200)),
            row("warm", True, (0, 2, 2), usage(None)),
            row("warm", False, (3, 5, 5), {}),
            row("priming", True, (9, 30, 60), dict(usage(0), peak_memory_bytes=999)),
        ]
        hosted = benchmark.summarize(rows, "openai")
        local = benchmark.summarize(rows, "mlx")
        latency = {
            "cold": {"runs": 3, "completed": 2, "median_ttft_seconds": 1,
                     "ttft_seconds_samples": 1, "median_end_to_end_seconds": 6,
                     "end_to_end_seconds_samples": 2, "median_startup_plus_end_to_end_seconds": 13,
                     "startup_plus_end_to_end_seconds_samples": 2},
            "warm": {"runs": 2, "completed": 1, "median_ttft_seconds": 0,
                     "ttft_seconds_samples": 1, "median_end_to_end_seconds": 2,
                     "end_to_end_seconds_samples": 1, "median_startup_plus_end_to_end_seconds": 2,
                     "startup_plus_end_to_end_seconds_samples": 1},
        }
        self.assertEqual(set(hosted), set(latency))
        self.assertEqual(local, {"cold": dict(latency["cold"], peak_mlx_memory_bytes=200),
                                 "warm": dict(latency["warm"], peak_mlx_memory_bytes=None)})
        # Failed requests can still incur cost or use memory, but never supply latency samples.
        self.assertAlmostEqual(hosted["cold"]["estimated_cost_usd"], 0.000510, places=12)
        self.assertEqual({k: v for k, v in hosted["cold"].items() if k != "estimated_cost_usd"},
                         dict(latency["cold"], cost_samples=2))
        self.assertEqual(hosted["warm"], dict(latency["warm"], estimated_cost_usd=None, cost_samples=0))

    def test_failed_only_and_empty_states_have_no_latency_medians(self):
        rows = [{"state": "cold", "answer": None, "stats": {}, "ttft_seconds": 1,
                 "end_to_end_seconds": 2, "startup_plus_end_to_end_seconds": 3}]
        summary = benchmark.summarize(rows, "openai")
        for state, runs in (("cold", 1), ("warm", 0)):
            with self.subTest(state=state):
                self.assertEqual(summary[state]["runs"], runs)
                self.assertEqual(summary[state]["completed"], 0)
                for metric in ("ttft_seconds", "end_to_end_seconds", "startup_plus_end_to_end_seconds"):
                    self.assertIsNone(summary[state][f"median_{metric}"])
                    self.assertEqual(summary[state][f"{metric}_samples"], 0)


class CombineRecordingsTests(unittest.TestCase):
    def setUp(self):
        self.recordings = []
        self.reviews = []
        for backend in ("mlx", "openai"):
            runs = []
            for state in ("cold", "warm"):
                for case_id, answerable in (("quota", True), ("missing-events", False)):
                    evidence = [f"{case_id}:event", "runbook:deployments"]
                    run_id = f"{backend}-{state}-{case_id}"
                    claims = [{"claim": "Quota blocked rollout.", "citations": evidence[:1]}] if answerable else []
                    failed = (backend, state, case_id) == ("openai", "warm", "missing-events")
                    answer = None if failed else {"status": "diagnosed" if answerable else "insufficient",
                                                  "conclusions": claims}
                    runs.append({"case_id": case_id, "state": state, "run_id": run_id,
                                 "answerable": answerable, "evidence_ids": evidence, "answer": answer})
                    if not failed:
                        judgments = [{"citation_supported": backend == "mlx" or state == "warm",
                                      "conclusion_correct": backend == "openai" or state == "cold",
                                      "note": "Check the quota event."} for _ in claims]
                        self.reviews.append({"run_id": run_id, "claims": judgments, "note": f"Review {run_id}"})
            self.recordings.append({"backend": backend, "retrieval_report_sha256": "a" * 64, "runs": runs})

    def test_combines_by_packet_and_grades_each_state_and_backend_independently(self):
        # Run order may differ; evidence order within each packet must not.
        self.recordings[1]["runs"].reverse()
        before = deepcopy((self.recordings, self.reviews))
        report = benchmark.combine_recordings(self.recordings, self.reviews[::-1])
        self.assertEqual(report["recordings"], before[0])
        self.assertEqual((self.recordings, self.reviews), before)
        self.assertEqual(set(report["quality"]), {"cold", "warm"})
        expected_by_state = {"cold": {"mlx": (1, 1, 2, 0), "openai": (0, 1, 1, 0)},
                             "warm": {"mlx": (1, 0, 1, 0), "openai": (1, 1, 1, 1)}}
        review_notes = {review["run_id"]: review["note"] for review in self.reviews}
        for state, expected in expected_by_state.items():
            quality = report["quality"][state]
            self.assertEqual(set(quality["backends"]), set(expected))
            self.assertCountEqual([(row["backend"], row["run_id"]) for row in quality["runs"]],
                                  [(backend, f"{backend}-{state}-{case_id}") for backend in expected
                                   for case_id in ("quota", "missing-events")])
            for row in quality["runs"]:
                self.assertNotIn("mode", row)
                self.assertEqual(row["note"], review_notes.get(row["run_id"]))
                self.assertEqual(row["failed"], row["run_id"] not in review_notes)
            for backend, counts in expected.items():
                with self.subTest(state=state, backend=backend):
                    arm = quality["backends"][backend]
                    for metric, count, denominator in zip(
                        ("citation_support", "conclusion_correctness", "full_correct", "failure"),
                        counts, (1, 1, 2, 2),
                    ):
                        self.assertEqual(arm[metric], {"count": count, "denominator": denominator,
                                                       "rate": count / denominator})

    def test_requires_both_backends_and_the_same_frozen_report(self):
        with self.assertRaisesRegex(ValueError, "one recording for each backend"):
            benchmark.combine_recordings([self.recordings[0]] * 2, self.reviews)
        self.recordings[1]["retrieval_report_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "frozen retrieval report"):
            benchmark.combine_recordings(self.recordings, self.reviews)

    def test_requires_the_same_cases_and_ordered_evidence_across_backends_and_states(self):
        for change, state in (("missing case", "cold"), ("different backend order", "cold"),
                              ("different state order", "warm")):
            recordings = deepcopy(self.recordings)
            if change == "missing case":
                recordings[1]["runs"].pop(0)
            elif change == "different backend order":
                recordings[1]["runs"][0]["evidence_ids"].reverse()
            else:
                # Both warm backends still agree with one another, but not with cold.
                for recording in recordings:
                    for row in recording["runs"]:
                        if row["state"] == "warm":
                            row["evidence_ids"].reverse()
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, f"ordered evidence for {state}"):
                benchmark.combine_recordings(recordings, self.reviews)

    def test_rejects_missing_duplicate_and_extra_reviews(self):
        with self.assertRaisesRegex(ValueError, "Missing review for mlx-cold-quota"):
            benchmark.combine_recordings(self.recordings, self.reviews[1:])
        with self.assertRaisesRegex(ValueError, "Duplicate review run ID"):
            benchmark.combine_recordings(self.recordings, self.reviews + self.reviews[:1])
        for run_id in ("unknown-run", "openai-warm-missing-events"):
            extra = {"run_id": run_id, "claims": [], "note": "Extra review"}
            with self.subTest(run_id=run_id), self.assertRaisesRegex(ValueError, "failed or unknown run"):
                benchmark.combine_recordings(self.recordings, self.reviews + [extra])


class RunBackendTests(unittest.TestCase):
    def test_fresh_cold_models_and_one_primed_warm_model(self):
        packets = [({"id": name, "question": f"Why {name}?", "answerable": False},
                    [{"id": f"{name}:event"}, {"id": "runbook:deployments"}]) for name in ("a", "b")]
        events = []
        answer = {"status": "insufficient", "conclusions": [], "missing_evidence": "Need events."}
        models = []

        def factory():
            index = len(models)
            events.append(("load", index))
            model = Mock(last_stats={"ttft_seconds": 0.5})
            model.close.side_effect = lambda: events.append(("close", index))
            models.append(model)
            return model

        def run(question, model, workflow, metrics, *, max_steps):
            case, initial = next(packet for packet in packets if packet[0]["question"] == question)
            self.assertIs(workflow.func, benchmark.run_diagnosis)
            self.assertEqual(workflow.keywords, {"case": case, "initial": initial})
            self.assertEqual(max_steps, 1)
            events.append(("run", models.index(model), case["id"]))
            return RunResult(str(len(events)), "fake-session", RunStatus.COMPLETED, json.dumps(answer), 2, {})

        runner = Mock(backend="fake")
        runner.new.side_effect = lambda: events.append(("new",))
        runner.run.side_effect = run
        clock = Mock(perf_counter=Mock(side_effect=[0, 3, 10, 14, 20, 25]))
        with patch.object(benchmark, "time", clock), patch("sys.stderr", new_callable=io.StringIO):
            rows, priming = benchmark.run_backend(packets, factory, runner)

        self.assertEqual(events, [("load", 0), ("new",), ("run", 0, "a"), ("close", 0),
                                  ("load", 1), ("new",), ("run", 1, "b"), ("close", 1),
                                  ("load", 2), ("new",), ("run", 2, "a"),
                                  ("new",), ("run", 2, "a"), ("new",), ("run", 2, "b"), ("close", 2)])
        self.assertEqual([(row["state"], row["case_id"], row["load_seconds"]) for row in rows],
                         [("cold", "a", 3), ("cold", "b", 4), ("warm", "a", 0), ("warm", "b", 0)])
        self.assertEqual((priming["state"], priming["case_id"], priming["load_seconds"]), ("priming", "a", 5))
        self.assertEqual([row["startup_plus_end_to_end_seconds"] for row in [*rows, priming]], [5, 6, 2, 2, 7])
        self.assertEqual(len({row["run_id"] for row in [*rows, priming]}), 5)
        for row in [*rows, priming]:
            self.assertEqual(row["evidence_ids"], [f"{row['case_id']}:event", "runbook:deployments"])
            self.assertEqual(row["answer"], answer)
            self.assertEqual(row["ttft_seconds"], 0.5)


if __name__ == "__main__":
    unittest.main()
