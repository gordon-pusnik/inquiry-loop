"""Offline tests for bounded incident synthesis, persisted traces, and human grading."""

from copy import deepcopy
from functools import partial
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from inquiry_loop import agent
from benchmarks.benchmark_diagnosis import RETRIEVAL_REPORT, grade_recording, load_packets
from benchmarks.benchmark_incidents import RUNBOOKS
from inquiry_loop.incidents.incident_diagnosis import parse_diagnosis, run_diagnosis
from inquiry_loop.incidents.incident_grading import grade_runs
from inquiry_loop.request_metrics import RequestMetrics
from inquiry_loop.response_types import ModelResponse, RunStatus
from inquiry_loop.session_store import SessionRunner, SessionStore


class BenchmarkInputsTests(unittest.TestCase):
    def test_frozen_starting_packets_match_hybrid_report_order(self):
        report_path = RETRIEVAL_REPORT
        report = json.loads(report_path.read_text(encoding="utf-8"))
        expected = {row["id"]: [p["id"] for p in row["retrieval"]["hybrid"]["ranked"]]
                    for row in report["cases"]}
        self.assertEqual({case["id"]: [p["id"] for p in passages]
                          for case, passages in load_packets()}, expected)
        with tempfile.TemporaryDirectory() as directory:
            altered = Path(directory) / "retrieval.json"
            report["config"]["corpus_sha256"]["incident_cases.json"] = "wrong"
            altered.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_packets(altered)

    def test_review_cannot_be_applied_to_a_different_recorded_run(self):
        answer = {"status": "insufficient", "conclusions": [], "missing_evidence": "Need events."}
        recording = {"runs": [{"case_id": "test", "mode": mode, "run_id": mode,
                               "answerable": False, "answer": answer}
                              for mode in ("fixed", "diagnostic")]}
        reviews = [{"case_id": "test", "mode": mode, "run_id": mode,
                    "claims": [], "note": "Missing events."}
                   for mode in ("fixed", "diagnostic")]
        self.assertEqual(grade_recording(recording, reviews)["grades"]["arms"]["fixed"]["runs"], 1)
        reviews[0]["run_id"] = "different-run"
        with self.assertRaisesRegex(ValueError, "does not match the recorded run ID"):
            grade_recording(recording, reviews)


QUESTION = "Why did the catalog deployment stall?"
OBSERVATIONS = [
    {"id": "catalog:deployment", "text": "ProgressDeadlineExceeded; no new pods are available."},
    {"id": "catalog:replicas", "text": "The new ReplicaSet reports FailedCreate: CPU quota exceeded."},
    {"id": "catalog:quota", "text": "The namespace has used its entire CPU quota."},
    {"id": "catalog:old", "text": "The previous revision still has three available pods."},
]
CASE = {"id": "catalog", "observations": OBSERVATIONS}
FOREIGN = {"id": "billing:events", "text": "Billing pods report ImagePullBackOff."}
OTHER_CASE = {"id": "billing", "observations": [FOREIGN]}
ABSTENTION = {"status": "insufficient", "conclusions": [],
              "missing_evidence": "Need the new ReplicaSet's creation events."}


def conclusion(claim, *citations):
    return {"claim": claim, "citations": list(citations)}


def diagnosis(*claims):
    return {"status": "diagnosed", "conclusions": list(claims), "missing_evidence": None}


QUOTA_CLAIM = conclusion("CPU quota blocked creation of the new pods.",
                         "catalog:replicas", "catalog:quota")


class ScriptedModel:
    def __init__(self, answer):
        response = (answer if isinstance(answer, ModelResponse) else
                    ModelResponse([], json.dumps(answer), {"generation_tokens": 20, "generation_tps": 10}))
        self.respond = Mock(side_effect=[response])


class ParseDiagnosisTests(unittest.TestCase):
    def setUp(self):
        self.ids = ["catalog:replicas", "catalog:quota"]
        self.claim = conclusion("The namespace is at its quota.", "catalog:quota")

    def test_accepts_citations_per_conclusion_at_the_limit_and_valid_abstention(self):
        answer = diagnosis(QUOTA_CLAIM, self.claim,
                           conclusion("The new ReplicaSet could not create pods.", "catalog:replicas"))
        self.assertEqual(parse_diagnosis(json.dumps(answer), self.ids), answer)
        self.assertEqual(parse_diagnosis(json.dumps(ABSTENTION), []), ABSTENTION)

    def test_each_conclusion_must_have_its_own_distinct_supplied_ids(self):
        invalid = {
            "missing": {"claim": "Another conclusion"},
            "empty": conclusion("Another conclusion"),
            "duplicate": conclusion("Another conclusion", self.ids[0], self.ids[0]),
            "unknown among valid": conclusion("Another conclusion", self.ids[0], FOREIGN["id"]),
            "non-string ID": conclusion("Another conclusion", 1),
            "scalar citations": {"claim": "Another conclusion", "citations": self.ids[0]},
            "blank claim": conclusion("  ", self.ids[0]),
            "extra conclusion key": dict(self.claim, confidence=1),
        }
        for label, bad_claim in invalid.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                parse_diagnosis(json.dumps(diagnosis(QUOTA_CLAIM, bad_claim)), self.ids)

    def test_rejects_invalid_status_shapes_and_non_strict_json(self):
        invalid = {
            "diagnosed without conclusions": json.dumps(diagnosis()),
            "four conclusions": json.dumps(diagnosis(*([self.claim] * 4))),
            "unknown status": json.dumps(dict(ABSTENTION, status="unknown")),
            "abstention with a claim": json.dumps(dict(ABSTENTION, conclusions=[self.claim])),
            "abstention without explanation": json.dumps(dict(ABSTENTION, missing_evidence=None)),
            "abstention with blank explanation": json.dumps(dict(ABSTENTION, missing_evidence=" \t")),
            "non-string explanation": json.dumps(dict(ABSTENTION, missing_evidence=42)),
            "top-level citations": json.dumps(dict(ABSTENTION, citations=self.ids)),
            "duplicate status": '{"status":"diagnosed",' + json.dumps(ABSTENTION)[1:],
            "duplicate nested key": json.dumps(diagnosis(self.claim)).replace('"claim":', '"claim":"other", "claim":', 1),
            "markdown wrapper": f"```json\n{json.dumps(ABSTENTION)}\n```",
        }
        for label, raw in invalid.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                parse_diagnosis(raw, self.ids)


class DiagnosisSessionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = SessionStore(Path(directory.name) / "incidents.sqlite3")
        self.addCleanup(self.store.close)
        self.runner = SessionRunner(self.store, "stub", "scripted-incident")
        runbooks = json.loads(RUNBOOKS.read_text(encoding="utf-8"))
        self.runbooks = [{key: row[key] for key in ("id", "text", "title", "url")} for row in runbooks[:5]]
        self.initial = [OBSERVATIONS[0], *self.runbooks[:4]]

    def run_case(self, model, *, case=CASE, question=QUESTION, initial=None, diagnostic=False):
        workflow = partial(run_diagnosis, case=case, initial=self.initial if initial is None else initial,
                           diagnostic=diagnostic)
        return self.runner.run(question, model, workflow, RequestMetrics(), max_steps=9)

    def trace(self, result, kinds):
        events = self.store.events(result.run_id)
        self.assertEqual([event["kind"] for event in events], kinds)
        return {event["kind"]: json.loads(event["payload_json"]) for event in events}

    def test_paired_prompts_differ_only_in_final_evidence_and_traces_bound_the_read(self):
        previous = self.run_case(
            ScriptedModel(diagnosis(conclusion(FOREIGN["text"], FOREIGN["id"]))),
            case=OTHER_CASE, question="Why are the billing pods unavailable?", initial=[FOREIGN],
        )
        self.assertEqual(previous.status, RunStatus.COMPLETED, previous.error)
        self.runner = SessionRunner(self.store, "stub", "scripted-incident", previous.session_id)

        fixed_model, diagnostic_model = ScriptedModel(ABSTENTION), ScriptedModel(diagnosis(QUOTA_CLAIM))
        fixed = self.run_case(fixed_model)
        diagnostic = self.run_case(diagnostic_model, diagnostic=True)
        for result, model, answer in ((fixed, fixed_model, ABSTENTION),
                                      (diagnostic, diagnostic_model, diagnosis(QUOTA_CLAIM))):
            self.assertEqual(result.session_id, previous.session_id)
            self.assertEqual(result.status, RunStatus.COMPLETED, result.error)
            self.assertEqual(json.loads(result.answer), answer)
            self.assertEqual(result.stats["calls"], 1)
            model.respond.assert_called_once()

        (fixed_prompt,), fixed_kwargs = fixed_model.respond.call_args
        (diagnostic_prompt,), diagnostic_kwargs = diagnostic_model.respond.call_args
        self.assertEqual(fixed_kwargs, {})
        self.assertEqual(diagnostic_kwargs, {})
        self.assertEqual([message["role"] for message in fixed_prompt], ["system", "user"])
        self.assertEqual([message["role"] for message in diagnostic_prompt], ["system", "user"])
        self.assertEqual(fixed_prompt[0], diagnostic_prompt[0])
        added = OBSERVATIONS[1:3]  # Skip the already-present observation; leave the fourth unread.
        self.assertEqual(json.loads(fixed_prompt[1]["content"]), {"question": QUESTION, "passages": self.initial})
        self.assertEqual(json.loads(diagnostic_prompt[1]["content"]),
                         {"question": QUESTION, "passages": self.initial + added})

        start = ["run_started", "incident_started"]
        finish = ["incident_evidence", "model_started", "model_response", "run_finished"]
        traces = {
            "fixed": self.trace(fixed, start + finish),
            "diagnostic": self.trace(diagnostic, start + ["tool_started", "tool_result"] + finish),
        }
        initial_ids = [row["id"] for row in self.initial]
        for mode, trace in traces.items():
            with self.subTest(mode=mode):
                self.assertEqual(trace["incident_started"], {"mode": mode, "initial_ids": initial_ids})
                final_ids = initial_ids + ([row["id"] for row in added] if mode == "diagnostic" else [])
                self.assertEqual(trace["incident_evidence"]["passage_ids"], final_ids)
                self.assertEqual(trace["model_started"], {"step": 1, "message_count": 2})
                self.assertEqual(trace["model_response"]["stats"]["generation_tokens"], 20)
                self.assertEqual(trace["run_finished"]["status"], "completed")
        call = traces["diagnostic"]["tool_started"]["call"]
        observed = traces["diagnostic"]["tool_result"]
        self.assertEqual(call["name"], "read_incident_observations")
        self.assertEqual(call["arguments"], {"case_id": CASE["id"], "limit": 2})
        self.assertEqual((observed["call_id"], observed["name"]), (call["id"], call["name"]))
        self.assertEqual(observed["observation"], {"status": "ok", "observations": added})

    def test_rejects_invalid_initial_evidence_before_synthesis(self):
        invalid = {
            "over limit": (self.initial + self.runbooks[4:], "at most five"),
            "duplicate": ([self.initial[0]] * 2, "repeat passage IDs"),
            "another case": ([FOREIGN], "outside the shared runbooks and this incident"),
        }
        for label, (initial, error) in invalid.items():
            model = ScriptedModel(ABSTENTION)
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, error):
                run_diagnosis(QUESTION, model, case=CASE, initial=initial, diagnostic=True)
            model.respond.assert_not_called()

    def test_rejects_citations_to_evidence_unavailable_in_the_run(self):
        for diagnostic, unavailable in (
            (False, OBSERVATIONS[1]),  # Local but not retrieved by the fixed arm.
            (True, OBSERVATIONS[3]),  # Local but beyond the two-observation cap.
            (True, self.runbooks[4]),  # Shared but not initially supplied.
            (True, FOREIGN),
        ):
            with self.subTest(diagnostic=diagnostic, citation=unavailable["id"]):
                model = ScriptedModel(diagnosis(conclusion("Claim based on missing evidence",
                                                         self.initial[0]["id"], unavailable["id"])))
                result = self.run_case(model, diagnostic=diagnostic)
                model.respond.assert_called_once()
                self.assertEqual(result.status, RunStatus.FAILED)
                self.assertIn("supplied passage IDs", result.error)
                self.assertEqual(result.answer, "")
                events = {event["kind"]: json.loads(event["payload_json"])
                          for event in self.store.events(result.run_id)}
                self.assertNotIn(unavailable["id"], events["incident_evidence"]["passage_ids"])
                self.assertEqual(events["run_finished"]["status"], "failed")

    def test_blocks_model_tools_even_if_the_response_contains_valid_abstention(self):
        call = {"id": "unexpected", "type": "function", "function": {"name": "list_dir", "arguments": "{}"}}
        for calls, reason in (([call], None), ([], "tool_calls"), ([], "function_call")):
            with self.subTest(calls=calls, reason=reason):
                model = ScriptedModel(ModelResponse(calls, json.dumps(ABSTENTION), finish_reason=reason))
                action = Mock()
                with patch.dict(agent.TOOLS, {"list_dir": action}):
                    result = self.run_case(model)
                action.assert_not_called()
                model.respond.assert_called_once()
                self.assertEqual(result.status, RunStatus.FAILED)
                self.assertIn("No model-requested tool was executed", result.error)
                self.assertEqual(result.answer, "")
                trace = self.trace(result, ["run_started", "incident_started", "incident_evidence",
                                            "model_started", "model_response", "run_finished"])
                self.assertEqual(trace["model_response"]["tool_calls"], calls)
                self.assertEqual(trace["model_response"]["finish_reason"], reason)
                self.assertEqual(trace["run_finished"]["status"], "failed")
                self.assertEqual(json.loads(self.store.get_session(result.session_id)["history_json"]), [])


def run_row(case_id, mode, answerable, answer):
    return {"case_id": case_id, "mode": mode, "run_id": f"{case_id}-{mode}",
            "answerable": answerable, "answer": answer}


def review(row, *judgments):
    return {"case_id": row["case_id"], "mode": row["mode"], "note": "Reviewed against incident evidence.",
            "claims": [{"citation_supported": supported, "conclusion_correct": correct, "note": note}
                       for supported, correct, note in judgments]}


class GradeRunsTests(unittest.TestCase):
    def setUp(self):
        unsupported = conclusion("The old revision still has three available pods.", "catalog:quota")
        self.rows = [
            run_row("quota", "fixed", True, diagnosis(QUOTA_CLAIM, unsupported)),
            run_row("quota", "diagnostic", True, diagnosis(QUOTA_CLAIM)),
            run_row("readiness", "fixed", True, ABSTENTION),
            run_row("readiness", "diagnostic", True, None),
            run_row("unknown", "fixed", False, ABSTENTION),
            run_row("unknown", "diagnostic", False, ABSTENTION),
        ]
        supported = (True, True, "The cited events and quota both substantiate the claim.")
        self.reviews = [
            review(self.rows[0], supported, (False, True, "True in the incident, but not in the cited quota record.")),
            review(self.rows[1], supported),
            *(review(self.rows[i]) for i in (2, 4, 5)),
        ]

    def test_distinct_scores_and_denominators_do_not_treat_failure_as_abstention(self):
        report = grade_runs(self.rows, list(reversed(self.reviews)))
        self.assertEqual([run["run_id"] for run in report["runs"]], [row["run_id"] for row in self.rows])
        mixed, correct, inappropriate, failed, abstained, _ = report["runs"]
        self.assertEqual([(c["index"], c["claim"], c["citation_supported"], c["conclusion_correct"])
                          for c in mixed["claims"]],
                         [(0, QUOTA_CLAIM["claim"], 1, 1),
                          (1, self.rows[0]["answer"]["conclusions"][1]["claim"], 0, 1)])
        self.assertEqual(mixed["full_correct"], 0)
        self.assertEqual(correct["full_correct"], 1)
        self.assertEqual(inappropriate["inappropriate_abstention"], 1)
        self.assertEqual(inappropriate["abstention_correct"], 0)
        self.assertEqual(abstained["status"], "insufficient")
        self.assertFalse(abstained["failed"])
        self.assertEqual(abstained["abstention_correct"], 1)
        self.assertEqual(abstained["full_correct"], 1)
        self.assertTrue(failed["failed"])
        self.assertIsNone(failed["status"])
        self.assertIsNone(failed["note"])
        self.assertEqual(failed["claims"], [])
        for score in ("abstention_correct", "inappropriate_abstention", "full_correct"):
            self.assertEqual(failed[score], 0, score)

        def assert_metrics(mode, **expected):
            for name, (count, denominator) in expected.items():
                with self.subTest(mode=mode, metric=name):
                    self.assertEqual(report["arms"][mode][name], {
                        "count": count, "denominator": denominator, "rate": count / denominator,
                    })

        assert_metrics("fixed", citation_support=(1, 2), conclusion_correctness=(2, 2),
                       abstention_correctness=(2, 3), inappropriate_abstention=(1, 2),
                       failure=(0, 3), coverage=(1, 3), full_correct=(1, 3))
        assert_metrics("diagnostic", citation_support=(1, 1), conclusion_correctness=(1, 1),
                       abstention_correctness=(2, 2), inappropriate_abstention=(0, 1),
                       failure=(1, 3), coverage=(1, 3), full_correct=(2, 3))
        self.assertEqual(report["arms"]["diagnostic"]["zero_conclusion_runs"], 1)

    def test_supported_but_incorrect_conclusion_does_not_receive_full_credit(self):
        reviews = deepcopy(self.reviews)
        reviews[1]["claims"][0].update(
            conclusion_correct=False,
            note="The events support a quota error, but the reference ties the stall to a different cause.",
        )
        report = grade_runs(self.rows, reviews)
        arm = report["arms"]["diagnostic"]
        self.assertEqual(arm["citation_support"]["rate"], 1.0)
        self.assertEqual(arm["conclusion_correctness"]["rate"], 0.0)
        self.assertEqual(arm["abstention_correctness"]["rate"], 1.0)
        self.assertEqual(report["runs"][1]["full_correct"], 0)

    def test_no_claims_and_no_completed_answerable_runs_have_undefined_rates(self):
        report = grade_runs(self.rows[-2:], self.reviews[-2:])
        for mode, arm in report["arms"].items():
            for metric in ("citation_support", "conclusion_correctness", "inappropriate_abstention"):
                with self.subTest(mode=mode, metric=metric):
                    self.assertEqual(arm[metric], {"count": 0, "denominator": 0, "rate": None})

    def test_invalid_or_misaligned_review_is_rejected(self):
        def changed_claim(**changes):
            reviews = deepcopy(self.reviews)
            reviews[0]["claims"][0].update(changes)
            return reviews

        for label, reviews, error in (
            ("missing", self.reviews[1:], "missing"),
            ("for failed run", [*self.reviews, review(self.rows[3])], "extra"),
            ("duplicate", [*self.reviews, self.reviews[0]], "duplicate case/mode"),
            ("wrong claim count", [review(self.rows[0]), *self.reviews[1:]], "expected 2 claim reviews"),
            ("blank run note", [dict(self.reviews[0], note=" "), *self.reviews[1:]], "review.note"),
            ("blank claim note", changed_claim(note=" "), "review.note"),
            ("integer support", changed_claim(citation_supported=1), "expected a boolean"),
            ("string correctness", changed_claim(conclusion_correct="false"), "expected a boolean"),
        ):
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, error):
                grade_runs(self.rows, reviews)

        rows = deepcopy(self.rows)
        rows[0]["answer"]["conclusions"][0]["citations"] = []
        with self.assertRaisesRegex(ValueError, "citation support requires at least one citation"):
            grade_runs(rows, self.reviews)

    def test_invalid_run_pair_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing paired run"):
            grade_runs(self.rows[1:], self.reviews)
        rows = deepcopy(self.rows)
        rows[1]["answerable"] = False
        with self.assertRaisesRegex(ValueError, "must share a boolean answerable label"):
            grade_runs(rows, self.reviews)


if __name__ == "__main__":
    unittest.main()
