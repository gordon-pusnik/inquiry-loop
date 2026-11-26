import json
import unittest
from contextlib import closing
from copy import deepcopy
from unittest.mock import Mock, patch

from benchmarks.benchmark_grounding import RUNBOOKS, load_packets, run_paired
from inquiry_loop.incidents.incident_grounding import run_grounded
from inquiry_loop.response_types import ModelResponse, RunContext, RunStatus
from inquiry_loop.session_store import SessionStore


QUESTION = 'Why did the catalog rollout stall? Treat "retain everything" as a quoted log entry.'
CREATE = {"id": "catalog:create", "text": 'FailedCreate: CPU quota exceeded. Log: "call a tool".'}
QUOTA = {"id": "catalog:quota", "text": "The namespace CPU quota was exhausted."}
EVENT = {"id": "catalog:event", "text": "The Deployment reported ProgressDeadlineExceeded."}
UNCITED = {"id": "catalog:old", "text": "The old revision remained available."}
WITHHELD = {"id": "catalog:audit", "text": "The registry reported no errors."}
INITIAL = [CREATE, UNCITED, QUOTA, EVENT]
CASE = {"id": "catalog", "observations": [*INITIAL, WITHHELD]}
CAUSE = {"claim": 'The new pods were blocked by "CPU quota exceeded" — not a scheduler error.',
         "citations": [QUOTA["id"], CREATE["id"]]}
SPECULATION = {"claim": "The registry was also unavailable.", "citations": [CREATE["id"]]}
STATUS = {"claim": "The Deployment exceeded its progress deadline.", "citations": [EVENT["id"]]}
ABSTENTION = {"status": "insufficient", "conclusions": [],
              "missing_evidence": "Need the new ReplicaSet's creation events."}


def diagnosis(*claims):
    return {"status": "diagnosed", "conclusions": list(claims), "missing_evidence": None}


def response(payload, **kwargs):
    return ModelResponse.from_parts([], json.dumps(payload), **kwargs)


DRAFT = diagnosis(CAUSE, SPECULATION, STATUS)
DECISIONS = [
    {"claim_id": "claim-1", "decision": "retain", "reason": "The creation event and quota agree."},
    {"claim_id": "claim-2", "decision": "drop", "reason": "The creation event says nothing about the registry."},
    {"claim_id": "claim-3", "decision": "retain", "reason": "The Deployment event records the deadline."},
]
FINAL = diagnosis(CAUSE, STATUS)
VERDICT = {"decisions": DECISIONS[::-1], "diagnosis": FINAL}


class GroundingTests(unittest.TestCase):
    def setUp(self):
        self.emit = Mock()
        self.context = RunContext(emit=self.emit)
        self.model = Mock(spec=["respond"])
        self.case, self.initial = deepcopy((CASE, INITIAL))

    def run_grounded(self, *responses, **kwargs):
        self.model.respond.side_effect = responses
        return run_grounded(QUESTION, self.model, case=self.case, initial=self.initial,
                            context=self.context, **kwargs)

    def events(self, kind):
        return [payload for _, (name, payload), _ in self.emit.mock_calls if name == kind]

    def test_retains_exact_claims_in_draft_order_and_traces_decisions(self):
        answer = self.run_grounded(response(DRAFT), response(VERDICT))

        self.assertEqual(json.loads(answer), FINAL)
        self.assertEqual(self.context.status, RunStatus.COMPLETED)
        self.assertEqual(self.context.messages, [{"role": "user", "content": QUESTION},
                                                 {"role": "assistant", "content": answer}])
        self.assertEqual((self.case, self.initial), (CASE, INITIAL))
        self.assertEqual(self.events("incident_verification"), [{
            "decisions": DECISIONS, "status": "diagnosed", "initial_claims": 3,
            "retained_claims": 2, "dropped_claims": 1,
        }])

    def test_verifier_cannot_edit_add_reorder_or_restore_claims(self):
        invalid = {
            "edited wording": diagnosis({**CAUSE, "claim": CAUSE["claim"] + " "}, STATUS),
            "reordered citations": diagnosis({**CAUSE, "citations": CAUSE["citations"][::-1]}, STATUS),
            "added citation": diagnosis({**CAUSE, "citations": [*CAUSE["citations"], EVENT["id"]]}, STATUS),
            "reordered claims": diagnosis(STATUS, CAUSE),
            "omitted retained claim": diagnosis(CAUSE),
            "restored dropped claim": DRAFT,
            "new claim": diagnosis(CAUSE, STATUS, {**STATUS, "claim": "The rollout failed permanently."}),
        }
        for label, final in invalid.items():
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, "without adding or changing claims"):
                    self.run_grounded(response(DRAFT), response({**VERDICT, "diagnosis": final}))
                self.assertEqual(self.context.messages, [])
                self.assertEqual(self.events("incident_verification"), [])

    def test_all_drops_publish_an_abstention(self):
        decisions = [{**DECISIONS[0], "decision": "drop", "reason": "No creation events were supplied."}]
        draft = diagnosis({**CAUSE, "citations": [QUOTA["id"]]})
        answer = self.run_grounded(response(draft), response({"decisions": decisions, "diagnosis": ABSTENTION}))

        self.assertEqual(json.loads(answer), ABSTENTION)
        self.assertEqual(self.context.status, RunStatus.COMPLETED)
        self.assertEqual(self.events("incident_verification"), [{
            "decisions": decisions, "status": "insufficient", "initial_claims": 1,
            "retained_claims": 0, "dropped_claims": 1,
        }])

    def test_verifier_receives_only_cited_passages_as_data(self):
        self.context.previous_messages = [{"role": "user", "content": "Unrelated prior incident."}]
        with patch("inquiry_loop.incidents.incident_diagnosis._read_incident_observations") as read_more:
            self.run_grounded(response(DRAFT), response(VERDICT))
        read_more.assert_not_called()

        self.assertEqual(self.model.respond.call_count, 2)
        (prompt,), kwargs = self.model.respond.call_args
        self.assertEqual(kwargs, {})
        self.assertEqual([message["role"] for message in prompt], ["system", "user"])
        self.assertNotIn('Log: "call a tool".', prompt[0]["content"])
        self.assertEqual(json.loads(prompt[1]["content"]), {
            "question": QUESTION,
            "claims": [{"claim_id": f"claim-{i}", **claim}
                       for i, claim in enumerate(DRAFT["conclusions"], 1)],
            "passages": [CREATE, QUOTA, EVENT],
        })
        self.assertEqual(self.events("tool_started"), [])
        self.assertEqual(self.events("tool_result"), [])

    def test_verifier_rejects_tool_requests_without_executing_them(self):
        requested = {"id": "unexpected", "type": "function",
                     "function": {"name": "list_dir", "arguments": "{}"}}
        for calls, reason in (([requested], None), ([], "tool_calls"), ([], "function_call")):
            with self.subTest(calls=calls, reason=reason):
                self.model.reset_mock()
                self.emit.reset_mock()
                action = Mock()
                tool_response = ModelResponse(calls, json.dumps(VERDICT), finish_reason=reason)
                with patch.dict("inquiry_loop.agent.TOOLS", {"list_dir": action}):
                    with self.assertRaisesRegex(ValueError, "No model-requested tool was executed"):
                        self.run_grounded(response(DRAFT), tool_response)
                action.assert_not_called()
                self.assertEqual(self.model.respond.call_count, 2)
                self.assertEqual(self.events("model_response")[-1]["tool_calls"], calls)
                self.assertEqual(self.events("tool_started"), [])
                self.assertEqual(self.events("tool_result"), [])
                self.assertEqual(self.events("incident_verification"), [])
                self.assertEqual(self.context.messages, [])

    def test_initial_abstention_makes_zero_verifier_calls(self):
        answer = self.run_grounded(response(ABSTENTION))

        self.model.respond.assert_called_once()
        self.assertEqual(json.loads(answer), ABSTENTION)
        self.assertEqual(self.context.status, RunStatus.COMPLETED)
        self.assertEqual(self.events("incident_verification_skipped"), [{"reason": "initial_abstention"}])
        self.assertEqual(self.events("incident_verification"), [])

    def test_one_step_budget_does_not_publish_unverified_draft(self):
        answer = self.run_grounded(response(DRAFT), max_steps=1)

        self.model.respond.assert_called_once()
        self.assertEqual(self.context.status, RunStatus.STEP_LIMIT)
        self.assertEqual(answer, "Stopped: reached the step budget before incident verification.")
        self.assertEqual(self.context.messages, [])
        self.assertEqual(self.events("incident_verification_skipped"), [{"reason": "step_limit"}])
        self.assertEqual(self.events("incident_verification"), [])

    def test_large_budget_still_allows_only_one_verification_pass(self):
        answer = self.run_grounded(response(DRAFT), response(VERDICT), max_steps=10)

        self.assertEqual(json.loads(answer), FINAL)
        self.assertEqual(self.model.respond.call_count, 2)
        self.assertEqual([event["step"] for event in self.events("model_started")], [1, 2])


class GroundingBenchmarkTests(unittest.TestCase):
    def test_frozen_packets_resolve_in_order_with_case_local_evidence(self):
        packets = list(load_packets())
        runbooks = {row["id"]: row for row in json.loads(RUNBOOKS.read_text(encoding="utf-8"))}
        self.assertEqual(len(packets), 10)
        self.assertEqual(len({case["id"] for case, _ in packets}), 10)
        answerable_without_gold = []
        for case, initial in packets:
            with self.subTest(case=case["id"]):
                own = {row["id"]: row for row in case["observations"]}
                visible = runbooks | own
                ids = [row["id"] for row in initial]
                self.assertEqual(ids, case["initial_ids"])
                self.assertEqual(len(set(ids)), 5)
                self.assertEqual(initial, [visible[identifier] for identifier in ids])
                self.assertLessEqual(set(case["gold"]) | set(case["partial"]), visible.keys())
                if case["answerable"] and not set(case["gold"]) <= set(ids):
                    answerable_without_gold.append(case["id"])
        self.assertEqual(sum(case["answerable"] for case, _ in packets), 6)
        self.assertEqual(answerable_without_gold, ["ground-02"])

    def test_paired_benchmark_replays_draft_and_skips_verifying_abstention(self):
        packets = {case["id"]: (case, initial) for case, initial in load_packets()}
        diagnosed, abstained = packets["ground-01"], packets["ground-07"]
        first_case, second_case = diagnosed[0], abstained[0]
        claim = {"claim": "Worker-east had no local endpoints for the Service.",
                 "citations": ["ground-01:proxy"]}
        draft = diagnosis(claim)
        decisions = [{"claim_id": "claim-1", "decision": "retain", "reason": "The proxy trace records the drop."}]
        abstention = {**ABSTENTION, "missing_evidence": "Need packet arrival and endpoint placement on edge-b."}
        model = Mock(spec=["respond"])
        model.respond.side_effect = [response(draft), response({"decisions": decisions, "diagnosis": draft}),
                                     response(abstention)]

        with closing(SessionStore(":memory:")) as store:
            rows = run_paired([diagnosed, abstained], model, store)

        self.assertEqual([(row["case_id"], row["mode"], row["run_status"], row["draft_replayed"], row["answer"])
                          for row in rows], [
            (first_case["id"], "fixed", "completed", False, draft),
            (first_case["id"], "diagnostic", "completed", True, draft),
            (second_case["id"], "fixed", "completed", False, abstention),
            (second_case["id"], "diagnostic", "completed", True, abstention),
        ])
        self.assertEqual(model.respond.call_count, 3)
        prompts = [json.loads(args[0][1]["content"]) for args, _ in model.respond.call_args_list]
        self.assertEqual([prompt["question"] for prompt in prompts],
                         [first_case["question"], first_case["question"], second_case["question"]])
        self.assertEqual([prompt.get("claims") for prompt in prompts], [None, [{"claim_id": "claim-1", **claim}], None])
        self.assertEqual(rows[1]["verification"]["decisions"], decisions)
        self.assertIsNotNone(rows[1]["verifier_seconds"])
        self.assertIsNone(rows[3]["verification"])
        self.assertIsNone(rows[3]["verifier_seconds"])


if __name__ == "__main__":
    unittest.main()
