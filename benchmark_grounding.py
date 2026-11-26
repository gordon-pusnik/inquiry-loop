"""Compare a frozen diagnosis draft with the same draft after one verification pass."""

import argparse
from contextlib import closing
from functools import partial
import json
from pathlib import Path
from statistics import median

from benchmark_diagnosis import MODEL, ROOT, SETTINGS, _sha256, grade_recording
from benchmark_inference import PRICE, PRICE_SOURCE, _hosted_factory, estimate_cost
from incident_diagnosis import run_diagnosis
from incident_grounding import run_grounded
from request_metrics import RequestMetrics
from response_types import ModelResponse, RunStatus
from session_store import SessionRunner, SessionStore

CASES = ROOT / "incident_grounding_cases.json"
RUNBOOKS = ROOT / "incident_runbooks.json"


def load_packets():
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    shared = {row["id"]: row for row in json.loads(RUNBOOKS.read_text(encoding="utf-8"))}
    if len(cases) != 10 or sum(case["answerable"] for case in cases) != 6:
        raise ValueError("Expected ten frozen cases, six corpus-answerable.")
    seen = set(shared)
    for case in cases:
        key = case["id"]
        own = {row["id"]: row for row in case["observations"]}
        if (key in seen or type(case["answerable"]) is not bool or not own
                or len(own) != len(case["observations"])
                or any(not identifier.startswith(key + ":") for identifier in own)
                or seen.intersection(own)):
            raise ValueError(f"Invalid case ID or observations: {key}.")
        seen.update((key, *own))
        visible = shared | own
        initial = case["initial_ids"]
        gold, partial_ids = set(case["gold"]), set(case["partial"])
        if (len(initial) != 5 or len(set(initial)) != 5 or not set(initial) <= visible.keys()
                or not (gold | partial_ids) <= visible.keys()
                or (case["answerable"] and (not gold.intersection(own) or not gold.intersection(shared)))
                or (not case["answerable"] and gold)):
            raise ValueError(f"Invalid evidence packet or labels: {key}.")
        yield case, [visible[identifier] for identifier in initial]


class ReplayDraft:
    def __init__(self, model, draft, elapsed):
        self.model, self.draft, self.elapsed = model, draft, elapsed

    def respond(self, messages):
        if self.draft is not None:
            draft, self.draft = self.draft, None
            return draft
        return self.model.respond(messages)


def _events(store, run_id):
    return [(event["kind"], json.loads(event["payload_json"])) for event in store.events(run_id)]


def _missing_draft(request, model, **kwargs):
    raise ValueError("Verification requires a valid fixed-workflow draft.")


def run_paired(packets, model, store):
    rows = []
    runner = SessionRunner(store, "openai", MODEL)
    for case, initial in packets:
        draft_event = None
        for mode, workflow in (("fixed", run_diagnosis), ("diagnostic", run_grounded)):
            runner.new()
            replay = mode == "diagnostic" and draft_event is not None
            adapter = (ReplayDraft(model, ModelResponse.from_parts(
                draft_event["tool_calls"], draft_event["text"], draft_event["stats"], draft_event["finish_reason"]),
                draft_event["elapsed"]) if replay else model)
            run = (partial(workflow, case=case, initial=initial)
                   if mode == "fixed" or replay else _missing_draft)
            result = runner.run(case["question"], adapter, run, RequestMetrics(),
                                max_steps=1 if mode == "fixed" else 2)
            events = _events(store, result.run_id)
            calls = [p for kind, p in events if kind == "model_response"]
            if mode == "fixed" and result.status == RunStatus.COMPLETED:
                draft_event = calls[0]
            costs = [estimate_cost(p["stats"]) for p in calls]
            rows.append({"case_id": case["id"], "mode": mode, "run_id": result.run_id,
                         "answerable": case["answerable"],
                         "gold_in_packet": set(case["gold"]) <= set(case["initial_ids"]) if case["answerable"] else None,
                         "answer": json.loads(result.answer) if result.status == RunStatus.COMPLETED else None,
                         "run_status": result.status.value, "initial_ids": case["initial_ids"],
                         "verification": next((p for kind, p in events if kind == "incident_verification"), None),
                         "draft_replayed": replay,
                         "elapsed_seconds": result.elapsed + (adapter.elapsed - calls[0]["elapsed"] if replay else 0),
                         "verifier_seconds": calls[1]["elapsed"] if len(calls) == 2 else None,
                         "estimated_cost_usd": sum(costs) if costs and all(c is not None for c in costs) else None})
    return rows


def grade(report, reviews):
    result = grade_recording(report, reviews)
    for mode, metrics in result["grades"]["arms"].items():
        completed = [r for r in result["runs"] if r["mode"] == mode and r["answer"] is not None]
        costs = [r["estimated_cost_usd"] for r in completed if r["estimated_cost_usd"] is not None]
        times = [r["elapsed_seconds"] for r in completed]
        support = metrics["citation_support"]
        metrics.update(unsupported_conclusions=support["denominator"] - support["count"],
                       median_elapsed_seconds=median(times) if times else None,
                       estimated_cost_usd=sum(costs) if costs else None, cost_samples=len(costs))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run")
    run.add_argument("--db", type=Path, default=ROOT / ".agent" / "incident-grounding.sqlite3")
    score = sub.add_parser("grade")
    score.add_argument("runs", type=Path)
    score.add_argument("reviews", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.action == "grade":
            result = grade(json.loads(args.runs.read_text()), json.loads(args.reviews.read_text()))
        else:
            packets = list(load_packets())
            with closing(_hosted_factory()) as model, closing(SessionStore(args.db)) as store:
                rows = run_paired(packets, model, store)
            result = {"schema_version": 1, "model": MODEL, "settings": SETTINGS,
                      "fixture_sha256": {p.name: _sha256(p) for p in (CASES, RUNBOOKS)},
                      "pricing": {"usd_per_million_tokens": PRICE, "source": PRICE_SOURCE},
                      "modes": {"fixed": "fixed", "diagnostic": "verified"}, "runs": rows}
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(1, f"Grounding benchmark failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
