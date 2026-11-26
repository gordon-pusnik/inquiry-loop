"""Compare fixed and bounded incident diagnosis, or grade a recorded pilot offline."""

import argparse
from contextlib import closing
from functools import partial
import hashlib
import json
from pathlib import Path

from benchmarks.benchmark_incidents import CASES, CUTOFF, RUNBOOKS
from inquiry_loop.incidents.incident_diagnosis import run_diagnosis
from inquiry_loop.incidents.incident_grading import grade_runs
from inquiry_loop.incidents.incident_retrieval import load_corpus
from inquiry_loop.request_metrics import RequestMetrics
from inquiry_loop.response_types import RunStatus
from inquiry_loop.session_store import SessionRunner, SessionStore

from inquiry_loop.paths import BENCHMARK_RESULTS, STATE_DIR

RETRIEVAL_REPORT = BENCHMARK_RESULTS / "incident_retrieval_report.json"
MODEL = "gpt-4o-mini-2024-07-18"
SETTINGS = {"temperature": 0, "max_completion_tokens": 512,
            "response_format": {"type": "json_object"}}


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_packets(retrieval_path=RETRIEVAL_REPORT):
    runbooks, cases = load_corpus(RUNBOOKS, CASES)
    held_out = {case["id"]: case for case in cases if case["split"] == "held-out"}
    report = json.loads(retrieval_path.read_text(encoding="utf-8"))
    hashes = {path.name: _sha256(path) for path in (RUNBOOKS, CASES)}
    if (report["split"] != "held-out" or report["config"]["cutoff"] != CUTOFF
            or report["config"]["corpus_sha256"] != hashes):
        raise ValueError("Retrieval report does not match the held-out corpus and cutoff.")
    rows = report["cases"]
    if len(rows) != len(held_out) or {row["id"] for row in rows} != held_out.keys():
        raise ValueError("Retrieval report does not cover each held-out case exactly once.")
    packets = []
    for row in rows:
        case = held_out[row["id"]]
        visible = {p["id"]: p for p in runbooks + case["observations"]}
        ids = [p["id"] for p in row["retrieval"]["hybrid"]["ranked"]]
        if (not ids or len(ids) > CUTOFF or len(set(ids)) != len(ids)
                or not set(ids) <= visible.keys() or row["question"] != case["question"]):
            raise ValueError(f"Invalid frozen evidence packet for {case['id']}.")
        packets.append((case, [visible[identifier] for identifier in ids]))
    return packets


def _trace(store, run_id):
    evidence = {}
    kinds = []
    added_ids = []
    for event in store.events(run_id):
        kind = event["kind"]
        kinds.append(kind)
        if kind == "incident_started":
            evidence["initial_ids"] = json.loads(event["payload_json"])["initial_ids"]
        elif kind == "tool_result":
            payload = json.loads(event["payload_json"])
            added_ids.extend(row["id"] for row in payload["observation"]["observations"])
        elif kind == "incident_evidence":
            evidence["final_ids"] = json.loads(event["payload_json"])["passage_ids"]
    return {"event_kinds": kinds, "added_ids": added_ids, **evidence}


def run_paired(packets, model, store, backend, model_id):
    rows = []
    runner = SessionRunner(store, backend, model_id)
    for index, (case, initial) in enumerate(packets):
        # Alternate arm order; never reuse a conversation between cases or arms.
        modes = ("fixed", "diagnostic") if index % 2 == 0 else ("diagnostic", "fixed")
        for mode in modes:
            runner.new()
            run_fn = partial(run_diagnosis, case=case, initial=initial, diagnostic=mode == "diagnostic")
            result = runner.run(case["question"], model, run_fn, RequestMetrics(), max_steps=1)
            answer = json.loads(result.answer) if result.status == RunStatus.COMPLETED else None
            rows.append({"case_id": case["id"], "mode": mode, "answerable": case["answerable"],
                         "session_id": result.session_id, "run_id": result.run_id,
                         "run_status": result.status.value, "answer": answer,
                         "trace": _trace(store, result.run_id)})
    return rows


def grade_recording(report, reviews):
    rows = report["runs"]
    run_ids = {(row["case_id"], row["mode"]): row["run_id"] for row in rows}
    grades = grade_runs(rows, reviews)
    for review in reviews:
        key = review["case_id"], review["mode"]
        if review.get("run_id") != run_ids[key]:
            raise ValueError(f"Review for {key!r} does not match the recorded run ID.")
    return {**report, "grades": grades}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    run = subparsers.add_parser("run", help="Run the pinned OpenAI pilot; sends only public/synthetic evidence")
    run.add_argument("--db", type=Path, default=STATE_DIR / "incident-diagnosis.sqlite3")
    run.add_argument("--case", action="append", help="Limit to named held-out cases (default: all)")
    grade = subparsers.add_parser("grade", help="Score recorded runs with explicit semantic reviews")
    grade.add_argument("runs", type=Path)
    grade.add_argument("reviews", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.action == "grade":
            report = json.loads(args.runs.read_text(encoding="utf-8"))
            reviews = json.loads(args.reviews.read_text(encoding="utf-8"))
            print(json.dumps(grade_recording(report, reviews), indent=2))
            return 0

        packets = load_packets()
        if args.case:
            selected = set(args.case)
            if not selected <= {case["id"] for case, _ in packets}:
                raise ValueError("Only held-out case IDs from the frozen retrieval report may be selected.")
            packets = [packet for packet in packets if packet[0]["id"] in selected]
        from inquiry_loop.backends.openai_eng import OpenAIModel

        with closing(OpenAIModel(tool_schemas=[], model=MODEL)) as model, closing(SessionStore(args.db)) as store:
            model.client = model.client.with_options(timeout=30, max_retries=0)
            model.respond = partial(model.respond, **SETTINGS)
            rows = run_paired(packets, model, store, "openai", MODEL)
        print(json.dumps({"schema_version": 1, "model": MODEL, "settings": SETTINGS,
                          "retrieval_report_sha256": _sha256(RETRIEVAL_REPORT), "runs": rows}, indent=2))
        return 0 if any(row["answer"] is not None for row in rows) else 1
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as exc:
        parser.exit(1, f"Diagnosis benchmark failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
