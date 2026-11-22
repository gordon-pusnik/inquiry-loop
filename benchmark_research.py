"""Replay frozen excerpts and report evidence coverage against human labels."""

import argparse
import json
import statistics
import sys

from jev_eng import CRITERIA, JevClient
from research import evidence_sources, observe_evidence
from research_cases import CASES


def presence_baseline(question, as_of, sources):
    return {"status": "ok", "label": "supported", "model": "source-presence"}


def summarize(rows):
    judged = [row for row in rows if row["status"] == "ok"]
    matrix = {expected: dict.fromkeys(CRITERIA, 0) for expected in CRITERIA}
    for row in judged:
        matrix[row["expected"]][row["label"]] += 1
    agreement = sum(matrix[label][label] for label in CRITERIA)
    return {
        "cases": len(rows),
        "judged": len(judged),
        "skipped_no_evidence": sum(row["status"] == "skipped" for row in rows),
        "errors": sum(row["status"] == "error" for row in rows),
        "agreement": agreement,
        "agreement_rate": agreement / len(judged) if judged else None,
        "false_supported": sum(matrix[label]["supported"] for label in ("partial", "insufficient")),
        "confusion_matrix": matrix,
        "median_latency_ms": statistics.median(row["elapsed_ms"] for row in judged) if judged else None,
    }


def replay(cases, judge, *, evaluator):
    rows = []
    for case in cases:
        sources = evidence_sources({"status": "ok", "results": case.results})
        judgment = observe_evidence(case.question, case.as_of, sources, judge)
        rows.append({"id": case.id, "split": case.split, "expected": case.expected,
                     "evidence_rubric": case.evidence_rubric, **judgment})
    return {"evaluator": evaluator, "summary": summarize(rows), "by_split": {
        split: summarize([row for row in rows if row["split"] == split])
        for split in dict.fromkeys(row["split"] for row in rows)
    }, "cases": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Send frozen public cases to hosted Jev")
    parser.add_argument("--split", choices=("dev", "held-out"), help="Replay only this split")
    parser.add_argument("--case", choices=[case.id for case in CASES], help="Replay one named case")
    args = parser.parse_args(argv)
    cases = [case for case in CASES
             if (args.split is None or case.split == args.split) and (args.case is None or case.id == args.case)]
    if not cases:
        parser.error("No cases match the requested filters")

    judge, evaluator = presence_baseline, "source-presence"
    if args.live:
        client = JevClient()
        if not client.configured:
            print("Set TYPESAFE_API_KEY in the environment or project .env to run the live Jev report.", file=sys.stderr)
            return 1
        judge, evaluator = client.judge, "jev"
    report = replay(cases, judge, evaluator=evaluator)
    print(json.dumps(report, indent=2))
    return int(report["summary"]["errors"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
