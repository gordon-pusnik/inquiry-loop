"""Human grading for the paired incident benchmark.

Judge citation support against the passages actually cited and available to the
run: they must substantiate the specific claim in the incident context. A real
passage ID or a plausible conclusion is not enough. Judge conclusion correctness
separately against the incident evidence and reference rationale; a correct claim
can be unsupported, and a misleading citation does not settle correctness. Record
a short reason for each judgment pair and an overall note for each reviewed run.
The grader checks structure and alignment, not the semantics of the notes or the
evidence. An empty citation list cannot receive a positive support judgment.

For an answerable case the expected status is 'diagnosed'; otherwise it is
'insufficient'. Abstention correctness is strictly a status match. For an
insufficient case the reviewer may discuss whether the named missing observation
is correct in the run note; this does not affect the status score.
"""


_MODES = ("fixed", "diagnostic")


def _object(value, where):
    if not isinstance(value, dict):
        raise ValueError(f"{where}: expected an object")
    return value


def _list(value, where):
    if not isinstance(value, list):
        raise ValueError(f"{where}: expected a list")
    return value


def _text(value, where):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: expected non-blank text")
    return value


def _bool(value, where):
    if type(value) is not bool:
        raise ValueError(f"{where}: expected a boolean")
    return value


def _index(items, name):
    indexed = {}
    for i, item in enumerate(_list(items, name)):
        where = f"{name}[{i}]"
        item = _object(item, where)
        case_id = _text(item.get("case_id"), f"{where}.case_id")
        mode = item.get("mode")
        if mode not in _MODES:
            raise ValueError(f"{where}.mode: expected 'fixed' or 'diagnostic'")
        key = (case_id, mode)
        if key in indexed:
            raise ValueError(f"{name}: duplicate case/mode {key!r}")
        indexed[key] = item
    return indexed


def _grade_run(key, row, review):
    case_id, mode = key
    where = f"run {key!r}"
    run_id = row.get("run_id")
    if type(run_id) is not int:
        _text(run_id, f"{where}.run_id")
    expected = "diagnosed" if row["answerable"] else "insufficient"
    failed = row["answer"] is None
    status = note = None
    claims = []
    if not failed:
        answer = _object(row["answer"], f"{where}.answer")
        status = answer.get("status")
        if status not in ("diagnosed", "insufficient"):
            raise ValueError(f"{where}.answer.status: expected 'diagnosed' or 'insufficient'")
        conclusions = _list(answer.get("conclusions"), f"{where}.answer.conclusions")
        judgments = _list(review.get("claims"), f"{where} review.claims")
        if len(conclusions) != len(judgments):
            raise ValueError(f"{where}: expected {len(conclusions)} claim reviews, got {len(judgments)}")
        note = _text(review.get("note"), f"{where} review.note")
        for i, (conclusion, judgment) in enumerate(zip(conclusions, judgments)):
            slot = f"{where} conclusion[{i}]"
            conclusion = _object(conclusion, slot)
            claim = _text(conclusion.get("claim"), f"{slot}.claim")
            citations = _list(conclusion.get("citations"), f"{slot}.citations")
            judgment = _object(judgment, f"{slot} review")
            supported = _bool(judgment.get("citation_supported"), f"{slot} review.citation_supported")
            correct = _bool(judgment.get("conclusion_correct"), f"{slot} review.conclusion_correct")
            if supported and not citations:
                raise ValueError(f"{slot}: citation support requires at least one citation")
            claims.append({
                "index": i, "claim": claim, "citation_supported": int(supported),
                "conclusion_correct": int(correct),
                "note": _text(judgment.get("note"), f"{slot} review.note"),
            })

    status_correct = status == expected
    full_correct = (
        status_correct and (status != "diagnosed" or bool(claims))
        and all(claim["citation_supported"] and claim["conclusion_correct"] for claim in claims)
    )
    return {
        "case_id": case_id, "mode": mode, "run_id": run_id,
        "answerable": row["answerable"], "expected_status": expected, "status": status,
        "failed": failed, "abstention_correct": int(status_correct),
        "inappropriate_abstention": int(expected == "diagnosed" and status == "insufficient"),
        "full_correct": int(full_correct), "claims": claims, "note": note,
    }


def _metric(count, denominator):
    return {"count": count, "denominator": denominator,
            "rate": count / denominator if denominator else None}


def _summarize(runs):
    completed = [run for run in runs if not run["failed"]]
    claims = [claim for run in completed for claim in run["claims"]]
    return {
        "runs": len(runs),
        "answerable_runs": sum(run["answerable"] for run in runs),
        "completed_runs": len(completed),
        "zero_conclusion_runs": sum(not run["claims"] for run in completed),
        "conclusions": len(claims),
        "citation_support": _metric(sum(claim["citation_supported"] for claim in claims), len(claims)),
        "conclusion_correctness": _metric(sum(claim["conclusion_correct"] for claim in claims), len(claims)),
        "abstention_correctness": _metric(sum(run["abstention_correct"] for run in completed), len(completed)),
        "inappropriate_abstention": _metric(
            sum(run["inappropriate_abstention"] for run in completed),
            sum(run["answerable"] for run in completed),
        ),
        "failure": _metric(len(runs) - len(completed), len(runs)),
        "coverage": _metric(sum(run["status"] == "diagnosed" for run in completed), len(runs)),
        "full_correct": _metric(sum(run["full_correct"] for run in runs), len(runs)),
    }


def grade_runs(rows, reviews):
    """Return a JSON-serializable {'arms': {...}, 'runs': [...]} report.

    Each case needs one row per arm with the same boolean answerable label.
    Rows carry case_id, mode, run_id, answerable, and answer (None on failure).
    Each completed row needs exactly one review keyed by (case_id, mode), with
    a non-blank note and a claims list in answer.conclusions order. Each claim
    review needs boolean citation_supported and conclusion_correct values and
    a non-blank note. Invalid or mismatched inputs raise ValueError.

    Runs are returned in input order with per-claim 0/1 scores and zero-based
    indices. Failed runs have null status and note, no claims, and a 0 status
    score. Both arms are always present. Each rate is reported alongside its
    numerator (count) and denominator, with null for a zero denominator:

    * Claim metrics use all reviewed conclusions. Runs with no conclusions
      contribute neither successes nor errors to those metrics.
    * Abstention correctness uses completed runs; inappropriate abstention
      (an insufficient answer on an answerable case) uses completed answerable
      runs. Failures are not abstentions.
    * Failure, coverage (diagnosed status), and full correctness use all runs.
      Full correctness requires the right status and both scores on every
      emitted conclusion. A diagnosis also needs at least one conclusion; a
      correct abstention can have none. Failures never count as fully correct.
    """
    rows_by_key = _index(rows, "rows")
    reviews_by_key = _index(reviews, "reviews")
    required_reviews = set()
    for key, row in rows_by_key.items():
        case_id, mode = key
        answerable = _bool(row.get("answerable"), f"run {key!r}.answerable")
        other = (case_id, "diagnostic" if mode == "fixed" else "fixed")
        if other not in rows_by_key:
            raise ValueError(f"rows: missing paired run {other!r}")
        if rows_by_key[other].get("answerable") is not answerable:
            raise ValueError(f"rows: paired runs for {case_id!r} must share a boolean answerable label")
        if "answer" not in row:
            raise ValueError(f"run {key!r}: missing answer (use None on failure)")
        if row["answer"] is not None:
            required_reviews.add(key)

    missing = sorted(required_reviews - reviews_by_key.keys())
    extra = sorted(reviews_by_key.keys() - required_reviews)
    if missing or extra:
        raise ValueError(f"reviews: missing {missing!r}; extra {extra!r}")

    runs = [_grade_run(key, row, reviews_by_key.get(key)) for key, row in rows_by_key.items()]
    return {
        "arms": {mode: _summarize([run for run in runs if run["mode"] == mode]) for mode in _MODES},
        "runs": runs,
    }
