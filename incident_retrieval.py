"""Validate frozen incident fixtures and replay offline passage retrieval."""

from collections import Counter
from contextlib import closing
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
from urllib.parse import urlsplit


_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*")
_WORD = re.compile(r"[^\W_]+")


def _objects(value, name):
    if not isinstance(value, list) or not value or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"{name} must be a nonempty array of objects")
    return value


def _text(row, field, where):
    value = row.get(field)
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{where}: {field} must be a nonempty string without NULs")
    return value


def _register_id(row, seen, where):
    identifier = _text(row, "id", where)
    if not _ID.fullmatch(identifier):
        raise ValueError(f"{where}: invalid ID {identifier!r}")
    if identifier in seen:
        raise ValueError(f"{where}: duplicate ID {identifier!r}")
    seen.add(identifier)
    return identifier


def _validate_url(url, where):
    try:
        parts = urlsplit(url)
        # Accessing port also rejects malformed and out-of-range ports.
        parts.port
        valid = (parts.scheme in {"http", "https"} and parts.hostname
                 and parts.username is None and not any(c.isspace() or ord(c) < 32 for c in url))
    except ValueError:
        valid = False
    if not valid:
        raise ValueError(f"{where}: url must be an absolute HTTP(S) source URL without credentials")


def _references(case, field, visible):
    values = case.get(field)
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError(f"{case['id']}: {field} must be an array of passage IDs")
    unique = set(values)
    if len(unique) != len(values):
        raise ValueError(f"{case['id']}: duplicate {field} reference")
    if outside := unique - visible:
        raise ValueError(f"{case['id']}: {field} references unavailable passages: {sorted(outside)!r}")
    return unique


def load_corpus(runbooks_path, cases_path):
    """Load the frozen pilot fixtures, raising ValueError for invalid contents."""
    runbooks = _objects(json.loads(Path(runbooks_path).read_text(encoding="utf-8")), "runbooks")
    cases = _objects(json.loads(Path(cases_path).read_text(encoding="utf-8")), "cases")

    seen_ids = set()
    for runbook in runbooks:
        identifier = _register_id(runbook, seen_ids, "runbook")
        for field in ("title", "url", "section", "license", "attribution", "snapshot", "text", "sha256"):
            _text(runbook, field, identifier)
        _validate_url(runbook["url"], identifier)
        digest = hashlib.sha256(runbook["text"].encode("utf-8")).hexdigest()
        if runbook["sha256"].lower() != digest:
            raise ValueError(f"{identifier}: sha256 does not match the UTF-8 text")

    runbook_ids = seen_ids.copy()
    questions = set()
    family_splits = {}
    counts = Counter()
    for case in cases:
        identifier = _register_id(case, seen_ids, "case")
        family = " ".join(_text(case, "family", identifier).casefold().split())
        split = _text(case, "split", identifier)
        if split not in {"dev", "held-out"}:
            raise ValueError(f"{identifier}: invalid split {split!r}")
        if family_splits.setdefault(family, split) != split:
            raise ValueError(f"{identifier}: family {family!r} spans splits")
        question = " ".join(_text(case, "question", identifier).casefold().split())
        if question in questions:
            raise ValueError(f"{identifier}: duplicate question")
        questions.add(question)
        _text(case, "rationale", identifier)
        answerable = case.get("answerable")
        if not isinstance(answerable, bool):
            raise ValueError(f"{identifier}: answerable must be a boolean")
        counts[split, answerable] += 1

        observation_ids = set()
        for observation in _objects(case.get("observations"), f"{identifier} observations"):
            obs_id = _register_id(observation, seen_ids, f"{identifier} observation")
            _text(observation, "text", obs_id)
            observation_ids.add(obs_id)
        visible = runbook_ids | observation_ids
        gold = _references(case, "gold", visible)
        partial = _references(case, "partial", visible)
        if len(gold) > 5:
            raise ValueError(f"{identifier}: gold may contain at most five passages")
        if answerable and (not gold & runbook_ids or not gold & observation_ids or partial):
            raise ValueError(f"{identifier}: answerable cases need both evidence types in gold and no partial labels")
        if not answerable and gold:
            raise ValueError(f"{identifier}: insufficient cases must have empty gold")

    if sum(counts["dev", value] for value in (True, False)) != 8:
        raise ValueError("expected 8 dev cases")
    if counts["held-out", True] != 12 or counts["held-out", False] != 4:
        raise ValueError("expected 12 answerable and 4 insufficient held-out cases")
    return runbooks, cases


def _keyword_rank(passages, question, depth):
    # Treat punctuation and FTS operators as ordinary question text, not MATCH syntax.
    terms = dict.fromkeys(_WORD.findall(question))
    if not terms or not depth:
        return []
    query = " OR ".join(f'"{term}"' for term in terms)
    # Filtering results in a global index would still leak other incidents into IDF.
    with closing(sqlite3.connect(":memory:")) as db:
        try:
            db.execute("CREATE VIRTUAL TABLE evidence USING fts5(id UNINDEXED, text)")
        except sqlite3.OperationalError as exc:
            if "no such module: fts5" in str(exc):
                raise RuntimeError("incident keyword retrieval requires SQLite FTS5") from exc
            raise
        db.executemany("INSERT INTO evidence VALUES (?, ?)", passages.items())
        rows = db.execute(
            "SELECT id, bm25(evidence) AS distance FROM evidence WHERE evidence MATCH ? "
            "ORDER BY distance ASC, id ASC LIMIT ?", (query, depth)
        )
        return [(identifier, -distance) for identifier, distance in rows]


def _vector(values, name):
    if isinstance(values, (str, bytes, dict)) or getattr(values, "ndim", 1) != 1:
        raise ValueError(f"{name} must be a one-dimensional numeric vector")
    try:
        vector = tuple(float(value) for value in values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a one-dimensional numeric vector") from exc
    if not vector or not all(map(math.isfinite, vector)):
        raise ValueError(f"{name} must be nonempty and finite")
    return vector


def _dense_rank(passages, vectors, query_vector, depth):
    query = _vector(query_vector, "query_vector")
    scores = []
    for identifier in passages:
        try:
            values = vectors[identifier]
        except KeyError as exc:
            raise ValueError(f"missing vector for visible passage {identifier!r}") from exc
        vector = _vector(values, f"vector for {identifier!r}")
        if len(vector) != len(query):
            raise ValueError(f"vector dimension mismatch for {identifier!r}")
        scores.append((identifier, math.fsum(a * b for a, b in zip(query, vector))))
    return sorted(scores, key=lambda item: (-item[1], item[0]))[:depth]


def _nonnegative_int(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def rank_case(runbooks, case, vectors, query_vector, *, cutoff=5, candidate_depth=10, rrf_k=60):
    """Return keyword and RRF results from shared runbooks and this case only.

    Each retriever contributes at most candidate_depth passages. Vectors must
    cover the visible passages and be pre-normalized; other entries are ignored.
    Scores are higher-is-better for both rankings.
    """
    for value, name in ((cutoff, "cutoff"), (candidate_depth, "candidate_depth"), (rrf_k, "rrf_k")):
        _nonnegative_int(value, name)
    passages = {}
    for group in (runbooks, case["observations"]):
        for passage in group:
            identifier = passage["id"]
            if identifier in passages:
                raise ValueError(f"duplicate visible passage ID {identifier!r}")
            passages[identifier] = passage["text"]
    depth = min(candidate_depth, len(passages))
    if not cutoff or not depth:
        return {"keyword": [], "hybrid": []}

    keyword = _keyword_rank(passages, case["question"], depth)
    dense = _dense_rank(passages, vectors, query_vector, depth)
    fused = {}
    for ranking in (keyword, dense):
        for rank, (identifier, _) in enumerate(ranking, start=1):
            fused[identifier] = fused.get(identifier, 0.0) + 1 / (rrf_k + rank)
    hybrid = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
    return {
        "keyword": [{"id": identifier, "score": score} for identifier, score in keyword[:cutoff]],
        "hybrid": [{"id": identifier, "score": score} for identifier, score in hybrid[:cutoff]],
    }


def score_case(case, ranked, cutoff=5):
    """Score unique gold hits in the first cutoff slots; exclude insufficient cases."""
    _nonnegative_int(cutoff, "cutoff")
    if not case["answerable"]:
        return {"recall": None, "full_gold": None}
    gold = set(case["gold"])
    if not gold:
        raise ValueError("answerable cases must have nonempty gold")
    found = gold.intersection(result["id"] for result in ranked[:cutoff])
    return {"recall": len(found) / len(gold), "full_gold": found == gold}
