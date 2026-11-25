"""Offline tests for the frozen incident corpus and retrieval replay."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import tempfile
import unittest
from urllib.parse import urlsplit

from benchmark_incidents import replay
from incident_retrieval import load_corpus, rank_case, score_case


FIXTURE_DIR = Path(__file__).resolve().parent
RUNBOOKS_PATH = FIXTURE_DIR / "incident_runbooks.json"
CASES_PATH = FIXTURE_DIR / "incident_cases.json"


class CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runbooks, cls.cases = load_corpus(RUNBOOKS_PATH, CASES_PATH)

    def test_real_fixture_counts(self):
        self.assertEqual(len(self.runbooks), 10)
        self.assertEqual(len(self.cases), 24)
        self.assertEqual(
            Counter((case["split"], case["answerable"]) for case in self.cases),
            {("dev", True): 6, ("dev", False): 2,
             ("held-out", True): 12, ("held-out", False): 4},
        )

    def test_real_runbook_provenance_and_hashes(self):
        for runbook in self.runbooks:
            with self.subTest(runbook=runbook["id"]):
                for field in ("title", "section", "license", "attribution", "snapshot"):
                    self.assertTrue(runbook[field].strip(), field)
                source = urlsplit(runbook["url"])
                self.assertIn(source.scheme, {"http", "https"})
                self.assertTrue(source.hostname)
                self.assertIsNone(source.username)
                digest = hashlib.sha256(runbook["text"].encode("utf-8")).hexdigest()
                self.assertEqual(runbook["sha256"].lower(), digest)

    def test_real_case_review_integrity(self):
        ids = [runbook["id"] for runbook in self.runbooks]
        runbook_ids = set(ids)
        questions = []
        family_splits = defaultdict(set)
        for case in self.cases:
            with self.subTest(case=case["id"]):
                ids.append(case["id"])
                ids.extend(obs["id"] for obs in case["observations"])
                questions.append(" ".join(case["question"].casefold().split()))
                family = " ".join(case["family"].casefold().split())
                family_splits[family].add(case["split"])
                observation_ids = {obs["id"] for obs in case["observations"]}
                visible = runbook_ids | observation_ids
                self.assertTrue(case["rationale"].strip())
                gold, partial = set(case["gold"]), set(case["partial"])
                self.assertEqual(len(gold), len(case["gold"]))
                self.assertEqual(len(partial), len(case["partial"]))
                self.assertLessEqual(gold | partial, visible)
                if case["answerable"]:
                    self.assertLessEqual(len(gold), 5)
                    self.assertTrue(gold & runbook_ids)
                    self.assertTrue(gold & observation_ids)
                    self.assertFalse(partial)
                else:
                    self.assertFalse(gold)
                    self.assertTrue(partial)
        self.assertEqual(len(ids), len(set(ids)), "IDs must be globally unique")
        self.assertEqual(len(questions), len(set(questions)), "Questions must be unique")
        for family, splits in family_splits.items():
            with self.subTest(family=family):
                self.assertEqual(len(splits), 1)


class BenchmarkTests(unittest.TestCase):
    def test_replay_scores_only_answerable_questions_in_selected_split(self):
        runbooks, cases = load_corpus(RUNBOOKS_PATH, CASES_PATH)
        report = replay(runbooks, cases, lambda texts: [(1.0,)] * len(texts), "held-out")
        self.assertEqual((report["questions"], report["answerable"], report["insufficient"]), (16, 12, 4))
        self.assertEqual([row["id"] for row in report["cases"]],
                         [case["id"] for case in cases if case["split"] == "held-out"])
        for method, metrics in report["metrics"].items():
            scores = [case["retrieval"][method] for case in report["cases"]]
            self.assertEqual(sum(score["recall"] is None for score in scores), 4)
            self.assertAlmostEqual(metrics["recall_at_5"],
                                   sum(score["recall"] or 0 for score in scores) / 12)
            self.assertEqual(metrics["full_gold_count"], sum(score["full_gold"] is True for score in scores))


class InvalidCorpusTests(unittest.TestCase):
    def setUp(self):
        self.runbooks = json.loads(RUNBOOKS_PATH.read_text(encoding="utf-8"))
        self.cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))

    def assert_invalid_corpus(self, message):
        with tempfile.TemporaryDirectory() as directory:
            runbooks_path = Path(directory) / RUNBOOKS_PATH.name
            cases_path = Path(directory) / CASES_PATH.name
            runbooks_path.write_text(json.dumps(self.runbooks), encoding="utf-8")
            cases_path.write_text(json.dumps(self.cases), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, message):
                load_corpus(runbooks_path, cases_path)

    def test_rejects_corrupted_hash(self):
        runbook = self.runbooks[0]
        digest = runbook["sha256"]
        runbook["sha256"] = ("1" if digest[0] == "0" else "0") + digest[1:]
        self.assert_invalid_corpus(rf"{re.escape(runbook['id'])}: sha256 does not match")

    def test_rejects_cross_case_gold_and_partial_labels(self):
        source = self.cases[0]
        foreign_id = source["observations"][0]["id"]
        for field, answerable in (("gold", True), ("partial", False)):
            case = next(case for case in self.cases
                        if case is not source and case["answerable"] == answerable)
            with self.subTest(field=field, case=case["id"]):
                case[field].append(foreign_id)
                try:
                    self.assert_invalid_corpus(
                        rf"{re.escape(case['id'])}: {field} references unavailable passages"
                        rf".*{re.escape(foreign_id)}"
                    )
                finally:
                    case[field].pop()

    def test_rejects_normalized_family_leakage_across_splits(self):
        dev = next(case for case in self.cases if case["split"] == "dev")
        held_out = next(case for case in self.cases if case["split"] == "held-out")
        held_out["family"] = f"  {dev['family'].upper()}  "
        self.assert_invalid_corpus(r"family .* spans splits")


class RankingTests(unittest.TestCase):
    def test_bm25_orders_by_relevance_then_passage_id(self):
        runbooks = [
            {"id": "rb:b", "text": "timeout filler"},
            {"id": "rb:a", "text": "timeout filler"},
            {"id": "rb:z", "text": "timeout timeout"},
            {"id": "rb:unmatched", "text": "healthy service"},
        ]
        case = {"question": "timeout?", "observations": []}
        vectors = {row["id"]: (0.0, 1.0) for row in runbooks}
        keyword = rank_case(runbooks, case, vectors, (1.0, 0.0))["keyword"]

        self.assertEqual([row["id"] for row in keyword], ["rb:z", "rb:a", "rb:b"])
        self.assertGreater(keyword[0]["score"], keyword[1]["score"])
        self.assertEqual(keyword[1]["score"], keyword[2]["score"])
        self.assertGreater(keyword[2]["score"], 0)

    def test_rrf_breaks_equal_fused_scores_by_passage_id(self):
        runbooks = [
            {"id": "rb:z", "text": "timeout timeout"},
            {"id": "rb:a", "text": "timeout filler"},
        ]
        case = {"question": "timeout", "observations": []}
        vectors = {"rb:z": (0.0, 1.0), "rb:a": (1.0, 0.0)}
        result = rank_case(runbooks, case, vectors, (1.0, 0.0), candidate_depth=2, rrf_k=10)

        # Opposite keyword and dense orders give both passages the same RRF score.
        self.assertEqual([row["id"] for row in result["keyword"]], ["rb:z", "rb:a"])
        self.assertEqual([row["id"] for row in result["hybrid"]], ["rb:a", "rb:z"])
        for row in result["hybrid"]:
            self.assertAlmostEqual(row["score"], 1 / 11 + 1 / 12)

    def test_rank_case_ignores_other_cases_observations_in_global_vectors(self):
        runbooks = [{"id": "rb:shared", "text": "general troubleshooting"}]
        cases = [
            {"question": "timeout", "observations": [{"id": "a:log", "text": "timeout"}]},
            {"question": "timeout", "observations": [{"id": "b:log", "text": "timeout"}]},
        ]
        global_vectors = {
            "rb:shared": (-0.6, -0.8), "a:log": (0.0, 1.0), "b:log": (1.0, 0.0),
        }
        # In each replay the other case's vector is the best dense match globally.
        for case, query_vector in zip(cases, [(1.0, 0.0), (0.0, 1.0)]):
            own_id = case["observations"][0]["id"]
            with self.subTest(observation=own_id):
                local_vectors = {key: global_vectors[key] for key in ("rb:shared", own_id)}
                local = rank_case(runbooks, case, local_vectors, query_vector, candidate_depth=1)
                result = rank_case(runbooks, case, global_vectors, query_vector, candidate_depth=1)
                self.assertEqual(result, local)
                for method in ("keyword", "hybrid"):
                    self.assertEqual([row["id"] for row in result[method]], [own_id])
                self.assertAlmostEqual(result["hybrid"][0]["score"], 2 / 61)


class ScoreCaseTests(unittest.TestCase):
    def test_recall_counts_unique_gold_within_ranked_slots(self):
        case = {"answerable": True, "gold": ["rb:guide", "a:log"]}
        ranked = [{"id": identifier} for identifier in
                  ("rb:guide", "rb:guide", "noise:1", "noise:2", "noise:3", "a:log")]
        self.assertEqual(score_case(case, ranked), {"recall": 0.5, "full_gold": False})
        self.assertEqual(score_case(case, ranked, cutoff=6), {"recall": 1.0, "full_gold": True})

    def test_recall_is_zero_when_no_gold_is_retrieved(self):
        case = {"answerable": True, "gold": ["a:log"]}
        self.assertEqual(score_case(case, [{"id": "rb:noise"}]),
                         {"recall": 0.0, "full_gold": False})

    def test_rejects_answerable_case_with_empty_gold(self):
        with self.assertRaisesRegex(ValueError, "answerable cases must have nonempty gold"):
            score_case({"answerable": True, "gold": []}, [])

    def test_insufficient_case_is_unscored_even_when_partial_evidence_is_found(self):
        case = {"answerable": False, "gold": [], "partial": ["a:log"]}
        self.assertEqual(score_case(case, [{"id": "a:log"}]),
                         {"recall": None, "full_gold": None})


if __name__ == "__main__":
    unittest.main()
