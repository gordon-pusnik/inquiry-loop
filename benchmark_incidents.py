"""Replay the frozen diagnostic corpus against keyword and hybrid retrieval."""

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
from statistics import mean

from incident_retrieval import load_corpus, rank_case, score_case

ROOT = Path(__file__).resolve().parent
RUNBOOKS = ROOT / "incident_runbooks.json"
CASES = ROOT / "incident_cases.json"
DEFAULT_CACHE = ROOT / ".agent" / "incident-embeddings"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL_FILES = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
               "special_tokens_map.json", "vocab.txt")
MAX_TOKENS = 256  # The pinned SentenceTransformer's configured sequence length.
CUTOFF = 5
CANDIDATE_DEPTH = 10
RRF_K = 60
METHODS = ("keyword", "hybrid")


def _model_snapshot(cache, *, download=False):
    if download:
        from huggingface_hub import snapshot_download

        return snapshot_download(MODEL, revision=REVISION, cache_dir=cache,
                                 allow_patterns=MODEL_FILES, token=False)
    snapshot = cache / f"models--{MODEL.replace('/', '--')}" / "snapshots" / REVISION
    if not all((snapshot / name).is_file() for name in MODEL_FILES):
        raise ValueError(f"Incomplete model snapshot in {cache}; run with --download-model first.")
    return snapshot


class MiniLMEncoder:
    def __init__(self, cache):
        snapshot = _model_snapshot(cache)
        import torch
        import transformers
        self.torch = torch
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        self.model = transformers.AutoModel.from_pretrained(
            snapshot, local_files_only=True, use_safetensors=True).to("cpu").eval()
        self.versions = {"torch": torch.__version__, "transformers": transformers.__version__}

    def encode(self, texts):
        torch = self.torch
        vectors = []
        with torch.inference_mode():
            for start in range(0, len(texts), 16):
                inputs = self.tokenizer(texts[start:start + 16], padding=True, truncation=True,
                                        max_length=MAX_TOKENS, return_tensors="pt")
                hidden = self.model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                vectors.extend(torch.nn.functional.normalize(pooled, dim=1).tolist())
        return vectors


def replay(runbooks, cases, encode, split):
    selected = [case for case in cases if case["split"] == split]
    passages = runbooks + [obs for case in selected for obs in case["observations"]]
    texts = [row["text"] for row in passages] + [case["question"] for case in selected]
    encoded = encode(texts)
    if len(encoded) != len(texts):
        raise ValueError("encoder returned the wrong number of vectors")
    vectors = {row["id"]: vector for row, vector in zip(passages, encoded[:len(passages)], strict=True)}
    results = []
    for case, query in zip(selected, encoded[len(passages):], strict=True):
        rankings = rank_case(runbooks, case, vectors, query, cutoff=CUTOFF,
                             candidate_depth=CANDIDATE_DEPTH, rrf_k=RRF_K)
        results.append({
            "id": case["id"], "question": case["question"], "answerable": case["answerable"],
            "gold": case["gold"], "partial": case["partial"], "rationale": case["rationale"],
            "retrieval": {method: {"ranked": ranked, **score_case(case, ranked, cutoff=CUTOFF)}
                          for method, ranked in rankings.items()},
        })

    answerable = [row for row in results if row["answerable"]]
    metrics = {}
    for method in METHODS:
        scores = [row["retrieval"][method] for row in answerable]
        full = sum(score["full_gold"] for score in scores)
        metrics[method] = {"recall_at_5": mean(score["recall"] for score in scores),
                           "full_gold_count": full, "full_gold_rate": full / len(scores)}
    return {
        "split": split, "questions": len(results), "answerable": len(answerable),
        "insufficient": len(results) - len(answerable), "metrics": metrics,
        "hybrid_minus_keyword_recall_at_5": (
            metrics["hybrid"]["recall_at_5"] - metrics["keyword"]["recall_at_5"]),
        "cases": results,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "held-out"), default="held-out")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--download-model", action="store_true", help="Fetch the pinned weights and exit")
    args = parser.parse_args(argv)
    try:
        if args.download_model:
            print(_model_snapshot(args.cache_dir, download=True))
            return 0
        runbooks, cases = load_corpus(RUNBOOKS, CASES)
        encoder = MiniLMEncoder(args.cache_dir)
        report = replay(runbooks, cases, encoder.encode, args.split)
    except ImportError as exc:
        parser.exit(1, f"Missing benchmark dependency: {exc}. Install requirements-incident.txt.\n")
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Incident benchmark failed: {exc}\n")
    config = {
        "cutoff": CUTOFF, "candidate_depth": CANDIDATE_DEPTH, "rrf_k": RRF_K,
        "keyword": "SQLite FTS5 BM25", "dense": "CPU mean pooling, L2 normalization, exact cosine",
        "embedding_model": MODEL, "embedding_revision": REVISION, "max_tokens": MAX_TOKENS,
        "versions": {"sqlite": sqlite3.sqlite_version, **encoder.versions},
        "corpus_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (RUNBOOKS, CASES)},
    }
    print(json.dumps({"schema_version": 1, "config": config, **report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
