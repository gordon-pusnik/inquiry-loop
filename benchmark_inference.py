"""Measure fixed-evidence incident diagnosis on MLX and streamed OpenAI."""

import argparse
from contextlib import closing
from functools import partial
from importlib.metadata import version
import json
from pathlib import Path
import platform
from statistics import median
import subprocess
import sys
import time

from benchmark_diagnosis import MODEL, RETRIEVAL_REPORT, ROOT, SETTINGS, _sha256, load_packets
from incident_diagnosis import run_diagnosis
from incident_grading import grade_runs
from request_metrics import RequestMetrics
from response_types import RunStatus
from session_store import SessionRunner, SessionStore

LOCAL_REPO = "mlx-community/Qwen3-4B-4bit"
LOCAL_REVISION = "4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25"
LOCAL_SETTINGS = {"max_tokens": SETTINGS["max_completion_tokens"], "max_context_tokens": 4096,
                  "prefill_step_size": 512, "cache_limit_mb": 512}
PRICE = {"input": 0.15, "cached_input": 0.075, "output": 0.60}
PRICE_SOURCE = "https://developers.openai.com/api/docs/models/gpt-4o-mini"


def estimate_cost(stats):
    prompt, output = (stats.get(key) for key in ("prompt_tokens", "generation_tokens"))
    cached = stats.get("cached_prompt_tokens")
    if cached is None:
        return None
    if (any(type(n) is not int or n < 0 for n in (prompt, output, cached)) or cached > prompt):
        return None
    return ((prompt - cached) * PRICE["input"] + cached * PRICE["cached_input"]
            + output * PRICE["output"]) / 1_000_000


def _local_factory(snapshot):
    from mlx_eng import MLXModel

    return MLXModel(tool_schemas=[], model_path=snapshot, **LOCAL_SETTINGS)


def _hosted_factory():
    from openai_eng import OpenAIModel

    model = OpenAIModel(tool_schemas=[], model=MODEL)
    model.client = model.client.with_options(timeout=30, max_retries=0)
    model.respond = partial(model.respond_stream, **SETTINGS)
    return model


def _measure(runner, model, packet, state, load_seconds):
    case, initial = packet
    print(f"{runner.backend} {state}: {case['id']}", file=sys.stderr, flush=True)
    runner.new()
    run = partial(run_diagnosis, case=case, initial=initial)
    result = runner.run(case["question"], model, run, RequestMetrics(), max_steps=1)
    stats = model.last_stats
    answer = json.loads(result.answer) if result.status == RunStatus.COMPLETED else None
    ttft = stats.get("ttft_seconds")
    return {"case_id": case["id"], "answerable": case["answerable"], "state": state,
            "run_id": result.run_id, "answer": answer, "run_status": result.status.value,
            "error_type": result.error.split(":", 1)[0] if result.error else None,
            "evidence_ids": [row["id"] for row in initial], "stats": stats,
            "load_seconds": load_seconds, "ttft_seconds": ttft,
            "end_to_end_seconds": result.elapsed,
            "startup_plus_end_to_end_seconds": load_seconds + result.elapsed}


def _load(factory):
    started = time.perf_counter()
    model = factory()
    return model, time.perf_counter() - started


def run_backend(packets, factory, runner):
    rows = []
    for packet in packets:
        model, load = _load(factory)
        with closing(model):
            rows.append(_measure(runner, model, packet, "cold", load))
    model, load = _load(factory)
    with closing(model):
        priming = _measure(runner, model, packets[0], "priming", load)
        for packet in packets:
            rows.append(_measure(runner, model, packet, "warm", 0.0))
    return rows, priming


def summarize(rows, backend):
    result = {}
    for state in ("cold", "warm"):
        group = [r for r in rows if r["state"] == state]
        completed = [r for r in group if r["answer"] is not None]
        summary = {"runs": len(group), "completed": len(completed)}
        for key in ("ttft_seconds", "end_to_end_seconds", "startup_plus_end_to_end_seconds"):
            values = [r[key] for r in completed if r[key] is not None]
            summary[f"median_{key}"] = median(values) if values else None
            summary[f"{key}_samples"] = len(values)
        if backend == "mlx":
            peaks = [r["stats"]["peak_memory_bytes"] for r in group
                     if r["stats"].get("peak_memory_bytes") is not None]
            summary["peak_mlx_memory_bytes"] = max(peaks, default=None)
        else:
            costs = [cost for r in group if (cost := estimate_cost(r["stats"])) is not None]
            summary["estimated_cost_usd"] = sum(costs) if costs else None
            summary["cost_samples"] = len(costs)
        result[state] = summary
    return result


def combine_recordings(recordings, reviews):
    if len(recordings) != 2 or {r["backend"] for r in recordings} != {"mlx", "openai"}:
        raise ValueError("Provide one recording for each backend.")
    if len({r["retrieval_report_sha256"] for r in recordings}) != 1:
        raise ValueError("Recordings disagree on the frozen retrieval report.")
    indexed_reviews = {r["run_id"]: r for r in reviews}
    if len(indexed_reviews) != len(reviews):
        raise ValueError("Duplicate review run ID.")
    combined = {}
    used = set()
    reference = None
    for state in ("cold", "warm"):
        rows, adapted = [], []
        for recording in recordings:
            backend = recording["backend"]
            group = [r for r in recording["runs"] if r["state"] == state]
            evidence = {r["case_id"]: r["evidence_ids"] for r in group}
            if not evidence or len(evidence) != len(group) or (reference is not None and reference != evidence):
                raise ValueError(f"Unmatched cases or ordered evidence for {state}.")
            reference = evidence
            mode = "fixed" if backend == "mlx" else "diagnostic"  # Reuse the two-arm grading rubric.
            for row in group:
                rows.append({**row, "mode": mode})
                if row["answer"] is not None:
                    review = indexed_reviews.get(row["run_id"])
                    if review is None:
                        raise ValueError(f"Missing review for {row['run_id']}.")
                    adapted.append({**review, "case_id": row["case_id"], "mode": mode})
                    used.add(row["run_id"])
        grades = grade_runs(rows, adapted)
        modes = {"fixed": "mlx", "diagnostic": "openai"}
        combined[state] = {"backends": {modes[k]: v for k, v in grades["arms"].items()},
                           "runs": [{**{k: v for k, v in r.items() if k != "mode"},
                                     "backend": modes[r["mode"]]} for r in grades["runs"]]}
    if used != indexed_reviews.keys():
        raise ValueError("Review references a failed or unknown run.")
    return {"recordings": recordings, "quality": combined}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run")
    run.add_argument("backend", choices=("mlx", "openai"))
    run.add_argument("--snapshot", type=Path,
                     default=Path.home() / ".cache/huggingface/hub/models--mlx-community--Qwen3-4B-4bit/snapshots" / LOCAL_REVISION)
    run.add_argument("--case", action="append", help="Run only the named held-out case(s).")
    run.add_argument("--db", type=Path, default=ROOT / ".agent" / "incident-inference.sqlite3")
    grade = sub.add_parser("grade")
    grade.add_argument("mlx")
    grade.add_argument("openai")
    grade.add_argument("reviews")
    args = parser.parse_args(argv)
    try:
        if args.action == "grade":
            report = combine_recordings([_read(args.mlx), _read(args.openai)], _read(args.reviews))
        else:
            packets = load_packets()
            if args.case:
                if set(args.case) - {case["id"] for case, _ in packets}:
                    raise ValueError("Only held-out case IDs may be selected.")
                packets = [p for p in packets if p[0]["id"] in args.case]
            if args.backend == "mlx":
                if args.snapshot.name != LOCAL_REVISION or not (args.snapshot / "config.json").is_file():
                    raise ValueError(f"Supply the cached MLX snapshot directory for revision {LOCAL_REVISION}.")
                factory = partial(_local_factory, args.snapshot)
                model_id = f"{LOCAL_REPO}@{LOCAL_REVISION}"
                settings = {**LOCAL_SETTINGS, "temperature": 0, "enable_thinking": False}
                versions = {pkg: version(pkg) for pkg in ("mlx", "mlx-lm")}
            else:
                factory, model_id, settings = _hosted_factory, MODEL, SETTINGS
                versions = {"openai": version("openai")}
            with closing(SessionStore(args.db)) as store:
                rows, priming = run_backend(packets, factory, SessionRunner(store, args.backend, model_id))
            report = {"schema_version": 1, "backend": args.backend, "model": model_id,
                      "settings": settings, "versions": {"python": platform.python_version(), **versions},
                      "host": {"os": platform.platform(), "chip": subprocess.check_output(
                          ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
                          "memory_bytes": int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))}
                      if platform.system() == "Darwin" else {"os": platform.platform()},
                      "retrieval_report_sha256": _sha256(RETRIEVAL_REPORT), "runs": rows,
                      "priming": priming, "summary": summarize(rows, args.backend)}
            if args.backend == "openai":
                report["pricing"] = {"usd_per_million_tokens": PRICE, "source": PRICE_SOURCE}
        print(json.dumps(report, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as exc:
        parser.exit(1, f"Inference benchmark failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
