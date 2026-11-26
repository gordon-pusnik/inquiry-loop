"""Locations shared by the CLI and benchmarks in a source checkout."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
STATE_DIR = PROJECT_ROOT / ".agent"
BENCHMARK_FIXTURES = PROJECT_ROOT / "benchmarks" / "fixtures"
BENCHMARK_RESULTS = PROJECT_ROOT / "benchmarks" / "results"
