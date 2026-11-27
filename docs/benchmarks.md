# Tests and benchmarks

Run all commands from the repository root. See the [quickstart](../README.md#quickstart) for the base dependencies.

## Regression tests

```bash
python3 -m unittest discover -v
python3 -m unittest tests.test_research tests.test_research_enforcement tests.test_search tests.test_benchmark_search -v
```

The regression suite uses mocks for provider calls and model generation. Run `python3 -m unittest tests.test_llama -v` for llama.cpp server selection, tool calls, session resume, and research coverage. The optional MLX chat-template check uses Jinja2 (`python3 -m pip install Jinja2`) and a cached Qwen3-4B tokenizer. The project has been exercised with live OpenAI, MLX, llama.cpp, Strata, Tavily, and TypeSafe integrations. AirLLM remains experimental.

## Search and research

```bash
mkdir -p .agent

# Evaluate Tavily results on 20 public queries.
python3 -m benchmarks.benchmark_search --provider tavily > .agent/tavily-search-report.json

# Replay labeled evidence cases without a provider call.
python3 -m benchmarks.benchmark_research

# Replay one public case against Jev using the configured TypeSafe key.
python3 -m benchmarks.benchmark_research --live --case gpt-names
```

Set `TAVILY_API_KEY` for the Tavily run and `TYPESAFE_API_KEY` for the live Jev run. For the Tavily benchmark, mark `useful_top_5` for each query, then run `python3 -m benchmarks.benchmark_search --score .agent/tavily-search-report.json`. The research benchmark measures label agreement, false `supported` judgments, and latency on frozen excerpts. Jev matched all 18 judged cases in the initial pilot, skipped two empty cases, and had a median latency of 195 ms. A proposed 75% cutoff would have triggered extra searches for four of five adequately supported held-out cases. See the [Jev research plan](plans/JEV_RESEARCH_PLAN.md#first-milestone-results) for the dataset and evaluation details.

## Incident runbook benchmarks

These are separate evaluation scripts, not a mode in `agent.py`. The incident workflow explains synthetic incidents from cited evidence; it does not query clusters or run remediation. All four evaluation milestones are complete. **Citation verification is off by default** and runs only in the verified arm of the grounding benchmark. It is separate from Jev’s shadow check in web research.

| Evaluation | Result | Artifacts |
| --- | --- | --- |
| **Retrieval** | On 12 answerable held-out cases, BM25 scored **80.56% Recall@5** and hybrid **83.33%**. Both recovered all labeled passages in 7/12; four insufficient cases were reported separately. | [Method and setup](plans/INCIDENT_RUNBOOK_PLAN.md#milestone-1-results) · [Report](../benchmarks/results/incident_retrieval_report.json) |
| **Cited diagnosis** | The fixed and bounded read-only workflows each scored **10/16 fully correct**. The lookup added observations in three cases without improving that total. Citation support, conclusion correctness, and abstention were graded separately. | [Trace analysis](plans/INCIDENT_RUNBOOK_PLAN.md#milestone-2-results) · [Report](../benchmarks/results/incident_diagnosis_report.json) |
| **Local vs. hosted inference** | On the same 16 cases per condition, warm MLX and OpenAI median times to first text were **5.22 s** and **0.50 s**; median end-to-end times were **9.49 s** and **1.69 s**. Warm fully correct counts were **4/16** and **10/16**. MLX peaked at **2.94 GiB of allocator memory** on the 32 GiB M1 Pro. Estimated OpenAI cost was **$0.00772** for 32 measured requests, excluding priming. | [Cold and warm results, settings, and workload guidance](plans/INCIDENT_RUNBOOK_PLAN.md#milestone-3-results) · [Report](../benchmarks/results/incident_inference_report.json) |
| **Citation verification pilot** | On 10 fresh cases, unsupported conclusions fell from **7/16 to 6/15** and fully correct runs rose from **3/10 to 4/10**, with no unnecessary abstentions in either arm. Median latency rose from **2.30 s to 5.49 s**. The pilot did not justify enabling verification by default. | [Analysis](plans/INCIDENT_RUNBOOK_PLAN.md#milestone-4-results) · [Report](../benchmarks/results/incident_grounding_report.json) |

To regrade the published diagnosis and grounding results without model calls or credentials:

```bash
python3 -m benchmarks.benchmark_diagnosis grade benchmarks/results/incident_diagnosis_report.json benchmarks/results/incident_diagnosis_reviews.json
python3 -m benchmarks.benchmark_grounding grade benchmarks/results/incident_grounding_report.json benchmarks/results/incident_grounding_reviews.json
```

### Replay the benchmarks

| Module | How to replay |
| --- | --- |
| `benchmarks.benchmark_incidents` | [Download the pinned encoder and replay retrieval](plans/INCIDENT_RUNBOOK_PLAN.md#reproduce). Requires SQLite FTS5 and [optional dependencies](../benchmarks/requirements.txt); replay is offline after the one-time download. |
| `benchmarks.benchmark_diagnosis` | [Run the paired diagnosis pilot](plans/INCIDENT_RUNBOOK_PLAN.md#milestone-2-results). |
| `benchmarks.benchmark_inference` | [Run the local and hosted comparison](plans/INCIDENT_RUNBOOK_PLAN.md#milestone-3-results). MLX requires Apple Silicon, Metal access, and the pinned snapshot. |
| `benchmarks.benchmark_grounding` | [Run the citation verification pilot](plans/INCIDENT_RUNBOOK_PLAN.md#milestone-4-results). |

Hosted replays require `OPENAI_API_KEY` and incur token charges. New runs need fresh semantic reviews. Keep local replays and traces in `.agent/` so the published artifacts in [`benchmarks/results/`](../benchmarks/results/) stay intact.

### Regrade published inference recordings

The published report embeds both original recordings. Extract them to `.agent/` and apply the published reviews without making any model calls:

```bash
mkdir -p .agent
python3 - <<'PY'
import json
from pathlib import Path

report = json.loads(Path('benchmarks/results/incident_inference_report.json').read_text())
for recording in report['recordings']:
    path = Path('.agent') / ('published-inference-' + recording['backend'] + '.json')
    path.write_text(json.dumps(recording))
PY
python3 -m benchmarks.benchmark_inference grade \
  .agent/published-inference-mlx.json .agent/published-inference-openai.json \
  benchmarks/results/incident_inference_reviews.json
```
