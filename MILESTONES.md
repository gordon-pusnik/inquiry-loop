# Incident Runbook Milestones

Use cited runbooks to explain incidents and measure local and hosted inference on the same tasks. See the [feature plan](INCIDENT_RUNBOOK_PLAN.md) for background.

## Milestone 1: Build a diagnostic corpus and retrieval baseline

**Status: Complete**

### Acceptance criteria

- [x] Index public runbooks and synthetic incidents, with labeled relevant passages and cases where evidence is insufficient.
- [x] Compare keyword and hybrid retrieval on the same held-out questions. Record recall at a fixed cutoff.

The corpus contains 10 [public Kubernetes runbook excerpts](incident_runbooks.json) and 24 [synthetic incidents](incident_cases.json) with passage-level evidence labels. The comparison used the same 16 held-out questions: on the 12 answerable questions, BM25 scored **80.56% Recall@5** and hybrid scored **83.33% Recall@5**. The four insufficient-evidence questions were replayed and reported separately, not counted in recall. See the [benchmark report](incident_retrieval_report.json) for per-question results.

## Milestone 2: Grade citations, conclusions, and abstentions

**Status: Complete**

### Acceptance criteria

- [x] Require an evidence citation for each diagnostic conclusion. Score citation support, conclusion correctness, and abstention separately.
- [x] Compare a fixed workflow with a bounded, read-only diagnostic step. Use existing traces to inspect any gains and failures.

Both workflows completed all 16 held-out cases. The bounded lookup added evidence in three cases, but both arms had 10/16 fully correct runs. See the [trace analysis](INCIDENT_RUNBOOK_PLAN.md#milestone-2-results), [scores and trace summaries](incident_diagnosis_report.json), and [semantic review notes](incident_diagnosis_reviews.json).

## Milestone 3: Benchmark local MLX and hosted inference

**Status: Complete**

### Acceptance criteria

- [x] Run the same cases with recorded model versions and settings. Record answer quality, time to first token, and end-to-end latency, including warm and cold runs.
- [x] Document peak local memory and hosted cost where measurable. Explain which workloads fit the M1 Pro and which benefit from hosted inference.

Each backend completed the same 16 held-out cases under both conditions. See the [results and workload guidance](INCIDENT_RUNBOOK_PLAN.md#milestone-3-results), [per-run benchmark report](incident_inference_report.json), and [semantic review notes](incident_inference_reviews.json).

## Milestone 4: Improve evidence grounding and calibrate abstention

**Status: Complete (pilot; not enabled by default)**

### Acceptance criteria

- [x] Compare the fixed workflow with one bounded citation-verification pass that can drop unsupported conclusions and abstain with a specific evidence request.
- [x] Evaluate both workflows on the same fresh, frozen held-out cases and evidence packets.
- [x] Report unsupported conclusions, fully correct runs, unnecessary abstentions, latency, and cost.

The verifier removed one unsupported conclusion without dropping any supported claims in this 10-case pilot. Six unsupported conclusions remained. See the [analysis](INCIDENT_RUNBOOK_PLAN.md#milestone-4-results), [report](incident_grounding_report.json), and [semantic reviews](incident_grounding_reviews.json).
