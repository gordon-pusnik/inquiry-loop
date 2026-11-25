# Incident Runbook Milestones

Use cited runbooks to explain incidents and measure local and hosted inference on the same tasks. See the [feature plan](INCIDENT_RUNBOOK_PLAN.md) for background.

## Milestone 1: Build a diagnostic corpus and retrieval baseline

**Status: Complete**

### Acceptance criteria

- [x] Index public runbooks and synthetic incidents, with labeled relevant passages and cases where evidence is insufficient.
- [x] Compare keyword and hybrid retrieval on the same held-out questions. Record recall at a fixed cutoff.

The corpus contains 10 [public Kubernetes runbook excerpts](incident_runbooks.json) and 24 [synthetic incidents](incident_cases.json) with passage-level evidence labels. The comparison used the same 16 held-out questions: on the 12 answerable questions, BM25 scored **80.56% Recall@5** and hybrid scored **83.33% Recall@5**. The four insufficient-evidence questions were replayed and reported separately, not counted in recall. See the [benchmark report](incident_retrieval_report.json) for per-question results.

## Milestone 2: Grade citations, conclusions, and abstentions

**Status: Not started**

### Acceptance criteria

- [ ] Require an evidence citation for each diagnostic conclusion. Score citation support, conclusion correctness, and abstention separately.
- [ ] Compare a fixed workflow with a bounded, read-only diagnostic step. Use existing traces to inspect any gains and failures.

## Milestone 3: Benchmark local MLX and hosted inference

**Status: Not started**

### Acceptance criteria

- [ ] Run the same cases with recorded model versions and settings. Record answer quality, time to first token, and end-to-end latency, including warm and cold runs.
- [ ] Document peak local memory and hosted cost where measurable. Explain which workloads fit the M1 Pro and which benefit from hosted inference.
