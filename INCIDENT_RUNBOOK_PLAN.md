# Cited incident explanations and inference comparison

**Status** — Milestone 1 complete. Cited explanations and the local/hosted inference comparison are not implemented.

## Feature goal

Given a synthetic operational incident and a diagnostic question, retrieve evidence from a versioned corpus and explain what happened, what the evidence supports, and what remains unknown. Cite the incident observations that support the explanation and the public runbook passages that explain the mechanism or recommend checks. A runbook describing a possible cause is not proof that the cause occurred. If the evidence cannot distinguish plausible causes, say so and identify the missing observation. The workflow explains incidents; it does not execute commands or remediation.

Build this in three milestones:

1. **Diagnostic corpus and retrieval baseline (this proposal).** Freeze labeled fixtures and compare keyword and hybrid retrieval offline, without answer generation.
2. **Cited explanations.** Feed retrieved excerpts to an existing model adapter. Reuse the research workflow’s citation validation pattern, but score whether each material claim is supported, not just whether its citation ID exists. Evaluate appropriate abstention on insufficient cases.
3. **Local versus hosted inference.** Replay identical questions, ordered evidence packets, instructions, and output budgets with one local backend (MLX or local Strata) and OpenAI. Use the same quality rubric for both. Report end-to-end latency, answer quality, abstention, and provider cost when available. Record model IDs, hardware, settings, and warm-up policy. Report local decode speed separately from hosted request throughput rather than comparing them as equivalent. This is a comparison of the selected configurations, not an isolation of the effect of hosting.

## Milestone 1 — Diagnostic corpus and retrieval baseline

### Corpus and labels

- **Pilot size:** 8–12 public runbook pages or troubleshooting guides and 24 synthetic incident/question pairs: 8 development and 16 held-out. Target 12 answerable and 4 insufficient cases in the held-out split. Focus on routine operational failures such as rollout readiness, queue backlog, and disk pressure. Include exact error strings, paraphrased symptoms, and distractors that share symptoms but suggest a different cause.
- **Freeze and cite sources:** Prefer official docs or maintained public runbooks with licenses that permit storing excerpts. Record title, canonical URL and section, license and attribution, revision or snapshot date, and content hash. Check the license before including text. Use only public documentation and fictional telemetry; no production logs or credentials.
- **Index both evidence types:** Split runbooks into heading-bounded passages and synthetic incidents into discrete observations (e.g., timeline, metric, or log excerpt). Assign stable passage IDs, preserve source provenance, and index the same text and metadata for both methods. For each question, expose only its own incident observations plus the shared runbooks. Keep questions, gold labels, and author-only case notes out of the index.
- **Label evidence before evaluation:** For each question, record answerability, gold passage IDs, and a short rationale. Answerable questions must link observed symptoms to an applicable runbook; note when both evidence types are necessary. For insufficient cases, list any useful partial passages separately and state what observation is missing. Example: a runbook covers both resource pressure and a stalled rollout, but the incident contains only a latency increase and neither differentiating signal.
- **Protect the held-out split:** Group variants of the same incident and near-duplicate questions into the same split. Shared runbook passages are allowed; duplicated incident scenarios across splits are not. Use only development cases to tune chunking or retrieval settings. Freeze corpus, labels, and configuration before the first held-out evaluation; document later revisions rather than silently retuning the baseline.

### Retrieval and scoring

| Method | Proposed implementation |
| --- | --- |
| Keyword | BM25 with SQLite FTS5, subject to confirming FTS5 support in the target Python build. |
| Hybrid | Combine the BM25 and local dense embedding rankings using reciprocal rank fusion. Pin the embedding model revision, candidate depth, and fusion settings. Use an offline exact similarity search for this small corpus; no vector service is needed. |

Run both methods on the same held-out questions, corpus snapshot, and case filters. Use **passage Recall@5** as the primary metric: for each answerable question, divide the number of unique gold passage IDs in the first five results by the total gold passage count, then average across answerable questions. Keep gold sets small enough that five results can cover them. Also report how often the top five contain the full gold set; finding a generic runbook passage alone should not disguise a missed incident observation. Break ties deterministically by passage ID.

Keep insufficient questions in the replay, but do not score them as zero recall or credit a retriever for always returning five results. Report their count and show the returned passages alongside the missing evidence. This milestone measures retrieval, not whether a model can abstain. Report both scores regardless of which method wins; the pilot is too small to establish a general improvement.

### Deliverables and acceptance checks

- A versioned corpus manifest, frozen passages and incident observations, and development/held-out labels. Validate unique IDs, valid gold references, expected split counts, and case isolation.
- A repeatable benchmark in the style of `benchmark_research.py`, with a documented setup and run command. After obtaining pinned local embedding weights, indexing and replay should run without network access or API keys.
- A report with corpus/config versions, question counts, per-question ranked IDs and scores, aggregate Recall@5 for both methods and the difference, full-gold coverage, and representative misses and insufficient cases.
- Automated checks for passage provenance, label integrity, cross-incident filtering, ranking tie behavior, and recall calculation (including excluding insufficient cases from the primary denominator).

**Out of scope for milestone 1:** CLI mode changes, Tavily/Jev calls, live incident ingestion, generated explanations, hosted inference, and remediation.

## Milestone 1 results

The frozen corpus contains [10 Kubernetes v1.34 runbook excerpts](incident_runbooks.json) and [24 synthetic incidents](incident_cases.json) with 43 discrete observations. The development split has 6 answerable and 2 insufficient questions. The held-out split has 12 answerable and 4 insufficient questions. The methods used the same case-visible passages and questions, a cutoff of 5, and a candidate depth of 10. Hybrid used equal-weight reciprocal rank fusion with `k=60` and a pinned, CPU-only MiniLM encoder. The held-out [JSON report](incident_retrieval_report.json) includes the corpus hashes, settings, ranked IDs, scores, and review labels.

| Split | Method | Mean passage Recall@5 | Full gold in top 5 |
| --- | --- | ---: | ---: |
| Dev | BM25 | 88.89% | 4/6 |
| Dev | Hybrid | 88.89% | 4/6 |
| Held-out | BM25 | 80.56% | 7/12 |
| Held-out | Hybrid | 83.33% | 7/12 |

Hybrid gained **2.78 percentage points** of held-out recall, but did not increase full-evidence coverage. It recovered the volume observation for `inc-16` and the init-container runbook for `inc-22`, but displaced the Service debugging runbook for `inc-12`. Both methods missed the relevant runbook and logs for the paraphrased initialization question `inc-23`. The pilot is too small and narrow to justify a general claim that hybrid is better.

The four held-out insufficient cases ran through both methods and are unscored. For example, both rankings include the DNS guide and the client error for `inc-11`, but the incident still lacks the namespace and lookup results needed to distinguish a name-resolution scope issue from a DNS failure. The report records the missing evidence alongside each ranking; it does not measure abstention.

### Reproduce

Use a Python environment with SQLite FTS5. Install the optional benchmark dependencies and download the pinned local embedding snapshot once (about 92 MB):

```bash
python3 -m pip install -r requirements-incident.txt
python3 benchmark_incidents.py --download-model
```

Then run either split without network access or API keys. The model files are read from `.agent/incident-embeddings` by default; use `--cache-dir` on both commands to choose a different cache. The checked-in report uses SQLite 3.53.4, PyTorch 2.15.0, and Transformers 5.22.2.

```bash
python3 benchmark_incidents.py --split dev
python3 benchmark_incidents.py --split held-out > incident_retrieval_report.json
python3 -m unittest test_incident_retrieval
```
