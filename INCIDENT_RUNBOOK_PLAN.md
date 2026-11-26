# Cited incident explanations and inference comparison

**Status** — Milestones 1–4 complete. The Milestone 4 verifier is evaluated but not enabled by default. See [milestone status](MILESTONES.md).

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

## Milestone 2 results

A paired pilot ran the same 16 held-out cases with `gpt-4o-mini-2024-07-18` at temperature 0, JSON mode, and a 512-token completion cap. Both arms received the same frozen hybrid top-five evidence and used one synthesis call with the same instructions. The diagnostic arm made one deterministic, read-only lookup for at most two omitted observations from the same synthetic incident. It could not run commands, query a cluster, read other incidents, or ask for additional steps. Each arm started a fresh session; the order alternated by case.

Every emitted conclusion had to cite at least one passage actually sent to the model. A model-assisted, unblinded review then judged whether those passages substantiated the claim and whether the conclusion was correct against the full case. Abstention was scored independently against the corpus answerability label; the missing-evidence explanation was reviewed qualitatively. Claim metrics exclude abstentions, so denominator differences matter. The fully correct metric requires the expected status and all emitted conclusions to be correct and supported; it does not establish that an otherwise correct answer was exhaustive.

| Metric | Fixed | Bounded lookup |
| --- | ---: | ---: |
| Citation support (per emitted conclusion) | 11/17 | 11/19 |
| Conclusion correctness (per emitted conclusion) | 14/17 | 15/19 |
| Diagnose/abstain status correct | 14/16 | 15/16 |
| Correct abstentions on insufficient cases | 3/4 | 3/4 |
| Fully correct runs | 10/16 | 10/16 |
| Failed runs | 0/16 | 0/16 |

The SQLite `tool_result` and `incident_evidence` events show the lookup added evidence in exactly three cases:

- **`inc-20`:** The budget observation let the model name the three-pod minimum. The fixed answer already correctly identified the disruption budget as the immediate blocker.
- **`inc-22`:** The logs exposed the missing configuration file. Both arms still mislabeled `PodInitializing` as the pod state rather than the application container's waiting reason shown in the incident.
- **`inc-23`:** The fixed arm abstained on a corpus-answerable case. With logs, the diagnostic arm identified the relevant init step but incorrectly said it returned HTTP 503; it received the 503 from the schema endpoint instead.

`inc-11` and `inc-14` changed abstention outcomes in opposite directions despite identical final evidence within each pair. Those changes cannot be attributed to the lookup. This single pilot shows that extra evidence can add useful specifics, but it did not establish an improvement in fully correct diagnoses or citation support.

The [report](incident_diagnosis_report.json) includes run IDs, trace summaries, answers, and separate grades. The [review file](incident_diagnosis_reviews.json) binds the semantic judgments to the recorded runs. Regrade the published run offline or make a new hosted run with a configured `OPENAI_API_KEY` (only public and synthetic evidence is sent):

```bash
python3 benchmark_diagnosis.py grade incident_diagnosis_report.json incident_diagnosis_reviews.json
python3 benchmark_diagnosis.py run > .agent/incident-diagnosis-runs.json
python3 -m unittest test_incident_diagnosis
```

A new run needs its own reviews. Full trace events are stored locally in `.agent/incident-diagnosis.sqlite3` by default; identify a run by its `run_id` from the report and query `events` in `id` order or use `SessionStore.events(run_id)`.

## Milestone 3 results

Both backends streamed one tool-free response on each of the same 16 held-out cases per condition. They used the fixed workflow, the same instructions, the same ordered hybrid top-five evidence, temperature 0, and a 512-token cap. The MLX configuration was `mlx-community/Qwen3-4B-4bit` at revision `4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25` (4,096-token context ceiling, thinking off, MLX 0.32.4, MLX-LM 0.32.2). Hosted used `gpt-4o-mini-2024-07-18` with API JSON mode and OpenAI SDK 3.50.0. Both outputs passed the same schema and citation-ID checks. This compares those configurations, not hosting in isolation: model, tokenizer, output lengths, and hosted JSON mode differ.

**Cold** means a fresh MLX model instance from cached weights or a new hosted client/connection for each case. Neither OS caches nor provider state were flushed. **Warm** reuses one primed instance or client, with fresh model conversation state for every case and a fresh MLX prompt cache. Runs were sequential with cold first; the warm OpenAI pass received 24,960 provider-cached input tokens, versus zero reported on the cold pass. Effects from connection reuse, prompt caching, and run order are not isolated.

| Configuration | Completed | Median time to first text | Median end-to-end | Median with startup | Fully correct |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLX cold | 16/16 | 5.29 s | 8.56 s | 9.36 s | 4/16 |
| MLX warm | 16/16 | 5.22 s | 9.49 s | 9.49 s | 4/16 |
| OpenAI cold | 16/16 | 0.65 s | 1.86 s | 1.86 s | 9/16 |
| OpenAI warm | 16/16 | 0.50 s | 1.69 s | 1.69 s | 10/16 |

Time to first text is the client-side time from inference call entry to the first nonblank visible streamed output (a practical TTFT measure). End-to-end includes evidence assembly, one streamed response, and validation, but excludes retrieval replay, startup, and final trace persistence. The startup column includes the model/client initialization for cold requests; warm initialization and priming are recorded separately. This one pass does not establish that warm MLX is slower: the result is sensitive to local load and run order. MLX also produced longer answers in this pilot (median 171.5 vs 87.5 output tokens; tokenizers differ).

| Quality metric | MLX cold / warm | OpenAI cold / warm |
| --- | ---: | ---: |
| Citation support per emitted conclusion | 15/33 · 15/33 | 11/18 · 11/17 |
| Correctness per emitted conclusion | 22/33 · 22/33 | 14/18 · 14/17 |
| Correct decision to diagnose or abstain | 15/16 · 15/16 | 13/16 · 14/16 |

The unblinded, model-assisted reviews used the Milestone 2 rubric. MLX abstained correctly on `inc-11`, where hosted inferred an unsupported namespace cause. Hosted identified the scheduler/PVC blocker in `inc-15`, where MLX abstained. Some MLX answers reached the right diagnosis but split the supporting observations across separate claims, leaving individual citations incomplete. Review counts are descriptive; the cold and warm passes reuse cases and are not independent quality samples.

**Memory and cost.** Peak MLX allocator memory was **3,153,384,424 bytes (2.94 GiB)** in both conditions on the **32 GiB Apple M1 Pro**; this includes resident MLX allocations during generation, not total process or system memory. Hosted token usage priced at [GPT-4o mini list rates](https://developers.openai.com/api/docs/models/gpt-4o-mini) ($0.15 input, $0.075 cached input, $0.60 output per million; checked 2026-11-25) came to **$0.00480 cold** and **$0.00292 warm**, or **$0.00772 for 32 measured requests** ($0.00791 including priming). These are estimates, not billing data; local power cost and hosted memory were not measured.

**Which workloads fit.** The measured 4-bit 4B model fits short, sequential, offline incident triage on this M1 Pro (up to 2,000 input and 235 output tokens observed); it keeps evidence local and is a reasonable option when several seconds of latency and human review are acceptable. Hosted was better suited in this pilot to interactive, latency-sensitive work and more frequently produced fully correct, cited diagnoses, but needs connectivity and a policy that permits sending evidence off-device. Larger local models, long contexts, and parallel requests were not measured; do not extrapolate this memory result to them. Neither backend’s output should be taken as proof of a diagnosis without checking the cited evidence.

The [inference report](incident_inference_report.json) includes per-run settings, usage, latency, model fingerprints where reported, and quality scores. [Semantic reviews](incident_inference_reviews.json) are keyed by run ID. Exact historical grades reproduce offline from the two recordings embedded in the report. For new runs, use the pinned snapshot listed above, a Metal-capable local terminal, and configure `OPENAI_API_KEY` for the hosted run (only public and synthetic evidence is sent):

```bash
HF_HUB_OFFLINE=1 python3 benchmark_inference.py run mlx > .agent/incident-inference-mlx.json
python3 benchmark_inference.py run openai > .agent/incident-inference-openai.json
python3 benchmark_inference.py grade .agent/incident-inference-mlx.json .agent/incident-inference-openai.json incident_inference_reviews.json
```

The grade command above expects the original recorded run IDs for the published reviews. Create new reviews for new recordings. SQLite traces are stored locally in `.agent/incident-inference.sqlite3` by default.

## Milestone 4 results

The [fresh held-out set](incident_grounding_cases.json) contains ten synthetic incidents: six answerable from the full corpus and four insufficient. Each case has a frozen, manually selected five-passage packet from the same runbooks and its own observations. This evaluates synthesis and verification, not retrieval ranking. One answerable case (`ground-02`) deliberately omits a decisive observation; it is still scored against corpus answerability and reported as missing gold evidence in its packet.

Both arms use the exact same draft from `gpt-4o-mini-2024-07-18` (temperature 0, JSON mode, 512-token completion cap). The fixed arm publishes the draft; the verified arm replays it and makes at most one additional call, using only the cited subset of the frozen packet. It can retain each claim verbatim or drop it; it cannot rewrite claims, supply new claims or citations, or query external evidence. If nothing supported answers the question, it abstains and names the observation needed. An initial abstention passes through unchanged. Unblinded, model-assisted reviews reused the Milestone 2 rubric.

| Metric | Fixed | Verified |
| --- | ---: | ---: |
| Unsupported conclusions | 7/16 | 6/15 |
| Correct conclusions | 9/16 | 9/15 |
| Correct diagnose-or-abstain status | 7/10 | 8/10 |
| Unnecessary abstentions | 0/6 | 0/6 |
| Fully correct runs | 3/10 | 4/10 |
| Median workflow latency | 2.30 s | 5.49 s |
| Estimated token cost for 10 cases | $0.00238 | $0.00449 |

The trace shows one changed answer. In `ground-08`, the fixed arm asserted a mount failure without evidence. The verified arm dropped that claim, abstained, and asked for attachment and mount status after rescheduling. It did not remove any supported claims in this sample. It still missed several errors: `ground-02` inferred a scheduling blocker from a Bound PVC without seeing the scheduler event, and `ground-03` treated a dashboard percentage as the trigger even though the HPA used a raw value. The result is directional: one improved case and six remaining unsupported claims do not justify enabling the verifier by default.

Verified-arm latency substitutes the measured original synthesis call for the near-zero replay call; cost includes the original draft plus verification, using provider-reported usage at the same list rates as Milestone 3. The estimates compare running each workflow independently; the pilot itself only paid for each shared draft once. The [per-case report](incident_grounding_report.json) and [review notes](incident_grounding_reviews.json) retain the decisions, timing, and grades. SQLite traces are stored locally in `.agent/incident-grounding.sqlite3`.

```bash
# Regrade the original recording offline.
python3 benchmark_grounding.py grade incident_grounding_report.json incident_grounding_reviews.json
# Start a new pilot with a configured OPENAI_API_KEY (public and synthetic evidence only).
python3 benchmark_grounding.py run > .agent/incident-grounding-runs.json
```

New runs require new semantic reviews keyed to their run IDs.
