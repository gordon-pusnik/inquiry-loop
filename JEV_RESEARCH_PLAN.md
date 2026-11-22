# Jev evidence checking for AI research

**Status** — First milestone complete. The research baseline, Jev shadow mode, and live pilot are verified. Enforcement remains proposed.

Add an opt-in research mode to the existing CLI for public questions about AI releases, papers, and benchmarks. Tavily retrieves evidence, Jev judges whether the excerpts cover the question, and the chosen LLM writes a cited answer. Python owns the search budget and decides what to do with the judgment. The portfolio deliverable is a reproducible comparison showing where this workflow improves answers and what it costs.

## First milestone results

- Live baseline verified with Tavily and OpenAI. The sample returned both gpt-oss names and the Apache 2.0 license with a link to OpenAI's announcement. It also described the models as open-source, while the announcement calls them open-weight. That wording drift is an example for claim-level evaluation. [OpenAI announcement](https://openai.com/index/introducing-gpt-oss/)
- Automated verification passed all 101 tests on Python 3.13. Research tests cover baseline and shadow prompt parity, SQLite trace events, the TypeSafe HTTP contract, malformed responses, and citation validation.
- The offline replay ran 20 cases, with eight development and twelve held-out cases. The source-presence comparator agreed on 7 of 18 judged cases. It marked 11 partial or insufficient cases `supported`. Two cases with no evidence were skipped. The fixtures use curated excerpts rather than a sample of live Tavily traffic.
- Live `jev-1.14.0` matched all seven judged development cases. Before running the held-out split, set a pilot cutoff of `P(supported) >= 0.75` when Jev selects `supported`. The two supported development cases scored 0.98 and 0.80. All five other cases scored at most 0.02. The 0.75 cutoff leaves some room below the weaker supported example. It is an evaluation setting for this small pilot.
- Jev matched all 11 judged held-out cases. Across both splits, the model matched 18 of 18 labels, skipped two empty cases, and returned zero provider errors. Median judgment latency was 195 ms. The 0.75 cutoff would have accepted only one of the five genuinely supported held-out cases, triggering a follow-up for the other four. Keep shadow mode and revisit the cutoff with more varied examples before enforcement.
- A live CLI run recorded `supported` with a probability of 0.89 in the SQLite trace between the Tavily result and LLM synthesis. The retrieval included OpenAI's announcement. The synthesized answer cited two secondary sources and repeated the open-source wording, so source selection and claim fidelity remain relevant to the next evaluation.
- Local results are in `.agent/jev-shadow-dev.json`, `.agent/jev-shadow-held-out.json`, and `.agent/jev-shadow-live.sqlite3`. To reproduce the reports, set `JEV_MODEL=jev-1.14.0` and run `python3 benchmark_research.py --live --split dev` and `python3 benchmark_research.py --live --split held-out`.

## Proposed enforcement workflow

1. Accept a standalone public research question. Search Tavily and keep up to five results with usable URLs and excerpts. Preserve the link associated with each excerpt and deduplicate across searches.
2. Ask Jev whether the excerpts directly support an answer to the question. Include an as-of date when the question depends on timing. The Jev `Choice` response supplies a label and per-label probabilities. [TypeSafe Choice documentation](https://docs.typesafe.ai/primitives/choice)
3. Apply the routing policy below. If a follow-up is warranted, ask the existing LLM for one focused query, validate it, and perform at most one additional Tavily search. Judge the combined evidence once. A missing, invalid, or repeated follow-up query ends the search.
4. Give the synthesizing LLM the retained evidence and the evidence status. Ask it to cite only retained sources and identify what remains uncertain. Verify citation references against the retained URL list. Human evaluation measures whether the citations actually support the claims.

| Evidence judgment | Application action |
| --- | --- |
| `supported` with sufficient probability | Write a cited answer using the available evidence. |
| `partial`, `insufficient`, or low probability for `supported` | Try one focused follow-up if the budget permits. Then report supported findings and identify unresolved claims. |
| Empty or unusable results | Skip the Jev call. Treat the evidence as insufficient and allow the single follow-up. |
| Search provider error | Report the search failure and any previously supported findings. Keep provider failures distinct from missing evidence. |
| Jev timeout, error, or invalid response | Record the failure and use the baseline research behavior. Treat coverage as unassessed and make the limitation visible in the answer. |

`Supported` means the excerpts directly cover the material parts of the question. `Partial` means they cover some of those parts or contain unresolved conflicts. `Insufficient` means they provide no direct support. An announcement about a future launch counts as evidence of the announcement. An availability claim needs an excerpt that establishes availability for the audience being asked about.

## Project integration

- **CLI.** `--research` and interactive `/research` use shadow checking by default. Interactive research runs stay in the active session, and `/new` returns to default mode. `--evidence-check off` runs the single-request baseline. Proposed `enforce` mode would apply the routing policy as an opt-in after evaluation.
- **Orchestration.** Put the workflow in `research.py`. Reuse `web_search.search`, `ModelResponse`, `RequestMetrics`, and the existing `SessionRunner` and `RunContext` interfaces. The research runner owns search calls and final synthesis. Tool calls returned by the synthesizing LLM have no execution path in this workflow. Keep the existing CLI entry point and `run_agent()` available for general tasks.
- **Jev client.** Put the provider-specific request and response handling in `jev_eng.py`. Use the existing HTTP and `.env` dependencies, a bounded timeout, and a separate `TYPESAFE_API_KEY`. Validate the expected question ID, label, and probability distribution. Record the resolved model ID. For benchmark reproducibility, use a versioned model ID when available. [TypeSafe quick start](https://docs.typesafe.ai/introduction/quickstart)
- **Data boundary.** Send TypeSafe only the standalone public question, relevant date, and bounded public source excerpts. Keep prior session turns, local file observations, and credentials out of that payload. Treat excerpts as untrusted data and prevent provider output from choosing executable tools or extending the request budget. Explain in CLI help and the README that a local LLM can still use the hosted Jev service when enabled.
- **Tracing.** Add an evidence event to the existing SQLite event stream using `RunContext.emit`. Record mode, search attempt, source count, label, probabilities, applied threshold, route, model ID, latency, usage when available, and sanitized error category. Reuse the existing search observations for source content. Keep `--no-trace` functional.

## Implementation and evaluation

1. **Build the baseline.** Implement the traced research path with one Tavily search and cited synthesis. Save about 20 dated public research cases with frozen search results, expected coverage labels, and expected claim-level evidence. Include current releases, planned availability, conflicting reports, benchmark scope, irrelevant links, and missing evidence. Prepare follow-up fixtures for cases where a second search could help.
2. **Add Jev in shadow mode.** Verify one live request with public data and mock the provider in automated tests. Compare Jev’s coverage labels against human labels. Use a small development subset to choose a tentative threshold for the `supported` probability, then freeze it before looking at the held-out cases. Treat the pilot results as directional. [TypeSafe confidence documentation](https://docs.typesafe.ai/confidence)
3. **Add opt-in enforcement.** Test each route with fake search and Jev clients. Cover empty results, duplicate queries, conflicts, low probability, provider errors, malformed responses, invalid citations, and excerpts containing instructions. Verify that the research mode attempts at most two searches and two Jev judgments per question, that the baseline sends no Jev requests, and that the general agent still passes its regression suite.
4. **Compare and document.** Replay the same cases through the baseline and enforcement modes with the same synthesis model, rubric, and initial search results. Include a simple two-search baseline on the same cases to see how much improvement comes from extra retrieval alone. Report unsupported material claims, answers that unnecessarily withheld supported information, follow-up search rate, latency, and per-run cost where usage is available. Publish the case counts, representative failures, and CLI commands in the README.

The first milestone is a repeatable baseline and a shadow report. Move to enforcement when the held-out run reduces unsupported claims and the report makes the effect on useful answers, latency, and cost clear. Use a side-by-side demo with one directly supported question and one where the first results establish an announcement while leaving availability unresolved.
