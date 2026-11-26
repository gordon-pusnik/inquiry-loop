# CLI guide

Run these commands from the repository root. See the [quickstart](../README.md#quickstart) for installation, or run `python3 agent.py --help` for all options.

## Configuration

| Setting | Used by |
| --- | --- |
| `OPENAI_API_KEY` | OpenAI model requests. `AGENT_MODEL` overrides the default `gpt-4o-mini`. |
| `TAVILY_API_KEY` | Ranked search with Tavily, the general-agent default and the research-mode provider. |
| `SEARCH_PROVIDER` | General-agent search only. Set `instant-answer` for limited, keyless lookup instead of Tavily. |
| `TYPESAFE_API_KEY` | Jev evidence checks. `JEV_MODEL` selects a model version and defaults to `jev-latest`. |

Set credentials in `.env` or in the environment. Existing environment variables take precedence. `.env` and the default `.agent/` trace directory are ignored by Git. Keep [`.env.example`](../.env.example) limited to placeholders.

## General agent

```bash
python3 agent.py "List the files here and summarize requirements.txt"
python3 agent.py --llm=mlx --interactive
```

The general agent can use `list_dir`, `read_file`, `search_file`, `write_file`, `run_shell`, and `search_web`. File reads return a continuation offset for large files. The CLI reports partial inspection when the recorded reads leave gaps. `run_shell` launches an allowlisted executable directly with output and time limits. The default run budget is 16 model responses, including the final answer. Use `--max-steps` to change it.

To check search by itself or ask the general agent for sources, run:

```bash
python3 -c 'from inquiry_loop.providers.web_search import search_web; print(search_web("Python asyncio Queue shutdown documentation", max_results=3))'
python3 agent.py "Find the official Python asyncio Queue.shutdown documentation and cite its URL."
```

## Research mode

```bash
python3 agent.py --research 'Which gpt-oss models did OpenAI announce in August 2025, and under what license?'
```

One-shot `--research` requires a standalone question and a real synthesis backend; it cannot be combined with `--interactive`, `--resume`, or `--stub`. In an interactive session, type `/research` and then enter a standalone public question. Each question can be up to 400 characters and 50 words. Research turns stay in the active session. Enter `/new` to start a fresh session in the general agent mode.

```text
You> /research
You> What is the capital of Canada?
```

Jev shadow checking runs by default. A line such as `Jev evidence check (shadow) — supported (72%, jev-1.14.0)` appears below the answer. The percentage is Jev’s probability for the evidence-coverage label, not the probability that the answer is correct. Use `--research --evidence-check off` or interactive `/research off` to skip Jev.

To opt into one focused follow-up search:

```bash
python3 agent.py --research --evidence-check enforce --evidence-threshold 0.60 \
  'Which gpt-oss models were announced, and when did they become available?'
```

In an interactive session, use `/research enforce` or `/research enforce 0.65`. The provisional cutoff is `0.60` unless overridden. Enforcement considers a follow-up when Jev returns `partial`, `insufficient`, `supported` below the cutoff, or when Tavily returns no usable excerpts. The model proposes one query targeting the evidence gap. Python rejects invalid or repeated queries, merges up to five new usable excerpts without duplicating URLs, and rechecks only if usable evidence was added. The CLI shows why it searched again and any available before-and-after judgments. The synthesis prompt asks the model to separate supported findings from what remains unresolved.

A Jev error alone never triggers a search; coverage is reported as unassessed. An unsuccessful follow-up leaves the initial excerpts available. Enforcement is capped at two searches, two Jev checks, and two LLM calls. With `--max-steps 1`, it reserves the single model call for the answer. Enforcement is not yet validated to improve citation support on held-out questions; see the [Jev research plan](plans/JEV_RESEARCH_PLAN.md).

## Interactive sessions

Use the Left and Right arrow keys to move within a question before submitting it. This works in a terminal when Python has `readline` support.

| Command | Behavior |
| --- | --- |
| `/research [off\|shadow\|enforce [threshold]]` | Switch to research in the current session. With no argument, use `shadow`. |
| `/new` | Start a clean session in the general agent mode while keeping the model loaded. |
| `/sessions` | List saved sessions and backend identities. |
| `/resume ID` | Restore a completed conversation checkpoint using a full session ID or unique prefix. |
| `/stats [ID]` | Inspect run statuses, durations, model calls, and token throughput. |
| `/help`, `/quit` | List commands or exit. |

The default database is `.agent/sessions.sqlite3`. Use `--db PATH` to choose another location or `--no-trace` to work without persistence. With tracing disabled, `/research` and `/new` still switch modes. Resume requires the original backend, model identity, and working directory. It restores the last completed checkpoint without replaying tool calls. From the command line, use `python3 agent.py --resume ID --interactive`, or give `--resume ID` a follow-up request. Select the original backend and model if they differ from the defaults.

Each run’s timing summary starts with the selected backend (for example, `mlx processed request in …`) and reports elapsed time and tokens per second when available. Interactive mode reports model initialization separately. Local decode speed and API request throughput use different timing bases.

## Model backends

| Backend | Run command | Setup |
| --- | --- | --- |
| **OpenAI** | `python3 agent.py --llm=openai -i` | Set `OPENAI_API_KEY`. This is the default backend. |
| **MLX-LM** | `python3 agent.py --llm=mlx -i` | Apple Silicon with compatible MLX weights and a tokenizer that supports native tool calls. Use `--model` for a model repository or `--model-path` for a local directory. |
| **Strata** | `python3 agent.py --llm=strata -i` | Start Strata and load a model. Copy the serving URL and model ID from its Serving screen. |
| **AirLLM** | `python3 agent.py --llm=airllm --model-path /path/to/Qwen3-snapshot -i` | Experimental Apple Silicon backend. Install compatible AirLLM and MLX packages separately and supply a local dense Qwen3 snapshot. |
| **Offline stub** | `python3 agent.py --stub --no-trace` | Runs a fixed file-tool demonstration with Python’s standard library. |

For MLX, set `MLX_MODEL` in `.env` or select a model on the command line. The built-in fallback is `mlx-community/Qwen3-4B-4bit`. The example config selects `mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit`. First use may download the selected weights, which are kept in the Hugging Face cache. `MLX_MAX_TOKENS` and `MLX_MAX_CONTEXT_TOKENS` control output and total context. The prompt must fit within the context after reserving the output budget.

For Strata, set `STRATA_BASE_URL` and `STRATA_MODEL` in `.env`, or pass `--base-url` and `--model`. The default endpoint is `http://127.0.0.1:8080/v1`, and the client automatically selects the model when the server lists exactly one. Set `STRATA_API_KEY` if the server requires it. `STRATA_MAX_TOKENS` sets the output budget. See the [Strata integration notes](plans/STRATA_BACKEND_PLAN.md) for details.

## Scope and data handling

The general agent’s file tools operate with the running user’s permissions. They can access paths outside the project and overwrite existing files. Use a dedicated working directory or an OS sandbox if you need to limit that access.

With hosted inference, tool observations can be sent to the configured provider. MLX, AirLLM, and a local Strata endpoint keep model execution on your machine, but using Tavily or Jev still makes network requests: Tavily receives web queries, and TypeSafe receives the public research question and excerpts when Jev is enabled. Hosted incident benchmarks send only public runbook excerpts and synthetic case evidence. SQLite traces can retain requests, excerpts, tool output, and answers.
