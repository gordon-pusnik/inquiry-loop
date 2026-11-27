# CLI guide

Run these commands from the repository root. See the [quickstart](../README.md#quickstart) for installation, or run `python3 agent.py --help` for all options.

## Configuration

| Setting | Used by |
| --- | --- |
| `OPENAI_API_KEY` | OpenAI model requests. `AGENT_MODEL` overrides the default `gpt-4o-mini`. |
| `OPENAI_REASONING_EFFORT` | Initial reasoning effort for GPT-6.2 Sol and Astra. Values are `low`, `medium`, `high`, and `max`. Defaults to `medium`. |
| `LLAMA_MODEL_PATH` | Default local GGUF file for `--llm=llama`. `--model-path` overrides it for one run. |
| `LLAMA_SERVER` | Optional path to the unified `llama` or standalone `llama-server` executable. |
| `TAVILY_API_KEY` | Ranked search with Tavily, the general-agent default and the research-mode provider. |
| `SEARCH_PROVIDER` | General-agent search only. Set `instant-answer` for limited, keyless lookup instead of Tavily. |
| `TYPESAFE_API_KEY` | Jev evidence checks. `JEV_MODEL` selects a model version and defaults to `jev-latest`. |

Set credentials in `.env` or in the environment. Existing environment variables take precedence. `.env` and the default `.agent/` trace directory are ignored by Git. Keep [`.env.example`](../.env.example) limited to placeholders.

## General agent

```bash
python3 agent.py "List the files here and summarize requirements.txt"
python3 agent.py --llm=mlx --interactive
```

The general agent can use `list_dir`, `read_file`, `search_file`, `write_file`, `run_shell`, and `search_web`. File reads return a continuation offset for large files. The CLI reports partial inspection when the recorded reads leave gaps. `run_shell` launches allowlisted executables, including `git`, with output and time limits. In `/code` mode, you can ask the agent to run commands such as `git status --short`, `git diff`, or `git log -5 --oneline`. The default run budget is 16 model responses, including the final answer. Use `--max-steps` to change it.

To check search by itself or ask the general agent for sources, run:

```bash
python3 -c 'from inquiry_loop.providers.web_search import search_web; print(search_web("Python asyncio Queue shutdown documentation", max_results=3))'
python3 agent.py "Find the official Python asyncio Queue.shutdown documentation and cite its URL."
```

## Research mode

```bash
python3 agent.py --research 'Which gpt-oss models did OpenAI announce in August 2025, and under what license?'
```

One-shot `--research` requires a standalone question and a real synthesis backend; it cannot be combined with `--interactive`, `--resume`, or `--stub`. In an interactive session, type `/research` and then enter a standalone public question. Each question can be up to 400 characters and 50 words. The CLI saves research turns in a separate research conversation and presents each question to the model on its own. Enter `/new` to start a fresh session in the general agent mode.

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
| `/model openai\|mlx\|llama` | Switch to that engine's configured model and keep the active research settings. |
| `/effort [low\|medium\|high\|max]` | Show or set reasoning effort for a supported OpenAI model. |
| `/chat` | Enter text chat and restore its conversation for the current model. Chat makes a single model call per turn and uses no tools. |
| `/code` | Enter the general agent and restore its tool-using conversation. This is the default at launch. |
| `/search [--mlx\|--gguf] TERM` | Find up to five local models on Hugging Face. |
| `/download OWNER/REPO [FILENAME]` | Save a model snapshot or selected file to the Hugging Face cache. |
| `/status` | Show the active engine, model ID or local file path, mode, and reasoning effort. Works with `--no-trace`. |
| `/research [off\|shadow\|enforce [threshold]]` | Switch to research in the current session. With no argument, use `shadow`. |
| `/new` | Start a clean session in the general agent mode while keeping the model loaded. |
| `/sessions` | List saved sessions and backend identities. |
| `/resume ID` | Restore a completed conversation checkpoint using a full session ID or unique prefix. |
| `/stats [ID]` | Inspect run statuses, durations, model calls, and token throughput. |
| `/help`, `/quit` | List commands or exit. |

For example, with a local GGUF loaded:

```text
You> /status
Engine: llama
Model: /path/to/model.gguf
Mode: chat
Reasoning effort: unavailable
```

For llama.cpp, `/status` shows the path selected by `--model-path` or `LLAMA_MODEL_PATH`, including the GGUF filename when it comes from a Hugging Face snapshot. For MLX, it shows the model repository ID or local model directory.

Use `/model openai`, `/model mlx`, or `/model llama` to switch without exiting. The command keeps the current mode and uses the selected engine's `.env` settings. Launch flags such as `--model-path` apply to the initial load. The CLI keeps a local model resident during a switch to OpenAI and unloads it before loading a different local model. Traced conversations stay with the backend, model, and mode that created them. Returning to that combination restores its active conversation.

The default database is `.agent/sessions.sqlite3`. Use `--db PATH` to choose another location or `--no-trace` to work without persistence. With tracing disabled, mode switches still work and each request starts with fresh history. Resume requires the original backend, model identity, and working directory. It restores the mode and last completed checkpoint without replaying tool calls. Research resumes in shadow mode in a new CLI process. From the command line, use `python3 agent.py --resume ID --interactive`, or give `--resume ID` a follow-up request. Select the original backend and model if they differ from the defaults.

Each run’s timing summary starts with the selected backend (for example, `mlx processed request in …`) and reports elapsed time and tokens per second when available. Interactive mode reports model initialization separately. Local decode speed and API request throughput use different timing bases.

### Compare a local model with OpenAI

With your MLX model configured and `AGENT_MODEL=gpt-6.2-sol` set in `.env`, start with `python3 agent.py --llm=mlx --interactive`. Enter the same prompt on both engines.

```text
You> /chat
You> Explain SQLite transactions in two sentences.
You> /model openai
You> /effort low
You> Explain SQLite transactions in two sentences.
You> /model mlx
You> /status
```

After the final switch you are back in your MLX chat. Any follow-up uses that chat's history. Use a fresh `/research` question for a cited comparison. Each research request uses only that question and its retrieved evidence.

## Find and download models

In an interactive session, `/search TERM` lists up to five Hugging Face model IDs, alternating MLX and GGUF matches ranked by downloads within each format. Use `/search --mlx TERM` or `/search --gguf TERM` to select one format. Use `/download OWNER/REPO` to cache a model by ID. Both commands contact Hugging Face directly. After installing the Python dependencies, run `python3 agent.py --stub --no-trace --interactive` to use them before you have a model.

```text
You> /search Qwen3-30B-A3B-Instruct-2507
You> /download mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit
You> /search --gguf Qwen3-30B-A3B-Instruct-2507
You> /download bartowski/Qwen_Qwen3-30B-A3B-Instruct-2507-GGUF
You> /download bartowski/Qwen_Qwen3-30B-A3B-Instruct-2507-GGUF Qwen_Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf
```

A repo with multiple GGUF variants lists filenames so you can select a quantization in the next command. For split GGUFs, the selected variant's shards download together and the command prints the first shard's path. Other model repos download as snapshots unless you specify a file. Downloads go to the Hugging Face cache, normally `~/.cache/huggingface/hub`. For a gated repo, request access on Hugging Face and authenticate with `hf auth login`.

For your next session, set `LLAMA_MODEL_PATH` to the printed GGUF path and start with `--llm=llama`, or set `MLX_MODEL` to the MLX repo ID and start with `--llm=mlx`.

Outside Inquiry Loop, use `hf models ls --search TERM --filter mlx --limit 5` (or `--filter gguf`) and `hf download OWNER/REPO [FILENAME]`. [Hugging Face CLI guide](https://huggingface.co/docs/huggingface_hub/guides/cli)

## Model backends

| Backend | Run command | Setup |
| --- | --- | --- |
| **OpenAI** | `python3 agent.py --llm=openai -i` | Set `OPENAI_API_KEY`. This is the default backend. |
| **MLX-LM** | `python3 agent.py --llm=mlx -i` | Apple Silicon with compatible MLX weights and a tokenizer that supports native tool calls. Use `--model` for a model repository or `--model-path` for a local directory. |
| **llama.cpp** | `python3 agent.py --llm=llama --model-path /path/to/model.gguf -i` | Install the unified `llama` CLI or `llama-server` and use a local instruct GGUF with a tool-aware chat template. |
| **Strata** | `python3 agent.py --llm=strata -i` | Start Strata and load a model. Copy the serving URL and model ID from its Serving screen. |
| **AirLLM** | `python3 agent.py --llm=airllm --model-path /path/to/Qwen3-snapshot -i` | Experimental Apple Silicon backend. Install compatible AirLLM and MLX packages separately and supply a local dense Qwen3 snapshot. |
| **Offline stub** | `python3 agent.py --stub --no-trace` | Runs a fixed file-tool demonstration with Python’s standard library. |

For MLX, set `MLX_MODEL` in `.env` or select a model on the command line. The built-in fallback is `mlx-community/Qwen3-4B-4bit`. The example config selects `mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit`. First use may download the selected weights, which are kept in the Hugging Face cache. `MLX_MAX_TOKENS` and `MLX_MAX_CONTEXT_TOKENS` control output and total context. The prompt must fit within the context after reserving the output budget.

For Strata, set `STRATA_BASE_URL` and `STRATA_MODEL` in `.env`, or pass `--base-url` and `--model`. The default endpoint is `http://127.0.0.1:8080/v1`, and the client automatically selects the model when the server lists exactly one. Set `STRATA_API_KEY` if the server requires it. `STRATA_MAX_TOKENS` sets the output budget. See the [Strata integration notes](plans/STRATA_BACKEND_PLAN.md) for details.

### OpenAI models and reasoning effort

Set `AGENT_MODEL=gpt-6.2-sol` or `AGENT_MODEL=gpt-6.2-astra` in `.env`. Both use the OpenAI Responses API for chat, coding tools, and research synthesis. The default `gpt-4o-mini` uses Chat Completions. [OpenAI model guide](https://developers.openai.com/api/docs/guides/latest-model)

```bash
AGENT_MODEL=gpt-6.2-sol OPENAI_REASONING_EFFORT=medium python3 agent.py --llm=openai --interactive
```

Use `/effort` to see the current effort and `/effort low`, `/effort medium`, `/effort high`, or `/effort max` to change it. The setting applies to subsequent requests and persists when you switch away from the model and back in the same process. Local models and legacy OpenAI models report effort as unavailable.

The Responses adapter sends requests with `store=false`. Tracing stores replayable Responses items, including encrypted reasoning, in the local SQLite checkpoint so chat and code can resume. Research still sends each question and its evidence independently. [OpenAI reasoning guide](https://developers.openai.com/api/docs/guides/reasoning#preserve-reasoning-without-stored-responses)

### Local GGUF models with llama.cpp

Install the [unified llama CLI](https://llama.app/docs/installation) or use a [llama.cpp release](https://github.com/ggml-org/llama.cpp/releases) for your platform. Inquiry Loop looks for `llama-server` and then `llama` on your PATH. You can set `LLAMA_SERVER` to an explicit executable path, such as `LLAMA_SERVER=~/.local/bin/llama` in `.env`. Inquiry Loop invokes `llama serve` for the unified CLI, binds to loopback, and stops the server when the session exits. For general agent tasks, choose a GGUF with a [tool-aware Jinja chat template](https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md).

```bash
python3 agent.py --llm=llama --model-path /path/to/model.gguf "List the files here"
python3 agent.py --llm=llama --model-path /path/to/model.gguf --interactive
python3 agent.py --llm=llama --model-path /path/to/model.gguf --research --evidence-check enforce "What is the population of Phoenix?"
```

Set `LLAMA_MODEL_PATH` in `.env` to omit `--model-path`. Saved sessions use the resolved file path to identify the model. Set `LLAMA_CONTEXT_SIZE` (default `8192`), `LLAMA_MAX_TOKENS` (default `2048`), `LLAMA_GPU_LAYERS` (default `999`, use `0` for CPU), or `LLAMA_TIMEOUT` (default `300` seconds) to suit the model and hardware.

When a llama.cpp request runs out of context, Inquiry Loop retries with fewer complete prior conversation turns. It keeps the current request and its tool results together and reports how many turns it omitted. The full history stays in the saved session. If the current task alone exceeds the window, narrow the task or increase `LLAMA_CONTEXT_SIZE`.

The llama.cpp equivalents of `MLX_MODEL`, `MLX_MAX_TOKENS`, and `MLX_MAX_CONTEXT_TOKENS` are `LLAMA_MODEL_PATH`, `LLAMA_MAX_TOKENS`, and `LLAMA_CONTEXT_SIZE`. For the Qwen3 30B Instruct-2507 model above, download the [Q4_K_M GGUF](https://huggingface.co/bartowski/Qwen_Qwen3-30B-A3B-Instruct-2507-GGUF/blob/main/Qwen_Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf) to the same Hugging Face cache as the MLX weights.

```bash
hf download bartowski/Qwen_Qwen3-30B-A3B-Instruct-2507-GGUF \
  Qwen_Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf \
  --cache-dir "$HOME/.cache/huggingface/hub"
```

Use the `.gguf` snapshot path printed by `hf download` for `LLAMA_MODEL_PATH`.

## Scope and data handling

The general agent’s file tools operate with the running user’s permissions. They can access paths outside the project and overwrite existing files. Use a dedicated working directory or an OS sandbox if you need to limit that access.

With hosted inference, tool observations can be sent to the configured provider. MLX, llama.cpp, AirLLM, and a local Strata endpoint keep model execution on your machine. Tavily receives web queries, and TypeSafe receives the public research question and excerpts when Jev is enabled. Hosted incident benchmarks send only public runbook excerpts and synthetic case evidence. SQLite traces can retain requests, excerpts, tool output, and answers.
