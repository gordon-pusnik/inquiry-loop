# Inquiry Loop

Inquiry Loop is a Python CLI for experimenting with local and hosted language models, including local GGUF models through llama.cpp. Use it to inspect files with a tool-using agent, ask cited web research questions, or trace runs in SQLite. A separate benchmark suite evaluates evidence retrieval, cited incident diagnoses, and local versus hosted inference.

## Quickstart

Run commands from the repository root with Python 3.11 or newer. Try the scripted tool loop without installing dependencies or setting credentials:

```bash
python3 agent.py --stub --no-trace "List the files here"
```

To use a real model backend, install the dependencies and create `.env` if it doesn't already exist:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
[ -f .env ] || cp .env.example .env
```

Set `OPENAI_API_KEY` for the default backend, then try a one-shot task or start an interactive session:

```bash
python3 agent.py "List the files here and summarize requirements.txt"
python3 agent.py --interactive
```

At the interactive prompt, type `/status` to see the active engine and model. [Interactive commands](docs/cli.md#interactive-sessions)

### Run a local GGUF with llama.cpp

Install the unified `llama` CLI or `llama-server` and download an instruct GGUF with a tool-aware chat template. Inquiry Loop starts the local server when the session opens and stops it when the session exits. [Installation, Hugging Face downloads, and configuration](docs/cli.md#local-gguf-models-with-llamacpp)

```bash
python3 agent.py --llm=llama --model-path /path/to/model.gguf --interactive
```

Set `LLAMA_MODEL_PATH` in `.env` to omit `--model-path`. If the unified binary is at `~/.local/bin/llama`, set `LLAMA_SERVER=~/.local/bin/llama` to select it explicitly.

### Find and download models

After installing the Python dependencies, run `python3 agent.py --stub --no-trace --interactive` to open the prompt before you have a model.

```text
You> /search Qwen3-30B-A3B-Instruct-2507
You> /download mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit
```

Search returns up to five MLX and GGUF matches. Use `/search --mlx TERM` or `/search --gguf TERM` to filter. Downloads go to `~/.cache/huggingface/hub` by default. If a GGUF repository offers multiple variants, use `/download OWNER/REPO FILENAME` to choose one. For a downloaded GGUF, set `LLAMA_MODEL_PATH` to the printed path and start a session with `python3 agent.py --llm=llama --interactive`. [Model discovery and downloads](docs/cli.md#find-and-download-models)

### Run cited web research

Set `TAVILY_API_KEY` for web search. Set `TYPESAFE_API_KEY` for the default Jev evidence check, or skip Jev as shown here. With `LLAMA_MODEL_PATH` configured, add `--llm=llama` to synthesize the answer locally.

```bash
python3 agent.py --research --evidence-check off \
  'Which gpt-oss models did OpenAI announce in August 2025, and under what license?'
```

The general agent runs tools with your user permissions. Hosted models may receive tool output; web research can contact Tavily and TypeSafe even with a local model. Use `--no-trace` to skip local SQLite persistence. See [data handling](docs/cli.md#scope-and-data-handling) before using sensitive inputs.

## Documentation

- [CLI and configuration](docs/cli.md): backends, research modes, interactive sessions, and data handling
- [Architecture](docs/architecture.md): workflows and repository layout
- [Tests and benchmarks](docs/benchmarks.md): commands, findings, and recorded results
- Plans for [incident runbooks](docs/plans/INCIDENT_RUNBOOK_PLAN.md), [Jev research](docs/plans/JEV_RESEARCH_PLAN.md), and [Strata](docs/plans/STRATA_BACKEND_PLAN.md)
- [Product architecture and milestones](PRODUCT_MILESTONES.md)

## License

Copyright 2026 Gordon Pusnik and contributors. Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).
