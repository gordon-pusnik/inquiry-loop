# Inquiry Loop

Inquiry Loop is a Python CLI for experimenting with local and hosted language models. Use it to inspect files with a tool-using agent, ask cited web research questions, or trace runs in SQLite. A separate benchmark suite evaluates evidence retrieval, cited incident diagnoses, and local versus hosted inference.

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

For cited web research, also set `TAVILY_API_KEY`. Set `TYPESAFE_API_KEY` for the default Jev evidence check, or skip Jev as shown here:

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
