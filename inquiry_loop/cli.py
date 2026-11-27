"""Command-line entry point and interactive session commands."""

from __future__ import annotations

import argparse
from functools import partial
import os
import sqlite3
import sys
import time

from inquiry_loop.agent import MAX_STEPS, create_model, run_agent
from inquiry_loop.backends.openai_responses_eng import REASONING_EFFORTS
from inquiry_loop.backends.stub_eng import StubModel
from inquiry_loop.engine import chat
from inquiry_loop.interactive_models import InteractiveModels, SWITCHABLE, close_model
from inquiry_loop.request_metrics import RequestMetrics
from inquiry_loop.response_types import RunContext, RunStatus
from inquiry_loop.session_store import DEFAULT_DB, SessionRunner, SessionStore


def _evidence_threshold(value):
    try:
        cutoff = float(value)
    except ValueError:
        raise ValueError('Evidence threshold must be greater than 0 and at most 1.') from None
    if not 0 < cutoff <= 1:
        raise ValueError('Evidence threshold must be greater than 0 and at most 1.')
    return cutoff


def _research_workflow(on_evidence, mode, cutoff=None):
    from inquiry_loop.research import run_research

    if mode == 'off':
        return partial(run_research, verbose=True)

    from inquiry_loop.providers.jev_eng import JevClient

    judge = JevClient()
    if not judge.configured:
        print(f"Jev {mode} check needs TYPESAFE_API_KEY. Evidence coverage cannot be assessed.", file=sys.stderr)
    options = {} if cutoff is None else {'supported_cutoff': cutoff}
    return partial(run_research, judge=judge.judge, on_evidence=on_evidence, verbose=True,
                   enforce=mode == 'enforce', **options)


def _research_settings(mode, cutoff=None):
    settings = {'evidence_check': mode}
    if mode == 'enforce':
        from inquiry_loop.research import DEFAULT_SUPPORTED_CUTOFF

        settings['supported_cutoff'] = DEFAULT_SUPPORTED_CUTOFF if cutoff is None else cutoff
    return settings


def _run_chat(request, model, *, max_steps, metrics, context=None):
    context = context if context is not None else RunContext()
    messages = [*context.previous_messages, {"role": "user", "content": request}]
    context.emit('model_started', {'step': 1, 'message_count': len(messages)})
    started = time.monotonic()
    response = chat(messages, model, context=context)
    context.emit('model_response', {'step': 1, 'elapsed': time.monotonic() - started,
                                    'tool_calls': response.tool_calls, 'text': response.text,
                                    'stats': response.stats, 'finish_reason': response.finish_reason})
    if metrics is not None:
        metrics.record(response.stats)
    return context.finish(RunStatus.COMPLETED, response.text, messages)


def _print_run_output(answer, evidence_checks, metrics, elapsed, backend):
    print(answer)
    if metrics.history_turns_omitted:
        count = metrics.history_turns_omitted
        turns = 'turn' if count == 1 else 'turns'
        print(f'\nContext: omitted {count} older conversation {turns} from the model prompt. '
              'Saved history was retained.')
    if evidence_checks:
        from inquiry_loop.research import format_evidence_check

        for judgment in evidence_checks:
            print(f"\n{format_evidence_check(judgment)}")
    print(metrics.summary(elapsed, backend))


def _model_identity(model, backend):
    identity = next((str(value) for value in (getattr(model, 'model_id', None),
                    getattr(model, 'model', None), getattr(model, 'model_path', None))
                     if isinstance(value, (str, os.PathLike))), type(model).__name__)
    if backend == 'strata' and isinstance(getattr(model, 'base_url', None), str):
        identity += f' @ {model.base_url}'
    return identity


def _print_model_status(models, mode, research_settings):
    if models.model is None:
        print('No active model. Choose an engine with /model.')
        return
    model_path = getattr(models.model, 'model_path', None) if models.backend == 'llama' else None
    label = model_path or _model_identity(models.model, models.backend)
    print(f'Engine: {models.backend}\nModel: {label}')
    mode_label = f"research ({research_settings['evidence_check']})" if mode == 'research' else mode
    print(f'Mode: {mode_label}')
    _print_reasoning_effort(models)


def _print_reasoning_effort(models):
    effort = getattr(models.model, 'reasoning_effort', None) if models.backend == 'openai' else None
    print(f"Reasoning effort: {effort if effort in REASONING_EFFORTS else 'unavailable'}")


def run_interactive(model, initial_request: str = "", max_steps: int = MAX_STEPS,
                    runner: SessionRunner | None = None, *, backend: str, model_options=None) -> None:
    """Own the interactive models and keep traced sessions tied to their model."""
    models = InteractiveModels(create_model, backend, model, model_options)
    try:
        _interactive_loop(models, initial_request, max_steps, runner)
    finally:
        models.close()


def _interactive_loop(models, initial_request, max_steps, runner):
    try:
        import readline  # Importing enables terminal line editing for input().
    except ImportError:
        pass

    if runner is None:
        print("Interactive mode: /model, /effort, /chat, /code, /research, /search, /download, /new, /status, /help, /quit. Each task starts fresh (tracing disabled).")
    else:
        print("Interactive mode: /model, /effort, /chat, /code, /research, /search, /download, /new, /status, /sessions, /resume ID, /stats [ID], /help, /quit.")
        if runner.sid:
            print(f"Resumed session {runner.sid}; enter a new request to continue. No tools were replayed.")
    pending = initial_request.strip()
    mode = runner.mode if runner is not None else 'code'
    research_settings = _research_settings('shadow')
    research_workflow = None
    evidence_checks = []
    store = runner.store if runner is not None else None
    runners = {(runner.backend, runner.model, mode): runner} if runner is not None else {}

    def select_runner():
        if store is None or models.model is None:
            return None
        backend = models.backend
        identity = _model_identity(models.model, backend)
        key = (backend, identity, mode)
        if key not in runners:
            runners[key] = SessionRunner(store, backend, identity, mode=mode)
        return runners[key]

    try:
        while True:
            request = pending
            pending = ""
            if not request:
                # libedit counts the prompt width when redrawing history.
                print()
                request = input("You> ").strip()
            if request.lower() in ("/quit", "/exit", "quit", "exit"):
                break
            if not request:
                continue
            if request.startswith('/'):
                command, *arguments = request.split()
                try:
                    if command == '/help' and not arguments:
                        print('/model openai|mlx|llama: switch to the configured model')
                        print('/effort [low|medium|high|max]: show or set the active OpenAI reasoning effort')
                        print('/search [--mlx|--gguf] TERM: find local models on Hugging Face')
                        print('/download OWNER/REPO [FILENAME]: cache a Hugging Face model or file')
                        print('/status: show the active engine, model, and mode')
                        print('/chat: text chat; /code: general agent with tools')
                        print('/research [off|shadow|enforce [threshold]]: switch to research; /new: clean conversation in default mode; /sessions: list saved sessions; /resume ID: restore a completed checkpoint; /stats [ID]: run outcomes and timings; /quit: exit.')
                        continue
                    if command == '/status':
                        if arguments:
                            raise ValueError('Usage: /status')
                        _print_model_status(models, mode, research_settings)
                        continue
                    if command == '/effort':
                        if arguments:
                            if len(arguments) != 1 or arguments[0] not in REASONING_EFFORTS:
                                raise ValueError('Usage: /effort [low|medium|high|max]')
                            if (models.backend != 'openai'
                                    or getattr(models.model, 'reasoning_effort', None) not in REASONING_EFFORTS):
                                raise ValueError('Reasoning effort is unavailable for the active model.')
                            models.model.reasoning_effort = arguments[0]
                        _print_reasoning_effort(models)
                        continue
                    if command == '/model':
                        if len(arguments) != 1 or arguments[0] not in SWITCHABLE:
                            raise ValueError('Usage: /model openai|mlx|llama')
                        try:
                            models.switch(arguments[0])
                        finally:
                            runner = select_runner()
                        _print_model_status(models, mode, research_settings)
                        continue
                    if command in ('/chat', '/code'):
                        if arguments:
                            raise ValueError(f'Usage: {command}')
                        mode = command[1:]
                        runner = select_runner()
                        print(f'{mode.capitalize()} mode.')
                        continue
                    if command in ('/search', '/download'):
                        try:
                            from inquiry_loop.providers.hub_models import run_model_command
                        except ImportError as exc:
                            raise ValueError("Install Hugging Face Hub with python3 -m pip install -r requirements.txt.") from exc

                        run_model_command(command, arguments)
                        continue
                    if command == '/research':
                        if isinstance(models.model, StubModel):
                            raise ValueError('/research requires a real synthesis backend.')
                        evidence_mode, *threshold = arguments or ['shadow']
                        if evidence_mode not in ('off', 'shadow', 'enforce') or (threshold and
                                (evidence_mode != 'enforce' or len(threshold) != 1)):
                            raise ValueError('Usage: /research [off|shadow|enforce [threshold]]')
                        cutoff = _evidence_threshold(threshold[0]) if threshold else None
                        research_workflow = _research_workflow(evidence_checks.append, evidence_mode, cutoff)
                        research_settings = _research_settings(evidence_mode, cutoff)
                        mode = 'research'
                        runner = select_runner()
                        print(f'Research mode (evidence check: {evidence_mode}). Enter a standalone public question.')
                        continue
                    if command == '/new' and not arguments:
                        if models.model is None:
                            raise ValueError('No active model. Choose an engine with /model.')
                        mode = 'code'
                        runner = select_runner()
                        if runner is not None:
                            print(f'New session {runner.new()} (default mode)')
                        else:
                            print('Default mode (tracing disabled).')
                        continue
                    if store is None:
                        raise ValueError('Session commands require tracing; restart without --no-trace.')
                    if command == '/sessions' and not arguments:
                        sessions = store.list_sessions()
                        print('Saved sessions:' if sessions else 'No saved sessions.')
                        for session in sessions:
                            title = ' '.join(session['title'].split())
                            print(f"{session['id']}  {session['backend']} / {session['model']}  [{session['mode']}] {title}")
                    elif command == '/resume' and len(arguments) == 1:
                        if models.model is None:
                            raise ValueError('No active model. Choose an engine with /model.')
                        identity = _model_identity(models.model, models.backend)
                        runner = SessionRunner(store, models.backend, identity, arguments[0])
                        mode = runner.mode
                        runners[(models.backend, identity, mode)] = runner
                        print(f'Resumed session {runner.sid}; enter a new request. No tools were replayed.')
                    elif command == '/stats' and len(arguments) <= 1:
                        stats_runner = runner or SessionRunner(store, models.backend, None)
                        print(stats_runner.stats_text(arguments[0] if arguments else None))
                    else:
                        raise ValueError('Unknown command or arguments. Use /help for session commands.')
                except ValueError as exc:
                    print(f'Error: {exc}')
                continue
            # The stub's script represents a whole task, unlike real backends
            # whose loaded weights/client are reusable across fresh tasks.
            if models.model is None:
                _print_model_status(models, mode, research_settings)
                continue
            if isinstance(models.model, StubModel):
                models.model.i = 0
            metrics = RequestMetrics()
            evidence_checks.clear()
            started = time.perf_counter()
            try:
                run_fn = run_agent
                if mode == 'chat':
                    run_fn = _run_chat
                elif mode == 'research':
                    if research_workflow is None:
                        research_workflow = _research_workflow(evidence_checks.append, 'shadow')
                    run_fn = research_workflow
                if runner is None:
                    answer = run_fn(request, models.model, max_steps=max_steps, metrics=metrics)
                else:
                    previous_sid = runner.sid
                    settings = research_settings if mode == 'research' else None
                    result = runner.run(request, models.model, run_fn, metrics, max_steps, run_settings=settings)
                    if runner.sid != previous_sid:
                        print(f'Session {runner.sid}')
                    print(f'Run {result.run_id} [{result.status.value}]')
                    if result.error:
                        print(f'Error: {result.error}')
                        continue
                    answer = result.answer
                elapsed = time.perf_counter() - started
            except Exception as exc:
                print(f"Error: {exc}")
                continue
            _print_run_output(answer, evidence_checks, metrics, elapsed, models.backend)
    except (EOFError, KeyboardInterrupt):
        print()
    print("Interactive session ended.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the tool-using agent.")
    parser.add_argument("--llm", choices=("openai", "airllm", "strata", "mlx", "llama"), default="openai",
                        help="LLM backend (default: openai)")
    parser.add_argument("--model-path", help="Local model directory for AirLLM/MLX, or GGUF file for --llm=llama")
    parser.add_argument("--base-url", help="Strata API base URL (default: http://127.0.0.1:8080/v1)")
    parser.add_argument("--model", help="Strata API model ID or MLX-LM model repository")
    parser.add_argument("--stub", action="store_true", help="Use the scripted offline backend")
    parser.add_argument("-i", "--interactive", action="store_true",
                        help="Keep the backend loaded; retain conversation in a traced session")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS,
                        help=f"Model-response budget per task, including its final answer (default: {MAX_STEPS})")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="SQLite trace database path")
    parser.add_argument("--no-trace", action="store_true", help="Disable persistence and keep tasks independent")
    parser.add_argument("--resume", help="Resume a saved session ID or unique prefix without replaying tools")
    parser.add_argument("--research", action="store_true", help="Run one public research question using Tavily and Jev")
    parser.add_argument("--evidence-check", choices=("off", "shadow", "enforce"), default=None,
                        help="Research only. Shadow (default) sends public evidence to TypeSafe; enforce can search again")
    parser.add_argument("--evidence-threshold", metavar="PROBABILITY",
                        help="Enforce only. Follow up below this supported probability (default: 0.60)")
    parser.add_argument("request", nargs="*", help="Task for the agent")
    args = parser.parse_args(argv)
    if args.resume and args.no_trace:
        parser.error("--resume cannot be combined with --no-trace")
    if args.resume and not args.interactive and not args.request:
        parser.error("--resume requires a follow-up request or --interactive")
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.model_path and (args.llm not in ("airllm", "mlx", "llama") or args.stub):
        parser.error("--model-path requires --llm=airllm, --llm=mlx, or --llm=llama without --stub")
    if args.base_url is not None and (args.llm != "strata" or args.stub):
        parser.error("--base-url requires --llm=strata without --stub")
    if args.model is not None and (args.llm not in ("strata", "mlx") or args.stub):
        parser.error("--model requires --llm=strata or --llm=mlx without --stub")
    if args.llm == "mlx" and args.model and args.model_path:
        parser.error("Use --model or --model-path for MLX, not both")
    if args.evidence_check is not None and not args.research:
        parser.error("--evidence-check requires --research")
    cutoff = None
    if args.evidence_threshold is not None:
        if not args.research or args.evidence_check != 'enforce':
            parser.error('--evidence-threshold requires --research --evidence-check enforce')
        try:
            cutoff = _evidence_threshold(args.evidence_threshold)
        except ValueError as exc:
            parser.error(str(exc))
    if args.research and (not args.request or args.interactive or args.resume or args.stub):
        parser.error("--research requires a standalone question and a real synthesis backend")
    model = None
    store = None
    started = time.perf_counter()
    metrics = RequestMetrics()
    evidence_checks = []
    try:
        run_fn = run_agent
        run_settings = None
        if args.research:
            from inquiry_loop.research import research_question

            research_question(" ".join(args.request))
            evidence_mode = args.evidence_check or 'shadow'
            run_fn = _research_workflow(evidence_checks.append, evidence_mode, cutoff)
            run_settings = _research_settings(evidence_mode, cutoff)
        options = {"stub": args.stub, "model_path": args.model_path}
        if args.llm == "strata":
            options.update(base_url=args.base_url, model=args.model)
        elif args.llm == "mlx":
            options.update(model=args.model)
        model = create_model(args.llm, **options)
        backend = 'stub' if args.stub else args.llm
        runner = None
        if not args.no_trace:
            store = SessionStore(args.db)
            runner = SessionRunner(store, backend, _model_identity(model, backend), args.resume,
                                   mode='research' if args.research else 'code')
        if args.interactive:
            print(f"Initialized backend in {time.perf_counter() - started:.2f} seconds.")
            try:
                run_interactive(model, " ".join(args.request), max_steps=args.max_steps, runner=runner,
                                backend=backend, model_options=options)
            finally:
                model = None  # The interactive loop owns backend cleanup.
        else:
            request = " ".join(args.request) or "What is in this directory?"
            if runner is None:
                answer = run_fn(request, model, max_steps=args.max_steps, metrics=metrics)
            else:
                if args.resume and runner.mode == 'chat':
                    run_fn = _run_chat
                elif args.resume and runner.mode == 'research':
                    run_fn = _research_workflow(evidence_checks.append, 'shadow')
                    run_settings = _research_settings('shadow')
                result = runner.run(request, model, run_fn, metrics, args.max_steps, run_settings=run_settings)
                print(f'Session {result.session_id}; run {result.run_id} [{result.status.value}]')
                if result.error:
                    print(f'Error: {result.error}')
                    return 1
                answer = result.answer
            elapsed = time.perf_counter() - started
            _print_run_output(answer, evidence_checks, metrics, elapsed, backend)
            if runner is not None and result.status != RunStatus.COMPLETED:
                return 2
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except (ImportError, RuntimeError, ValueError, OSError, sqlite3.Error) as exc:
        parser.exit(1, f"Error: {exc}\n")
    finally:
        try:
            close_model(model)
        finally:
            if store is not None:
                store.close()
    return 0
