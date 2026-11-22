"""Session checkpoints and traces, using temporary SQLite databases and models."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import agent
from request_metrics import RequestMetrics
from response_types import ModelResponse, RunContext, RunStatus
from session_store import SessionRunner, SessionStore


def tool(name='list_dir', **args):
    return {'id': 'call', 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}


class Model:
    model_id = 'test-model'

    def __init__(self, replies=None):
        self.replies = iter(replies) if replies is not None else None
        self.histories = []
        self.close = Mock()

    def respond(self, messages):
        self.histories.append(copy.deepcopy(messages))
        reply = next(self.replies) if self.replies is not None else ModelResponse([], 'answer', {'generation_tokens': 10, 'generation_tps': 5})
        if isinstance(reply, BaseException):
            raise reply
        return reply


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'traces.sqlite3'
        self.store = SessionStore(self.path)
        self.addCleanup(self.store.close)
        self.runner = SessionRunner(self.store, 'mlx', 'test-model')

    def run_task(self, model=None, request='test', max_steps=16):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.runner.run(request, model or Model(), agent.run_agent, RequestMetrics(), max_steps)

    def test_checkpoint_survives_reopen_and_resume(self):
        first = self.run_task(request='remember this')
        reopened = SessionStore(self.path)
        self.addCleanup(reopened.close)
        runner = SessionRunner(reopened, 'mlx', 'test-model', first.session_id[:12])
        model = Model()
        runner.run('follow-up', model, agent.run_agent, RequestMetrics(), 16)
        history = model.histories[0]
        self.assertEqual([m['role'] for m in history], ['system', 'user', 'assistant', 'user'])
        self.assertEqual(history[1]['content'], 'remember this')
        self.assertEqual(history[-1]['content'], 'follow-up')
        self.assertEqual(len(self.store.runs(first.session_id)), 2)

    def test_events_include_tool_results_stats_and_order(self):
        model = Model([ModelResponse([tool()], ''), ModelResponse([], 'finished', {'generation_tokens': 20, 'generation_tps': 10})])
        with patch.dict(agent.TOOLS, {'list_dir': lambda: {'entries': ['file']}}):
            result = self.run_task(model)
        events = self.store.events(result.run_id)
        self.assertEqual([e['kind'] for e in events], ['run_started', 'model_started', 'model_response', 'tool_started', 'tool_result', 'model_started', 'model_response', 'run_finished'])
        self.assertEqual(json.loads(events[4]['payload_json'])['observation']['entries'], ['file'])
        self.assertEqual(result.status, RunStatus.COMPLETED)
        self.assertFalse(result.stats['complete'])  # First tool turn lacked timing.
        self.assertEqual(json.loads(events[-2]['payload_json'])['stats']['generation_tokens'], 20)

    def test_failed_run_preserves_completed_checkpoint(self):
        first = self.run_task(request='good')
        result = self.run_task(Model([ValueError('output budget')]), request='failed request')
        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertIn('output budget', result.error)
        history = json.loads(self.store.get_session(first.session_id)['history_json'])
        self.assertEqual([m['content'] for m in history], ['good', 'answer'])
        self.assertEqual(json.loads(self.store.events(result.run_id)[-1]['payload_json'])['status'], 'failed')

    def test_interrupted_tool_is_not_replayed_on_resume(self):
        action = Mock(side_effect=KeyboardInterrupt)
        with patch.dict(agent.TOOLS, {'list_dir': action}), self.assertRaises(KeyboardInterrupt):
            self.run_task(Model([ModelResponse([tool()], '')]))
        run = self.store.runs(self.runner.sid)[0]
        self.assertEqual(run['status'], 'interrupted')
        self.assertEqual([e['kind'] for e in self.store.events(run['id'])][-2:], ['tool_started', 'run_finished'])
        SessionRunner(self.store, 'mlx', 'test-model', self.runner.sid)
        action.assert_called_once()
        self.assertEqual(json.loads(self.store.get_session(self.runner.sid)['history_json']), [])

    def test_budget_stop_has_status_and_executes_no_terminal_tools(self):
        action = Mock()
        with patch.dict(agent.TOOLS, {'list_dir': action}):
            result = self.run_task(Model([ModelResponse([tool()], '')]), max_steps=1)
        self.assertEqual(result.status, RunStatus.STEP_LIMIT)
        action.assert_not_called()
        self.assertEqual(json.loads(self.store.get_session(result.session_id)['history_json']), [])

    def test_repeated_tool_error_stop_has_separate_status(self):
        replies = [ModelResponse([tool('unknown')], '')] * 4
        result = self.run_task(Model(replies))
        self.assertEqual(result.status, RunStatus.TOOL_ERROR_LIMIT)

    def test_tool_errors_can_recover_to_normal_answer(self):
        result = self.run_task(Model([ModelResponse([tool('unknown')], ''), ModelResponse([], 'limited assessment')]))
        self.assertEqual(result.status, RunStatus.COMPLETED)
        self.assertIn('unknown tool', self.store.events(result.run_id)[4]['payload_json'])

    def test_new_session_clears_history_without_unloading_model(self):
        model = Model()
        self.run_task(model)
        sid = self.runner.sid
        self.runner.new()
        self.run_task(model)
        self.assertNotEqual(self.runner.sid, sid)
        self.assertEqual([len(h) for h in model.histories], [2, 2])
        model.close.assert_not_called()

    def test_resume_rejects_backend_model_and_directory_mismatch(self):
        self.run_task()
        for backend, model in (('openai', 'test-model'), ('mlx', 'different-model')):
            with self.subTest(backend=backend), self.assertRaisesRegex(ValueError, 'backend/model'):
                SessionRunner(self.store, backend, model, self.runner.sid)
        with patch('session_store.os.getcwd', return_value='/different/location'), self.assertRaisesRegex(ValueError, 'working directory'):
            self.runner.resume(self.runner.sid)

    def test_literal_and_ambiguous_session_prefixes(self):
        self.runner.new()
        with self.assertRaises(ValueError):
            self.store.get_session('%')
        self.runner.new()
        with self.assertRaises(ValueError):
            self.store.get_session('')

    def test_stats_include_status_model_kind_and_weighted_rate(self):
        result = self.run_task()
        text = self.runner.stats_text()
        for expected in ('mlx / test-model', '[completed]', 'tokens/s=5.00', 'kind=generation', 'model calls=1'):
            self.assertIn(expected, text)
        self.assertEqual(self.store.runs(result.session_id)[0]['status'], 'completed')

    def test_control_system_prompt_does_not_persist(self):
        model = Model()
        self.run_task(model, max_steps=1)
        self.run_task(model)
        self.assertEqual(sum(m['role'] == 'system' for m in model.histories[-1]), 1)
        self.assertNotIn('final response turn', str(model.histories[-1]))

    def test_unknown_schema_is_rejected_without_changes(self):
        path = Path(self.directory.name) / 'future.sqlite3'
        db = sqlite3.connect(path)
        db.execute('PRAGMA user_version=999')
        db.close()
        with self.assertRaisesRegex(ValueError, 'schema'):
            SessionStore(path)
        db = sqlite3.connect(path)
        try:
            self.assertEqual(db.execute('PRAGMA user_version').fetchone()[0], 999)
        finally:
            db.close()

    def test_shared_response_copies_stats_and_normalizes_calls(self):
        stats = {'generation_tokens': 12}
        response = ModelResponse.from_parts([tool()], 'text', stats, 'tool_calls')
        stats['generation_tokens'] = 99
        self.assertEqual(response.stats['generation_tokens'], 12)
        calls, text = response
        self.assertEqual(text, 'text')
        self.assertEqual(calls[0]['function']['name'], 'list_dir')
        self.assertIsInstance(agent.StubModel().respond([]), ModelResponse)

    def test_openai_truncation_is_failed_without_executing_partial_calls(self):
        from openai_eng import OpenAIModel
        model = OpenAIModel.__new__(OpenAIModel)
        model.model, model.tool_schemas, model.client = 'test-model', [], Mock()
        model.client.chat.completions.create.return_value = SimpleNamespace(choices=[
            SimpleNamespace(finish_reason='length', message=SimpleNamespace(
                content='partial', tool_calls=[tool('write_file', path='unused', content='partial')]))])
        action = Mock()
        with patch.dict(agent.TOOLS, {'write_file': action}):
            result = self.run_task(model)
        self.assertEqual(result.status, RunStatus.FAILED)
        action.assert_not_called()
        self.assertIn('output-token limit', result.error)

    def test_cli_commands_and_continuing_history(self):
        model = Model()
        output = io.StringIO()
        inputs = ['first', 'second', '/stats', '/sessions', '/new', 'third', '/help', '/bogus', '/quit']
        with patch('agent.create_model', return_value=model), patch('builtins.input', side_effect=inputs), contextlib.redirect_stdout(output):
            self.assertEqual(agent.main(['--llm=mlx', '--interactive', '--db', str(self.path)]), 0)
        self.assertEqual([len(h) for h in model.histories], [2, 4, 2])
        self.assertEqual(len(self.store.list_sessions()), 2)
        self.assertIn('Saved sessions:', output.getvalue())
        self.assertIn('kind=generation', output.getvalue())
        self.assertIn('Unknown command', output.getvalue())
        model.close.assert_called_once()

    def test_cli_resume_command_and_flag_wait_for_new_input(self):
        result = self.run_task(request='original')
        for arguments, commands in ((['--resume', result.session_id], ['/quit']),
                                    ([], ['/resume ' + result.session_id, '/quit'])):
            model = Model()
            with self.subTest(arguments=arguments), patch('agent.create_model', return_value=model), patch('builtins.input', side_effect=commands), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(agent.main(['--llm=mlx', '--interactive', '--db', str(self.path), *arguments]), 0)
            self.assertEqual(model.histories, [])

    def test_failed_cli_request_exits_nonzero_and_saves_failure(self):
        model = Model([ValueError('bad output')])
        with patch('agent.create_model', return_value=model), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(agent.main(['--llm=mlx', '--db', str(self.path), 'request']), 1)
        run = self.store.runs(self.store.list_sessions()[0]['id'])[0]
        self.assertEqual(run['status'], 'failed')

    def test_same_session_cannot_run_concurrently(self):
        sid = self.runner.new()
        other_store = SessionStore(self.path)
        self.addCleanup(other_store.close)
        other = SessionRunner(other_store, 'mlx', 'test-model', sid)
        model = Model()
        with self.store.session_lock(sid), self.assertRaisesRegex(ValueError, 'another process'):
            other.run('task', model, agent.run_agent, RequestMetrics(), 16)
        self.assertEqual(model.histories, [])
        self.assertEqual(self.store.runs(sid), [])

    def test_abandoned_run_is_interrupted_without_tool_replay(self):
        sid = self.runner.new()
        rid = self.store.start_run(sid, 'abandoned')
        self.store.event(rid, 'tool_started', {'name': 'write_file'})
        self.run_task()
        runs = {r['id']: r for r in self.store.runs(sid)}
        self.assertEqual(runs[rid]['status'], 'interrupted')
        self.assertIn('may already have occurred', runs[rid]['error'])
        self.assertEqual([e['kind'] for e in self.store.events(rid)], ['tool_started', 'run_finished'])

    def test_no_trace_creates_no_database_and_rejects_resume(self):
        path = Path(self.directory.name) / 'unused.sqlite3'
        with patch('agent.create_model', return_value=Model()), contextlib.redirect_stdout(io.StringIO()):
            agent.main(['--no-trace', '--db', str(path), 'task'])
        self.assertFalse(path.exists())
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            agent.main(['--no-trace', '--resume', 'id', '-i'])


if __name__ == '__main__':
    unittest.main()
