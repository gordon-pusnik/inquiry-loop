"""Regression tests for complete source inspection and bounded review loops."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from inquiry_loop import agent, cli


class ScriptModel:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.histories = []

    def respond(self, messages):
        self.histories.append(list(messages))
        return next(self.actions)


def call(name, **arguments):
    return ([{'id': 'call', 'function': {'name': name, 'arguments': json.dumps(arguments)}}], '')


class FileReviewTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = str(Path(self.directory.name) / 'source.txt')
        Path(self.path).write_text('é🙂first\nsecond needle\nlast needle', encoding='utf-8')

    def test_unicode_pages_reconstruct_source(self):
        pages, offset = [], 0
        while True:
            page = agent.read_file(self.path, offset, 7)
            pages.append(page['content'])
            if page['eof']:
                self.assertIsNone(page['next_offset'])
                break
            offset = page['next_offset']
        self.assertEqual(''.join(pages), Path(self.path).read_text())

    def test_invalid_ranges(self):
        for kwargs in ({'offset': -1}, {'offset': True}, {'limit': 0}, {'limit': 12001}, {'offset': 10000}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                agent.read_file(self.path, **kwargs)

    def test_search_returns_usable_offsets(self):
        result = agent.search_file(self.path, 'needle', max_results=1)
        self.assertTrue(result['matches_truncated'])
        match = result['matches'][0]
        self.assertEqual(match['line'], 2)
        self.assertEqual(agent.read_file(self.path, match['offset'], 6)['content'], 'needle')
        self.assertFalse(result['scan_truncated'])

    def test_search_has_bounded_scan_and_snippets(self):
        Path(self.path).write_text('x' * 1_000_001)
        result = agent.search_file(self.path, 'x', 2)
        self.assertTrue(result['scan_truncated'])
        self.assertEqual(len(result['matches']), 2)
        self.assertTrue(all(len(m['snippet']) <= 300 for m in result['matches']))

    def test_partial_review_is_flagged(self):
        model = ScriptModel([call('read_file', path=self.path, limit=7), ([], 'Assessment')])
        answer = agent.run_agent('Review source', model, verbose=False)
        self.assertIn('File inspection was partial', answer)

    def test_complete_review_has_no_partial_footer(self):
        model = ScriptModel([call('read_file', path=self.path, limit=7), call('read_file', path=self.path, offset=7), ([], 'Assessment')])
        self.assertEqual(agent.run_agent('Review source', model, verbose=False), 'Assessment')

    def test_eof_with_unread_gap_is_partial(self):
        model = ScriptModel([call('read_file', path=self.path, offset=10), ([], 'Assessment')])
        self.assertIn('File inspection was partial', agent.run_agent('Review', model, verbose=False))

    def test_final_turn_requests_answer_without_executing_tools(self):
        model = ScriptModel([call('list_dir'), call('write_file', path=self.path, content='bad')])
        with patch.dict(agent.TOOLS, {'list_dir': lambda: {}}):
            answer = agent.run_agent('Review', model, max_steps=2, verbose=False)
        self.assertIn('no further tools were executed', answer)
        self.assertIn('final response turn', model.histories[-1][-1]['content'])
        self.assertNotEqual(Path(self.path).read_text(), 'bad')

    def test_repeated_errors_request_final_assessment(self):
        model = ScriptModel([call('missing')] * 3 + [([], 'Unable to complete')])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            answer = agent.run_agent('Review', model)
        self.assertEqual(answer, 'Unable to complete')
        self.assertIn('Tool error:', output.getvalue())
        self.assertIn('final response turn', model.histories[-1][-1]['content'])

    def test_pipeline_error_explains_pagination(self):
        result = agent.run_shell('cat source.txt | head -c 4000')
        self.assertFalse(result['retryable'])
        self.assertIn('next_offset', result['error'])

    def test_cli_step_budget_is_forwarded(self):
        with patch.object(cli, 'create_model'), patch.object(cli, 'run_agent', return_value='done') as run, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["--no-trace", '--stub', '--max-steps', '24', 'review']), 0)
        self.assertEqual(run.call_args.kwargs['max_steps'], 24)

    def test_invalid_step_budget(self):
        with self.assertRaises(ValueError):
            agent.run_agent('Review', ScriptModel([]), max_steps=0)


if __name__ == '__main__':
    unittest.main()
