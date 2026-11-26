import json
import os
import socket
import ssl
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit

from inquiry_loop import agent
from inquiry_loop.providers import web_search
from inquiry_loop.backends.stub_eng import _FakeCall


@patch.dict(os.environ, {"SEARCH_PROVIDER": "tavily", "TAVILY_API_KEY": "test-token"})
@patch('inquiry_loop.providers.web_search.build_opener')
class SearchTests(unittest.TestCase):
    @staticmethod
    def response(opener, data):
        opener.return_value.open.return_value.__enter__.return_value.read.return_value = (
            data if isinstance(data, bytes) else json.dumps(data).encode()
        )

    def test_tavily_request_ranking_and_normalization(self, opener):
        self.response(opener, {"results": [
            {"title": " First ", "url": "https://example.com/a#part", "content": "x" * 1050},
            {"url": "https://example.com/a"}, {"url": "javascript:bad"}, {"url": "https:///no-host"},
            {"title": "Second", "url": "https://example.org", "content": "Snippet"},
            {"url": "https://example.net"},
        ]})
        result = agent.search_web('test & query', 2)
        self.assertEqual((result['status'], result['provider']), ('ok', 'tavily'))
        self.assertEqual([r['title'] for r in result['results']], ['First', 'Second'])
        self.assertEqual(result['results'][0]['url'], 'https://example.com/a#part')
        self.assertEqual(len(result['results'][0]['snippet']), 1000)
        call = opener.return_value.open.call_args
        request = call.args[0]
        url = urlsplit(request.full_url)
        self.assertEqual((url.hostname, url.path, url.query), ('api.tavily.com', '/search', ''))
        self.assertEqual(request.get_method(), 'POST')
        self.assertEqual(json.loads(request.data), {'query': 'test & query', 'max_results': 2,
                                                   'search_depth': 'basic', 'auto_parameters': False,
                                                   'include_answer': False, 'include_raw_content': False})
        self.assertEqual(request.get_header('Content-type'), 'application/json')
        self.assertEqual(request.get_header('Authorization'), 'Bearer test-token')
        self.assertNotIn('test-token', request.full_url + request.data.decode() + json.dumps(result))
        self.assertEqual(result['results'][1]['snippet'], 'Snippet')
        self.assertEqual(call.kwargs['timeout'], web_search.SEARCH_TIMEOUT)
        handler = opener.call_args.args[1]
        self.assertIsNone(handler.redirect_request(request, None, 302, '', {}, 'https://example.org'))

    def test_explicit_keyless_lookup_preserves_direct_answers(self, opener):
        self.response(opener, {"Heading": "Python", "AbstractURL": "https://example.com", "AbstractText": "Summary",
                               "RelatedTopics": [{"Topics": [{"Text": "Duplicate", "FirstURL": "https://example.com"},
                                                            {"Text": "Related", "FirstURL": "https://example.org"}]}]})
        with patch.dict(os.environ, {"SEARCH_PROVIDER": "instant-answer"}):
            result = agent.search_web('python')
            self.assertEqual([r['snippet'] for r in result['results']], ['Summary', 'Related'])
            request = opener.return_value.open.call_args.args[0]
            self.assertEqual(urlsplit(request.full_url).hostname, 'api.duckduckgo.com')
            self.assertIsNone(request.get_header('Authorization'))
            self.assertEqual(request.get_method(), 'GET')
            self.response(opener, {"Answer": "42", "AnswerType": "calc"})
            result = agent.search_web('test')
            self.assertEqual((result['status'], result['results'], result['answer']), ('ok', [], '42'))

    def test_no_results_and_malformed_responses(self, opener):
        self.response(opener, {'results': []})
        self.assertEqual(agent.search_web('test')['status'], 'no_results')
        for data in (b'<html>blocked</html>', [], {}, {'results': 1}):
            self.response(opener, data)
            self.assertEqual(agent.search_web('test')['error_code'], 'invalid_response')
        self.response(opener, b'x' * (web_search.SEARCH_RESPONSE_LIMIT + 1))
        self.assertEqual(agent.search_web('test')['error_code'], 'response_too_large')

    def test_input_and_configuration_errors_do_not_search(self, opener):
        for query, count in (('', 5), ('x' * 401, 5), ('x ' * 51, 5), ('test', True), ('test', 11)):
            self.assertEqual(agent.search_web(query, count)['status'], 'error')
        with patch.dict(os.environ, {'TAVILY_API_KEY': ''}):
            self.assertEqual(agent.search_web('test')['error_code'], 'missing_credentials')
        with patch.dict(os.environ, {'SEARCH_PROVIDER': 'unknown'}):
            self.assertEqual(agent.search_web('test')['error_code'], 'invalid_provider')
        opener.assert_not_called()

    def test_default_and_dotenv_credentials(self, opener):
        self.response(opener, {'results': []})
        with patch.dict(os.environ, {}, clear=True), patch('dotenv.dotenv_values', return_value={'TAVILY_API_KEY': 'file-token'}):
            self.assertEqual(agent.search_web('test')['provider'], 'tavily')
            self.assertEqual(opener.return_value.open.call_args.args[0].get_header('Authorization'), 'Bearer file-token')

    def test_network_errors_are_classified_without_leaking_secrets(self, opener):
        cases = [(HTTPError('', status, 'test-token', {}, None), code, retryable)
                 for status, code, retryable in ((302, 'http', False), (401, 'auth', False), (403, 'auth', False),
                                                  (429, 'rate_limit', True), (432, 'quota', False),
                                                                                                    (433, 'quota', False), (503, 'http', True))]
        cases += [(TimeoutError(), 'timeout', True),
                  (URLError(ssl.SSLCertVerificationError('untrusted')), 'tls', False),
                  (URLError(socket.gaierror('unknown host')), 'dns', False),
                  (URLError('test-token'), 'connection', False)]
        for exc, code, retryable in cases:
            with self.subTest(code=code):
                opener.return_value.open.side_effect = exc
                result = agent.search_web('test')
                self.assertEqual((result['error_code'], result['retryable']), (code, retryable))
                self.assertNotIn('test-token', json.dumps(result))

    def test_shared_agent_loop_receives_citable_results(self, opener):
        self.response(opener, {'results': [{'title': 'Source', 'url': 'https://example.com'}]})
        testcase = self

        class ScriptedModel:
            def respond(self, messages):
                if messages[-1]['role'] == 'user':
                    return [_FakeCall('search_1', 'search_web', {'query': 'test'})], ''
                observation = json.loads(messages[-1]['content'])
                testcase.assertEqual(observation['provider'], 'tavily')
                return [], observation['results'][0]['url']

        self.assertEqual(agent.run_agent('Find a source', ScriptedModel(), verbose=False), 'https://example.com')
