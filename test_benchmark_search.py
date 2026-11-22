import unittest
from unittest.mock import call, patch

import benchmark_search


class SearchBenchmarkTests(unittest.TestCase):
    @patch.object(benchmark_search, 'CASES', (('docs', 'example query', 'an official source'),))
    @patch('benchmark_search.search')
    def test_paired_collection_and_manual_scoring(self, search):
        search.side_effect = [
            {'status': 'ok', 'results': [{'url': 'https://example.com'}]},
            {'status': 'no_results', 'results': []},
        ]
        report = benchmark_search.collect(benchmark_search.PROVIDERS)
        self.assertEqual(search.call_args_list, [call('example query', provider='tavily'),
                                                 call('example query', provider='instant-answer')])
        self.assertIsNone(benchmark_search.summarize(report)['tavily']['useful_top_5_rate'])
        report['runs'][0]['useful_top_5'] = True
        report['runs'][1]['useful_top_5'] = False
        summary = benchmark_search.summarize(report)
        self.assertEqual(summary['tavily']['useful_top_5_rate'], 1)
        self.assertEqual(summary['instant-answer']['useful_top_5_rate'], 0)
        self.assertEqual(summary['instant-answer']['queries_without_links'], 1)

    @patch('benchmark_search.search', return_value={'error_code': 'missing_credentials', 'error': 'Configure key'})
    def test_missing_key_stops_before_sending_baseline_queries(self, search):
        with self.assertRaisesRegex(RuntimeError, 'tavily: Configure key'):
            benchmark_search.collect(benchmark_search.PROVIDERS)
        search.assert_called_once()
