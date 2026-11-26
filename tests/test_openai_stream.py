import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from inquiry_loop.backends.openai_eng import OpenAIModel
from inquiry_loop.response_types import ModelResponse


def chunk(content=None, *, finish=None, usage=None, tool_calls=None,
          function_call=None, model=None, fingerprint=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls, function_call=function_call)
    choices = [] if usage is not None else [SimpleNamespace(delta=delta, finish_reason=finish)]
    return SimpleNamespace(choices=choices, usage=usage, model=model, system_fingerprint=fingerprint)


def usage(prompt=20, output=8, cached=4):
    return SimpleNamespace(prompt_tokens=prompt, completion_tokens=output,
                           prompt_tokens_details=SimpleNamespace(cached_tokens=cached))


class OpenAIStreamTests(unittest.TestCase):
    def setUp(self):
        self.model = OpenAIModel.__new__(OpenAIModel)
        self.model.model = "requested-alias"
        self.model.tool_schemas = []
        self.model.client = Mock()
        self.create = self.model.client.chat.completions.create
        self.now = 100.0
        clock = patch("inquiry_loop.backends.openai_eng.time", SimpleNamespace(perf_counter=lambda: self.now))
        clock.start()
        self.addCleanup(clock.stop)

    def set_stream(self, *chunks, error=None):
        def iterate():
            for item in chunks:
                self.now += 0.5
                yield item
            self.now += 0.5
            if error is not None:
                raise error

        def close():
            self.now += 10

        stream = Mock(close=Mock(side_effect=close))
        stream.__iter__ = lambda _: iterate()

        def create(**kwargs):
            self.now += 1.0
            return stream

        self.create.side_effect = create
        return stream

    def test_stream_collects_text_usage_and_request_timings(self):
        stream = self.set_stream(
            chunk(model="resolved-version"),  # role-only
            chunk(""),
            chunk(" \n"),
            chunk("Hello", fingerprint="fp_test"),
            chunk(" world"),
            chunk(finish="stop"),
            chunk(usage=usage()),
        )
        messages = [{"role": "user", "content": "Say hello"}]
        stream_options = {"include_usage": False}
        response = self.model.respond_stream(messages, max_completion_tokens=32,
                                             stream=False, stream_options=stream_options)

        self.create.assert_called_once_with(model="requested-alias", messages=messages,
                                            max_completion_tokens=32, stream=True,
                                            stream_options={"include_usage": True})
        self.assertEqual(stream_options, {"include_usage": False})
        stream.close.assert_called_once_with()
        self.assertIsInstance(response, ModelResponse)
        self.assertEqual((response.text, response.tool_calls, response.finish_reason),
                         (" \nHello world", [], "stop"))
        self.assertEqual(response.stats, {
            "requested_model": "requested-alias", "resolved_model": "resolved-version",
            "system_fingerprint": "fp_test", "prompt_tokens": 20,
            "generation_tokens": 8, "cached_prompt_tokens": 4, "finish_reason": "stop",
            "ttft_seconds": 3.0, "generation_seconds": 5.0,
            "generation_tps": 1.6, "speed_kind": "request",
        })
        self.assertEqual(response.stats, self.model.last_stats)
        self.assertIsNot(response.stats, self.model.last_stats)

    def test_rejects_tool_configuration_before_request(self):
        self.model.last_stats = {"generation_tokens": 99}
        self.model.tool_schemas = [{"type": "function"}]
        with self.assertRaisesRegex(ValueError, "does not support tools"):
            self.model.respond_stream([])
        self.assertEqual(self.model.last_stats, {})
        self.model.tool_schemas = []
        for key in ("tools", "functions"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "does not support tools"):
                self.model.respond_stream([], **{key: [{"name": "test"}]})
        self.create.assert_not_called()

    def test_rejects_tool_deltas_even_with_stop_and_retains_usage(self):
        for options in ({"tool_calls": [SimpleNamespace(index=0)]},
                        {"function_call": SimpleNamespace(name="legacy")}):
            with self.subTest(options=options):
                stream = self.set_stream(chunk("before"), chunk(**options), chunk(finish="stop"),
                                         chunk(usage=usage()))
                with self.assertRaisesRegex(ValueError, "returned a tool call"):
                    self.model.respond_stream([])
                self.assertEqual(self.model.last_stats["ttft_seconds"], 1.5)
                self.assertEqual(self.model.last_stats["generation_tokens"], 8)
                stream.close.assert_called_once_with()

    def test_rejects_unsuccessful_finishes_but_keeps_final_usage(self):
        for finish, message in (("length", "output-token limit"),
                                ("content_filter", "finish_reason='content_filter'"),
                                ("tool_calls", "returned a tool call"),
                                (None, "finish_reason=None")):
            with self.subTest(finish=finish):
                stream = self.set_stream(chunk("partial", finish=finish), chunk(usage=usage()))
                with self.assertRaisesRegex(ValueError, message):
                    self.model.respond_stream([])
                self.assertEqual(self.model.last_stats["finish_reason"], finish)
                self.assertEqual(self.model.last_stats["ttft_seconds"], 1.5)
                self.assertEqual(self.model.last_stats["generation_seconds"], 2.5)
                self.assertEqual(self.model.last_stats["cached_prompt_tokens"], 4)
                stream.close.assert_called_once_with()

    def test_missing_or_invalid_counts_stay_unknown_and_zero_is_valid(self):
        for chunks, expected_cache in (([], None), ([chunk(usage=usage("20", True, 0))], 0)):
            with self.subTest(cached=expected_cache):
                stream = self.set_stream(chunk("\t ", finish="stop"), *chunks)
                response = self.model.respond_stream([])
                self.assertEqual(response.text, "\t ")
                for key in ("ttft_seconds", "resolved_model", "system_fingerprint",
                            "prompt_tokens", "generation_tokens", "generation_tps"):
                    self.assertIsNone(response.stats[key], key)
                self.assertEqual(response.stats["cached_prompt_tokens"], expected_cache)
                stream.close.assert_called_once_with()

    def test_iteration_error_closes_stream_and_preserves_collected_metrics(self):
        stream = self.set_stream(chunk("partial", fingerprint="fp_test"),
                                 chunk(usage=usage()), error=RuntimeError("dropped stream"))
        with self.assertRaisesRegex(RuntimeError, "dropped stream"):
            self.model.respond_stream([])
        stream.close.assert_called_once_with()
        self.assertEqual(self.model.last_stats["ttft_seconds"], 1.5)
        self.assertEqual(self.model.last_stats["generation_seconds"], 2.5)
        self.assertEqual(self.model.last_stats["prompt_tokens"], 20)
        self.assertEqual(self.model.last_stats["system_fingerprint"], "fp_test")


if __name__ == "__main__":
    unittest.main()
