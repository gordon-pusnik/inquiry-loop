"""MLX-LM backend tests without importing Metal or downloading model weights."""

import contextlib
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from inquiry_loop import agent, cli
from inquiry_loop.backends.mlx_eng import DEFAULT_MODEL, MLXModel
from inquiry_loop.research import run_research


class MLXTests(unittest.TestCase):
    def setUp(self):
        self.mx = Mock()
        self.mx.set_cache_limit.return_value = 12345
        self.tokenizer = Mock(
            has_tool_calling=True, tool_call_start="<tool_call>", tool_call_end="</tool_call>",
            think_start="<think>", think_end="</think>",
        )
        self.tokenizer.tool_parser.side_effect = lambda text, tools: json.loads(text)
        self.tokenizer.apply_chat_template.return_value = "formatted prompt"
        self.tokenizer.encode.return_value = [1, 2, 3]
        self.weights = object()
        self.load = Mock(return_value=(self.weights, self.tokenizer, {"max_position_embeddings": 40960}))
        self.generate = Mock()
        self.sampler = Mock(return_value="greedy")
        self.closed = []
        modules = {
            "mlx": SimpleNamespace(core=self.mx), "mlx.core": self.mx,
            "mlx_lm": SimpleNamespace(load=self.load, stream_generate=self.generate),
            "mlx_lm.sample_utils": SimpleNamespace(make_sampler=self.sampler),
            "dotenv": SimpleNamespace(load_dotenv=Mock()),
        }
        for patcher in (patch.dict(sys.modules, modules), patch.dict(os.environ, {}, clear=True),
                        patch("inquiry_loop.backends.mlx_eng.sys.platform", "darwin")):
            patcher.start()
            self.addCleanup(patcher.stop)

    def make_model(self, **options):
        model = MLXModel(tool_schemas=agent.TOOL_SCHEMAS, **options)
        self.addCleanup(model.close)
        return model

    def stream(self, text, finish="stop"):
        try:
            middle = len(text) // 2
            yield SimpleNamespace(text=text[:middle], finish_reason=None)
            yield SimpleNamespace(text=text[middle:], finish_reason=finish, prompt_tokens=3,
                                  generation_tokens=12, prompt_tps=100, generation_tps=30, peak_memory=3.0)
        finally:
            self.closed.append(True)

    def test_default_native_loader_and_memory_settings(self):
        model = self.make_model()
        self.load.assert_called_once_with(DEFAULT_MODEL, return_config=True)
        self.mx.set_cache_limit.assert_called_once_with(512 * 1024 * 1024)
        self.sampler.assert_called_once_with(temp=0.0)
        self.assertEqual(model.prefill_step_size, 512)
        self.assertEqual(model.max_tokens, 2048)
        self.assertEqual(model.max_context_tokens, 16384)
        model.close()
        self.assertIsNone(model.model)
        self.mx.set_cache_limit.assert_called_with(12345)

    def test_environment_explicit_and_local_model_selection(self):
        with patch.dict(os.environ, {"MLX_MODEL": "custom/model", "MLX_MAX_TOKENS": "128",
                                     "MLX_MAX_CONTEXT_TOKENS": "2048", "MLX_PREFILL_STEP_SIZE": "256",
                                     "MLX_CACHE_LIMIT_MB": "64"}):
            model = self.make_model()
            self.assertEqual((model.model_id, model.max_tokens, model.max_context_tokens,
                              model.prefill_step_size), ("custom/model", 128, 2048, 256))
            self.assertEqual(self.make_model(model="explicit/model").model_id, "explicit/model")
            with tempfile.TemporaryDirectory() as directory:
                Path(directory, "config.json").write_text("{}")
                self.assertEqual(self.make_model(model_path=directory).model_id, directory)

    def test_two_native_tool_turns_and_argument_object_history(self):
        replies = iter([
            '<think>private reasoning</think><tool_call>{"name":"list_dir","arguments":{"path":"."}}</tool_call>',
            '<tool_call>{"name":"read_file","arguments":{"path":"requirements.txt"}}</tool_call>',
            "Requirements contain OpenAI.",
        ])
        histories = []

        def render(history, **kwargs):
            for message in history:
                self.assertIsInstance(message["content"], str)
            histories.append(copy.deepcopy(history))
            return "formatted prompt"

        self.tokenizer.apply_chat_template.side_effect = render
        self.generate.side_effect = lambda *args, **kwargs: self.stream(next(replies))
        model = self.make_model()
        with patch.dict(agent.TOOLS, {
            "list_dir": Mock(return_value={"entries": ["requirements.txt"]}),
            "read_file": Mock(return_value={"content": "openai>=1.30"}),
        }):
            answer = agent.run_agent("List then read requirements", model, verbose=False)
        self.assertEqual(answer, "Requirements contain OpenAI.")
        self.assertEqual(histories[1][-2]["tool_calls"][0]["function"]["arguments"], {"path": "."})
        self.assertEqual(json.loads(histories[2][-1]["content"])["content"], "openai>=1.30")
        self.assertEqual(len(self.closed), 3)
        self.assertEqual(model.last_stats["generation_tps"], 30)
        self.assertFalse(self.tokenizer.apply_chat_template.call_args.kwargs["enable_thinking"])
        self.assertEqual(self.generate.call_args.kwargs["prefill_step_size"], 512)
        self.assertEqual(self.generate.call_args.kwargs["prompt"], [1, 2, 3])

    def test_enforced_research_omits_tool_schemas_for_both_model_calls(self):
        question = "What is the population of Phoenix?"
        query = "Phoenix city population census estimate"
        results = [
            {"title": "Earlier estimate", "url": "https://example.org/earlier", "snippet": "Phoenix had 1.6 million residents."},
            {"title": "Latest estimate", "url": "https://example.org/latest", "snippet": "Phoenix had 1.7 million residents."},
        ]
        search = Mock(side_effect=[{"status": "ok", "results": [result]} for result in results])
        judge = Mock(side_effect=[
            {"status": "ok", "label": "partial"},
            {"status": "ok", "label": "supported", "probabilities": {"supported": 0.9}},
        ])
        replies = iter([
            json.dumps({"gap": "Latest estimate", "query": query}),
            "The latest estimate is 1.7 million residents [S2].",
        ])
        self.generate.side_effect = lambda *args, **kwargs: self.stream(next(replies))
        model = self.make_model()

        answer = run_research(question, model, search_fn=search, judge=judge, enforce=True)

        planning, synthesis = self.tokenizer.apply_chat_template.call_args_list
        self.assertIsNone(planning.kwargs["tools"])
        self.assertIsNone(synthesis.kwargs["tools"])
        self.assertIn("[S1], [S2]", synthesis.args[0][0]["content"])
        self.assertEqual([call.args[0] for call in search.call_args_list], [question, query])
        self.assertIn("[S2](<https://example.org/latest>)", answer)
        self.assertNotIn(results[0]["url"], answer)

        self.generate.side_effect = lambda *args, **kwargs: self.stream("A general answer.")
        model.respond([])
        self.assertEqual(self.tokenizer.apply_chat_template.call_args.kwargs["tools"], agent.TOOL_SCHEMAS)

    def test_ttft_includes_prompt_work_and_waits_for_visible_text(self):
        now = 100.0

        def render(*args, **kwargs):
            nonlocal now
            now += 2
            return "formatted prompt"

        def encode(*args, **kwargs):
            nonlocal now
            now += 3
            return [1, 2, 3]

        def generate(*args, **kwargs):
            nonlocal now
            for chunk in ("", " \n", "<thi", "nk>private</thi", "nk>\n<tool_",
                          'call>{"name":"list_dir","arguments":{}}', "</tool_call>\n", "Hello"):
                now += 1
                yield SimpleNamespace(text=chunk, finish_reason=None)
            now += 10
            yield SimpleNamespace(text=" world", finish_reason="stop", peak_memory=3.0)

        def reset_peak():
            self.load.assert_called_once()
            self.tokenizer.apply_chat_template.assert_not_called()

        def get_peak():
            self.mx.synchronize.assert_called_once()
            self.mx.clear_cache.assert_not_called()
            return 3_000_000_123

        self.tokenizer.apply_chat_template.side_effect = render
        self.tokenizer.encode.side_effect = encode
        self.generate.side_effect = generate
        self.mx.reset_peak_memory.side_effect = reset_peak
        self.mx.get_peak_memory.side_effect = get_peak
        model = self.make_model()
        with patch("inquiry_loop.backends.mlx_eng.perf_counter", side_effect=lambda: now):
            response = model.respond([])

        self.assertEqual(response.text, "Hello world")
        self.assertEqual(len(response.tool_calls), 1)
        self.assertEqual(response.stats, model.last_stats)
        self.assertEqual(response.stats["ttft_seconds"], 13.0)
        self.assertEqual(response.stats["peak_memory_bytes"], 3_000_000_123)
        self.assertEqual(response.stats["peak_memory"], 3.0)
        self.mx.reset_peak_memory.assert_called_once()
        self.mx.get_peak_memory.assert_called_once()
        self.mx.clear_cache.assert_called_once()

    def test_literal_marker_prefix_uses_its_arrival_time(self):
        model = self.make_model()


        def generate(*args, **kwargs):
            yield SimpleNamespace(text="<", finish_reason=None)
            yield SimpleNamespace(text="", finish_reason="stop")

        self.generate.side_effect = generate
        with patch("inquiry_loop.backends.mlx_eng.perf_counter", side_effect=[10.0, 11.0]):
            response = model.respond([])
        self.assertEqual(response.text, "<")
        self.assertEqual(response.stats["ttft_seconds"], 1.0)

    def test_tool_only_response_has_no_ttft_and_memory_api_is_optional(self):
        model = self.make_model()
        self.mx.get_peak_memory.return_value = 512
        self.generate.side_effect = lambda *args, **kwargs: self.stream("Answer")
        first = model.respond([])
        self.assertIsNotNone(first.stats["ttft_seconds"])
        self.assertEqual(first.stats["peak_memory_bytes"], 512)

        del self.mx.get_peak_memory
        self.generate.side_effect = lambda *args, **kwargs: self.stream(
            '<tool_call>{"name":"list_dir","arguments":{}}</tool_call> \n')
        with patch("inquiry_loop.backends.mlx_eng.perf_counter", side_effect=[100.0]):
            second = model.respond([])
        self.assertEqual(second.text, "")
        self.assertEqual(len(second.tool_calls), 1)
        self.assertIsNone(second.stats["ttft_seconds"])
        self.assertIsNone(second.stats["peak_memory_bytes"])
        self.assertEqual(second.stats["peak_memory"], 3.0)
        self.assertEqual(self.mx.reset_peak_memory.call_count, 2)
        self.assertEqual(first.stats["peak_memory_bytes"], 512)

    def test_history_conversion_does_not_mutate_agent_messages(self):
        model = self.make_model()
        messages = [{"role": "assistant", "tool_calls": [{"function": {"arguments": '{"path":"."}'}}]}]
        history = model._messages(messages)
        self.assertEqual(history[0]["tool_calls"][0]["function"]["arguments"], {"path": "."})
        self.assertIsInstance(messages[0]["tool_calls"][0]["function"]["arguments"], str)

    def test_missing_and_null_content_normalized_without_mutating_history(self):
        model = self.make_model()
        messages = [
            {"role": "assistant", "tool_calls": []},
            {"role": "assistant", "content": None, "tool_calls": []},
            {"role": "tool", "content": "observed value"},
        ]
        history = model._messages(messages)
        self.assertEqual([m["content"] for m in history], ["", "", "observed value"])
        self.assertNotIn("content", messages[0])
        self.assertIsNone(messages[1]["content"])

    def test_downloaded_qwen_template_renders_tool_round_trip(self):
        # Optional integration check against the user's actual downloaded
        # template, with real Jinja rendering but no GPU/model/network access.
        try:
            from jinja2 import UndefinedError
            from jinja2.sandbox import SandboxedEnvironment
        except ImportError:
            self.skipTest("Jinja is not installed")
        configs = list((Path.home() / ".cache/huggingface/hub/models--mlx-community--Qwen3-4B-4bit/snapshots").glob("*/tokenizer_config.json"))
        if not configs:
            self.skipTest("Default Qwen3 tokenizer is not cached locally")
        source = json.loads(configs[0].read_text())["chat_template"]
        template = SandboxedEnvironment().from_string(source)
        history = [
            {"role": "system", "content": "Use tools."},
            {"role": "user", "content": "List files and read requirements.txt"},
            {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function", "function": {
                "name": "list_dir", "arguments": '{"path":"."}'}}]},
            {"role": "tool", "tool_call_id": "c1", "content": '{"entries":["requirements.txt"]}'},
        ]

        def render(messages):
            return template.render(messages=messages, tools=agent.TOOL_SCHEMAS,
                                   add_generation_prompt=True, enable_thinking=False)

        with self.assertRaises(UndefinedError):
            render(history)
        model = self.make_model()
        prompt = render(model._messages(history))
        self.assertIn('<tool_call>\n{"name": "list_dir"', prompt)
        self.assertIn('<tool_response>\n{"entries":["requirements.txt"]}', prompt)
        history[2]["content"] = None
        self.assertEqual(render(model._messages(history)), prompt)

    def test_incomplete_or_malformed_calls_are_not_executed(self):
        model = self.make_model()
        for text in ('<tool_call>{"name":"read_file"}', '<tool_call>bad</tool_call>',
                     '<tool_call>[]</tool_call>',
                     '<tool_call>{"name":"read_file","arguments":[]}</tool_call>',
                     '<think>unfinished', '<think>reasoning only</think>', ''):
            with self.subTest(text=text), self.assertRaises(ValueError):
                model._parse_response(text)

    def test_multiple_calls_have_unique_ids_and_preserve_literal_think_tags(self):
        model = self.make_model()
        text = '<tool_call>' + json.dumps({"name": "write_file", "arguments": {
            "path": "example.txt", "content": "<think>literal</think>"}}) + '</tool_call>'
        calls, _ = model._parse_response(text + text)
        self.assertNotEqual(calls[0]["id"], calls[1]["id"])
        self.assertEqual(json.loads(calls[0]["function"]["arguments"])["content"], "<think>literal</think>")

    def test_output_limit_and_generation_error_cleanup(self):
        model = self.make_model()
        self.mx.get_peak_memory.return_value = 2048
        self.generate.side_effect = lambda *args, **kwargs: self.stream("partial", finish="length")
        with self.assertRaisesRegex(ValueError, "output budget"):
            model.respond([])
        self.assertTrue(self.closed)
        self.mx.clear_cache.assert_called()
        self.assertIsNotNone(model.last_stats["ttft_seconds"])
        self.assertEqual(model.last_stats["peak_memory_bytes"], 2048)
        self.assertEqual(model.last_stats["generation_tokens"], 12)

        def broken(*args, **kwargs):
            yield SimpleNamespace(text="x", finish_reason=None)
            raise RuntimeError("generation failed")

        self.generate.side_effect = broken
        before = self.mx.clear_cache.call_count
        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            model.respond([])
        self.assertGreater(self.mx.clear_cache.call_count, before)

    def test_parse_failure_retains_metrics(self):
        model = self.make_model()
        self.mx.get_peak_memory.return_value = 1024
        self.generate.side_effect = lambda *args, **kwargs: self.stream("Answer <tool_call>bad</tool_call>")
        with patch("inquiry_loop.backends.mlx_eng.perf_counter", side_effect=[10.0, 11.0]):
            with self.assertRaisesRegex(ValueError, "invalid tool arguments"):
                model.respond([])
        self.assertEqual(model.last_stats["ttft_seconds"], 1.0)
        self.assertEqual(model.last_stats["peak_memory_bytes"], 1024)
        self.assertEqual(model.last_stats["generation_tokens"], 12)

    def test_context_overflow_does_not_generate_or_truncate(self):
        model = self.make_model(max_tokens=16, max_context_tokens=32)
        self.tokenizer.encode.return_value = list(range(17))
        with self.assertRaisesRegex(ValueError, "context budget"):
            model.respond([])
        self.generate.assert_not_called()
        self.tokenizer.encode.assert_called_once_with("formatted prompt", add_special_tokens=False)

    def test_unsupported_tokenizer_or_model_load_error_restores_cache(self):
        self.tokenizer.has_tool_calling = False
        with self.assertRaisesRegex(ValueError, "native tool-call"):
            self.make_model()
        self.mx.set_cache_limit.assert_called_with(12345)
        self.load.side_effect = RuntimeError("load failed")
        with self.assertRaisesRegex(RuntimeError, "load failed"):
            self.make_model()

    def test_invalid_budgets_model_path_and_selection(self):
        for options in ({"max_tokens": 0}, {"max_tokens": 32, "max_context_tokens": 32},
                        {"prefill_step_size": 0}, {"cache_limit_mb": -1},
                        {"model_path": "/missing/model"}, {"model": "repo", "model_path": "/path"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.make_model(**options)
        self.load.assert_not_called()

    def test_factory_and_cli_options(self):
        with patch("inquiry_loop.backends.mlx_eng.MLXModel") as backend:
            agent.create_model("mlx", model="custom/model")
            backend.assert_called_once_with(tool_schemas=agent.TOOL_SCHEMAS, model="custom/model", model_path=None)
        with patch("inquiry_loop.cli.create_model") as factory, patch("inquiry_loop.cli.run_agent", return_value="done"), \
                contextlib.redirect_stdout(io.StringIO()):
            cli.main(["--no-trace", "--llm=mlx", "--model=custom/model", "hello"])
            factory.assert_called_once_with("mlx", stub=False, model_path=None, model="custom/model")
        for argv in (["--llm=mlx", "--base-url=http://localhost/v1"],
                     ["--llm=mlx", "--model=repo", "--model-path=/path"],
                     ["--llm=mlx", "--stub", "--model=repo"]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                cli.main(argv)
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
