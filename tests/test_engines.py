import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import ANY, Mock, patch

from inquiry_loop import agent, cli
from inquiry_loop.backends.airllm_eng import AirLLMModel
from inquiry_loop.backends.openai_eng import OpenAIModel


class BackendTests(unittest.TestCase):
    def test_imports_do_not_load_model_dependencies(self):
        result = subprocess.run([
            sys.executable, "-c",
            "import sys, agent; "
            "from inquiry_loop.backends import openai_eng, airllm_eng, strata_eng, mlx_eng; "
            "assert not any(n in sys.modules for n in ('openai', 'mlx.core', 'airllm', 'mlx_lm'))",
        ], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cli_defaults_to_openai_and_selects_airllm(self):
        for argv, backend, path in ((["hello"], "openai", None),
                                    (["--llm=airllm", "--model-path", "/model", "hello"], "airllm", "/model")):
            with self.subTest(backend=backend), patch("inquiry_loop.cli.create_model") as factory, \
                    patch("inquiry_loop.cli.run_agent", return_value="done") as run, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(["--no-trace", *argv]), 0)
                factory.assert_called_once_with(backend, stub=False, model_path=path)
                run.assert_called_once_with("hello", factory.return_value, max_steps=agent.MAX_STEPS, metrics=ANY)

    def test_cli_rejects_invalid_backend_and_irrelevant_path(self):
        for argv in (["--llm=unknown"], ["--model-path", "/model"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                cli.main(argv)
            self.assertEqual(error.exception.code, 2)

    def test_factory_loads_selected_engine(self):
        with (patch("inquiry_loop.backends.openai_eng.configured_openai_model", return_value="gpt-4o-mini"),
              patch("inquiry_loop.backends.openai_eng.OpenAIModel") as engine):
            self.assertIs(agent.create_model(), engine.return_value)
            engine.assert_called_once_with(tool_schemas=agent.TOOL_SCHEMAS)
        with patch("inquiry_loop.backends.airllm_eng.AirLLMModel") as engine:
            self.assertIs(agent.create_model("airllm", model_path="/model"), engine.return_value)
            engine.assert_called_once_with(tool_schemas=agent.TOOL_SCHEMAS, model_path="/model")
        self.assertIsInstance(agent.create_model("airllm", stub=True), agent.StubModel)

    def test_openai_configuration_and_tool_feedback(self):
        client = Mock()
        call = {"id": "call_test", "type": "function", "function": {
            "name": "list_dir", "arguments": '{"path":"."}'}}
        client.chat.completions.create.side_effect = [
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[call], content=None))]),
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[], content="done"))]),
        ]
        modules = {"openai": SimpleNamespace(OpenAI=Mock(return_value=client)),
                   "dotenv": SimpleNamespace(load_dotenv=Mock())}
        with patch.dict(sys.modules, modules), patch.dict(os.environ, {"AGENT_MODEL": "configured-model"}):
            model = OpenAIModel(tool_schemas=agent.TOOL_SCHEMAS)
        with patch.dict(agent.TOOLS, {"list_dir": Mock(return_value={"entries": ["example"]})}):
            self.assertEqual(agent.run_agent("list", model, verbose=False), "done")
        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "configured-model")
        self.assertEqual(kwargs["tools"], agent.TOOL_SCHEMAS)
        self.assertFalse(kwargs["parallel_tool_calls"])
        self.assertEqual(json.loads(kwargs["messages"][-1]["content"])["entries"], ["example"])
        modules["dotenv"].load_dotenv.assert_called_once()

    def make_local_model(self, responses):
        model = AirLLMModel.__new__(AirLLMModel)
        model.tool_schemas = agent.TOOL_SCHEMAS
        model.max_new_tokens = 8
        model.max_context_tokens = 128
        model.mx = SimpleNamespace(array=lambda value: value)
        model.tokenizer = Mock(eos_token_id=99)
        model.tokenizer.return_value = {"input_ids": [[1, 2, 3]]}
        model.tokenizer.apply_chat_template.return_value = "prompt"
        model.tokenizer.decode.side_effect = responses
        self.closed_streams = []

        def generate(*args, **kwargs):
            try:
                for value in (4, 99, 5):
                    yield SimpleNamespace(item=lambda value=value: value)
            finally:
                self.closed_streams.append(True)

        model.model = Mock()
        model.model.model_generate.side_effect = generate
        return model

    def test_airllm_tool_feedback_eos_and_history(self):
        model = self.make_local_model([
            '{"tool_call":{"name":"list_dir","arguments":{"path":"."}}}',
            '{"answer":"done"}',
        ])
        tool = Mock(return_value={"entries": ["example"]})
        with patch.dict(agent.TOOLS, {"list_dir": tool}):
            self.assertEqual(agent.run_agent("list", model, verbose=False), "done")
        tool.assert_called_once_with(path=".")
        model.tokenizer.decode.assert_called_with([4], skip_special_tokens=True)
        self.assertEqual(len(self.closed_streams), 2)
        chat = model.tokenizer.apply_chat_template.call_args.args[0]
        history = json.loads(chat[1]["content"])
        self.assertEqual(history[-1]["role"], "tool")
        self.assertEqual(json.loads(history[-1]["content"])["entries"], ["example"])
        self.assertFalse(model.tokenizer.apply_chat_template.call_args.kwargs["enable_thinking"])

    def test_airllm_rejects_malformed_or_ambiguous_decisions(self):
        for text in ('not JSON', '[]', '{"answer":"ok","tool_call":{}}',
                     '{"tool_call":{"name":"write_file","arguments":"bad"}}',
                     '{"answer":123}', '{"tool_call":'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                AirLLMModel._parse_response(text)
        calls, answer = AirLLMModel._parse_response('<think>reasoning</think>{"answer":"ok"}')
        self.assertEqual((calls, answer), ([], "ok"))

    def test_airllm_does_not_truncate_history(self):
        model = self.make_local_model([])
        model.max_context_tokens = 10
        with self.assertRaisesRegex(ValueError, "context budget"):
            model.respond([{"role": "user", "content": "hello"}])
        model.model.model_generate.assert_not_called()

    def test_airllm_token_limit_closes_stream(self):
        model = self.make_local_model(['{"answer":"ok"}'])
        model.max_new_tokens = 1
        self.assertEqual(tuple(model.respond([])), ([], "ok"))
        self.assertEqual(len(self.closed_streams), 1)

    def test_airllm_model_config_applied_before_generation(self):
        backend = Mock()
        mx = SimpleNamespace()
        modules = {"mlx": SimpleNamespace(core=mx), "mlx.core": mx,
                   "airllm": SimpleNamespace(),
                   "airllm.airllm_llama_mlx": SimpleNamespace(AirLLMLlamaMlx=Mock(return_value=backend))}
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text(json.dumps({
                "model_type": "qwen3", "head_dim": 128, "rope_theta": 1000000.0}))
            with patch("inquiry_loop.backends.airllm_eng.sys.platform", "darwin"), patch.dict(sys.modules, modules), \
                    patch("inquiry_loop.backends.airllm_eng._install_qwen3_attention") as install:
                model = AirLLMModel(tool_schemas=[], model_path=directory)
            install.assert_called_once()
            self.assertEqual(model.model.model_args.head_dim, 128)
            self.assertEqual(model.model.model_args.rope_theta, 1000000.0)
            self.assertFalse(model.model.model_args.rope_traditional)


if __name__ == "__main__":
    unittest.main()
