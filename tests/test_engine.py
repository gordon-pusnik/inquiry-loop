import unittest
from unittest.mock import Mock, patch

from inquiry_loop.engine import chat, create_model
from inquiry_loop.response_types import ModelResponse


class EngineTests(unittest.TestCase):
    def test_chat_preserves_history_without_tools(self):
        messages = [
            {"role": "user", "content": "What's the capital of Ontario?"},
            {"role": "assistant", "content": "Toronto."},
            {"role": "user", "content": "Which lake is it on?"},
        ]
        model = Mock()
        model.respond.return_value = ModelResponse([], "Lake Ontario.", {"generation_tokens": 3}, "stop")

        response = chat(messages, model)

        model.respond.assert_called_once_with(messages, allow_tools=False)
        self.assertEqual(response.text, "Lake Ontario.")
        self.assertEqual(response.stats["generation_tokens"], 3)

    def test_chat_rejects_tool_messages_before_inference(self):
        model = Mock()
        user = {"role": "user", "content": "Continue"}
        for message in ({"role": "tool", "content": "result"},
                        {"role": "assistant", "content": "", "tool_calls": []}):
            with self.subTest(message=message), self.assertRaises(ValueError):
                chat([message, user], model)
        model.respond.assert_not_called()

    def test_chat_rejects_tool_responses(self):
        messages = [{"role": "user", "content": "Say hello."}]
        model = Mock()
        for response in (ModelResponse([{"name": "example"}], "Hello"),
                         ModelResponse([], "Hello", finish_reason="function_call")):
            with self.subTest(response=response), self.assertRaisesRegex(ValueError, "unexpected tool call"):
                model.respond.return_value = response
                chat(messages, model)

    def test_factory_sets_explicit_hosted_model_without_tool_schemas(self):
        with patch("inquiry_loop.backends.openai_eng.OpenAIModel") as backend:
            self.assertIs(create_model(model="example-model"), backend.return_value)

        backend.assert_called_once_with(model="example-model", tool_schemas=[])
