"""Shared model construction and text chat for non-agent callers."""

from inquiry_loop.response_types import ModelResponse


def create_model(llm: str = "openai", *, stub: bool = False, model_path=None,
                 base_url=None, model=None, tool_schemas=None):
    """Load only the selected backend. Tools require explicit schemas."""
    schemas = [] if tool_schemas is None else tool_schemas
    if stub:
        from inquiry_loop.backends.stub_eng import StubModel

        return StubModel()
    if llm == "openai":
        from inquiry_loop.backends.openai_eng import OpenAIModel

        options = {} if model is None else {"model": model}
        return OpenAIModel(tool_schemas=schemas, **options)
    if llm == "airllm":
        from inquiry_loop.backends.airllm_eng import AirLLMModel

        return AirLLMModel(tool_schemas=schemas, model_path=model_path)
    if llm == "strata":
        from inquiry_loop.backends.strata_eng import StrataModel

        return StrataModel(tool_schemas=schemas, base_url=base_url, model=model)
    if llm == "mlx":
        from inquiry_loop.backends.mlx_eng import MLXModel

        return MLXModel(tool_schemas=schemas, model=model, model_path=model_path)
    raise ValueError(f"Unknown LLM backend: {llm}")


def chat(messages: list[dict[str, str]], model) -> ModelResponse:
    """Run one text completion. The caller owns history, limits, and model lifetime."""
    if not messages:
        raise ValueError("Chat requires at least one message.")
    prompt = []
    for message in messages:
        role, content = message.get("role"), message.get("content")
        if role not in ("system", "user", "assistant") or not isinstance(content, str):
            raise ValueError("Chat messages require a system, user, or assistant role and text content.")
        if "tool_calls" in message or "function_call" in message:
            raise ValueError("Chat messages cannot contain tool or function calls.")
        prompt.append({"role": role, "content": content})
    if prompt[-1]["role"] != "user" or not prompt[-1]["content"].strip():
        raise ValueError("Chat must end with a nonempty user message.")

    response = ModelResponse.coerce(model.respond(prompt, allow_tools=False),
                                    getattr(model, "last_stats", None))
    if response.tool_calls or response.finish_reason in ("tool_calls", "function_call"):
        raise ValueError("Chat received an unexpected tool call.")
    if not response.text.strip():
        raise ValueError("Chat received an empty response.")
    return response
