import pytest
from unittest.mock import MagicMock
from azllm import azLLM  

def test_generate_parallel_success():
    llm = azLLM()
    prompt = "What is the capital of France?"
    client_models = ["openai:gpt-4o-mini::v1", "gemini:model-x::v2", "ollama:llama2::default"]
    expected_outputs = ["Paris", "Paris (Gemini)", "Paris (LLaMA)"]

    # Mock generate_text to return fake answers
    llm.generate_text = MagicMock(side_effect=expected_outputs)

    results = llm.generate_parallel(prompt, client_models)

    for idx, cmv in enumerate(client_models):
        key = f"{cmv}:{idx}"
        assert results[key] == expected_outputs[idx]

    assert llm.generate_text.call_count == len(client_models)

def test_generate_parallel_with_errors():
    llm = azLLM()
    prompt = "Tell me a joke"
    client_models = ["openai:gpt-4::v1", "grok:model-z::v1"]

    # First succeeds, second throws error
    def side_effect(client_model_version, prompt, kwargs, parse):
        if "grok" in client_model_version:
            raise RuntimeError("API timeout")
        return "Why did the chicken cross the road?"

    llm.generate_text = MagicMock(side_effect=side_effect)

    results = llm.generate_parallel(prompt, client_models)

    assert results["openai:gpt-4::v1:0"] == "Why did the chicken cross the road?"
    assert results["grok:model-z::v1:1"].startswith("Error: API timeout")

def test_generate_parallel_invalid_kwargs_length():
    llm = azLLM()
    with pytest.raises(ValueError, match="length of kwargs"):
        llm.generate_parallel("prompt", ["openai:gpt-4::v1"], kwargs=[{}, {}])

def test_generate_parallel_invalid_parse_length():
    llm = azLLM()
    with pytest.raises(ValueError, match="length of parse"):
        llm.generate_parallel("prompt", ["openai:gpt-4::v1"], parse=[True, False])
