import os
import pytest
from unittest.mock import MagicMock, patch
from azllm.clients.anthropic import AnthropicClient, DEFAULT_CONFIG 

# ------------------------
# 1. Test Default Config
# ------------------------

def test_get_default_config():
    assert AnthropicClient.get_default_config() == DEFAULT_CONFIG

# ------------------------
# 2. Test Initialization
# ------------------------

def test_anthropic_client_init_defaults():
    client = AnthropicClient()
    assert client.model == DEFAULT_CONFIG['model']
    assert client.system_message == DEFAULT_CONFIG['system_message']
    assert client.temperature == DEFAULT_CONFIG['temperature']
    assert client.max_tokens == DEFAULT_CONFIG['max_tokens']
    assert client.kwargs == DEFAULT_CONFIG['kwargs']

def test_anthropic_client_init_custom():
    custom_config = {
        'model': 'claude-3-7-sonnet-20250219',
        'parameters': {
            'system_message': 'Custom system message.',
            'temperature': 0.5,
            'max_tokens': 2000,
            'kwargs': {'top_p': 0.9}
        }
    }
    client = AnthropicClient(config=custom_config)
    assert client.model == 'claude-3-7-sonnet-20250219'
    assert client.system_message == 'Custom system message.'
    assert client.temperature == 0.5
    assert client.max_tokens == 2000
    assert client.kwargs == {'top_p': 0.9}
# ------------------------
# 3. Test get_api_key()
# ------------------------

def test_get_api_key_success(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key") 
    client = AnthropicClient()
    assert client.get_api_key() == "fake-key"

def test_get_api_key_missing(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = AnthropicClient()
    with pytest.raises(ValueError, match="A valid API Key for Anthropic is missing."):
        client.get_api_key()

# ------------------------
# 4. Test generate_text (mocked)
# ------------------------

@patch("azllm.clients.anthropic.OpenAI")  
def test_generate_text_mocked(mock_openai):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Mock response"))]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai.return_value = mock_client

    client = AnthropicClient()
    client.get_api_key = MagicMock(return_value="mock-key")
    client.client = None

    result = client.generate_text("Hello, AI!")
    assert result == "Mock response"
    mock_openai.assert_called_once_with(api_key="mock-key", base_url="https://api.anthropic.com/v1/")
    mock_client.chat.completions.create.assert_called_once()


def test_batch_generate_success():
    prompts = ["Hello", "How are you?", "Tell me a joke"]
    expected_responses = ["Hi!", "I'm good!", "Why did the chicken..."]

    client = AnthropicClient()
    
    client.generate_text = MagicMock(side_effect=expected_responses)

    kwargs = [{}] * len(prompts)  
    parse = [False] * len(prompts)  
    
    responses = client.batch_generate(prompts, kwargs=kwargs, parse=parse)

    assert responses == expected_responses
    assert client.generate_text.call_count == len(prompts)
    client.generate_text.assert_any_call("Hello", {}, False)


def test_batch_generate_with_errors():
    prompts = ["Hello", "Bad prompt", "Another one"]
    
    def side_effect(prompt, kwargs, parse):
        if prompt == "Bad prompt":
            raise RuntimeError("Something went wrong")
        return f"Response to: {prompt}"
    
    client = AnthropicClient()
    client.generate_text = MagicMock(side_effect=side_effect)

    kwargs = [{}] * len(prompts)  
    parse = [False] * len(prompts)  

    responses = client.batch_generate(prompts, kwargs=kwargs, parse=parse)

    assert responses == [
        "Response to: Hello",
        "Error: Something went wrong",
        "Response to: Another one"
    ]
    assert client.generate_text.call_count == 3