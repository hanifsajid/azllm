import os
import pytest
from unittest.mock import MagicMock, patch
from azllm.clients.deepseek import DeepSeekClient, DEFAULT_CONFIG 

# ------------------------
# 1. Test Default Config
# ------------------------

def test_get_default_config():
    assert DeepSeekClient.get_default_config() == DEFAULT_CONFIG

# ------------------------
# 2. Test Initialization
# ------------------------

def test_deepseek_client_init_defaults():
    client = DeepSeekClient()
    assert client.model == DEFAULT_CONFIG['model']
    assert client.system_message == DEFAULT_CONFIG['system_message']
    assert client.temperature == DEFAULT_CONFIG['temperature']
    assert client.max_tokens == DEFAULT_CONFIG['max_tokens']
    assert client.kwargs == DEFAULT_CONFIG['kwargs']

def test_deepseek_client_init_custom():
    custom_config = {
        'model': 'deepseek-chat',
        'parameters': {
            'system_message': 'Custom system message.',
            'temperature': 0.5,
            'max_tokens': 2000,
            'kwargs': {'top_p': 0.9}
        }
    }
    client = DeepSeekClient(config=custom_config)
    assert client.model == 'deepseek-chat'
    assert client.system_message == 'Custom system message.'
    assert client.temperature == 0.5
    assert client.max_tokens == 2000
    assert client.kwargs == {'top_p': 0.9}
# ------------------------
# 3. Test get_api_key()
# ------------------------

def test_get_api_key_success(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key") 
    client = DeepSeekClient()
    assert client.get_api_key() == "fake-key"

def test_get_api_key_missing(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    client = DeepSeekClient()
    with pytest.raises(ValueError, match="A valid API Key for DeepSeek is missing."):
        client.get_api_key()

# ------------------------
# 4. Test generate_text (mocked)
# ------------------------

@patch("azllm.clients.deepseek.OpenAI")  
def test_generate_text_mocked(mock_openai):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Mock response"))]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai.return_value = mock_client

    client = DeepSeekClient()
    client.get_api_key = MagicMock(return_value="mock-key")
    client.client = None

    result = client.generate_text("Hello, AI!")
    assert result == "Mock response"
    mock_openai.assert_called_once_with(api_key="mock-key", base_url="https://api.deepseek.com/v1")
    mock_client.chat.completions.create.assert_called_once()


def test_batch_generate_success():
    prompts = ["Hello", "How are you?", "Tell me a joke"]
    expected_responses = ["Hi!", "I'm good!", "Why did the chicken..."]

    client = DeepSeekClient()
    
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
    
    client = DeepSeekClient()
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