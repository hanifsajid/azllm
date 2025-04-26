import os
import tempfile
import shutil
import pytest
import yaml
from unittest.mock import MagicMock, patch

from azllm.configmanager import create_custom_file, save_custom_configs
from azllm import configmanager 

ACCEPTABLE_CLIENTS = ['openai', 'deepseek', 'grok', 'gemini', 'anthropic', 'ollama', 'fireworks']

@pytest.fixture
def temp_custom_dir(monkeypatch):
    """Fixture to mock 'custom_configs' directory creation and cleanup."""
    temp_dir = tempfile.mkdtemp()
    monkeypatch.setattr(configmanager, "create_custom_file", MagicMock())
    monkeypatch.setattr(configmanager, "save_custom_configs", MagicMock())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_load_custom_config(monkeypatch):
    """Mock `load_custom_config` to return a fake config."""
    dummy_config = {
        "openai": {
            "models": [
                {
                    "model": "gpt-4o-mini",
                    "version": "v1",
                    "parameters": {"temperature": 0.7}
                }
            ]
        }
    }
    monkeypatch.setattr(configmanager, "load_custom_config", MagicMock(return_value=dummy_config))
    return dummy_config


# --- Tests ---

def test_init_with_custom_config(temp_custom_dir, mock_load_custom_config):
    cfg = configmanager.azLLMConfigs(config_file="test_config.yaml", custom=True)
    assert "openai" in cfg.custom_configs
    assert isinstance(cfg.custom_configs["openai"]["models"], list)


@patch("azllm.clients.openai.OpenAIClient.get_default_config", return_value={"model": "mock"})
@patch("azllm.clients.deepseek.DeepSeekClient.get_default_config", return_value={"model": "mock"})
@patch("azllm.clients.grok.GrokClient.get_default_config", return_value={"model": "mock"})
@patch("azllm.clients.anthropic.AnthropicClient.get_default_config", return_value={"model": "mock"})
@patch("azllm.clients.gemini.GeminiClient.get_default_config", return_value={"model": "mock"})
@patch("azllm.clients.ollama.OllamaClient.get_default_config", return_value={"model": "mock"})
@patch("azllm.clients.fireworks.FireworksClient.get_default_config", return_value={"model": "mock"})
def test_get_default_configs_all(mock_fw, mock_ol, mock_ge, mock_an, mock_gr, mock_ds, mock_oa):
    cfg = configmanager.azLLMConfigs(custom=False)
    all_configs = cfg.get_default_configs()
    assert isinstance(all_configs, dict)
    assert all(client in all_configs for client in ACCEPTABLE_CLIENTS)


def test_get_default_configs_invalid():
    cfg = configmanager.azLLMConfigs(custom=False)
    with pytest.raises(ValueError, match="Default configuration for client type 'foobar'"):
        cfg.get_default_configs("foobar")


def test_update_existing_model(temp_custom_dir, mock_load_custom_config, monkeypatch):
    monkeypatch.setattr(configmanager, "save_custom_configs", MagicMock())
    cfg = configmanager.azLLMConfigs(config_file="custom.yaml", custom=True)

    cfg.update_custom_configs("openai", {
        "gpt-4o-mini": {
            "version": "v1",
            "parameters": {"temperature": 0.9}
        }
    })

    updated_model = next(
        m for m in cfg.custom_configs["openai"]["models"]
        if m["model"] == "gpt-4o-mini" and m["version"] == "v1"
    )
    assert updated_model["parameters"]["temperature"] == 0.9


def test_add_new_model(temp_custom_dir, mock_load_custom_config, monkeypatch):
    monkeypatch.setattr(configmanager,"save_custom_configs", MagicMock())
    cfg = configmanager.azLLMConfigs(config_file="custom.yaml", custom=True)

    cfg.update_custom_configs("openai", {
        "new-model": {
            "version": "v1",
            "parameters": {"temperature": 0.6}
        }
    })

    new_model = next(
        m for m in cfg.custom_configs["openai"]["models"]
        if m["model"] == "new-model"
    )
    assert new_model["version"] == "v1"
    assert new_model["parameters"]["temperature"] == 0.6


def test_update_invalid_client(monkeypatch):
    monkeypatch.setattr(configmanager,"load_custom_config", MagicMock(return_value={}))
    monkeypatch.setattr(configmanager, "create_custom_file", MagicMock())
    monkeypatch.setattr(configmanager, "save_custom_configs", MagicMock())
    cfg = configmanager.azLLMConfigs(config_file="invalid.yaml", custom=True)

    with pytest.raises(ValueError, match="Client type fakeclient is not working"):
        cfg.update_custom_configs("fakeclient", {
            "model-x": {
                "parameters": {"temperature": 1.0}
            }
        })