import os
import shutil
import tempfile
import yaml
from pathlib import Path

import pytest
from azllm import utils


@pytest.fixture
def temp_dir():
    """Creates a temporary directory for testing and cleans it up after."""
    dirpath = tempfile.mkdtemp()
    yield dirpath
    shutil.rmtree(dirpath)

def test_create_custom_file(temp_dir):
    config_dir_name = "test_config_dir"
    config_file_name = "config.yaml"
    template_content = {'openai': { 'gpt-4o-mini': {'version': 'v1', 'parameters': { 'system_message': 'You are an advanced AI assistant', 'temperature': 0.6 }}}}

    # Change working directory temporarily
    original_cwd = os.getcwd()
    os.chdir(temp_dir)

    try:
        utils.create_custom_file(config_dir_name, config_file_name, template_content)

        config_path = Path(temp_dir) / config_dir_name / config_file_name

        # Check if the file was created
        assert config_path.exists(), "Config file was not created"

        # Check the contents of the file
        with open(config_path, 'r') as f:
            content = yaml.safe_load(f)
        assert content == template_content, "File contents do not match template"
    finally:
        os.chdir(original_cwd)


def test_save_custom_configs(temp_dir):
    config_file_name = "config.yaml"
    custom_data = {'deepseek': { 'deepseek-chat': {'version': 'v1', 'parameters': { 'system_message': 'You are a DeepSeek advanced AI assistant', 'temperature': 0.4 }}}}
    
    # Change working directory temporarily
    original_cwd = os.getcwd()
    os.chdir(temp_dir)

    try:
        # Call the function (it will write to custom_configs/config.yaml)
        utils.save_custom_configs(config_file_name, custom_data)

        config_path = Path(temp_dir) / 'custom_configs' / config_file_name

        # Assert that file was created
        assert config_path.exists(), "Config file was not created"

        # Assert contents are correct
        with open(config_path, 'r') as f:
            saved_content = yaml.safe_load(f)
        assert saved_content == custom_data, "Saved config data does not match expected"
    finally:
        os.chdir(original_cwd)

def test_load_valid_custom_config(temp_dir):
    config_dir = "my_config"
    config_file = "config.yaml"
    config_data = {"option": True, "details": {"level": 5}}

    # Create directory and file
    config_path = Path(temp_dir) / config_dir
    config_path.mkdir(parents=True)
    file_path = config_path / config_file
    with open(file_path, 'w') as f:
        yaml.dump(config_data, f)

    # Temporarily change to temp_dir
    original_cwd = os.getcwd()
    os.chdir(temp_dir)

    try:
        loaded_config = utils.load_custom_config(config_dir, config_file)
        assert loaded_config == config_data
    finally:
        os.chdir(original_cwd)

def test_load_missing_config_file(temp_dir, capsys):
    config_dir = "non_existent_dir"
    config_file = "missing.yaml"

    # Change to temp_dir
    original_cwd = os.getcwd()
    os.chdir(temp_dir)

    try:
        result = utils.load_custom_config(config_dir, config_file)
        captured = capsys.readouterr()
        assert result is None
        assert "was not found" in captured.out
    finally:
        os.chdir(original_cwd)

def test_load_invalid_yaml(temp_dir, capsys):
    config_dir = "bad_config"
    config_file = "broken.yaml"
    
    config_path = Path(temp_dir) / config_dir
    config_path.mkdir(parents=True)
    file_path = config_path / config_file

    # Write invalid YAML content
    with open(file_path, 'w') as f:
        f.write("invalid: [unclosed_list\n")

    # Change to temp_dir
    original_cwd = os.getcwd()
    os.chdir(temp_dir)

    try:
        result = utils.load_custom_config(config_dir, config_file)
        captured = capsys.readouterr()
        assert result is None
        assert "Error parsing YAML file" in captured.out or "An error occurred" in captured.out
    finally:
        os.chdir(original_cwd)
