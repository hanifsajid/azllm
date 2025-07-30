import yaml
import os
from pathlib import Path
import json
from typing import Type, Optional
from pydantic import BaseModel


def load_custom_config(config_dir: str, config_file: str) -> dict:
    """
    Load a custom YAML configuration from a specified directory.

    Args:
        config_dir (str): The name of the directory where the config file is stored.
        config_file (str): The YAML file name to be loaded.

    Returns:
        dict: Parsed YAML content as a dictionary, or None if loading fails.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If YAML file contains invalid format.
    """
    try:
        cwd = Path(os.getcwd())  
        config_dir = cwd / config_dir
        config_path = config_dir / config_file

        if not config_path.exists():
            raise FileNotFoundError(f"The configuration file '{config_file}' was not found in the directory '{config_dir}'.")

        with open(config_path, 'r') as file:
            try:
                return yaml.safe_load(file)
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing YAML file '{config_file}': {e}")
    
    except FileNotFoundError as fnf_error:
        print(fnf_error)
        return None
    except PermissionError:
        print(f"Permission denied while trying to read the file '{config_file}' in '{config_dir}'.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the configuration: {e}")
        return None
    
def save_custom_configs(config_file: str, custom_configs):
        """
        Save a custom configuration dictionary to a YAML file.

        Args:
            config_file (str): The name of the YAML file to write to.
            custom_configs (dict): The dictionary containing updated configuration data.

        Returns:
            None
        """

        config_dir = Path('custom_configs')
        config_file = config_dir / config_file
        
        config_dir.mkdir(parents=True, exist_ok=True)
    
        with open(config_file, 'w') as f:
            yaml.dump(custom_configs, f)
            print(f"Custom configurations saved to {config_file}")

 

def create_custom_file(config_dir_name: str, config_file_name: str, template_content: dict) -> None:
    """Create an empty directory with a template YAML file for custom configs."""
    """
    Create a new configuration directory and YAML file with template content if it doesn't exist.

    Args:
        config_dir_name (str): Directory to create for storing the config file.
        config_file_name (str): Name of the config YAML file.
        template_content (dict): Initial template content to write to the file.

    Returns:
        None
    """

    cwd = Path(os.getcwd())  

    config_dir = cwd / config_dir_name
    config_dir.mkdir(parents=True, exist_ok=True)

    template_file = config_dir / config_file_name

    if not template_file.exists():
        with open(template_file, 'w') as file:
            yaml.dump(template_content, file, default_flow_style=False)

        print(f"Template YAML file created at {template_file}. You can now customize it.")
    else:
        print(f"The template YAML file already exists at {template_file}.")


class StructuredOutput:
    def _build_prompt(self, schema: Type[BaseModel]) -> str:
        return (
            "You are an API that returns structured data.\n"
            "Return only a valid JSON object that matches this schema:\n"
            f"{json.dumps(schema.model_json_schema(), indent=2)}\n"
            "Do not include any extra text. Just the JSON."
        )

    def extract_json(self, text: str) -> dict:
        """
        Extracts a JSON object from text by tracking opening/closing braces.
        Avoids relying on unsupported recursive regex (?R).
        """
        start = text.find("{")
        if start == -1:
            raise ValueError("No opening brace found in response.")

        open_braces = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                open_braces += 1
            elif text[i] == "}":
                open_braces -= 1
                if open_braces == 0:
                    json_str = text[start:i + 1]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON: {e}")
        raise ValueError("Could not find matching closing brace for JSON.")

    def format_system_message(
        self,
        response_format: Type[BaseModel],
        user_system_prompt: Optional[str] = None) -> dict:
        """
        Formats a system message combining a user prompt and a schema prompt.
        """
        structured_prompt = self._build_prompt(response_format)

        if user_system_prompt:
            combined_prompt = f"{user_system_prompt.strip()}\n\n{structured_prompt}"
        else:
            combined_prompt = structured_prompt

        return {"role": "system", "content": combined_prompt}
      

template_content = {'openai':
                     {'models': [
                         {'model': 'gpt-4o-mini',
                          'version': 'default',
                          'parameters': {
                              'system_message': 'You are a helpful assistant. You always start the output with "HELLO, I AM AI ASSITANT HS"',
                              'temperature': 0.7,
                              'max_tokens': 1024,
                              'frequency_penalty': 0,
                              'presence_penalty': 0
                        }
                    }
                ]
            }
        }

__all__ = []