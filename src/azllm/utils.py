import yaml
import os
from pathlib import Path


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