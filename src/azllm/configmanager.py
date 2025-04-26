
from .utils import create_custom_file, load_custom_config, save_custom_configs, template_content
from typing import Dict, Any 

from .clients.openai import OpenAIClient
from .clients.deepseek import DeepSeekClient
from .clients.grok import GrokClient
from .clients.anthropic import AnthropicClient
from .clients.gemini import GeminiClient
from .clients.fireworks import FireworksClient
from .clients.ollama import OllamaClient



ACCEPTABLE_CLIENTS = ['openai', 'deepseek', 'grok', 'gemini', 'anthropic', 'ollama', 'fireworks']


class azLLMConfigs:
    """
    A configuration manager for handling custom and default configurations 
    of supported LLM clients.
    
    Attributes:
        config_file (str): Name of the configuration file.
        custom (bool): Flag to determine if custom configurations are used.
        custom_configs (dict): Dictionary holding custom configuration data.
    
        
    Usage:
        To use default configurations:
            >>> cfg = azLLMConfigs()
            >>> cfg.get_default_configs('openai')

        Retrieve default config for all supported clients
            >>> cfg.get_default_configs('all')

    Working with Custom Configurations:
    -----------------------------------

    When `custom=True`, the class will:
    
    - Create a local configuration file if it doesn't exist at: `custom_configs/config.yaml`
    - Load existing configurations from that file.
    - Allow updating or adding new model configurations per client.
    
    Example of internal structure of a custom config (custom_configs/config.yaml):

    ::
        
        deepseek:
            models:
                - model: deepseek-chat
                version: v2
                parameters:
                    frequency_penalty: 0
                    max_tokens: 1024
                    presence_penalty: 0
                    system_message: You are an advanced AI assistant.
                    temperature: 0.7

    Raises:
        ValueError: If a client is unsupported or input format is incorrect.
    """
    def __init__(self, config_file='config.yaml', custom: bool = False):
        """
        Initializes the azLLMConfigs instance.

        If `custom` is True, it creates a custom configuration file (if not present)
        and loads to `custom_configs/config.yaml` of your current directory.

        Args:
            config_file (str): The YAML file to store/load custom configurations.
            custom (bool): If True, loads and manages custom configurations.
        """



        self.config_file = config_file
        self.custom = custom
        
        if self.custom:
            create_custom_file('custom_configs', self.config_file, template_content)
            self.custom_configs = load_custom_config('custom_configs', self.config_file)
        # else:
        #     raise ValueError("Custom configuration is not enabled or available.")
    
    def get_default_configs(self, client: str = 'all'):
        """
        Retrieves the default configuration(s) for one or all LLM clients.

        Args:
            client (str): Client name (e.g., 'openai') or 'all' for all clients.
        
        Returns:
            dict: Dictionary of default configurations.
        
        Raises:
            ValueError: If the specified client is unsupported.
        """
        self.clients = {
            'openai': OpenAIClient.get_default_config,
            'deepseek': DeepSeekClient.get_default_config,
            'grok': GrokClient.get_default_config,
            'anthropic': AnthropicClient.get_default_config,
            'gemini': GeminiClient.get_default_config,
            'ollama': OllamaClient.get_default_config,
            'fireworks': FireworksClient.get_default_config,
        }
        if client == 'all':
            return {key: self.clients[key]() for key in self.clients}  
        elif client in self.clients:
            return self.clients[client]()  
        else:
            raise ValueError(f"Default configuration for client type '{client}' is not found. Please confirm you are using one of the supporting clients.")

    
    def update_custom_configs(self, client_type: str, models_to_update_or_add: Dict[str, Dict[str, Any]]) -> None:
        """
        Updates or adds custom configurations for models under a specific client.
        
        If the client or model doesn't exist in the configuration, it is added.
        If the model exists, its configuration is updated.

        Args:
            client_type (str): The LLM client identifier (e.g., 'openai').
            models_to_update_or_add (dict): A dictionary where keys are model names and 
                                            values are dictionaries with 'version' and 'parameters'.

        Raises:
            ValueError: If the client type is not supported.
        """
    
        if client_type not in ACCEPTABLE_CLIENTS:
            raise ValueError(f"Client type {client_type} is not working, please contact us to add.")
        
        if client_type not in self.custom_configs:
            self.custom_configs[client_type] = {
                'models': []  
            }
            print(f"Client {client_type} added to custom_configs.")

        client_config = self.custom_configs[client_type]
        models_list = client_config['models']

        for model_name, model_info in models_to_update_or_add.items():
            version = model_info.get('version', 'default') 
            new_parameters = model_info['parameters']
            
            existing_model = next((m for m in models_list if m['model'] == model_name and m['version'] == version), None)
            
            if existing_model:
                existing_model['parameters'] = new_parameters
                print(f"Updated model {model_name} with version {version} for client {client_type}.")
            else:
                models_list.append({
                    'model': model_name,
                    'version': version,
                    'parameters': new_parameters
                })
                print(f"Added new model {model_name} with version {version} for client {client_type}.")
        save_custom_configs(self.config_file, self.custom_configs)


__all__ = ['azLLMConfigs']
