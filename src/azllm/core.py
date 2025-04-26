from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from .utils import load_custom_config
from .base import UNIClient


from .clients.openai import OpenAIClient
from .clients.deepseek import DeepSeekClient
from .clients.grok import GrokClient
from .clients.anthropic import AnthropicClient
from .clients.gemini import GeminiClient
from .clients.ollama import OllamaClient
from .clients.fireworks import FireworksClient



class azLLM:
    """
    Main interface to interact with multiple LLM clients via unified configuration and execution methods.
    
    Attributes:
        config_file (str): Path to the configuration file.
        custom (bool): Whether to use custom configuration.
        config (dict): Loaded configuration from file if custom is True.
        clients (dict): Mapping of client names to their respective classes.
    """
    def __init__(self, config_file ='config.yaml', custom: str = False):
        """
        Initializes the azLLM instance and loads configurations.

        Args:
            config_file (str): Path to the configuration file.
            custom (bool): Whether to use custom configuration.
        """
        self.config_file = config_file
        self.custom = custom

        if self.custom:
            self.config = load_custom_config('custom_configs', self.config_file)
        else:
            self.config = None

        self.clients = {
            'openai': OpenAIClient,
            'deepseek': DeepSeekClient,
            'grok': GrokClient,
            'anthropic': AnthropicClient,
            'gemini': GeminiClient,
            'ollama': OllamaClient,
            'fireworks': FireworksClient,
        }

    def get_client(self, client_name: str, model_config: Dict[str, Any] = None) -> UNIClient: 

        """
        Returns an instance of the specified client initialized with optional model configuration.

        Args:
            client_name (str): Name of the LLM client.
            model_config (dict, optional): Configuration for the specific model.

        Returns:
            UNIClient: Initialized client instance.

        Raises:
            ValueError: If the client is not supported or configuration is invalid.

        Example:
            >>> azllm = azLLM()
            >>> client = azllm.get_client('openai')
            >>> isinstance(client, OpenAIClient)
            True
        """ 

        if client_name not in self.clients:
            raise ValueError(f"Client {client_name} not found.")

        if model_config is not None:
            try:
                return self.clients[client_name](model_config)
            except Exception as e:
                raise ValueError(f"Invalid model configuration: {e}")
        return self.clients[client_name]()
    
    def split_client_model_version(self, cmv: str):
        """
        Splits a client model version string into individual components.

        This method takes a formatted string representing a client, model, and optionally a version, 
        and parses it into three separate components: client, model, and version. 
        If the string format is invalid or incomplete, a ValueError is raised.

        Args:
            cmv (str): The client-model-version string. The format can be:
                - 'client:model::version'
                - 'client:model' (in which case 'default' will be used for version)
                - 'client' (in which case 'default' will be used for both model and version)

        Returns:
            tuple: A tuple containing three elements:
                - client (str): The client name.
                - model (str): The model name (or 'default' if not specified).
                - version (str): The version name (or 'default' if not specified).

        Raises:
            ValueError: If the input string does not match the expected format, 
                        or if it is empty.

        Example:
            >>> azllm = azLLM()
            >>> azllm.split_client_model_version("openai:gpt-3::v1")
            ('openai', 'gpt-3', 'v1')

            >>> azllm.split_client_model_version("openai:gpt-3")
            ('openai', 'gpt-3', 'default')

            >>> azllm.split_client_model_version("openai")
            ('openai', 'default', 'default')
        """
        if not cmv:
            raise ValueError("Empty model identifier.")
        try:
            if "::" in cmv:
                client, rest = cmv.split(":", 1)
                model, version = rest.rsplit("::", 1)
            elif ":" in cmv:
                client, model = cmv.split(":", 1)
                version = "default"
            else:
                client, model, version = cmv, "default", "default"
            return client, model, version
        except Exception:
            raise ValueError(f"Invalid format: '{cmv}'. Expected 'client:model::version' or similar.")


    def get_model_config(self, client_name:str, model_name: str, version:str = 'default') -> Dict[str, Any]:
        """
        Retrieves the configuration of a specific model from custom configurations.

        Args:
            client_name (str): Name of the client.
            model_name (str): Name of the model.
            version (str): Model version.

        Returns:
            dict: Configuration of the model.

        Raises:
            ValueError: If client or model config is not found.

        Example:
            >>> azllm = azLLM(custom=True)
            >>> model_config = azllm.get_model_config('openai', 'gpt-3', 'v1')
            >>> isinstance(model_config, dict)
            True
        """
        client_configs = self.config.get(client_name)
        if not client_configs:
            raise ValueError(f"Client configs for '{client_name}' not found.")
        models_configs = client_configs.get('models', {})
        model_config = None 
        for configs_model_i in models_configs:
            if configs_model_i['model'] == model_name and configs_model_i['version'] == version:
                model_config = configs_model_i
                break
            elif model_name == 'default' and version == 'default' and configs_model_i['version'] == 'default':
                model_config = configs_model_i
                break
        if model_config is None:
            raise ValueError(f"Model configuration for '{client_name}', '{model_name}', '{version}' not found.")

        return model_config
    
    def generate_text(self, client_model_version: str, prompt: str, kwargs: dict = None, parse: bool = False) -> str:
        """
        Generates text using a specific client and model for a given prompt.

        Args:
            client_model_version (str): Format 'client:model::version'.
            prompt (str): Text prompt.
            kwargs (dict, optional): Additional generation parameters.
            parse (bool): Whether to parse the output.

        Returns:
            str: Generated text.

        Example:
            >>> azllm = azLLM()
            >>> result = azllm.generate_text("openai:gpt-4o-mini::v1", "Hello, how are you?")
            >>> isinstance(result, str)
            True
        """
        kwargs = kwargs or {} 

        try:
            if self.custom and self.config:
                client_name, model, version = self.split_client_model_version(client_model_version)
                model_config = self.get_model_config(client_name, model, version) 
                client = self.get_client(client_name, model_config)
            else:
                client_name, _, _ = self.split_client_model_version(client_model_version)
                client = self.get_client(client_name)

            return client.generate_text(prompt, kwargs, parse)
        
        except ValueError as e:
            raise ValueError(f"Error in generating text for client model version '{client_model_version}': {str(e)}") from e
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred while generating text: {str(e)}") from e

    def batch_generate(self, client_model_version: str, prompts: List[str], kwargs: List[dict] = None, parse: List[bool] = None) -> List[str]: 
        """
        Generates text for multiple prompts using the specified client and model.

        Args:
            client_model_version (str): Format 'client:model::version'.
            prompts (List[str]): List of prompts.
            kwargs (List[dict], optional): Parameters per prompt.
            parse (List[bool], optional): Parse flag per prompt.

        Returns:
            List[str]: List of generated texts.
        
        Example:
            >>> azllm = azLLM()
            >>> results = azllm.batch_generate("openai:gpt-4o-mini::v1", ["How are you?", "What's the weather?"])
            >>> isinstance(results, list)
            True
            
        """
        try:
            if self.custom and self.config:
                client_name, model, version = self.split_client_model_version(client_model_version)
                model_config = self.get_model_config(client_name, model, version) 
                client = self.get_client(client_name, model_config)
            else:
                client_name, _, _ = self.split_client_model_version(client_model_version)
                client = self.get_client(client_name)

            return client.batch_generate(prompts, kwargs, parse)
        
        except ValueError as e:
            raise ValueError(f"Error in generating text for client model version '{client_model_version}': {str(e)}") from e
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred while generating text: {str(e)}") from e


    def generate_parallel(self, prompt: str, clients_models_versions: list, kwargs: List[dict] = None, parse: List[bool] = None) -> dict:
        """
        Generate text in parallel using different clients and models for the same prompt.

        Args:
            prompt (str): Input prompt.
            clients_models_versions (List[str]): List of 'client:model::version' strings.
            kwargs (List[dict], optional): Additional parameters per client.
            parse (List[bool], optional): Parse flag per client.

        Returns:
            dict: Mapping of 'client:model::version:index' to generated text or error message.
        
        Example:
            >>> azllm = azLLM()
            >>> results = azllm.generate_parallel("Hello!", ["openai:gpt-4o-mini::v1", "grok:default::default"])
            >>> isinstance(results, dict)
            True
        """
        
        kwargs = kwargs if kwargs is not None else [{}] * len(clients_models_versions)
        parse = parse if parse is not None else [False] * len(clients_models_versions)
        
        if len(kwargs) != len(clients_models_versions):
            raise ValueError("The length of kwargs must match the length of clients_models_versions.")
        if len(parse) != len(clients_models_versions):
            raise ValueError("The length of parse must match the length of clients_models_versions.")
        
        results = {}

        with ThreadPoolExecutor(max_workers=len(clients_models_versions)) as executor:
            futures = {
                executor.submit(self.generate_text, client_model_version, prompt, kwargs[idx] if kwargs[idx] else {}, parse[idx]): idx
                for idx, client_model_version in enumerate(clients_models_versions)
            }
        
            for future in as_completed(futures):
                idx = futures[future]
                client_model_version = clients_models_versions[idx]  
                
                try:
                    result = future.result()
                    results[f"{client_model_version}:{idx}"] = result
                    
                except Exception as e:
                    results[f"{client_model_version}:{idx}"] = f"Error: {str(e)}"
        return results


__all__ = ['azLLM']