from openai import OpenAI
import os
from typing import List, Dict, Any
from azllm.base import UNIClient
from dotenv import load_dotenv
load_dotenv()


DEFAULT_CONFIG = {'model': 'tinyllama:1.1b',
                  'system_message': 'You are an advanced AI assistant.',
                  'temperature': 1,
                  'max_tokens': 4096,
                  'frequency_penalty': 0,
                  'presence_penalty': 0,
                  'kwargs': {}}

class OllamaClient:
    """
    A client for using Ollama models via an OpenAI-compatible API wrapper.

    This class wraps a local instance of Ollama (typically hosted at http://localhost:11434)
    using the OpenAI Python SDK for consistency with other OpenAI-style models.

    Attributes:
        model (str): Model identifier to use (e.g., 'tinyllama:1.1b').
        parameters (dict): Dict of generation parameters like temperature, etc.
        system_message (str): The assistant's system message to guide behavior.
        temperature (float): Controls randomness in output.
        max_tokens (int): Maximum number of tokens to generate.
        frequency_penalty (float): Penalizes repeated tokens.
        presence_penalty (float): Encourages discussion of new topics.
        kwargs (dict): Additional parameters passed to the API.
    """

    def __init__(self, config: Dict[str, Any]= None):
        """
        Initializes the OllamaClient with the given or default configuration.

        Args:
            config (dict, optional): Custom configuration dictionary. If not provided,
                                     DEFAULT_CONFIG is used.
        """
        
        config = config or {}
        self.api_key: str = None
        self.client = None

        self.model = config.get('model', DEFAULT_CONFIG['model']) 
        self.parameters = config.get('parameters', {}) 
        self.system_message = self.parameters.get('system_message', DEFAULT_CONFIG['system_message']) 
        self.temperature = self.parameters.get('temperature', DEFAULT_CONFIG['temperature']) 
        self.max_tokens = self.parameters.get('max_tokens', DEFAULT_CONFIG['max_tokens']) 
        self.frequency_penalty = self.parameters.get('frequency_penalty', DEFAULT_CONFIG['frequency_penalty']) 
        self.presence_penalty = self.parameters.get('presence_penalty', DEFAULT_CONFIG['presence_penalty']) 
        self.kwargs = self.parameters.get('kwargs', DEFAULT_CONFIG['kwargs'])
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """
        Get the default configuration for Ollama models.

        Returns:
            dict: A dictionary containing default model settings.
        """
        return DEFAULT_CONFIG
    
    def get_client(self):  
        """
        Lazily initializes and returns the OpenAI-compatible client configured for local Ollama.

        Returns:
            OpenAI: An instance of the OpenAI client pointing to local Ollama API.
        """
        if self.client is None:
            self.client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1",)
        return self.client
    
    def generate_text(self, prompt: str, kwargs:dict = None, parse: bool = False) -> str:
        """
        Generates a single completion response from the Ollama model.

        Args:
            prompt (str): The prompt to send to the model.
            kwargs (dict, optional): Overrides for generation parameters.
            parse (bool, optional): If True, use the beta `.parse()` method (experimental).

        Returns:
            str: The model's generated text.

        Raises:
            RuntimeError: If the API request fails or is malformed.
        """
        client = self.get_client()

        kwargs = kwargs or {}

        # Temporary override (default behavior)
        system_message = kwargs.pop("system_message", self.system_message)

        base_params = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty
        }
    
        base_params.update(self.kwargs)
        base_params.update(kwargs)

        try:
            if parse:
                response = client.beta.chat.completions.parse(**base_params)
                return response.choices[0].message
            else:
                response = client.chat.completions.create(**base_params)
                return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Error generating text: {str(e)}")
    
    
    def batch_generate(self, prompts: List[str], kwargs: List[dict] = None, parse: List[bool] = None) -> List[str]:
        """
        Generate responses for multiple prompts in sequence.

        Args:
            prompts (List[str]): List of input prompts.
            kwargs (List[dict], optional): Optional list of parameter overrides per prompt.
            parse (List[bool], optional): Optional list indicating whether to use `parse` per prompt.

        Returns:
            List[str]: List of generated message contents or error messages.
        
        Raises:
            ValueError: If input list lengths are mismatched.
        """
        responses = []

        kwargs = kwargs if kwargs is not None else [{}] * len(prompts)
        parse = parse if parse is not None else [False] * len(prompts)

        if len(kwargs) != len(prompts):
            raise ValueError("The length of kwargs dictionaries must match the number of prompts.")
        if len(parse) != len(prompts):
            raise ValueError("The length of parse list must match the number of prompts.")
        
        for idx, prompt in enumerate(prompts):
            try:
                response = self.generate_text(prompt, kwargs[idx], parse[idx]) 
                responses.append(response)
            except Exception as e:
                responses.append(f"Error: {str(e)}")
        return responses

__all__ = []