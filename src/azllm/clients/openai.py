from openai import OpenAI
import os
import threading
from typing import List, Dict, Any
from azllm.base import UNIClient
from dotenv import load_dotenv
load_dotenv()


DEFAULT_CONFIG = {'model': 'gpt-4o-mini',
                  'system_message': 'You are an advanced AI assistant.',
                  'temperature': 1,
                  'max_tokens': 4096,
                  'frequency_penalty': 0,
                  'presence_penalty': 0,
                  'kwargs': {}}

class OpenAIClient:
    """
    A wrapper for the OpenAI client that supports configuration-based initialization and unified text generation.

    Attributes:
        model (str): Model name to use (e.g., 'gpt-4o-mini').
        parameters (dict): Parameter overrides for temperature, max_tokens, etc.
        system_message (str): System prompt to set the behavior of the assistant.
        temperature (float): Sampling temperature for response generation.
        max_tokens (int): Maximum number of tokens to generate.
        frequency_penalty (float): Penalty to reduce frequency of repeated tokens.
        presence_penalty (float): Penalty to encourage new topic generation.
        kwargs (dict): Additional keyword arguments passed to the OpenAI API.
    """
    def __init__(self, config: Dict[str, Any]= None):
        """
        Initialize the OpenAI client with optional custom configuration.

        Args:
            config (dict, optional): Custom configuration dictionary.
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

        self._lock = threading.Lock()  # Used for thread-safe updates
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """
        Get the default model configuration.

        Returns:
            dict: Default configuration dictionary.
        """
        return DEFAULT_CONFIG
        
    def get_api_key(self) -> str:
        """
        Load the OpenAI API key from environment variables (.env file).

        Returns:
            str: API key string.

        Raises:
            ValueError: If the API key is not found.
        """
        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("A valid API Key for OpenAI is missing. Please set it in the .env file")
        return self.api_key
    
    def get_client(self):  
        """
        Lazily initialize and return the OpenAI client.

        Returns:
            OpenAI: Initialized OpenAI client instance.
        """
        if self.client is None:
            self.client = OpenAI(api_key=self.get_api_key())
        return self.client
    
    def generate_text(self, prompt: str, kwargs:dict = None, parse: bool = False) -> str:
        """
        Generate a single text response using the configured OpenAI model.

        Args:
            prompt (str): The input prompt for the model.
            kwargs (dict, optional): Additional parameters to override generation behavior.
            parse (bool, optional): Use the beta `parse` endpoint if True.

        Returns:
            str: The generated message content.

        Raises:
            RuntimeError: If an API or network error occurs during generation.
        """
        client = self.get_client()

        kwargs = kwargs or {}

        # Temporary override (default behavior)
        system_message = kwargs.pop("system_message", self.system_message)
        
        base_params = {
            "model": self.model,
            "messages": [
                # {"role": "system", "content": self.system_message}, 
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