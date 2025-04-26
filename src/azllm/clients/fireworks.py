from openai import OpenAI
import os
from typing import List, Dict, Any
from azllm.base import UNIClient
from dotenv import load_dotenv
load_dotenv()


DEFAULT_CONFIG = {'model': 'accounts/fireworks/models/llama4-scout-instruct-basic',
                  'system_message': 'You are an advanced AI assistant.',
                  'temperature': 1,
                  'max_tokens': 4096,
                  'frequency_penalty': 0,
                  'presence_penalty': 0,
                  'kwargs': {}}

class FireworksClient:
    """
    A client for interacting with the Fireworks AI models.

    This client interfaces with Fireworks AI's inference API and allows
    for customizable model interactions, text generation, and batch processing.

    Attributes:
        model (str): The model to be used for text generation.
        parameters (dict): Configuration for how to interact with the model.
        system_message (str): The system message to give context to the AI.
        temperature (float): Controls the randomness in model responses.
        max_tokens (int): Maximum number of tokens to generate in the response.
        frequency_penalty (float): Affects repetition in the response.
        presence_penalty (float): Controls how much the AI sticks to previously mentioned concepts.
        kwargs (dict): Additional settings for model configuration.
    """
    def __init__(self, config: Dict[str, Any]= None):
        """
        Initializes the FireworksClient with the provided configuration.

        Args:
            config (dict, optional): A dictionary containing user-defined settings for the client.
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
        Returns the default configuration for the FireworksClient.

        Returns:
            dict: The default configuration settings.
        """
        return DEFAULT_CONFIG
        
    def get_api_key(self) -> str:
        """
        Loads the API key for Fireworks AI from the environment variable.

        Returns:
            str: The Fireworks AI API key.

        Raises:
            ValueError: If the API key is not found.
        """
        if not self.api_key:
            self.api_key = os.getenv("FIREWORKS_API_KEY")
            if not self.api_key:
                raise ValueError("A valid API Key for Fireworks is missing. Please set it in the .env file")
        return self.api_key
    
    def get_client(self):  
        """
        Initializes the OpenAI-compatible client for Fireworks AI (lazy initialization).

        Returns:
            OpenAI: The client configured for Fireworks API interactions.
        """
        if self.client is None:
            self.client = OpenAI(api_key=self.get_api_key(), base_url= "https://api.fireworks.ai/inference/v1",)
        return self.client
    
    def generate_text(self, prompt: str, kwargs:dict = None, parse: bool = False) -> str:
        """
        Generate text based on a single prompt using the Fireworks AI model.

        Args:
            prompt (str): The input prompt to send to the model.
            kwargs (dict, optional): Additional configurations to override the default parameters.
            parse (bool, optional): If set to True, uses `parse()` method for structured responses.

        Returns:
            str: The generated text from the model.

        Raises:
            RuntimeError: If there’s an error in generating text from the model.
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