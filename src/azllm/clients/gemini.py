from openai import OpenAI
import os
from typing import List, Dict, Any
from azllm.base import UNIClient
from dotenv import load_dotenv
load_dotenv()


DEFAULT_CONFIG = {'model': 'gemini-2.0-flash',
                  'system_message': 'You are an advanced AI assistant.',
                  'temperature': 1,
                  'max_tokens': 4096,
                  'frequency_penalty': 0,
                  'presence_penalty': 0,
                  'kwargs': {}}

class GeminiClient:
    """
    A client for interacting with Google's Gemini models via an OpenAI-compatible SDK.

    This client wraps API calls to Gemini, providing lazy initialization,
    parameter customization, and batch request support.

    Attributes:
        model (str): Name of the Gemini model.
        parameters (dict): Configuration for generation behavior.
        system_message (str): System message to contextualize responses.
        temperature (float): Controls response creativity.
        max_tokens (int): Max number of tokens for the response.
        frequency_penalty (float): Currently unused in Gemini's setup.
        presence_penalty (float): Currently unused.
        kwargs (dict): Additional keyword arguments for request customization.
    """
    def __init__(self, config: Dict[str, Any]= None):
        """
        Initialize the Gemini client with the provided or default configuration.

        Args:
            config (dict, optional): User-defined configuration for the client.
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
        Return the default Gemini configuration.

        Returns:
            dict: Default settings for Gemini model interaction.
        """
        return DEFAULT_CONFIG
        
    def get_api_key(self) -> str:
        """
        Load the Gemini API key from the .env file.

        Returns:
            str: The Gemini API key.

        Raises:
            ValueError: If the API key is not set.
        """
        if not self.api_key:
            self.api_key = os.getenv("GEMINI_API_KEY")
            if not self.api_key:
                raise ValueError("A valid API Key for Gemini is missing. Please set it in the .env file")
        return self.api_key
    
    def get_client(self):  
        """
        Lazily initialize and return the OpenAI-compatible client for Gemini.

        Returns:
            OpenAI: Configured OpenAI client for Gemini API.
        """
        if self.client is None:
            self.client = OpenAI(api_key=self.get_api_key(), base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
        return self.client
    
    def generate_text(self, prompt: str, kwargs:dict = None, parse: bool = False) -> str:
        """
        Generate a single response from the Gemini model.

        Args:
            prompt (str): The input text prompt.
            kwargs (dict, optional): Overrides for generation parameters.
            parse (bool, optional): Use `.parse()` if supported (default: False).

        Returns:
            str: The generated text from the model.

        Raises:
            RuntimeError: If the generation fails or API throws an error.
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