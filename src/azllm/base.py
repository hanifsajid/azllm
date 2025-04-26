from abc import ABC, abstractmethod
from typing import List

class UNIClient(ABC): 
    @abstractmethod
    def generate_text(self, client_model: str, prompt: str) -> str:
        """Generate text based on a single prompt"""
        pass

    @abstractmethod
    def batch_generate(self, client_model: str, prompts: List[str]) -> List[str]:
        """Generate text for a list of mutliptle prompts"""
        pass

__all__ = []
