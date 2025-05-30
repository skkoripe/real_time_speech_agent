"""
LLM interface definition for Real-Time Speech Agent.
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Dict, Iterator, Optional


class LLMInterface(ABC):
    """Interface for Large Language Model services with streaming support."""
    
    @abstractmethod
    def process_text(self, input_text: str, context: Optional[Dict] = None) -> Iterator[str]:
        """
        Process input text and return a stream of response text chunks.
        
        This method takes a complete input text (e.g., from speech-to-text)
        and returns the LLM's response as a stream of text chunks as they
        are generated, enabling real-time text-to-speech conversion.
        
        Args:
            input_text: The complete input text to process
            context: Optional context information for the LLM
            
        Returns:
            Iterator[str]: Iterator yielding chunks of response text as they are generated
        """
        pass
    
    @abstractmethod
    async def process_text_async(self, input_text: str, context: Optional[Dict] = None) -> AsyncIterator[str]:
        """
        Asynchronously process input text and return a stream of response text chunks.
        
        Async version of process_text for use in async contexts.
        
        Args:
            input_text: The complete input text to process
            context: Optional context information for the LLM
            
        Returns:
            AsyncIterator[str]: AsyncIterator yielding chunks of response text as they are generated
        """
        pass
    
    @abstractmethod
    def update_context(self, context_updates: Dict) -> None:
        """
        Update the conversation context for the LLM.
        
        Args:
            context_updates: Dictionary containing context updates
        """
        pass
    
    @abstractmethod
    def reset_context(self) -> None:
        """
        Reset the conversation context for the LLM.
        
        This should be called when starting a new conversation.
        """
        pass
