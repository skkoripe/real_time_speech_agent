"""
Text-to-Speech interface definition for Real-Time Speech Agent.
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Iterator


class TextToSpeechInterface(ABC):
    """Interface for text-to-speech services with streaming support."""
    
    @abstractmethod
    def process_text_stream(self, text_stream: Iterator[str]) -> Iterator[bytes]:
        """
        Convert a stream of text chunks to a stream of audio chunks.
        
        This method takes text chunks as they arrive from an LLM in real-time
        and converts them to audio chunks that can be played immediately,
        minimizing latency in the interaction.
        
        Args:
            text_stream: Iterator yielding chunks of text as they arrive
            
        Returns:
            Iterator[bytes]: Iterator yielding chunks of audio data
        """
        pass
    
    @abstractmethod
    async def process_text_stream_async(self, text_stream: AsyncIterator[str]) -> AsyncIterator[bytes]:
        """
        Asynchronously convert a stream of text chunks to a stream of audio chunks.
        
        Async version of process_text_stream for use in async contexts.
        
        Args:
            text_stream: AsyncIterator yielding chunks of text as they arrive
            
        Returns:
            AsyncIterator[bytes]: AsyncIterator yielding chunks of audio data
        """
        pass
    
    @abstractmethod
    def set_voice(self, voice_id: str) -> None:
        """
        Set the voice to use for speech synthesis.
        
        Args:
            voice_id: Identifier for the voice to use
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """
        Reset the internal state of the synthesizer.
        
        This should be called when starting a new synthesis session.
        """
        pass
