"""
Speech-to-Text interface definition for Real-Time Speech Agent.
"""

from abc import ABC, abstractmethod
from typing import Optional


class SpeechToTextInterface(ABC):
    """Interface for speech-to-text services that handle real-time audio streams."""
    
    @abstractmethod
    def process_audio_stream(self, audio_bytes: bytes) -> Optional[str]:
        """
        Process audio data and return transcribed text if available.
        
        Args:
            audio_bytes: Audio data as bytes
            
        Returns:
            Optional[str]: Transcribed text if available, otherwise None
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """
        Reset the internal state of the transcriber.
        
        This should be called when starting a new transcription session.
        """
        pass
    
    @abstractmethod
    def finalize(self) -> str:
        """
        Finalize the current transcription and return the complete text.
        
        This should be called when the audio stream has ended.
        
        Returns:
            str: The complete transcribed text
        """
        pass
