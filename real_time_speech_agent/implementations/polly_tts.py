"""
AWS Polly-based Text-to-Speech implementation for Real-Time Speech Agent.
"""

import asyncio
import logging
import queue
import threading
from typing import AsyncIterator, Dict, Iterator, List, Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from ..interfaces.text_to_speech import TextToSpeechInterface


class PollyTextToSpeech(TextToSpeechInterface):
    """
    AWS Polly-based implementation of the TextToSpeechInterface.
    
    This implementation uses Amazon Polly's neural voices for high-quality
    text-to-speech synthesis with streaming support.
    """
    
    def __init__(
        self,
        voice_id: str = "Ruth",
        engine: str = "neural",
        language_code: str = "en-US",
        sample_rate: str = "16000",
        region_name: Optional[str] = None,
        profile_name: Optional[str] = None,
        buffer_size: int = 1024,
        streaming: bool = True,
    ):
        """
        Initialize the Polly TTS component.
        
        Args:
            voice_id: Polly voice ID to use (default: "Ruth")
            engine: Polly engine to use ("neural" or "standard", default: "neural")
            language_code: Language code for the voice (default: "en-US")
            sample_rate: Audio sample rate in Hz (default: "16000")
            region_name: AWS region name (default: None, uses boto3 default)
            profile_name: AWS profile name (default: None, uses boto3 default)
            buffer_size: Size of the audio buffer in bytes (default: 1024)
            streaming: Whether to use streaming synthesis (default: True)
        """
        self.voice_id = voice_id
        self.engine = engine
        self.language_code = language_code
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.streaming = streaming
        
        # Initialize Polly client
        session = boto3.Session(profile_name=profile_name, region_name=region_name)
        self.polly_client = session.client("polly")
        
        # Validate voice ID
        self._validate_voice()
        
        # Initialize state
        self.reset()
        
    def _validate_voice(self) -> None:
        """
        Validate that the selected voice is available.
        
        Raises:
            ValueError: If the voice is not available
        """
        try:
            response = self.polly_client.describe_voices(
                Engine=self.engine,
                LanguageCode=self.language_code
            )
            
            available_voices = [voice["Id"] for voice in response["Voices"]]
            if self.voice_id not in available_voices:
                logging.warning(
                    f"Voice '{self.voice_id}' not found for engine '{self.engine}' "
                    f"and language '{self.language_code}'. Available voices: {available_voices}"
                )
                
                # If the requested voice isn't available, use the first available voice
                if available_voices:
                    self.voice_id = available_voices[0]
                    logging.info(f"Using voice '{self.voice_id}' instead")
                else:
                    raise ValueError(f"No voices available for engine '{self.engine}' and language '{self.language_code}'")
                    
        except (BotoCoreError, ClientError) as error:
            logging.error(f"Error validating voice: {error}")
            # Continue anyway, will fail later if voice is invalid
    
    def reset(self) -> None:
        """
        Reset the internal state of the synthesizer.
        """
        self.text_buffer = ""
        self.audio_queue = queue.Queue()
        self.is_synthesizing = False
        self.synthesis_thread = None
        self.stop_event = threading.Event()
    
    def set_voice(self, voice_id: str) -> None:
        """
        Set the voice to use for speech synthesis.
        
        Args:
            voice_id: Polly voice ID to use
        """
        self.voice_id = voice_id
        self._validate_voice()
    
    def _synthesize_text(self, text: str) -> Iterator[bytes]:
        """
        Synthesize text to speech using Polly.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Iterator[bytes]: Iterator yielding audio chunks
        """
        if not text.strip():
            return
            
        try:
            if self.streaming:
                # Use streaming synthesis
                response = self.polly_client.synthesize_speech(
                    Engine=self.engine,
                    LanguageCode=self.language_code,
                    OutputFormat="pcm",
                    SampleRate=self.sample_rate,
                    Text=text,
                    TextType="text",
                    VoiceId=self.voice_id
                )
                
                # Get the streaming body
                stream = response.get("AudioStream")
                
                # Read and yield chunks
                chunk = stream.read(self.buffer_size)
                while chunk:
                    yield chunk
                    chunk = stream.read(self.buffer_size)
            else:
                # Use standard synthesis (get all audio at once)
                response = self.polly_client.synthesize_speech(
                    Engine=self.engine,
                    LanguageCode=self.language_code,
                    OutputFormat="pcm",
                    SampleRate=self.sample_rate,
                    Text=text,
                    TextType="text",
                    VoiceId=self.voice_id
                )
                
                # Get the audio data
                audio_data = response.get("AudioStream").read()
                
                # Yield in chunks
                for i in range(0, len(audio_data), self.buffer_size):
                    yield audio_data[i:i+self.buffer_size]
                
        except (BotoCoreError, ClientError) as error:
            logging.error(f"Error synthesizing speech: {error}")
            raise
    
    def process_text_stream(self, text_stream: Iterator[str]) -> Iterator[bytes]:
        """
        Convert a stream of text chunks to a stream of audio chunks.
        
        This method implements a streaming approach where text is accumulated
        until a natural break point (punctuation) is reached, then that segment
        is synthesized while continuing to accumulate the next segment.
        
        Args:
            text_stream: Iterator yielding chunks of text as they arrive
            
        Returns:
            Iterator[bytes]: Iterator yielding chunks of audio data
        """
        # Reset state
        self.reset()
        
        # Punctuation that indicates a good break point for synthesis
        break_chars = ['.', '!', '?', ',', ';', ':', '\n']
        
        # Buffer for accumulating text
        text_buffer = ""
        
        # Process text chunks as they arrive
        for text_chunk in text_stream:
            if not text_chunk:
                continue
                
            # Add chunk to buffer
            text_buffer += text_chunk
            
            # Check if we have enough text to synthesize
            if any(char in text_buffer for char in break_chars) and len(text_buffer) > 5:
                # Find the last break character
                last_break_idx = max(
                    text_buffer.rfind(char) for char in break_chars
                    if text_buffer.rfind(char) != -1
                )
                
                if last_break_idx == -1:
                    # No break character found, continue accumulating
                    continue
                
                # Split at the break character
                text_to_synthesize = text_buffer[:last_break_idx + 1]
                text_buffer = text_buffer[last_break_idx + 1:]
                
                # Synthesize the text segment
                for audio_chunk in self._synthesize_text(text_to_synthesize):
                    yield audio_chunk
        
        # Synthesize any remaining text
        if text_buffer:
            for audio_chunk in self._synthesize_text(text_buffer):
                yield audio_chunk
    
    async def process_text_stream_async(self, text_stream: AsyncIterator[str]) -> AsyncIterator[bytes]:
        """
        Asynchronously convert a stream of text chunks to a stream of audio chunks.
        
        Args:
            text_stream: AsyncIterator yielding chunks of text as they arrive
            
        Returns:
            AsyncIterator[bytes]: AsyncIterator yielding chunks of audio data
        """
        # Reset state
        self.reset()
        
        # Punctuation that indicates a good break point for synthesis
        break_chars = ['.', '!', '?', ',', ';', ':', '\n']
        
        # Buffer for accumulating text
        text_buffer = ""
        
        # Process text chunks as they arrive
        async for text_chunk in text_stream:
            if not text_chunk:
                continue
                
            # Add chunk to buffer
            text_buffer += text_chunk
            
            # Check if we have enough text to synthesize
            if any(char in text_buffer for char in break_chars) and len(text_buffer) > 5:
                # Find the last break character
                last_break_idx = max(
                    text_buffer.rfind(char) for char in break_chars
                    if text_buffer.rfind(char) != -1
                )
                
                if last_break_idx == -1:
                    # No break character found, continue accumulating
                    continue
                
                # Split at the break character
                text_to_synthesize = text_buffer[:last_break_idx + 1]
                text_buffer = text_buffer[last_break_idx + 1:]
                
                # Synthesize the text segment in a separate thread to avoid blocking
                loop = asyncio.get_event_loop()
                audio_chunks = await loop.run_in_executor(
                    None, 
                    lambda: list(self._synthesize_text(text_to_synthesize))
                )
                
                # Yield audio chunks
                for audio_chunk in audio_chunks:
                    yield audio_chunk
        
        # Synthesize any remaining text
        if text_buffer:
            loop = asyncio.get_event_loop()
            audio_chunks = await loop.run_in_executor(
                None, 
                lambda: list(self._synthesize_text(text_buffer))
            )
            
            for audio_chunk in audio_chunks:
                yield audio_chunk
