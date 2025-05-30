"""
Speech-to-Text implementation using Moonshine ONNX model.
"""

import time
import logging
from typing import Optional

import numpy as np

from ..interfaces.speech_to_text import SpeechToTextInterface

# Configure logging
logger = logging.getLogger(__name__)

# These will be imported from the moonshine-onnx package
# For now, we're assuming these imports will be available when the package is installed
try:
    from moonshine_onnx import MoonshineOnnxModel, load_tokenizer
    from silero_vad import VADIterator, load_silero_vad
except ImportError:
    raise ImportError(
        "moonshine_onnx package is required for MoonshineSTT. "
        "Please install it with: pip install moonshine-onnx"
    )


class MoonshineSTT(SpeechToTextInterface):
    """
    Speech-to-Text implementation using Moonshine ONNX model.
    
    This implementation uses the Moonshine model for transcription and Silero VAD
    for voice activity detection to determine when speech starts and ends.
    """
    
    def __init__(
        self,
        model_name: str = "moonshine/base",
        sampling_rate: int = 16000,
        vad_threshold: float = 0.5,
        min_silence_duration_ms: int = 2000,
        max_speech_duration_sec: int = 15
    ):
        """
        Initialize the Moonshine STT implementation.
        
        Args:
            model_name: Name of the Moonshine model to use ('moonshine/base' or 'moonshine/tiny')
            sampling_rate: Audio sampling rate (must be 16000 Hz for Moonshine)
            vad_threshold: Voice activity detection threshold (0.0-1.0)
            min_silence_duration_ms: Minimum silence duration in ms before considering speech ended
            max_speech_duration_sec: Maximum speech duration in seconds before forced processing
        """
        if sampling_rate != 16000:
            raise ValueError("Moonshine supports sampling rate 16000 Hz only.")
        
        logger.info(f"Initializing MoonshineSTT with model '{model_name}', VAD threshold {vad_threshold}, "
                   f"min silence {min_silence_duration_ms}ms, max speech {max_speech_duration_sec}s")
        
        self.sampling_rate = sampling_rate
        self.max_speech_duration_sec = max_speech_duration_sec
        
        # Initialize Moonshine model and tokenizer
        logger.info(f"Loading Moonshine model '{model_name}' (using ONNX runtime) ...")
        try:
            from moonshine_onnx import MoonshineOnnxModel, load_tokenizer
            self.model = MoonshineOnnxModel(model_name=model_name)
            self.tokenizer = load_tokenizer()
            logger.info("Moonshine model and tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Moonshine model: {e}")
            raise
        
        # Initialize Silero VAD
        logger.info("Initializing Silero VAD...")
        try:
            from silero_vad import VADIterator, load_silero_vad
            self.vad_model = load_silero_vad(onnx=True)
            self.vad_iterator = VADIterator(
                model=self.vad_model,
                sampling_rate=sampling_rate,
                threshold=vad_threshold,
                min_silence_duration_ms=min_silence_duration_ms
            )
            logger.info("Silero VAD initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Silero VAD: {e}")
            raise
        
        # Initialize state variables
        self.audio_buffer = np.array([], dtype=np.float32)
        self.is_speech_active = False
        self.speech_start_time = 0
        
        # Performance metrics
        self.inference_secs = 0
        self.number_inferences = 0
        self.speech_secs = 0
        
        # Warm up the model
        logger.info("Warming up the model...")
        self._warmup()
        logger.info("MoonshineSTT initialization complete")
    
    def _warmup(self):
        """Warm up the model with a zero array to initialize it."""
        self.transcribe(np.zeros(self.sampling_rate, dtype=np.float32))
    
    def transcribe(self, audio_data: np.ndarray) -> str:
        """
        Transcribe audio data to text using Moonshine model.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            str: Transcribed text
        """
        self.number_inferences += 1
        self.speech_secs += len(audio_data) / self.sampling_rate
        start_time = time.time()
        
        tokens = self.model.generate(audio_data[np.newaxis, :].astype(np.float32))
        text = self.tokenizer.decode_batch(tokens)[0]
        
        self.inference_secs += time.time() - start_time
        return text
    
    def _soft_reset_vad(self):
        """Soft reset the VAD iterator without affecting the model state."""
        self.vad_iterator.triggered = False
        self.vad_iterator.temp_end = 0
        self.vad_iterator.current_sample = 0
    
    def process_audio_stream(self, audio_bytes: bytes) -> Optional[str]:
        """
        Process a chunk of audio data and potentially return a transcription result.
        
        Args:
            audio_bytes: A chunk of audio data as bytes
            
        Returns:
            Optional[str]: A transcription result if available, otherwise None
        """
        # Convert bytes to float32 numpy array
        # Assuming the audio_bytes are already in the correct format (16-bit PCM)
        # This conversion might need adjustment based on the actual audio format
        audio_chunk = np.frombuffer(audio_bytes, dtype=np.float32)
        
        # Log audio chunk information
        logger.debug(f"Audio chunk received: {len(audio_chunk)} samples, mean amplitude: {np.abs(audio_chunk).mean():.6f}")
        
        # Add the chunk to our buffer
        self.audio_buffer = np.concatenate((self.audio_buffer, audio_chunk))
        logger.debug(f"Audio buffer size: {len(self.audio_buffer)} samples")
        
        # Check for voice activity
        speech_dict = self.vad_iterator(audio_chunk)
        
        if speech_dict:
            logger.info(f"VAD event: {speech_dict}")
            
            # Speech start detected
            if "start" in speech_dict and not self.is_speech_active:
                self.is_speech_active = True
                self.speech_start_time = time.time()
                logger.info("Speech start detected")
                return None  # No transcription yet, just started listening
            
            # Speech end detected
            if "end" in speech_dict and self.is_speech_active:
                self.is_speech_active = False
                speech_duration = time.time() - self.speech_start_time
                logger.info(f"Speech end detected, duration: {speech_duration:.2f}s")
                
                # Process the complete utterance
                logger.info(f"Transcribing audio buffer of {len(self.audio_buffer)} samples")
                transcription = self.transcribe(self.audio_buffer)
                logger.info(f"Transcription result: '{transcription}'")
                
                # Clear the buffer after processing
                self.audio_buffer = np.array([], dtype=np.float32)
                return transcription
        
        # Check if we've exceeded the maximum speech duration
        elif self.is_speech_active:
            speech_duration = time.time() - self.speech_start_time
            logger.debug(f"Ongoing speech, duration: {speech_duration:.2f}s")
            
            if speech_duration > self.max_speech_duration_sec:
                self.is_speech_active = False
                logger.info(f"Max speech duration ({self.max_speech_duration_sec}s) exceeded, processing what we have")
                
                # Process what we have so far
                transcription = self.transcribe(self.audio_buffer)
                logger.info(f"Transcription result: '{transcription}'")
                
                # Clear the buffer after processing
                self.audio_buffer = np.array([], dtype=np.float32)
                # Reset VAD to prepare for new speech
                self._soft_reset_vad()
                return transcription
        
        # No complete transcription available yet
        return None
    
    def reset(self) -> None:
        """
        Reset the internal state of the transcriber.
        """
        self.audio_buffer = np.array([], dtype=np.float32)
        self.is_speech_active = False
        self._soft_reset_vad()
    
    def finalize(self) -> str:
        """
        Finalize the current transcription and return the complete text.
        
        Returns:
            str: The complete transcribed text
        """
        # Process whatever is in the buffer
        if len(self.audio_buffer) > 0:
            transcription = self.transcribe(self.audio_buffer)
            self.audio_buffer = np.array([], dtype=np.float32)
            self.is_speech_active = False
            self._soft_reset_vad()
            return transcription
        return ""
    
    def get_performance_stats(self) -> dict:
        """
        Get performance statistics for the transcriber.
        
        Returns:
            dict: Dictionary containing performance statistics
        """
        if self.number_inferences == 0:
            return {
                "number_inferences": 0,
                "mean_inference_time": 0,
                "realtime_factor": 0
            }
        
        return {
            "number_inferences": self.number_inferences,
            "mean_inference_time": self.inference_secs / self.number_inferences,
            "realtime_factor": self.speech_secs / self.inference_secs
        }
