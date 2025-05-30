"""
Web server for real-time speech-to-speech agent.
"""

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import logging
import os
import uvicorn
import asyncio
import numpy as np
import base64, json
import hashlib
import time
from collections import deque

# Import interfaces and implementations from local modules
from .implementations.moonshine_stt import MoonshineSTT
from .implementations.bedrock_llm import BedrockLLM
from .implementations.polly_tts import PollyTextToSpeech

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Real-Time Speech Agent",
    description="Web interface for real-time speech-to-speech conversational AI",
    version="0.1.0",
)

# Set up templates directory
base_dir = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(base_dir, "templates"))

# System prompt for casual, conversational responses
SYSTEM_PROMPT = """
You are a friendly, casual assistant named Bentham. 

Style guidelines:
- Keep responses extremely brief, limited to 2 lines maximum
- Use a conversational, natural tone
- Use contractions (don't, can't, I'll, etc.)
- Use simple, everyday language
- Respond as if you're having a friendly chat
- Never exceed 2 lines in your response

Remember, you're having a casual conversation with very concise responses.
"""

# Initialize Moonshine STT
stt = MoonshineSTT(
    model_name="moonshine/base",
    vad_threshold=0.3,
    min_silence_duration_ms=500  # Shorter silence duration for more responsive transcription
)

# Initialize Bedrock LLM
llm = BedrockLLM(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    region_name="us-east-1",
    system_prompt=SYSTEM_PROMPT
)

# Initialize Polly TTS
tts = PollyTextToSpeech(
    voice_id="Ruth",  # Default voice
    engine="neural",
    language_code="en-US",
    sample_rate="16000",
    region_name="us-east-1"
)

# Buffer to store fingerprints of recently played TTS audio
recent_tts_fingerprints = deque(maxlen=10)  # Store last 10 audio fingerprints

# Add a function to create a simple audio fingerprint
def create_audio_fingerprint(audio_data):
    """Create a simple fingerprint of audio data for comparison."""
    # Use a hash of downsampled audio as a fingerprint
    # This is a simplified approach - production systems would use more sophisticated algorithms
    if len(audio_data) < 1000:
        return hashlib.md5(audio_data).hexdigest()
    
    # Downsample by taking every 10th byte for the fingerprint
    downsampled = audio_data[::10]
    return hashlib.md5(downsampled).hexdigest()

# Add a function to check if incoming audio matches TTS output
def is_tts_audio(audio_data):
    """Check if incoming audio matches our TTS output."""
    if not recent_tts_fingerprints:
        return False
        
    incoming_fingerprint = create_audio_fingerprint(audio_data)
    
    # Check if this fingerprint matches any recent TTS audio
    for tts_fingerprint in recent_tts_fingerprints:
        if incoming_fingerprint == tts_fingerprint:
            return True
    
    return False

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """Return a simple HTML page with audio capture functionality."""
    logger.info("Serving audio capture page")
    return templates.TemplateResponse("audio_capture.html", {"request": request})

@app.websocket("/ws/audio")
async def websocket_audio(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    # Reset STT at the beginning of the session only
    stt.reset()
    logger.info("STT reset for new session")
    
    # Reset LLM conversation history
    llm.reset_context()
    logger.info("LLM conversation history reset for new session")
    
    # Expected sample size for Moonshine (512 samples for 16000Hz)
    expected_samples = 512
    bytes_per_sample = 2  # 16-bit PCM = 2 bytes per sample
    expected_chunk_size = expected_samples * bytes_per_sample
    
    # Speech buffer for VAD-based processing
    speech_buffer = np.array([], dtype=np.float32)
    
    # Keep track of the complete transcript
    complete_transcript = ""
    
    # Variables for silence detection
    last_speech_time = time.time()
    silence_threshold = 0.01  # Threshold for considering audio as silence
    max_silence_duration = 2.0  # Maximum silence duration before considering end of utterance
    
    # Flag to track if we're currently processing with LLM
    processing_with_llm = False
    
    # Flag to track if we're currently speaking (for barge-in)
    is_speaking = False
    
    try:
        # Process audio chunks as they arrive
        while True:
            # Receive message (could be audio data or control message)
            message = await websocket.receive()
            
            # Check if this is a text message (control message)
            if "text" in message:
                try:
                    data = json.loads(message["text"])
                    if data.get("type") == "user_interrupt":
                        logger.info("Received user interrupt signal, stopping TTS")
                        # Set flag to stop TTS processing
                        is_speaking = False
                        continue
                except Exception as e:
                    logger.error(f"Error processing text message: {e}")
                    continue
            
            # If not a text message, assume it's binary audio data
            if "bytes" not in message:
                continue
                
            audio_chunk = message["bytes"]
            logger.info(f"Received audio chunk: {len(audio_chunk)} bytes")
            
            if len(audio_chunk) < 2:  # Skip very small chunks
                continue
                
            try:
                # Convert bytes to numpy array of int16 (16-bit PCM)
                audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
                
                # Check if this might be our own TTS audio
                if is_tts_audio(audio_chunk):
                    logger.info("Detected our own TTS audio, ignoring")
                    continue  # Skip processing this chunk
                
                # Convert to float32 in the range [-1, 1] as Moonshine expects
                audio_float = audio_array.astype(np.float32) / 32768.0
                
                # Check if this is speech or silence
                audio_level = np.abs(audio_float).mean()
                current_time = time.time()
                
                # If we detect speech, update the last speech time
                if audio_level > silence_threshold:
                    last_speech_time = current_time
                
                # Add to speech buffer
                speech_buffer = np.concatenate((speech_buffer, audio_float))
                
                # Keep a reasonable buffer size
                max_buffer_size = 16000 * 10  # 10 seconds at 16kHz (increased from 5 to handle pauses better)
                if len(speech_buffer) > max_buffer_size:
                    speech_buffer = speech_buffer[-max_buffer_size:]
                
                # Process in chunks of 512 samples (what Moonshine expects)
                while len(speech_buffer) >= expected_samples:
                    # Extract a chunk of the expected size
                    chunk_to_process = speech_buffer[:expected_samples]
                    speech_buffer = speech_buffer[expected_samples:]
                    
                    # Process through STT
                    logger.debug(f"Processing audio chunk of {len(chunk_to_process)} samples")
                    text = stt.process_audio_stream(chunk_to_process)
                    
                    # If we have transcription, send it back
                    if text:
                        logger.info(f"Transcribed text: '{text}'")
                        
                        # Add to complete transcript (with space if needed)
                        if complete_transcript and not complete_transcript.endswith(" "):
                            complete_transcript += " "
                        complete_transcript += text
                        
                        await websocket.send_json({
                            "type": "transcription",
                            "text": text,
                            "complete_transcript": complete_transcript
                        })
                
                # If we've been silent for too long, finalize any pending transcription
                if current_time - last_speech_time > max_silence_duration and not processing_with_llm:
                    # Try to finalize any remaining audio
                    final_text = stt.finalize()
                    if final_text:
                        logger.info(f"Finalized text after silence: '{final_text}'")
                        
                        # Add to complete transcript
                        if complete_transcript and not complete_transcript.endswith(" "):
                            complete_transcript += " "
                        complete_transcript += final_text
                        
                        await websocket.send_json({
                            "type": "transcription",
                            "text": final_text,
                            "complete_transcript": complete_transcript
                        })
                    
                    # Process the complete transcript with Bedrock LLM after silence
                    if complete_transcript:
                        logger.info(f"Processing with Bedrock LLM after silence: '{complete_transcript}'")
                        
                        # Set processing flag
                        processing_with_llm = True
                        
                        # Clear previous LLM response on client
                        await websocket.send_json({
                            "type": "clear_llm_response"
                        })
                        
                        # Process with LLM and stream responses
                        try:
                            # Buffer for accumulating text for TTS
                            text_buffer = ""
                            sentence_end_chars = ['.', '!', '?']
                            total_lines = 0
                            max_lines = 2  # Limit to 2 lines
                            
                            async for response_chunk in llm.process_text_async(complete_transcript):
                                logger.info(f"LLM response chunk: '{response_chunk}'")
                                
                                # Check if we've reached the line limit
                                if total_lines >= max_lines and '\n' in response_chunk:
                                    # Only take content up to the second line break
                                    parts = response_chunk.split('\n', 1)
                                    response_chunk = parts[0]
                                    logger.info(f"Truncating response to stay within 2-line limit")
                                    break
                                
                                # Count new lines in this chunk
                                new_lines = response_chunk.count('\n')
                                total_lines += new_lines
                                
                                # Send LLM response chunk to client
                                await websocket.send_json({
                                    "type": "llm_response",
                                    "text": response_chunk
                                })
                                
                                # Add to text buffer for TTS
                                text_buffer += response_chunk
                                
                                # Process complete sentences for TTS
                                while True:
                                    # Find the end of a sentence
                                    sentence_end = -1
                                    for char in sentence_end_chars:
                                        pos = text_buffer.find(char)
                                        if pos != -1 and (sentence_end == -1 or pos < sentence_end):
                                            sentence_end = pos
                                    
                                    # If we found the end of a sentence
                                    if sentence_end != -1:
                                        # Extract the sentence (including the punctuation)
                                        sentence = text_buffer[:sentence_end + 1].strip()
                                        
                                        # Remove the sentence from the buffer
                                        text_buffer = text_buffer[sentence_end + 1:].strip()
                                        
                                        # Only process non-empty sentences
                                        if sentence:
                                            logger.info(f"Processing TTS for sentence: '{sentence}'")
                                            
                                            # Synthesize the complete sentence
                                            try:
                                                # Use the _synthesize_text method directly
                                                audio_data = b''
                                                for audio_chunk in tts._synthesize_text(sentence):
                                                    audio_data += audio_chunk
                                                
                                                # Create and store fingerprint of TTS audio
                                                fingerprint = create_audio_fingerprint(audio_data)
                                                recent_tts_fingerprints.append(fingerprint)
                                                logger.info(f"Stored TTS fingerprint: {fingerprint[:8]}... for sentence: '{sentence}'")
                                                
                                                # Set speaking flag
                                                is_speaking = True
                                                
                                                # Encode audio data as base64 and send to client
                                                encoded_audio = base64.b64encode(audio_data).decode('ascii')
                                                logger.info(f"Sending audio response: {len(audio_data)} bytes, encoded length: {len(encoded_audio)}")
                                                await websocket.send_json({
                                                    "type": "audio_response",
                                                    "data": encoded_audio,
                                                    "format": "pcm"  # Polly returns PCM format by default
                                                })
                                                
                                                # Reset speaking flag after sending
                                                is_speaking = False
                                            except Exception as e:
                                                logger.error(f"Error synthesizing speech: {e}")
                                    else:
                                        # No complete sentence found, break the loop
                                        break
                            
                            # Process any remaining text as a final chunk
                            if text_buffer.strip():
                                logger.info(f"Processing final TTS text: '{text_buffer}'")
                                try:
                                    audio_data = b''
                                    for audio_chunk in tts._synthesize_text(text_buffer):
                                        audio_data += audio_chunk
                                    
                                    # Create and store fingerprint of TTS audio
                                    fingerprint = create_audio_fingerprint(audio_data)
                                    recent_tts_fingerprints.append(fingerprint)
                                    logger.info(f"Stored TTS fingerprint: {fingerprint[:8]}... for final text: '{text_buffer}'")
                                    
                                    # Encode audio data as base64 and send to client
                                    encoded_audio = base64.b64encode(audio_data).decode('ascii')
                                    logger.info(f"Sending final audio response: {len(audio_data)} bytes, encoded length: {len(encoded_audio)}")
                                    await websocket.send_json({
                                        "type": "audio_response",
                                        "data": encoded_audio,
                                        "format": "pcm"  # Polly returns PCM format by default
                                    })
                                except Exception as e:
                                    logger.error(f"Error synthesizing final speech: {e}")
                                    
                        except Exception as e:
                            logger.error(f"Error processing with LLM: {e}")
                            await websocket.send_json({
                                "type": "error",
                                "message": f"Error processing with LLM: {str(e)}"
                            })
                        
                        # Reset processing flag
                        processing_with_llm = False
                    
                    # Don't reset STT here - we want to continue the transcript
                    # Just update the last speech time to avoid repeated finalizations
                    last_speech_time = current_time
                    
                    # Reset complete transcript after processing with LLM
                    complete_transcript = ""
                
            except Exception as e:
                logger.error(f"Error processing audio chunk: {e}")
                # Don't reset buffer on error, try again with next chunk
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error in WebSocket: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass

if __name__ == "__main__":
    logger.info("Starting Real-Time Speech Agent server")
    uvicorn.run(app, host="0.0.0.0", port=8000)
