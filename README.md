# Real-Time Speech Agent

A Python package for real-time speech-to-speech conversational AI agents.

## Overview

Real-Time Speech Agent provides a web-based interface for interacting with an AI assistant using voice. It handles:

- Real-time audio streaming from the browser microphone
- Speech-to-text transcription using Moonshine STT
- Processing text with a large language model (Amazon Bedrock)
- Text-to-speech synthesis using Amazon Polly

This package provides a ready-to-use web application for voice-based legal assistant interactions.

## Features

- Real-time streaming speech-to-text conversion
- Streaming LLM processing for natural language understanding
- Real-time streaming text-to-speech synthesis
- Web interface for browser-based voice interactions
- Voice activity detection for determining when speech starts and ends
- Configurable system prompt for the AI assistant

## Installation

```bash
pip install real-time-speech-agent
```

## Usage

### Starting the Web Server

```bash
python -m real_time_speech_agent.server
```

Then open your browser to http://localhost:8000 to access the voice interface.

### Configuration

You can customize the behavior of the speech agent by modifying the following parameters in the server.py file:

- `SYSTEM_PROMPT`: The system prompt for the AI assistant
- Voice settings for Amazon Polly
- Model settings for Amazon Bedrock
- Speech-to-text settings for Moonshine STT

## Requirements

- Python 3.8+
- AWS credentials with access to Amazon Bedrock and Amazon Polly
- Web browser with microphone access

## Dependencies

- fastapi: Web server framework
- uvicorn: ASGI server
- jinja2: HTML templating
- numpy: Numerical processing
- boto3: AWS SDK for Python
- moonshine-onnx: Speech-to-text model
- silero-vad: Voice activity detection

## License

[License information will be added]

## Contact

[Contact information will be added]
