from setuptools import setup, find_packages

setup(
    name="real_time_speech_agent",
    version="0.1.0",
    description="Real-time speech-to-speech agent for conversational AI",
    author="Bentham",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "fastapi>=0.95.0",
        "uvicorn>=0.21.0",
        "jinja2>=3.1.2",
        "numpy>=1.24.0",
        "boto3>=1.26.0",
        "moonshine-onnx>=0.1.0",
        "silero-vad>=0.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
        ],
    },
    python_requires=">=3.8",
)
