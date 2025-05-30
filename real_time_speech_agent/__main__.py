"""
Main entry point for the real-time speech agent.
"""

import uvicorn
from .server import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
