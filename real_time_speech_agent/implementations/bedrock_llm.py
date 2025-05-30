"""
LLM implementation using Amazon Bedrock.
"""

import json
from typing import AsyncIterator, Dict, Iterator, Optional

import boto3

from ..interfaces.llm_interface import LLMInterface


class BedrockLLM(LLMInterface):
    """
    LLM implementation using Amazon Bedrock with streaming support.
    
    This implementation uses Amazon Bedrock's Claude models to process text
    and return streaming responses.
    """
    
    def __init__(
        self,
        model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0",
        region_name: str = "us-east-1",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        max_history: int = 10,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize the Bedrock LLM implementation.
        
        Args:
            model_id: Bedrock model ID to use
            region_name: AWS region for Bedrock
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for text generation
            max_history: Maximum number of messages to keep in conversation history
            system_prompt: Optional system prompt for the model
        """
        self.bedrock_runtime = boto3.client(
            service_name="bedrock-runtime",
            region_name=region_name
        )
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_history = max_history
        self.system_prompt = system_prompt
        
        # Initialize conversation history
        self.conversation_history = []
    
    def process_text(self, input_text: str, context: Optional[Dict] = None) -> Iterator[str]:
        """
        Process input text and return a stream of response text chunks.
        
        Args:
            input_text: The complete input text to process
            context: Optional context information for the LLM
            
        Returns:
            Iterator[str]: Iterator yielding chunks of response text as they are generated
        """
        # Add user message to conversation history
        self.conversation_history.append({"role": "user", "content": input_text})
        
        # Prepare the request body with conversation history
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": self.conversation_history
        }
        
        # Add system prompt if provided
        if self.system_prompt:
            body["system"] = self.system_prompt
        
        # Add any additional context from the context parameter
        if context:
            # Handle context based on the model's requirements
            # For Claude, we might add it to the system prompt or as a separate message
            if "system_prompt_override" in context:
                body["system"] = context["system_prompt_override"]
        
        # Convert the request body to JSON
        body_json = json.dumps(body)
        
        try:
            # Invoke the model with streaming
            response = self.bedrock_runtime.invoke_model_with_response_stream(
                modelId=self.model_id,
                body=body_json
            )
            
            # Process the streaming response
            full_response = ""
            
            # Process each chunk as it arrives
            for event in response.get("body"):
                if "chunk" in event:
                    chunk_data = json.loads(event["chunk"]["bytes"])
                    
                    # Handle content_block_delta events which contain the actual text
                    if chunk_data.get("type") == "content_block_delta" and "delta" in chunk_data:
                        if chunk_data["delta"].get("type") == "text_delta" and "text" in chunk_data["delta"]:
                            text_chunk = chunk_data["delta"]["text"]
                            full_response += text_chunk
                            yield text_chunk
            
            # Add assistant response to conversation history
            self.conversation_history.append({"role": "assistant", "content": full_response})
            
            # Keep conversation history to a reasonable size
            if len(self.conversation_history) > self.max_history:
                self.conversation_history = self.conversation_history[-self.max_history:]
                
        except Exception as e:
            # Fallback to standard API if streaming fails
            try:
                response = self.bedrock_runtime.invoke_model(
                    modelId=self.model_id,
                    body=body_json
                )
                
                # Parse the response
                response_body = json.loads(response.get("body").read())
                full_response = response_body["content"][0]["text"]
                
                # Add assistant response to conversation history
                self.conversation_history.append({"role": "assistant", "content": full_response})
                
                # Keep conversation history to a reasonable size
                if len(self.conversation_history) > self.max_history:
                    self.conversation_history = self.conversation_history[-self.max_history:]
                
                # Yield the full response as a single chunk
                yield full_response
                
            except Exception as nested_e:
                error_message = f"Error processing text with Bedrock: {nested_e}"
                yield error_message
    
    async def process_text_async(self, input_text: str, context: Optional[Dict] = None) -> AsyncIterator[str]:
        """
        Asynchronously process input text and return a stream of response text chunks.
        
        This is a simple wrapper around the synchronous method for now.
        For production use, this should be implemented with async AWS SDK.
        
        Args:
            input_text: The complete input text to process
            context: Optional context information for the LLM
            
        Returns:
            AsyncIterator[str]: AsyncIterator yielding chunks of response text as they are generated
        """
        # This is a simple implementation that doesn't truly leverage async
        # In a production environment, you would use an async AWS SDK
        for chunk in self.process_text(input_text, context):
            yield chunk
    
    def update_context(self, context_updates: Dict) -> None:
        """
        Update the conversation context for the LLM.
        
        Args:
            context_updates: Dictionary containing context updates
        """
        # For Claude, we might update the system prompt or add a new message
        if "system_prompt" in context_updates:
            self.system_prompt = context_updates["system_prompt"]
        
        # Add a new message to the conversation history if provided
        if "message" in context_updates and "role" in context_updates:
            self.conversation_history.append({
                "role": context_updates["role"],
                "content": context_updates["message"]
            })
            
            # Keep conversation history to a reasonable size
            if len(self.conversation_history) > self.max_history:
                self.conversation_history = self.conversation_history[-self.max_history:]
    
    def reset_context(self) -> None:
        """
        Reset the conversation context for the LLM.
        """
        self.conversation_history = []
