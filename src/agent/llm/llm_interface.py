import os
from dotenv import load_dotenv
from typing import List, Dict, AsyncGenerator
from langchain.schema import HumanMessage, AIMessage
from langchain_aws.chat_models import ChatBedrockConverse
from langchain_anthropic.chat_models import ChatAnthropic
from langchain_openai.chat_models import ChatOpenAI
from openinference.instrumentation.langchain import LangChainInstrumentor

class LLMChat:
    def __init__(self, model_type: str, model_id: str, tracer_provider=None, callback_handler=None):
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
        
        # Common parameters
        common_params = {
            "temperature": float(os.environ.get("TEMPERATURE", 0.3)),
            "max_tokens": int(os.environ.get("MAX_TOKENS", 4096))
        }
        
        if model_type == "bedrock":
            self.model = ChatBedrockConverse(
                model=model_id,
                **common_params
            )
        elif model_type == "anthropic":
            self.model = ChatAnthropic(
                model=model_id,
                streaming=True,
                callbacks=[callback_handler] if callback_handler else None,
                **common_params
            )
        elif model_type == "openai":
            self.model = ChatOpenAI(
                model=model_id,
                streaming=True,
                callbacks=[callback_handler] if callback_handler else None,
                **common_params
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    def chat(self, prompt: str, message_history: List[Dict] = None) -> Dict:
        """
        Regular chat method for non-streaming responses
        """
        if message_history is None:
            message_history = []

        # Convert message history to LangChain format
        messages = []
        for msg in message_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        # Add current prompt
        messages.append(HumanMessage(content=prompt))

        try:
            # Get response from the model
            response = self.model.invoke(messages)
            assistant_message = response.content

            # Handle Claude 3 format if needed
            if isinstance(assistant_message, list):
                assistant_message = "".join(
                    item.get('text', '') 
                    for item in assistant_message 
                    if item.get('type') == 'text'
                )

            # Update message history
            updated_history = message_history + [
                {
                    "role": "user",
                    "content": prompt
                },
                {
                    "role": "assistant",
                    "content": assistant_message
                }
            ]

            return {
                'response': assistant_message,
                'message_history': updated_history
            }

        except Exception as e:
            print(f"Error in chat: {str(e)}")
            raise e

    async def stream_chat(self, prompt: str, message_history: List[Dict] = None) -> AsyncGenerator:
        """
        Streams the chat response token by token using the model's native streaming capability
        """
        if message_history is None:
            message_history = []

        # Convert message history to LangChain format
        messages = []
        for msg in message_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        # Add current prompt
        messages.append(HumanMessage(content=prompt))

        try:
            complete_response = ""
            
            # Use the model's native streaming capability
            async for chunk in self.model.astream(messages):
                # Extract token from the chunk based on the response format
                if isinstance(chunk.content, list):
                    # Claude 3 format
                    for content_item in chunk.content:
                        if content_item.get('type') == 'text':
                            token = content_item.get('text', '')
                            if token:
                                complete_response += token
                                yield token
                else:
                    # Standard format
                    token = chunk.content
                    if token:
                        complete_response += token
                        yield token
            
            # Return final message with history
            updated_history = message_history + [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": complete_response}
            ]
            
            yield {
                'response': complete_response,
                'message_history': updated_history
            }

        except Exception as e:
            print(f"Error in stream_chat: {str(e)}")
            raise e

# Test the LLMChat class
if __name__ == "__main__":
    import asyncio
    
    # Regular chat tests
    try:
        print("Start LLM Chat Test")
        
        # Create chat instance
        chat = LLMChat(model_type="bedrock", model_id="anthropic.claude-3-sonnet-20240229-v1:0")
        
        # Test prompt
        test_prompt = "Hello, this is a test message. My name is yunwoong."
        print(f"\nFirst Prompt: {test_prompt}")
        
        # Response test
        response = chat.chat(test_prompt)
        print(f"First Response: {response['response']}")
        
        # Second prompt test (remembering the previous message)
        second_prompt = "What did I say my name was?"
        print(f"\nSecond Prompt: {second_prompt}")

        response2 = chat.chat(second_prompt, response['message_history'])
        print(f"Second Response: {response2['response']}")
        print(f"\nBasic Chat Test Passed ✅")
        
        # Streaming test
        async def test_streaming():
            print("\nStart Streaming Test")
            
            stream_prompt = "Tell me a short story about a robot"
            print(f"\nStreaming Prompt: {stream_prompt}")
            
            print("\nStreaming response:")
            async for token in chat.stream_chat(stream_prompt):
                if isinstance(token, str):
                    print(token, end='', flush=True)
                else:
                    print("\n\nFinal message history received ✅")
            
            print(f"\nStreaming Test Passed ✅")
        
        # Run streaming test
        asyncio.run(test_streaming())
        
        print("\nAll Tests Passed Successfully! ✅")
        
    except Exception as e:
        print(f"\nTest Failed ❌: {str(e)}")