import os
from dotenv import load_dotenv

load_dotenv()

from langchain_aws.chat_models import ChatBedrockConverse, ChatBedrock
from openinference.instrumentation.langchain import LangChainInstrumentor
from langchain.schema import HumanMessage, AIMessage
from typing import List, Dict

# Import additional models
from langchain_anthropic.chat_models import ChatAnthropic
from langchain_openai.chat_models import ChatOpenAI

class LLMChat:
    def __init__(self, model_type: str, model_id: str, tracer_provider=None):
        # Instrument LangChain
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
          
        # Initialize the appropriate chat model based on model_type
        if model_type == "bedrock":
            self.model = ChatBedrockConverse(
                model=model_id,
                temperature=float(os.environ.get("TEMPERATURE", 0.3)),
                max_tokens=int(os.environ.get("MAX_TOKENS", 4096)),
            )
        elif model_type == "anthropic":
            self.model = ChatAnthropic(
                model=model_id,
                temperature=float(os.environ.get("TEMPERATURE", 0.3)),
                max_tokens=int(os.environ.get("MAX_TOKENS", 4096)),
            )
        elif model_type == "openai":
            self.model = ChatOpenAI(
                model=model_id,
                temperature=float(os.environ.get("TEMPERATURE", 0.3)),
                max_tokens=int(os.environ.get("MAX_TOKENS", 4096)),
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    def chat(self, prompt: str, message_history: List[Dict] = None) -> Dict:
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
            # Get response from the model using invoke instead of direct call
            response = self.model.invoke(messages)
            assistant_message = response.content

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
        
# Test the LLMChat class
if __name__ == "__main__":
    # basic test
    try:
        print("Start LLM Chat Test")
        
        # create chat instance
        chat = LLMChat(model_type="bedrock", model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0")
        
        # test prompt
        test_prompt = "Hello, this is a test message. My name is yunwoong."
        print(f"\nFirst Prompt: {test_prompt}")
        
        # response test
        response = chat.chat(test_prompt)
        print(f"First Response: {response['response']}")
        
        # second prompt test (remembering the previous message)
        second_prompt = "What did I say in my name?"
        print(f"\nSecond Prompt: {second_prompt}")

        response2 = chat.chat(second_prompt, response['message_history'])
        print(f"Second Response: {response2['response']}")
        print(f"\nTest Passed ✅")
        
    except Exception as e:
        print(f"\nTest Failed ❌: {str(e)}")
