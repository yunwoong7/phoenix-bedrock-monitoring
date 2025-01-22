import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from typing import Dict, List
from src.agent.llm.llm_interface import LLMChat
from config import CONFIG

class ResponderModel(LLMChat):
    def __init__(self, tracer_provider=None):
        super().__init__(
            CONFIG['response_model']['model_type'],
            CONFIG['response_model']['model_id'],
            tracer_provider
        )
    
    async def stream_response(self, prompt: str, message_history: List[Dict] = None):
        async for token in self.stream_chat(prompt, message_history):
            yield token