# src/agent/llm/planner_model.py
from src.agent.llm.llm_interface import LLMChat
from config import CONFIG

class PlannerModel(LLMChat):
    def __init__(self, tracer_provider=None):
        super().__init__(CONFIG['planner_model']['model_type'], CONFIG['planner_model']['model_id'], tracer_provider)
    
    async def stream_plan(self, prompt: str, message_history=None):
        async for token in self.stream_chat(prompt, message_history):
            yield token