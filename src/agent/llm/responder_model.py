from src.agent.llm.llm_interface import LLMChat
from config import CONFIG

class ResponderModel(LLMChat):
    def __init__(self, tracer_provider=None):
        super().__init__(CONFIG['response_model']['model_type'], CONFIG['response_model']['model_id'], tracer_provider)