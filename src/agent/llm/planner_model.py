from src.agent.llm.llm_interface import LLMChat
from config import CONFIG

class PlannerModel(LLMChat):
    def __init__(self, tracer_provider=None):
        super().__init__(CONFIG['planner_model']['model_type'], CONFIG['planner_model']['model_id'], tracer_provider)
